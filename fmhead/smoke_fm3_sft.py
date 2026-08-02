"""smoke_fm3_sft.py -- exercise the REAL fmhead_sft training path for both fm3 variants.

Runs FMHeadSFTAutoVLA._fm_loss (the exact training-step loss used by the 8-GPU launcher)
on the REAL Qwen2.5-VL-3B (text-only, single process -- this box is 1 MIG slice with NVML/
vision restrictions, same limitation test_sft_grad.py documents) for BOTH new configs. Proves
the constraint plumbing (soft penalty & kinematic GT->control inversion) runs end-to-end with
genuine VLM hidden states: no dtype/shape errors, loss finite + decreasing, grad to FMHead.

FSDP note: the FSDP FULL_SHARD wrapping / mixed-precision / use_orig_params logic in
fmhead_sft.build_and_fit is UNCHANGED by fm3 (only the head's loss/decode differ); it is the
same harness that produced the GOLD 0.9088 head. Multi-GPU FSDP is validated by that harness;
here we validate the fm3 loss path with the real VLM.

Usage:
  PYTHONPATH=...AutoVLA:...AutoVLA/navsim:...fmhead \
    OMP_NUM_THREADS=8 FMHEAD_TEXT_ONLY=1 \
    /root/workspace/miniconda3/envs/autovla/bin/python smoke_fm3_sft.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

FMHEAD_DIR = os.path.dirname(os.path.abspath(__file__))
if FMHEAD_DIR not in sys.path:
    sys.path.insert(0, FMHEAD_DIR)

from fmhead_sft import AUTOVLA_ROOT, FMHeadSFTAutoVLA, load_config  # noqa: E402
from train_fmhead import DistCtx, make_textonly_batches  # noqa: E402

os.chdir(AUTOVLA_ROOT)


def run_variant(config_name, device):
    cfg = load_config(os.path.join(FMHEAD_DIR, "config", config_name))
    cfg["model"]["sft_model_path"] = None          # smoke: skip the 7GB base warm-start
    torch.manual_seed(0)
    model = FMHeadSFTAutoVLA(cfg).to(device)
    model.train()
    mode = model.fm_decoder.config.constraint_mode

    dctx = DistCtx()
    batches = make_textonly_batches(model.autovla, n=2, device=device, dctx=dctx,
                                    dataset_dir=cfg["data"]["train"]["json_dataset_path"][0])
    def pure_flow_loss(b):
        """FLOW-only loss (penalty weights zeroed) -> isolates the data-fitting signal from
        the additive smoothness bias, so 'loss decreases' is meaningful for the soft variant."""
        c = model.fm_decoder.config
        jw, cw = c.jerk_weight, c.curv_weight
        c.jerk_weight, c.curv_weight = 0.0, 0.0
        with torch.no_grad():
            l = float(model._fm_loss(b))
        c.jerk_weight, c.curv_weight = jw, cw
        return l

    flow0 = float(np.mean([pure_flow_loss(b) for b in batches]))   # before training
    opt = torch.optim.AdamW([p for p in model.fm_decoder.parameters() if p.requires_grad], lr=1e-3)
    losses = []
    n_steps = 80
    for step in range(n_steps):
        b = batches[step % len(batches)]
        model.zero_grad(set_to_none=True)
        loss = model._fm_loss(b)            # combined (flow + fm3 penalty)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    flow1 = float(np.mean([pure_flow_loss(b) for b in batches]))   # after training
    fm_grad = int(sum(int(p.grad is not None and p.grad.abs().sum() > 0)
                      for p in model.fm_decoder.parameters()))
    comb0, comb1 = float(np.mean(losses[:5])), float(np.mean(losses[-5:]))
    finite = bool(np.isfinite(losses).all() and np.isfinite([flow0, flow1]).all())
    # requirement (b): the TRAINING loss (combined objective the trainer minimizes) decreases.
    down = bool(comb1 < comb0)
    ok = bool(finite and down and fm_grad > 0)
    print(f"[{config_name}] mode={mode} | TRAIN-loss {comb0:.4f}->{comb1:.4f} (down={down}) "
          f"| flow-only {flow0:.4f}->{flow1:.4f} finite={finite} "
          f"fm_grad_params={fm_grad} -> {'PASS' if ok else 'FAIL'}", flush=True)
    del model
    torch.cuda.empty_cache()
    return {"config": config_name, "mode": mode, "train_loss_init": comb0,
            "train_loss_final": comb1, "flow_only_init": flow0, "flow_only_final": flow1,
            "finite": finite, "train_loss_down": down, "fm_grad_params": fm_grad, "pass": ok}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "smoke wants GPU"
    os.environ.setdefault("FMHEAD_TEXT_ONLY", "1")
    res = {}
    for cn in ("fmhead_fm3_lite.yaml", "fmhead_fm3_kin.yaml"):
        res[cn] = run_variant(cn, device)
    ok = all(v["pass"] for v in res.values())
    json.dump(res, open(os.path.join(FMHEAD_DIR, "smoke_fm3_sft_results.json"), "w"), indent=2)
    print(f"\n=== fm3 SFT-PATH SMOKE: {'PASS' if ok else 'FAIL'} -> smoke_fm3_sft_results.json ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
