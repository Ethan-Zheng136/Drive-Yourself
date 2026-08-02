"""test_sft_grad.py -- single-process (no-FSDP) unit test of the full-FT decoder swap.

This box has 1 MIG slice + NVML/vision restriction, so FSDP multi-GPU can't run here.
This test verifies the model/loss/trainable-flags logic WITHOUT FSDP, on the text-only
path (no pixel_values -> vision tower not invoked, which also dodges the NVML issue):

  (a) the FMHead flow loss computes,
  (b) gradients reach the LM backbone (some vlm.model param has non-None, nonzero grad),
  (c) gradients reach the FMHead,
  (d) vlm.visual stays frozen (requires_grad False, grad None/zero),
  (e) loss decreases over a few steps.

Memory note: to avoid holding AdamW state for the 3B LM on one MIG slice, the test
optimizer steps ONLY the FMHead; the LM backbone stays requires_grad=True so its
gradients ARE computed and verified (b) -- the real 8-GPU FSDP run trains the LM too.

Usage: OMP_NUM_THREADS=8 FMHEAD_TEXT_ONLY=1 python test_sft_grad.py
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

os.chdir(AUTOVLA_ROOT)  # codebook_cache_path is repo-relative


def grad_stats(params):
    n_with, n_total, tot = 0, 0, 0.0
    for p in params:
        n_total += 1
        if p.grad is not None:
            g = float(p.grad.detach().abs().sum())
            tot += g
            if g > 0:
                n_with += 1
    return n_with, n_total, tot


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(os.path.join(FMHEAD_DIR, "config/fmhead-sft-full.yaml"))
    # test-only: don't warm-start the 7GB base ckpt (subclass __init__ never does)
    cfg["model"]["sft_model_path"] = None
    torch.manual_seed(0)

    model = FMHeadSFTAutoVLA(cfg).to(device)
    model.train()

    n_lm = sum(p.numel() for p in model.autovla.vlm.model.parameters())
    n_lm_tr = sum(p.numel() for p in model.autovla.vlm.model.parameters() if p.requires_grad)
    n_vis = sum(p.numel() for p in model.autovla.vlm.visual.parameters())
    n_vis_tr = sum(p.numel() for p in model.autovla.vlm.visual.parameters() if p.requires_grad)
    n_fm_tr = sum(p.numel() for p in model.fm_decoder.parameters() if p.requires_grad)
    print(f"[params] LM backbone: {n_lm:,} ({n_lm_tr:,} trainable) | "
          f"vision: {n_vis:,} ({n_vis_tr:,} trainable) | FMHead trainable: {n_fm_tr:,}", flush=True)

    dctx = DistCtx()
    batches = make_textonly_batches(model.autovla, n=2, device=device, dctx=dctx,
                                    dataset_dir=cfg["data"]["train"]["json_dataset_path"][0])

    # optimize ONLY the FMHead (memory); LM grads still computed + checked
    opt = torch.optim.AdamW([p for p in model.fm_decoder.parameters() if p.requires_grad], lr=1e-3)

    losses = []
    lm_grad = fm_grad = vis_grad = None
    n_steps = 40
    for step in range(n_steps):
        b = batches[step % len(batches)]
        model.zero_grad(set_to_none=True)
        loss = model._fm_loss(b)
        loss.backward()
        # Capture grad flow on the LAST step. NOTE: FMHead uses AdaLN-Zero init, so the
        # gradient to the conditioning c (hence to the LM backbone) is exactly 0 on the
        # FIRST step (all c->loss paths pass through zero-init adaLN/final weights). Once
        # the head warms up (a few steps), those weights are nonzero and grad flows to
        # c -> LM backbone. So we assert LM grad AFTER warmup, not at init.
        if step == n_steps - 1:
            lm_grad = grad_stats(model.autovla.vlm.model.parameters())
            fm_grad = grad_stats(model.fm_decoder.parameters())
            vis_grad = grad_stats(model.autovla.vlm.visual.parameters())
        opt.step()
        losses.append(float(loss.item()))

    init_l, fin_l = float(np.mean(losses[:5])), float(np.mean(losses[-5:]))
    a_ok = np.isfinite(losses).all()
    b_ok = lm_grad[0] > 0          # LM backbone received grad
    c_ok = fm_grad[0] > 0          # FMHead received grad
    d_ok = (n_vis_tr == 0) and (vis_grad[2] == 0.0)  # vision frozen (no trainable, no grad)
    e_ok = fin_l < init_l

    print("\n=== FULL-FT DECODER-SWAP GRAD TEST ===", flush=True)
    print(f"(a) flow loss computes           : {'PASS' if a_ok else 'FAIL'} (init={init_l:.4f})")
    print(f"(b) grad reaches LM backbone      : {'PASS' if b_ok else 'FAIL'} "
          f"({lm_grad[0]}/{lm_grad[1]} params, sum|g|={lm_grad[2]:.3e})")
    print(f"(c) grad reaches FMHead           : {'PASS' if c_ok else 'FAIL'} "
          f"({fm_grad[0]}/{fm_grad[1]} params, sum|g|={fm_grad[2]:.3e})")
    print(f"(d) vision frozen (0 trainable/grad): {'PASS' if d_ok else 'FAIL'} "
          f"(trainable={n_vis_tr}, sum|g|={vis_grad[2]:.3e})")
    print(f"(e) loss decreases                : {'PASS' if e_ok else 'FAIL'} "
          f"({init_l:.4f} -> {fin_l:.4f})")
    ok = bool(a_ok and b_ok and c_ok and d_ok and e_ok)
    res = {"loss_init": init_l, "loss_final": fin_l,
           "lm_params_with_grad": lm_grad[0], "lm_grad_sum": lm_grad[2],
           "fm_params_with_grad": fm_grad[0], "fm_grad_sum": fm_grad[2],
           "vision_trainable": n_vis_tr, "vision_grad_sum": vis_grad[2],
           "lm_trainable_params": n_lm_tr, "fm_trainable_params": n_fm_tr,
           "checks": {"a_loss": bool(a_ok), "b_lm_grad": bool(b_ok), "c_fm_grad": bool(c_ok),
                      "d_vision_frozen": bool(d_ok), "e_loss_down": bool(e_ok)},
           "all_pass": ok}
    json.dump(res, open(os.path.join(FMHEAD_DIR, "sft_grad_test_results.json"), "w"), indent=2)
    print(f"ALL: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
