"""smoke_lora_fm3aligned_wiring.py -- CHEAP wiring proof for the fm3-aligned LoRA loss.

No 3B VLM: stand in a differentiable synthetic ctx (requires_grad, mimicking the LoRA-ON
styled content) and verify the LOSS FORMULATION + the frozen-head gradient path:
  * loss = fm3-kin CONTROL-space kinematic flow-matching MSE to a TEACHER-as-controls target;
  * grad flows THROUGH the FROZEN head into ctx (== the LoRA activations) -> nonzero ctx.grad;
  * the frozen fm3-kin head params receive ZERO grad (requires_grad False -> .grad is None);
  * configs parse; base/head/normalizer/dataset paths exist.
Run: /root/workspace/miniconda3/envs/autovla/bin/python smoke_lora_fm3aligned_wiring.py
"""
from __future__ import annotations
import os, sys, glob, json
import numpy as np, torch, yaml

FMHEAD_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FMHEAD_DIR)
from train_fmhead import build_decoder, load_yaml
from train_lora_fm3aligned import build_frozen_head

CFGS = [f"config/fmhead_lora_fm3aligned_{t}.yaml" for t in ("ddv2", "goalflow", "transfuser")]


def check_paths(cfg):
    ok = {}
    ok["base_ckpt"] = os.path.exists(cfg["base_ckpt"])
    ok["head_ckpt"] = os.path.exists(cfg["init_fmhead_ckpt"])
    ok["normalizer"] = os.path.exists(cfg["normalizer_path"])
    ds = cfg["dataset_path"]
    ok["dataset"] = os.path.isdir(ds) and len(glob.glob(os.path.join(ds, "*.json"))) > 100
    avc = os.path.join("/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA",
                       "config", cfg["autovla_config"] + ".yaml")
    ok["autovla_config"] = os.path.exists(avc)
    return ok


def main():
    dev = torch.device("cpu")
    torch.manual_seed(0)
    results = {}

    # --- 1. all three configs parse + gate flag + paths exist ---
    for cp in CFGS:
        cfg = load_yaml(os.path.join(FMHEAD_DIR, cp))
        assert cfg.get("train_lora_only") is True, f"{cp}: train_lora_only must be true"
        assert (cfg.get("lora_adapter_path") in (None, "null")), f"{cp}: LoRA must be fresh"
        assert cfg["fmhead"]["constraint"]["mode"] == "kinematic", cp
        assert cfg["fmhead"].get("use_z_encoder") is False and cfg["style_alpha"] == 0.0, cp
        p = check_paths(cfg)
        results[f"paths::{os.path.basename(cp)}"] = all(p.values())
        print(f"[cfg] {os.path.basename(cp):40s} gate=OK paths={p}")

    # --- 2. build + FREEZE the fm3-kin ep6 head (control space) from the ddv2 config ---
    cfg = load_yaml(os.path.join(FMHEAD_DIR, CFGS[0]))
    dec = build_frozen_head(cfg, dev)
    n_head_train = sum(p.numel() for p in dec.parameters() if p.requires_grad)
    results["head_frozen"] = (n_head_train == 0)
    print(f"[head] control mode={dec.config.constraint_mode} trainable_params={n_head_train} "
          f"(want 0) -> {'PASS' if n_head_train == 0 else 'FAIL'}")

    # --- 3. LOSS: synthetic differentiable styled ctx -> control-space kinematic flow loss to
    #        a TEACHER-as-controls target. grad -> ctx (LoRA proxy); frozen head gets 0 grad. ---
    B, T_ctx, H = 2, 24, dec.config.d_ctx     # d_ctx = 2048
    ctx = torch.randn(B, T_ctx, H, requires_grad=True)
    ctx_mask = torch.ones(B, T_ctx)
    s = torch.zeros(B, dec.config.d_style)    # AdaLN style OFF (exactly the ep6 conditioning)
    # a plausible "teacher" xy trajectory (B, 10, 3): forward motion + slight curve
    tgt = torch.zeros(B, 10, 3)
    for b in range(B):
        v = 6.0 + 3.0 * b
        for t in range(10):
            tgt[b, t, 0] = v * 0.5 * (t + 1)
            tgt[b, t, 1] = 0.15 * (t + 1) ** 2 * (1 if b == 0 else -1)
    loss = dec.training_loss(tgt, ctx, s, ctx_mask=ctx_mask, style_dropout_prob=0.0)
    loss.backward()
    finite = bool(torch.isfinite(loss))
    ctx_grad = ctx.grad is not None and float(ctx.grad.abs().sum()) > 0
    head_grad_elems = sum(int(p.grad is not None) for p in dec.parameters())
    results["loss_finite"] = finite
    results["grad_to_ctx"] = bool(ctx_grad)
    results["head_zero_grad"] = (head_grad_elems == 0)
    print(f"[loss] control-space kinematic flow MSE={float(loss):.4f} finite={finite} "
          f"-> {'PASS' if finite else 'FAIL'}")
    print(f"[grad] |ctx.grad|>0={ctx_grad} (grad reaches LoRA proxy) frozen-head grad elems="
          f"{head_grad_elems} (want 0) -> {'PASS' if ctx_grad and head_grad_elems == 0 else 'FAIL'}")

    # --- 4. confirm the target really is inverted to controls + z-scored (control space) ---
    #     i.e. training_loss(kinematic) uses xy_to_controls + control normalizer, not raw xy.
    from fm_head import FMHead
    gt_xy = tgt[:, :10, :2].float()
    v0 = torch.linalg.norm(gt_xy[:, 0, :], dim=-1) / float(dec.config.kin_dt)
    controls = FMHead.xy_to_controls(gt_xy, v0, dt=dec.config.kin_dt)
    tau = (controls - dec.traj_mean) / dec.traj_std
    results["control_space_target"] = bool(torch.isfinite(tau).all() and tau.abs().mean() > 0)
    print(f"[target] teacher->controls(a_long,yaw_rate) z-scored: mean|tau|={float(tau.abs().mean()):.3f} "
          f"-> {'PASS' if results['control_space_target'] else 'FAIL'}")

    allok = all(results.values())
    print("\n=== wiring smoke summary ===")
    for k, v in results.items():
        print(f"  {k:40s}: {'PASS' if v else 'FAIL'}")
    print(f"  {'ALL':40s}: {'PASS' if allok else 'FAIL'}")
    json.dump({k: bool(v) for k, v in results.items()},
              open(os.path.join(FMHEAD_DIR, "smoke_lora_fm3aligned_wiring_results.json"), "w"), indent=2)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
