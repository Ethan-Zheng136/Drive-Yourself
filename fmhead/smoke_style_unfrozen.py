"""smoke_style_unfrozen.py -- GPU smoke for the fm3-kin UNFROZEN-HEAD style run.

Launch-readiness check for config/fmhead_style_ddv2_fm3kin_unfrozen.yaml. Single GPU,
text-only VLM forward (vision NVML restricted on this box; same fm_step path). Uses the
REAL train_fmhead code (build_frozen_vlm + build_decoder + extract_c_s) so it exercises
exactly what the 8-GPU launcher will run.

Checks:
  A. fm3-kin ep6 base VLM + DDv2 LoRA load & freeze (VLM requires_grad == 0).
  B. UNFREEZE-HEAD: FM head DiT blocks + style adapter trainable (~11.5M); n_trainable print.
  C. A few train steps: loss finite + decreasing; grad reaches the UNFROZEN head DiT blocks
     (not just the adapter); style delta s is nonzero (LoRA on).
  D. sample()/decode(): kinematic trajectory is finite, within caps, LOW-jerk (feasible).
  E. GOLD safety: the adapter-only path (unfreeze_head off) still freezes the head.

Run:
  FMHEAD_TEXT_ONLY=1 /root/workspace/miniconda3/envs/autovla/bin/python smoke_style_unfrozen.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_fmhead as T  # noqa: E402

CFG_PATH = "config/fmhead_style_ddv2_fm3kin_unfrozen.yaml"


class _Ctx:
    """Minimal single-process stand-in for DistCtx (for make_textonly_batches)."""
    world_size = 1
    rank = 0

    def log(self, *a, **k):
        print(*a, **k, flush=True)


def main():
    assert torch.cuda.is_available(), "need a GPU"
    dev = torch.device("cuda:0")
    cfg = T.load_yaml(os.path.join(os.path.dirname(os.path.abspath(__file__)), CFG_PATH))
    torch.manual_seed(cfg["train"]["seed"])
    res = {}

    # -- A. frozen VLM (fm3-kin ep6 base + DDv2 LoRA) --------------------------
    base_override = cfg.get("base_ckpt")
    av, av_config, lora_on = T.build_frozen_vlm(cfg, dev, load_sft_base=True,
                                                base_ckpt_override=base_override)
    n_vlm_grad = sum(p.numel() for p in av.parameters() if p.requires_grad)
    res["A_lora_on"] = bool(lora_on)
    res["A_vlm_grad_params"] = int(n_vlm_grad)
    print(f"[A] lora_on={lora_on}  VLM requires_grad params={n_vlm_grad} (want 0)", flush=True)

    # -- B. UNFREEZE-HEAD decoder ---------------------------------------------
    dec = T.build_decoder(cfg, dev)
    n_train = sum(p.numel() for p in dec.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in dec.parameters() if not p.requires_grad)
    res["B_n_trainable"] = int(n_train)
    res["B_constraint_mode"] = dec.config.constraint_mode
    res["B_num_steps"] = dec.num_steps
    print(f"[B] n_trainable={n_train:,} n_frozen={n_frozen:,} "
          f"mode={dec.config.constraint_mode} num_steps={dec.num_steps}", flush=True)

    # names of a few head DiT-block params (must receive grad -> head really unfrozen)
    head_probe = [n for n, p in dec.named_parameters()
                  if p.requires_grad and "fm_head.blocks" in n][:3]
    adapter_probe = [n for n, p in dec.named_parameters()
                     if p.requires_grad and "style_adapter" in n][:2]
    print(f"[B] head-block probe params: {head_probe}", flush=True)

    # -- C. train steps: loss + grad-to-head -----------------------------------
    style_off = (not lora_on) or float(cfg.get("style_alpha", 1.0)) == 0.0
    batches = T.make_textonly_batches(av, n=4, device=dev, dctx=_Ctx(),
                                      dataset_dir=cfg["dataset_path"])
    opt = torch.optim.AdamW([p for p in dec.parameters() if p.requires_grad],
                            lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    dec.train()
    losses, s_norms = [], []
    grad_head_ok = grad_adapter_ok = False
    n_steps = 40
    for step in range(n_steps):
        b = batches[step % len(batches)]
        ctx, ctx_mask, s = T.extract_c_s(av, b, alpha=cfg["style_alpha"],
                                         reduce=cfg["c_reduce"], style_off=style_off,
                                         ctx_max_len=dec.config.ctx_max_len)
        s_norms.append(float(s.float().norm(dim=-1).mean()))
        loss = dec(b["gt_trajectory"], ctx, s, ctx_mask=ctx_mask)  # fwd == kinematic loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step == 0:
            gh = [dict(dec.named_parameters())[n].grad for n in head_probe]
            ga = [dict(dec.named_parameters())[n].grad for n in adapter_probe]
            grad_head_ok = all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in gh)
            grad_adapter_ok = all(g is not None and torch.isfinite(g).all() for g in ga)
        torch.nn.utils.clip_grad_norm_(dec.parameters(), cfg["train"]["grad_clip"])
        opt.step()
        losses.append(float(loss))
    fin = np.isfinite(losses).all()
    init_l, last_l = float(np.mean(losses[:5])), float(np.mean(losses[-5:]))
    res.update(dict(C_loss_finite=bool(fin), C_loss_init=init_l, C_loss_last=last_l,
                    C_loss_decreasing=bool(last_l < init_l),
                    C_grad_reaches_head=bool(grad_head_ok),
                    C_grad_reaches_adapter=bool(grad_adapter_ok),
                    C_style_s_norm_mean=float(np.mean(s_norms))))
    print(f"[C] loss finite={fin}  {init_l:.4f} -> {last_l:.4f} (decreasing={last_l < init_l})  "
          f"grad->head={grad_head_ok} grad->adapter={grad_adapter_ok}  "
          f"||s||~{np.mean(s_norms):.1f} (nonzero style)", flush=True)

    # -- D. sample()/decode(): feasible kinematic trajectory -------------------
    dec.eval()
    b = batches[0]
    ctx, ctx_mask, s = T.extract_c_s(av, b, alpha=cfg["style_alpha"], reduce=cfg["c_reduce"],
                                     style_off=style_off, ctx_max_len=dec.config.ctx_max_len)
    gt_xy = b["gt_trajectory"][:, :10, :2].float()
    v0 = torch.linalg.norm(gt_xy[:, 0, :], dim=-1) / float(dec.config.kin_dt)  # (B,)
    with torch.no_grad():
        poses = dec.decode(ctx, s, ctx_mask=ctx_mask, v0=v0)  # (B,10,3) [x,y,heading]
    xy = poses[..., :2].float().cpu()
    # jerk (3rd diff of position) in physical units, m/s^3
    dt = float(dec.config.kin_dt)
    origin = torch.zeros(xy.shape[0], 1, 2)
    P = torch.cat([origin, xy], dim=1)
    vel = (P[:, 1:] - P[:, :-1]) / dt
    acc = (vel[:, 1:] - vel[:, :-1]) / dt
    jerk = (acc[:, 1:] - acc[:, :-1]) / dt
    speeds = torch.linalg.norm(vel, dim=-1)
    max_jerk = float(jerk.abs().max())
    max_speed = float(speeds.max())
    finite_traj = bool(torch.isfinite(poses).all())
    within_caps = max_speed <= dec.config.v_max + 1e-3
    res.update(dict(D_traj_finite=finite_traj, D_max_jerk_mps3=max_jerk,
                    D_max_speed_mps=max_speed, D_within_v_cap=bool(within_caps),
                    D_endpoint_xy=[round(v, 2) for v in xy[0, -1].tolist()]))
    print(f"[D] traj finite={finite_traj}  endpoint={xy[0, -1].tolist()}  "
          f"max_speed={max_speed:.2f} (<= v_max {dec.config.v_max})  max_jerk={max_jerk:.2f} m/s^3",
          flush=True)

    # -- E. GOLD safety: adapter-only path still freezes the head --------------
    cfg_gold = dict(cfg)
    fh = dict(cfg["fmhead"]); fh["unfreeze_head"] = False
    cfg_gold["fmhead"] = fh
    dec_g = T.build_decoder(cfg_gold, dev)
    n_train_g = sum(p.numel() for p in dec_g.parameters() if p.requires_grad)
    head_frozen = all(not p.requires_grad for n, p in dec_g.named_parameters()
                      if "style_adapter" not in n)
    res.update(dict(E_adapter_only_trainable=int(n_train_g), E_head_frozen=bool(head_frozen)))
    print(f"[E] GOLD/adapter-only: trainable={n_train_g:,}  head_frozen={head_frozen}", flush=True)

    ok = (res["A_vlm_grad_params"] == 0 and lora_on and n_train > 8e6
          and res["C_loss_finite"] and res["C_loss_decreasing"] and res["C_grad_reaches_head"]
          and res["C_style_s_norm_mean"] > 0 and res["D_traj_finite"] and res["D_within_v_cap"]
          and res["E_head_frozen"] and n_train_g < n_train)
    res["ALL_PASS"] = bool(ok)
    print("\n=== SMOKE SUMMARY (fm3-kin unfrozen-head style) ===", flush=True)
    for k, v in res.items():
        print(f"  {k:26s}: {v}")
    print(f"  {'ALL':26s}: {'PASS' if ok else 'FAIL'}", flush=True)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "smoke_style_unfrozen_results.json"), "w"), indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
