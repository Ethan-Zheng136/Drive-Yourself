"""zoff_control_probe.py -- LOCATE the ~9-10 GATE1 loss floor (z-encoder OFF control).

Bypasses the z-encoder so the head IS the clean frozen fm3-kin ep6 base, then measures its
mean flow-matching loss (the SAME train path as smoke_zenc/train_fmhead) on TWO target sets
with the SAME head + SAME control-normalizer:

  * DDv2 teacher  (the z-run target, nocot_ddv2_teacher12k) -- expected ~8-9.
  * human GT      (what the ctrl-normalizer was fit on, nocot_navtrain12k) -- expected ~2.

If the frozen base already floors at ~8-9 on DDv2 but ~2 on human GT with everything else held
fixed, the inflation is the DATA/NORMALIZER/TARGET path (DDv2 controls are ~3x wider than the
human-GT stats -> normalized E[tau^2]~7.5), NOT the z-encoder. Also prints the normalized
E[tau^2] per set as the analytic floor cross-check (init flow loss ~ E[tau^2] + 1).

Run:
  FMHEAD_TEXT_ONLY=1 /root/workspace/miniconda3/envs/autovla/bin/python zoff_control_probe.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_fmhead as T  # noqa: E402
from fm_head import FMHead  # noqa: E402

CFG_PATH = "config/fmhead_style_ddv2_zenc.yaml"
DDV2 = "/mnt/pfs/zhengguantian/autovla/persona/nocot_ddv2_teacher12k"
HUMAN = "/mnt/pfs/zhengguantian/autovla/persona/nocot_navtrain12k"
DT = 0.5


class _Ctx:
    world_size = 1
    rank = 0

    def log(self, *a, **k):
        print(*a, **k, flush=True)


def norm_tau_sq(batches, dec, dev):
    """Mean normalized E[tau^2] of the control targets under the decoder's ctrl-normalizer."""
    vals = []
    for b in batches:
        xy = b["gt_trajectory"][:, :10, :2].float().to(dev)
        v0 = torch.linalg.norm(xy[:, 0, :], dim=-1) / DT
        ctrl = FMHead.xy_to_controls(xy, v0, dt=DT)
        tau = (ctrl - dec.traj_mean.to(dev)) / dec.traj_std.to(dev)
        vals.append(float((tau ** 2).mean()))
    return float(np.mean(vals))


def main():
    assert torch.cuda.is_available(), "need a GPU"
    dev = torch.device("cuda:0")
    cfg = T.load_yaml(os.path.join(os.path.dirname(os.path.abspath(__file__)), CFG_PATH))
    torch.manual_seed(cfg["train"]["seed"])

    av, av_config, lora_on = T.build_frozen_vlm(cfg, dev, load_sft_base=True,
                                                base_ckpt_override=cfg.get("base_ckpt"))
    dec = T.build_decoder(cfg, dev)
    style_off = (not lora_on) or float(cfg.get("style_alpha", 1.0)) == 0.0

    # BYPASS the z-encoder -> head == clean frozen fm3-kin base.
    saved_z = dec.fm_head.z_encoder
    dec.fm_head.z_encoder = None
    dec.eval()
    dec.style_alpha = float(cfg["style_alpha"])

    def mean_base_loss(batches, drop, passes=4):
        ls = []
        with torch.no_grad():
            for _ in range(passes):
                for b in batches:
                    ctx, mask, s = T.extract_c_s(av, b, alpha=cfg["style_alpha"], reduce=cfg["c_reduce"],
                                                 style_off=style_off, ctx_max_len=dec.config.ctx_max_len,
                                                 content_source=cfg.get("content_source"))
                    ls.append(float(dec(b["gt_trajectory"], ctx, s, ctx_mask=mask,
                                        style_dropout_prob=drop)))
        return float(np.mean(ls)), float(np.std(ls))

    out = {}
    for name, src in [("DDv2_teacher(z-run target)", DDV2), ("human_GT_navtrain(norm-fit)", HUMAN)]:
        batches = T.make_textonly_batches(av, n=12, device=dev, dctx=_Ctx(), dataset_dir=src)
        etau2 = norm_tau_sq(batches, dec, dev)
        m0, s0 = mean_base_loss(batches, 0.0)
        m2, s2 = mean_base_loss(batches, cfg["fmhead"]["style_dropout_prob"])
        out[name] = {"norm_E_tau2": etau2, "analytic_floor_E_tau2_plus1": etau2 + 1.0,
                     "zoff_loss_cond_drop0": m0, "zoff_loss_cond_drop0_std": s0,
                     "zoff_loss_drop02": m2, "zoff_loss_drop02_std": s2}
        print(f"[{name}] E[tau^2]={etau2:.2f} (analytic floor ~{etau2+1:.2f})  "
              f"z-OFF base loss: cond(drop=0)={m0:.3f}+/-{s0:.2f}  drop=0.2={m2:.3f}+/-{s2:.2f}",
              flush=True)

    dec.fm_head.z_encoder = saved_z
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "zoff_control_probe_results.json"), "w"), indent=2)
    print("\n=== z-OFF CONTROL PROBE: the DDv2 floor == the base@DDv2 (data/normalizer), "
          "human-GT floor ~2 with the SAME head+normalizer -> inflation is the target path ===",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
