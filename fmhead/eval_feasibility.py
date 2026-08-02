"""eval_feasibility.py -- decoded-trajectory-vs-GT L2 comparison: FMHead vs codebook.

Feasibility proof (part a): under the SAME frozen base AutoVLA weights, compare the
FMHead decoder against the codebook/token decoder on decoded-trajectory-vs-human-GT
L2 (ADE/FDE) over a held-out navtrain slice. (Part b, navtest PDMS, is produced by
`run_pdms_fmhead.sh` + `eval_persona.sh` — see README §12.)

Metrics per scene (metres, in the (x_forward, y_left) frame):
  * ADE = mean over the 10 waypoints of ||pred - gt||
  * FDE = ||pred[-1] - gt[-1]||  (endpoint)
FMHead draws N samples; we report:
  * FMHead-select : the select_by_score pick (deployment behaviour, s=0)
  * FMHead-mean   : mean over the N samples (central tendency)
  * FMHead-oracle : min over N (upper bound / multimodality headroom)
  * Codebook      : AutoVLA.predict() argmax decode (baseline)

Usage (real vision node, after training):
  python eval_feasibility.py --config config/fmhead_gt_feasibility.yaml \
     --ckpt /mnt/pfs/.../fmhead_ckpts_gt/<run>/fmhead_final.pt --n 500
Smoke on this box (text-only, structural):
  FMHEAD_TEXT_ONLY=1 python eval_feasibility.py --config config/fmhead_gt_feasibility.yaml \
     --ckpt <smoke ckpt> --n 4 --smoke --no_sft_base --no_codebook
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

FMHEAD_DIR = os.path.dirname(os.path.abspath(__file__))
if FMHEAD_DIR not in sys.path:
    sys.path.insert(0, FMHEAD_DIR)

from train_fmhead import (  # noqa: E402  (reuse the training helpers, no duplication)
    build_decoder, build_frozen_vlm, extract_c_s, load_yaml, make_textonly_batches,
    DistCtx,
)


def ade_fde(pred_xy: np.ndarray, gt_xy: np.ndarray):
    """pred_xy,(T,2) gt_xy,(T,2) -> (ADE, FDE) in metres."""
    d = np.linalg.norm(pred_xy - gt_xy, axis=-1)
    return float(d.mean()), float(d[-1])


@torch.no_grad()
def eval_slice(av, decoder, batches, style_off, alpha, c_reduce, device,
               num_samples, num_steps, do_codebook):
    """Return per-decoder ADE/FDE lists over the given batches."""
    T = decoder.config.horizon
    out = {"fmhead_select": {"ade": [], "fde": []},
           "fmhead_mean":   {"ade": [], "fde": []},
           "fmhead_oracle": {"ade": [], "fde": []},
           "codebook":      {"ade": [], "fde": []}}
    for b in batches:
        gt = b["gt_trajectory"][0, :T, :2].float().cpu().numpy()             # (T,2) metres
        ctx, ctx_mask, s = extract_c_s(av, b, alpha=alpha, reduce=c_reduce, style_off=style_off,
                                       ctx_max_len=decoder.config.ctx_max_len)
        # FMHead: N candidates (normalized -> metres inside decode's denorm on trajs)
        trajs = decoder.fm_head.sample(ctx, s, ctx_mask=ctx_mask, num_samples=num_samples,
                                       num_steps=num_steps, cfg_weight=decoder.cfg_weight)  # (1,N,T,2) norm
        trajs_m = decoder._denormalize(trajs)[0].float().cpu().numpy()         # (N,T,2) metres
        # select pick (deployment)
        best_xy = decoder.decode(ctx, s, ctx_mask=ctx_mask)[0, :, :2].float().cpu().numpy()  # (T,2) metres
        a, f = ade_fde(best_xy, gt); out["fmhead_select"]["ade"].append(a); out["fmhead_select"]["fde"].append(f)
        # mean over N
        a, f = ade_fde(trajs_m.mean(0), gt); out["fmhead_mean"]["ade"].append(a); out["fmhead_mean"]["fde"].append(f)
        # oracle: min-ADE over N
        ades = [ade_fde(trajs_m[k], gt)[0] for k in range(trajs_m.shape[0])]
        ki = int(np.argmin(ades)); a, f = ade_fde(trajs_m[ki], gt)
        out["fmhead_oracle"]["ade"].append(a); out["fmhead_oracle"]["fde"].append(f)
        # codebook baseline (real vision node only; needs generate())
        if do_codebook and "input_features" in b:
            try:
                poses, _ = av.predict(b["input_features"])
                pxy = poses[:T, :2].float().cpu().numpy()
                a, f = ade_fde(pxy, gt); out["codebook"]["ade"].append(a); out["codebook"]["fde"].append(f)
            except Exception as e:  # noqa: BLE001
                print(f"[codebook] predict failed on a scene: {type(e).__name__}: {e}", flush=True)
    return out


def summarize(out):
    rows = []
    for name in ("codebook", "fmhead_select", "fmhead_mean", "fmhead_oracle"):
        ade, fde = out[name]["ade"], out[name]["fde"]
        if not ade:
            continue
        rows.append((name, len(ade), float(np.mean(ade)), float(np.mean(fde))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/fmhead_gt_feasibility.yaml")
    ap.add_argument("--ckpt", type=str, required=True, help="trained FMHead .pt")
    ap.add_argument("--n", type=int, default=500, help="held-out scenes to evaluate")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no_sft_base", action="store_true")
    ap.add_argument("--no_codebook", action="store_true", help="skip codebook baseline (no-vision box)")
    ap.add_argument("--base_ckpt", type=str, default=None,
                    help="override base VLM ckpt (e.g. full-FT full_ft_base.ckpt); "
                         "None -> config sft_model_path (AutoVLA_PDMS_89)")
    args = ap.parse_args()

    dctx = DistCtx()  # single-process eval
    device = dctx.device
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(FMHEAD_DIR, args.config)
    cfg = load_yaml(cfg_path)
    text_only = os.environ.get("FMHEAD_TEXT_ONLY", "0") == "1"
    load_sft_base = cfg.get("load_sft_base", True) and not args.no_sft_base

    av, av_config, lora_on = build_frozen_vlm(cfg, device, load_sft_base=load_sft_base,
                                              base_ckpt_override=args.base_ckpt)
    style_off = (not lora_on) or str(cfg.get("style_mode", "")).lower() == "none" \
        or float(cfg.get("style_alpha", 1.0)) == 0.0
    decoder = build_decoder(cfg, device)
    ck = torch.load(args.ckpt, map_location=device)
    decoder.load_state_dict(ck["fm_decoder"])
    decoder.eval()
    print(f"[eval] style_off={style_off} ckpt_step={ck.get('step')} n={args.n} "
          f"codebook={not args.no_codebook}", flush=True)

    # held-out slice: take the LAST args.n scenes (train uses shuffled front)
    if text_only:
        batches = make_textonly_batches(av, n=args.n, device=device, dctx=dctx,
                                        dataset_dir=cfg.get("dataset_path"))
    else:
        from dataset_utils.sft_dataset import SFTDataset, DataCollator
        from torch.utils.data import DataLoader
        if cfg.get("dataset_path"):
            av_config["data"]["train"]["json_dataset_path"] = [cfg["dataset_path"]]
        ds = SFTDataset(av_config["data"]["train"], av_config["model"], av.processor,
                        using_cot=av_config["model"]["use_cot"])
        collate = DataCollator(processor=av.processor,
                               ignore_index=av_config["model"]["tokens"]["ignore_index"],
                               assistant_id=av_config["model"]["tokens"]["assistant_id"])
        idxs = list(range(max(0, len(ds) - args.n), len(ds)))
        loader = DataLoader(torch.utils.data.Subset(ds, idxs), batch_size=1, collate_fn=collate,
                            num_workers=2, shuffle=False)
        batches = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
                   for b in loader]

    out = eval_slice(av, decoder, batches, style_off, cfg["style_alpha"], cfg["c_reduce"],
                     device, cfg["fmhead"]["num_samples"], cfg["fmhead"]["num_steps"],
                     do_codebook=not args.no_codebook)
    rows = summarize(out)

    print("\n=== FEASIBILITY L2 (human GT, metres) ===")
    print(f"{'decoder':16s} {'n':>5s} {'ADE':>8s} {'FDE':>8s}")
    for name, n, ade, fde in rows:
        print(f"{name:16s} {n:5d} {ade:8.3f} {fde:8.3f}")
    res = {"ckpt": args.ckpt, "n": args.n, "style_off": bool(style_off),
           "rows": [{"decoder": r[0], "n": r[1], "ADE_m": round(r[2], 4), "FDE_m": round(r[3], 4)} for r in rows]}
    json.dump(res, open(os.path.join(FMHEAD_DIR, "eval_feasibility_results.json"), "w"), indent=2)
    print(f"-> {os.path.join(FMHEAD_DIR, 'eval_feasibility_results.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
