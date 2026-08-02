"""extract_full_ft_ckpt.py -- split a full-FT Lightning ckpt into (base VLM) + (FMHead).

The full-FT run (fmhead_sft.py) saves a Lightning checkpoint whose state_dict mixes:
  * autovla.*      -> the TRAINED VLM (LM backbone + lm_head trained, vision frozen)
  * fm_decoder.*   -> the trained FMHead decoder (+ normalizer buffers)

To EVALUATE the full-FT model you must use its OWN trained VLM as the base (NOT
AutoVLA_PDMS_89) plus its FMHead. This script writes two files next to the ckpt:

  <run>/full_ft_base.ckpt : {"state_dict": {autovla.* kept verbatim}}
      -> loadable exactly like AutoVLA_PDMS_89.ckpt: both the navsim agent
         (AutoVLAAgent.initialize) and train_fmhead.build_frozen_vlm load a base via
         torch.load(path)["state_dict"] then strip the "autovla." prefix.
  <run>/fm_decoder.pt : {"fm_decoder": {fm_decoder.* stripped}}
      -> matches the frozen-eval format: FMHeadDecoder.load_state_dict(ck["fm_decoder"]).

Usage:
  python extract_full_ft_ckpt.py --ckpt /mnt/pfs/.../<run>/epoch=4-loss=0.1762.ckpt
"""
from __future__ import annotations
import argparse, os, torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Lightning full-FT .ckpt")
    ap.add_argument("--outdir", default=None, help="default: same dir as --ckpt")
    args = ap.parse_args()
    outdir = args.outdir or os.path.dirname(args.ckpt)
    os.makedirs(outdir, exist_ok=True)

    print(f"[extract] loading {args.ckpt}", flush=True)
    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck["state_dict"] if "state_dict" in ck else ck

    base = {k: v for k, v in sd.items() if k.startswith("autovla.")}          # keep prefix
    fm = {k[len("fm_decoder."):]: v for k, v in sd.items() if k.startswith("fm_decoder.")}
    other = [k for k in sd if not k.startswith(("autovla.", "fm_decoder."))]
    print(f"[extract] state_dict={len(sd)}  autovla.*={len(base)}  fm_decoder.*={len(fm)}  other={len(other)}")
    if other:
        print(f"[extract] WARNING: {len(other)} keys with neither prefix (ignored): {other[:5]}")

    base_path = os.path.join(outdir, "full_ft_base.ckpt")
    fm_path = os.path.join(outdir, "fm_decoder.pt")
    torch.save({"state_dict": base}, base_path)
    torch.save({"fm_decoder": fm}, fm_path)
    print(f"[extract] wrote base -> {base_path}  ({len(base)} keys, all '{list(base.keys())[0].split('.')[0]}.*')")
    print(f"[extract] wrote fmhead -> {fm_path}  ({len(fm)} keys, sample={list(fm.keys())[:4]})")

    # quick self-check: base strips to vlm.* ; fm strips to head/normalizer keys
    stripped = [k.replace("autovla.", "", 1) for k in list(base.keys())[:3]]
    print(f"[extract] base after strip 'autovla.': {stripped}")
    print(f"[extract] fm dtypes: {set(str(v.dtype) for v in list(fm.values())[:10])}")
    return base_path, fm_path


if __name__ == "__main__":
    main()
