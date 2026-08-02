"""smoke_zenc_safety.py -- ADDITIVE/GOLD-safety check for the z-encoder + content_source
change (CPU, no VLM). Proves NOTHING existing was broken:

  [1] GOLD 0.9088 head loads with DEFAULT flags (no z-encoder / no adapter) strict=True,
      missing=0 unexpected=0, and decodes finite/deterministic (byte-identical module).
  [2] fm3-kin ep6 head (kinematic control-space) loads with DEFAULT flags strict=True,
      missing=0 unexpected=0, and decodes finite/deterministic.
  [3] z-encoder OFF (default) is byte-identical to before: a head with use_z_encoder=False
      has NO z_encoder params and its state_dict keys == the plain head's.
  [4] z-encoder ON loads the fm3-kin ep6 head strict=False with ONLY z_encoder keys missing
      (every base head key loads), and at s=0 the z-on velocity == z-bypass velocity EXACTLY
      (enc(0)=0) -> alpha=0/z=null starts at the clean neutral base.
  [5] content_source default == "styled" (legacy inference); build with "base" is accepted.

Run:
  /root/workspace/miniconda3/envs/autovla/bin/python smoke_zenc_safety.py
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autovla_fmhead import FMHeadDecoder, autovla_fmhead_config

GOLD = "/mnt/pfs/zhengguantian/autovla/persona/fmhead_sft_full/2026-07-14_15-49-24/fm_decoder.pt"
GOLD_NORM = "/root/workspace/fmhead/traj_norm_stats_gt.json"
FM3KIN_HEAD = "/mnt/pfs/zhengguantian/autovla/persona/fmhead_fm3_kin/2026-07-17_12-47-42/fullft_fm3_kin_ep6_head.pt"
CTRL_NORM = "/root/workspace/fmhead/traj_norm_stats_ctrl.json"

KIN = {"mode": "kinematic", "accel_max": 5.0, "yawrate_max": 1.0, "v_max": 25.0, "dt": 0.5,
       "jerk_weight": 0.01, "curv_weight": 0.0, "smooth_t_min": 0.5}


def _load(sd_path):
    ck = torch.load(sd_path, map_location="cpu", weights_only=False)
    return ck["fm_decoder"] if "fm_decoder" in ck else ck


def _decode_det(dec, seed=0, kin=False):
    """Deterministic finite decode with fixed inputs (medoid scorer, no VLM)."""
    torch.manual_seed(seed)
    B, Tctx, H = 2, 20, 2048
    ctx = torch.randn(B, Tctx, H); mask = torch.ones(B, Tctx); s = torch.zeros(B, H)
    v0 = torch.full((B,), 5.0) if kin else None
    dec.eval()
    with torch.no_grad():
        poses = dec.decode(ctx, s, ctx_mask=mask, v0=v0)  # (B,10,3)
    return poses


def main():
    res, ok = {}, True

    # [1] GOLD 0.9088 head, DEFAULT flags (xy, mode none) ----------------------
    cfg_gold = autovla_fmhead_config(hidden_size=256, depth=4, style_dropout_prob=0.0)
    dec_gold = FMHeadDecoder(cfg_gold, num_samples=16, num_steps=30, cfg_weight=1.0)
    dec_gold.load_normalizer(GOLD_NORM)
    m = dec_gold.load_state_dict(_load(GOLD), strict=True)  # raises if any mismatch
    has_z = hasattr(dec_gold.fm_head, "z_encoder") and dec_gold.fm_head.z_encoder is not None
    p1 = _decode_det(dec_gold, kin=False); p1b = _decode_det(dec_gold, kin=False)
    gold_ok = (len(m.missing_keys) == 0 and len(m.unexpected_keys) == 0
               and not has_z and torch.isfinite(p1).all() and torch.equal(p1, p1b))
    res["gold_strict_load"] = gold_ok
    print(f"[1] GOLD 0.9088: strict-load missing={len(m.missing_keys)} unexpected={len(m.unexpected_keys)} "
          f"z_encoder={has_z} decode_finite={bool(torch.isfinite(p1).all())} deterministic={bool(torch.equal(p1,p1b))} "
          f"-> {'PASS' if gold_ok else 'FAIL'}", flush=True)
    ok &= gold_ok

    # [2] fm3-kin ep6 head, DEFAULT flags (kinematic control-space) ------------
    cfg_kin = autovla_fmhead_config(hidden_size=256, depth=4, style_dropout_prob=0.0, constraint=KIN)
    dec_kin = FMHeadDecoder(cfg_kin, num_samples=16, num_steps=100, cfg_weight=1.0)
    dec_kin.load_normalizer(CTRL_NORM)
    m2 = dec_kin.load_state_dict(_load(FM3KIN_HEAD), strict=True)
    p2 = _decode_det(dec_kin, kin=True); p2b = _decode_det(dec_kin, kin=True)
    kin_ok = (len(m2.missing_keys) == 0 and len(m2.unexpected_keys) == 0
              and torch.isfinite(p2).all() and torch.equal(p2, p2b))
    res["fm3kin_strict_load"] = kin_ok
    print(f"[2] fm3-kin ep6: strict-load missing={len(m2.missing_keys)} unexpected={len(m2.unexpected_keys)} "
          f"decode_finite={bool(torch.isfinite(p2).all())} deterministic={bool(torch.equal(p2,p2b))} "
          f"-> {'PASS' if kin_ok else 'FAIL'}", flush=True)
    ok &= kin_ok

    # [3] z-encoder OFF == plain head keys (byte-identical state_dict shape) ----
    cfg_off = autovla_fmhead_config(hidden_size=256, depth=4, use_z_encoder=False)
    cfg_on = autovla_fmhead_config(hidden_size=256, depth=4, use_z_encoder=True)
    dec_off = FMHeadDecoder(cfg_off); dec_on = FMHeadDecoder(cfg_on)
    keys_off = set(dec_off.state_dict().keys()); keys_on = set(dec_on.state_dict().keys())
    z_keys = {k for k in keys_on if "z_encoder" in k}
    off_no_z = (keys_off == keys_on - z_keys) and len(z_keys) > 0 \
        and not any("z_encoder" in k for k in keys_off)
    res["zenc_off_byte_identical_keys"] = off_no_z
    print(f"[3] z-enc OFF: n_z_keys(on)={len(z_keys)} off_has_no_z={not any('z_encoder' in k for k in keys_off)} "
          f"keys(off)==keys(on)-z={keys_off == keys_on - z_keys} -> {'PASS' if off_no_z else 'FAIL'}", flush=True)
    ok &= off_no_z

    # [4] z-encoder ON loads fm3-kin base strict=False, ONLY z keys missing; s=0 identical --
    dec_zon = FMHeadDecoder(cfg_on, num_samples=16, num_steps=100, cfg_weight=1.0)
    dec_zon.load_normalizer(CTRL_NORM)
    m4 = dec_zon.load_state_dict(_load(FM3KIN_HEAD), strict=False)
    base_missing = [k for k in m4.missing_keys if "z_encoder" not in k]
    z_missing = [k for k in m4.missing_keys if "z_encoder" in k]
    # at s=0, enc(0)=0 -> z-on velocity must equal z-bypass velocity EXACTLY
    torch.manual_seed(0)
    B, Tctx, H = 2, 20, 2048
    ctx = torch.randn(B, Tctx, H); mask = torch.ones(B, Tctx); s0 = torch.zeros(B, H)
    x_t = torch.randn(B, 10, 2); t = torch.rand(B)
    dec_zon.eval()
    with torch.no_grad():
        v_z = dec_zon.fm_head(x_t, t, ctx, s0, ctx_mask=mask)
        saved = dec_zon.fm_head.z_encoder; dec_zon.fm_head.z_encoder = None
        v_plain = dec_zon.fm_head(x_t, t, ctx, s0, ctx_mask=mask)
        dec_zon.fm_head.z_encoder = saved
    s0_delta = (v_z - v_plain).abs().max().item()
    zon_ok = (len(base_missing) == 0 and len(z_missing) > 0 and len(m4.unexpected_keys) == 0
              and s0_delta == 0.0)
    res.update(zenc_on_base_missing=len(base_missing), zenc_on_z_missing=len(z_missing),
               zenc_s0_max_delta=s0_delta, zenc_on_load=zon_ok)
    print(f"[4] z-enc ON: fm3-kin base_missing={len(base_missing)} z_missing={len(z_missing)} "
          f"unexpected={len(m4.unexpected_keys)} s0(z vs bypass) max|Δ|={s0_delta:.1e} (must be 0) "
          f"-> {'PASS' if zon_ok else 'FAIL'}", flush=True)
    ok &= zon_ok

    # [5] content_source default == styled (legacy); "base" accepted -----------
    d_def = FMHeadDecoder(cfg_off)
    d_base = FMHeadDecoder(cfg_off, content_source="base")
    cs_ok = (d_def.content_source == "styled" and d_base.content_source == "base")
    res["content_source_default_styled"] = cs_ok
    print(f"[5] content_source: default={d_def.content_source!r} (legacy 'styled') "
          f"explicit={d_base.content_source!r} -> {'PASS' if cs_ok else 'FAIL'}", flush=True)
    ok &= cs_ok

    # [6] z_norm_delta STABILITY variant (ControlNet zero-conv): default OFF -> legacy z path is
    #     byte-identical (final Linear stays xavier, no extra keys). ON -> the z-encoder's FINAL
    #     Linear is ZERO-init (so at init the injected modulation is 0 for ANY delta -> forward
    #     == clean base) while its EARLIER layers stay nonzero (live gradient path -> it trains).
    cfg_znorm = autovla_fmhead_config(hidden_size=256, depth=4, style_dropout_prob=0.0,
                                      use_z_encoder=True, z_norm_delta=True, constraint=KIN)
    # no separate gate param anywhere (both legacy and norm_delta): keys are just net.{0,2,4}
    no_extra_gate = not any(k.endswith("z_encoder.out_gate") for k in dec_on.state_dict().keys()
                            ) and not any(k.endswith("z_encoder.out_gate")
                                          for k in FMHeadDecoder(cfg_znorm).state_dict().keys())
    # legacy final Linear must be NONZERO (xavier) -> legacy forward byte-identical
    legacy_final_nonzero = float(dec_on.fm_head.z_encoder.net[-1].weight.abs().max()) > 0.0
    dznorm = FMHeadDecoder(cfg_znorm, num_samples=16, num_steps=100, cfg_weight=1.0)
    dznorm.load_normalizer(CTRL_NORM)
    m6 = dznorm.load_state_dict(_load(FM3KIN_HEAD), strict=False)
    base_missing6 = [k for k in m6.missing_keys if "z_encoder" not in k]
    znet = dznorm.fm_head.z_encoder.net
    final_zero = float(znet[-1].weight.abs().max()) == 0.0          # zero-conv at init
    earlier_nonzero = float(znet[0].weight.abs().max()) > 0.0 and float(znet[2].weight.abs().max()) > 0.0
    # at init, final zero-conv => the z-path injects 0 modulation for ANY delta AND any gain, so
    # forward == the CLEAN BASE (head @ neutral s=0). Feed a LARGE nonzero delta (||~158|| like
    # the real DDv2 delta) through the z-path and compare to base@s=0: must be EXACTLY equal.
    torch.manual_seed(0)
    B, Tctx, H = 2, 20, 2048
    ctx = torch.randn(B, Tctx, H); mask = torch.ones(B, Tctx)
    s_big = torch.randn(B, H) * 158.0                 # large ||delta|| like the real DDv2 delta
    s0 = torch.zeros(B, H)
    x_t = torch.randn(B, 10, 2); t = torch.rand(B)
    dznorm.eval(); dznorm.fm_head.style_gain = 1.0
    with torch.no_grad():
        v_z = dznorm.fm_head(x_t, t, ctx, s_big, ctx_mask=mask)     # z-path, big delta, final=0
        saved = dznorm.fm_head.z_encoder; dznorm.fm_head.z_encoder = None
        v_base = dznorm.fm_head(x_t, t, ctx, s0, ctx_mask=mask)     # clean base @ s=0
        dznorm.fm_head.z_encoder = saved
    init_delta = (v_z - v_base).abs().max().item()
    znorm_ok = (no_extra_gate and legacy_final_nonzero and len(base_missing6) == 0
                and final_zero and earlier_nonzero and init_delta == 0.0)
    res.update(znorm_no_extra_gate=no_extra_gate, znorm_legacy_final_nonzero=legacy_final_nonzero,
               znorm_final_zero_init=final_zero, znorm_earlier_nonzero=earlier_nonzero,
               znorm_init_bitexact_maxdelta=init_delta, znorm_base_missing=len(base_missing6),
               znorm_ok=znorm_ok)
    print(f"[6] z_norm_delta (zero-conv): no_extra_gate={no_extra_gate} "
          f"legacy_final_nonzero={legacy_final_nonzero} final_zero_init={final_zero} "
          f"earlier_layers_nonzero={earlier_nonzero} base_missing={len(base_missing6)} "
          f"init(big delta) max|Delta|={init_delta:.1e} (must be 0) -> {'PASS' if znorm_ok else 'FAIL'}", flush=True)
    ok &= znorm_ok

    res["ALL_PASS"] = bool(ok)
    print(f"\n=== z-enc GOLD/ADDITIVE safety: {'PASS' if ok else 'FAIL'} ===", flush=True)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "smoke_zenc_safety_results.json"), "w"), indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
