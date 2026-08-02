"""Strict-load verify the UNFROZEN-head style ckpt as a StyleFMHead on the fm3-kin
(kinematic, control-space) config -- CPU, no VLM. Replicates the agent.initialize load:

  * auto-detect fm_head.style_adapter.* keys -> build WITH the adapter submodule
  * build with the SAME kinematic constraint the ep6 base was trained with
  * dec.load_state_dict(sd, strict=True) -> raises on ANY missing/unexpected key
  * confirm the style adapter is present AND active: alpha>0 (s=delta) changes the
    velocity field vs alpha=0 (s=0); s=0 stays finite (no NaN).
"""
import sys, os
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autovla_fmhead import FMHeadDecoder, autovla_fmhead_config

CKPT = os.environ.get(
    "FMHEAD_CKPT",
    "/mnt/pfs/zhengguantian/autovla/persona/fmhead_style_ddv2_fm3kin_unfrozen/"
    "2026-07-18_01-47-36_x8/fmhead_final.pt")
NORM = os.environ.get("FMHEAD_NORM", "/root/workspace/fmhead/traj_norm_stats_ctrl.json")
CONSTRAINT = {"mode": "kinematic", "accel_max": 5.0, "yawrate_max": 1.0,
              "v_max": 25.0, "dt": 0.5, "jerk_weight": 0.01}


def main():
    ok = True
    ck = torch.load(CKPT, map_location="cpu")
    assert "fm_decoder" in ck, f"ckpt has no fm_decoder key: {list(ck)[:5]}"
    sd = ck["fm_decoder"]
    ckpt_has_adapter = any(k.startswith("fm_head.style_adapter") for k in sd)
    use_style_adapter = True  # explicit (FMHEAD_USE_STYLE_ADAPTER=true at eval)

    cfg = autovla_fmhead_config(hidden_size=256, depth=4, style_dropout_prob=0.0,
                                use_style_adapter=use_style_adapter, constraint=CONSTRAINT)
    dec = FMHeadDecoder(cfg, num_samples=16, num_steps=100, cfg_weight=1.0)
    dec.load_normalizer(NORM)
    res = dec.load_state_dict(sd, strict=True)  # raises on mismatch
    missing = list(getattr(res, "missing_keys", []))
    unexpected = list(getattr(res, "unexpected_keys", []))
    has_mod = hasattr(dec.fm_head, "style_adapter")
    n_adapter_keys = sum(k.startswith("fm_head.style_adapter") for k in sd)
    print(f"[load] ckpt={os.path.basename(CKPT)}")
    print(f"[load] ckpt_has_adapter={ckpt_has_adapter} ({n_adapter_keys} adapter keys) "
          f"style_adapter_present={has_mod}")
    print(f"[load] constraint_mode={dec.config.constraint_mode} traj_dim={dec.config.traj_dim} "
          f"horizon={dec.config.horizon}")
    print(f"[load] strict-load OK: missing={len(missing)} unexpected={len(unexpected)}")
    ok &= ckpt_has_adapter and has_mod and not missing and not unexpected
    ok &= (dec.config.constraint_mode == "kinematic")

    # adapter active + no NaN: s=0 vs s=delta must differ; both finite
    torch.manual_seed(0)
    B, T, C = 2, 20, 2048
    ctx = torch.randn(B, T, C); mask = torch.ones(B, T)
    s0 = torch.zeros(B, C); sd_ = torch.randn(B, C) * 5.0
    x_t = torch.randn(B, 10, 2); t = torch.rand(B)
    dec.eval()
    with torch.no_grad():
        v0 = dec.fm_head(x_t, t, ctx, s0, ctx_mask=mask)
        vS = dec.fm_head(x_t, t, ctx, sd_, ctx_mask=mask)
    finite = bool(torch.isfinite(v0).all() and torch.isfinite(vS).all())
    delta = (v0 - vS).abs().mean().item()
    print(f"[active] finite(v[s=0],v[s=delta])={finite}  |Δ(s=0,s=delta)|={delta:.4f} "
          f"(must be >0 -> adapter active)")
    ok &= finite and (delta > 1e-4)

    print(f"\n=== unfrozen-head strict-load verification: {'PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
