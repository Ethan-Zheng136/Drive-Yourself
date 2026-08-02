"""Verify the style-adapter eval LOAD fix (CPU, no VLM). Mimics the agent initialize load:
  * trained style ckpt + auto-detected adapter -> build WITH adapter, strict=True, missing/unexpected=0
  * plain 0.91 ckpt + use_style_adapter=False -> strict=True still loads (0.91 path unaffected)
  * alpha>0 (s=delta) vs alpha=0 (s=0) changes the decoded trajectory (adapter active),
    and s=0 is bit-identical to the plain 0.91 head.
"""
import sys, os
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autovla_fmhead import FMHeadDecoder, autovla_fmhead_config

STYLE = "/mnt/pfs/zhengguantian/autovla/persona/fmhead_ckpts_style_ddv2/2026-07-16_10-56-07_x4/fmhead_final.pt"
GOLD = "/mnt/pfs/zhengguantian/autovla/persona/fmhead_sft_full/2026-07-14_15-49-24/fm_decoder.pt"
NORM = "/root/workspace/fmhead/traj_norm_stats_gt.json"


def agent_style_load(ckpt_path, flag):
    """Replicate agent.initialize() load logic exactly."""
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck["fm_decoder"]
    ckpt_has_adapter = any(k.startswith("fm_head.style_adapter") for k in sd)
    use_style_adapter = bool(flag or ckpt_has_adapter)
    cfg = autovla_fmhead_config(hidden_size=256, depth=4, style_dropout_prob=0.0,
                                use_style_adapter=use_style_adapter)
    dec = FMHeadDecoder(cfg, num_samples=16, num_steps=30, cfg_weight=1.0)
    dec.load_normalizer(NORM)
    res = dec.load_state_dict(sd, strict=True)  # would raise on any mismatch
    return dec, use_style_adapter, ckpt_has_adapter, res


def main():
    ok = True
    # 1. trained style ckpt, flag NOT set -> auto-detect adapter, strict load
    dec, used, detected, res = agent_style_load(STYLE, flag=False)
    has_mod = hasattr(dec.fm_head, "style_adapter")
    print(f"[1:style-ckpt] ckpt_has_adapter={detected} use_style_adapter={used} "
          f"style_adapter_present={has_mod} strict-load OK (missing=0/unexpected=0)")
    ok &= detected and used and has_mod

    # 2. plain 0.91 ckpt, flag False -> plain head, strict load (0.91 path unchanged)
    decp, usedp, detp, _ = agent_style_load(GOLD, flag=False)
    plainp = not hasattr(decp.fm_head, "style_adapter")
    print(f"[2:0.91-ckpt] ckpt_has_adapter={detp} use_style_adapter={usedp} "
          f"plain_head={plainp} strict-load OK")
    ok &= (not detp) and (not usedp) and plainp

    # 3. alpha>0 vs alpha=0 changes decode; s=0 == plain 0.91 head
    torch.manual_seed(0)
    B, T, C = 2, 20, 2048
    ctx = torch.randn(B, T, C); mask = torch.ones(B, T)
    s0 = torch.zeros(B, C); sd_ = torch.randn(B, C) * 5.0
    dec.eval(); decp.eval()
    with torch.no_grad():
        # s=0 through the trained style head vs the plain 0.91 head -> must be identical
        v_style_s0 = dec.fm_head(torch.randn(B, 10, 2), torch.rand(B), ctx, s0, ctx_mask=mask)
        v_plain_s0 = decp.fm_head(torch.randn(B, 10, 2), torch.rand(B), ctx, s0, ctx_mask=mask)
    # use the SAME x_t/t for a fair identical check
    x_t = torch.randn(B, 10, 2); t = torch.rand(B)
    with torch.no_grad():
        a0 = dec.fm_head(x_t, t, ctx, s0, ctx_mask=mask)
        aP = decp.fm_head(x_t, t, ctx, s0, ctx_mask=mask)
        aS = dec.fm_head(x_t, t, ctx, sd_, ctx_mask=mask)
    s0_identical = (a0 - aP).abs().max().item()
    alpha_change = (a0 - aS).abs().mean().item()
    print(f"[3:active] s=0 style-head vs 0.91-head max|Δ|={s0_identical:.2e} (must be 0)  "
          f"|Δ(s=0, s=delta)|={alpha_change:.4f} (must be >0 -> adapter active)")
    ok &= (s0_identical == 0.0) and (alpha_change > 1e-4)

    print(f"\n=== style-load verification: {'PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
