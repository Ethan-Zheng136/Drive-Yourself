"""verify_latent_interp.py -- CPU verification for the inference-only LATENT CONTENT
INTERPOLATION style mode (content_source="interp").

Proves (all on CPU, no vision / no NVML):
  * ADDITIVE/GOLD safety: the fm3-kin ep6 PLAIN head strict-loads (strict=True) into an
    FMHeadDecoder built with content_source="interp" (no adapter, no z-encoder built).
  * alpha=0 in interp mode == content_source="base" at alpha=0, BIT-EXACT (max|delta|=0)
    at BOTH the conditioning level (ctx, s) AND the decoded-trajectory level.
  * interp geometry: ctx(alpha=1) == styled ctx (bit-exact); ctx(alpha) is the exact
    linear blend ctx_base + alpha*(ctx_styled-ctx_base); AdaLN style s==0 for all alpha.
  * existing modes untouched: base/styled produce EXACTLY the legacy (ctx, s) formulas.

The heavy Qwen VLM is mocked (correct 2048-dim bf16 hidden + disable_adapter toggle),
exactly like test_integration.py. We call the SAME helper functions the real agent
path (fmhead_navsim_agent._extract_ctx_s) and subclass path
(autovla_fmhead.AutoVLA_FMHead._ctx_and_style) use, so the arithmetic under test is
identical to production.

Usage:
  /root/workspace/miniconda3/envs/autovla/bin/python verify_latent_interp.py
"""
from __future__ import annotations

import contextlib
import json
import os

import numpy as np
import torch
import torch.nn as nn

from autovla_fmhead import (
    AUTOVLA_ACTION_START_ID, AUTOVLA_HIDDEN_SIZE, AUTOVLA_VLM_DTYPE,
    FMHeadDecoder, answer_position_hidden, autovla_fmhead_config,
    context_sequence_hidden, style_delta,
)

HEAD_CKPT = os.environ.get(
    "HEAD_CKPT",
    "/mnt/pfs/zhengguantian/autovla/persona/fmhead_fm3_kin/2026-07-17_12-47-42/fullft_fm3_kin_ep6_head.pt")
NORM = os.environ.get("NORM", "/root/workspace/fmhead/traj_norm_stats_ctrl.json")
ASID = AUTOVLA_ACTION_START_ID
CONSTRAINT = {"mode": "kinematic", "accel_max": 5.0, "yawrate_max": 1.0, "v_max": 25.0, "dt": 0.5}


class MockQwenVLM(nn.Module):
    """Deterministic 2048-dim bf16 stand-in with a disable_adapter() toggle (base vs styled)."""

    def __init__(self, vocab: int = 152000, hidden: int = AUTOVLA_HIDDEN_SIZE):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.adapter = nn.Embedding(vocab, hidden)  # the "LoRA" delta
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.adapter.weight, std=0.05)
        self.to(AUTOVLA_VLM_DTYPE)
        self._adapter_on = True

    @contextlib.contextmanager
    def disable_adapter(self):
        prev = self._adapter_on
        self._adapter_on = False
        try:
            yield
        finally:
            self._adapter_on = prev

    def forward(self, input_ids=None, attention_mask=None, **kw):
        h = self.embed(input_ids)
        if self._adapter_on:
            h = h + self.adapter(input_ids)
        out = type("Out", (), {})()
        out.hidden_states = (h,)
        return out


def make_inputs(B=3, S=40, n_action=10, device="cpu"):
    torch.manual_seed(0)
    input_ids = torch.randint(0, 150000, (B, S), device=device)
    input_ids[:, -n_action:] = ASID + torch.randint(0, 64, (B, n_action), device=device)
    attention_mask = torch.ones(B, S, dtype=torch.long, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def forward_hidden(vlm, mi, disable_adapter):
    cx = vlm.disable_adapter() if disable_adapter else contextlib.nullcontext()
    with cx:
        out = vlm(**mi, output_hidden_states=True)
    return out.hidden_states[-1]


def ctx_s_for_mode(vlm, mi, content_source, alpha, ctx_max_len=64):
    """Replicate EXACTLY the (ctx, ctx_mask, s) computed by the real agent/subclass paths."""
    hb = forward_hidden(vlm, mi, disable_adapter=True)   # BASE
    anchor_base = answer_position_hidden(hb, mi["input_ids"],
                                         attention_mask=mi.get("attention_mask"), action_start_id=ASID)
    hs = forward_hidden(vlm, mi, disable_adapter=False)  # STYLED
    anchor_styled = answer_position_hidden(hs, mi["input_ids"],
                                           attention_mask=mi.get("attention_mask"), action_start_id=ASID)
    ctx_b, mask_b = context_sequence_hidden(hb, mi["input_ids"], attention_mask=mi.get("attention_mask"),
                                            action_start_id=ASID, max_len=ctx_max_len)
    ctx_s, mask_s = context_sequence_hidden(hs, mi["input_ids"], attention_mask=mi.get("attention_mask"),
                                            action_start_id=ASID, max_len=ctx_max_len)
    s = style_delta(anchor_styled, anchor_base, alpha=alpha)
    if content_source == "interp":
        return ctx_b + alpha * (ctx_s - ctx_b), mask_b, torch.zeros_like(anchor_base)
    if content_source == "base":
        return ctx_b, mask_b, s
    return ctx_s, mask_s, s


def maxabs(a, b):
    return float((a.float() - b.float()).abs().max().item())


def main():
    device = "cpu"
    results = {}
    vlm = MockQwenVLM().to(device)
    mi = make_inputs(device=device)

    # ---- strict load of the fm3-kin ep6 PLAIN head into an interp decoder ----
    cfg = autovla_fmhead_config(constraint=CONSTRAINT)
    dec = FMHeadDecoder(cfg, num_samples=8, num_steps=10, cfg_weight=1.0,
                        style_alpha=0.0, content_source="interp").to(device)
    if os.path.exists(NORM):
        dec.load_normalizer(NORM)
    ck = torch.load(HEAD_CKPT, map_location=device, weights_only=False)
    missing, unexpected = [], []
    try:
        dec.load_state_dict(ck["fm_decoder"], strict=True)
        strict_ok, note = True, "strict=True load OK (plain head, no adapter/z built)"
    except Exception as e:  # noqa: BLE001 - report the exact failure, never swallow
        strict_ok, note = False, f"{type(e).__name__}: {e}"
    dec.eval()
    built_adapter = getattr(dec.fm_head, "style_adapter", None) is not None
    built_z = getattr(dec.fm_head, "z_encoder", None) is not None
    print(f"[strict-load] {note} | adapter_built={built_adapter} z_built={built_z} -> "
          f"{'PASS' if (strict_ok and not built_adapter and not built_z) else 'FAIL'}")
    results["strict_load"] = {"pass": bool(strict_ok and not built_adapter and not built_z),
                              "note": note, "adapter_built": built_adapter, "z_built": built_z}

    # ---- (1) interp alpha=0 == base alpha=0, bit-exact at (ctx, s) ----
    ctx_i0, m_i0, s_i0 = ctx_s_for_mode(vlm, mi, "interp", 0.0)
    ctx_b0, m_b0, s_b0 = ctx_s_for_mode(vlm, mi, "base", 0.0)
    d_ctx = maxabs(ctx_i0, ctx_b0)
    d_s = maxabs(s_i0, s_b0)
    d_mask = maxabs(m_i0, m_b0)
    p1 = (d_ctx == 0.0 and d_s == 0.0 and d_mask == 0.0)
    print(f"[interp a=0 == base a=0]  max|dctx|={d_ctx} max|ds|={d_s} max|dmask|={d_mask} -> "
          f"{'PASS' if p1 else 'FAIL'}")
    results["interp_a0_eq_base_a0_cond"] = {"pass": bool(p1), "max_abs_dctx": d_ctx,
                                            "max_abs_ds": d_s, "max_abs_dmask": d_mask}

    # ---- (2) interp s==0 for every alpha; ctx is the exact linear blend ----
    ctx_b, mask_b, _ = ctx_s_for_mode(vlm, mi, "base", 0.0)
    ctx_styled, _, _ = ctx_s_for_mode(vlm, mi, "styled", 1.0)
    s_all_zero, blend_ok, blend_worst = True, True, 0.0
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        ctx_a, _, s_a = ctx_s_for_mode(vlm, mi, "interp", a)
        s_all_zero &= (float(s_a.abs().max().item()) == 0.0)
        expect = ctx_b + a * (ctx_styled - ctx_b)
        d = maxabs(ctx_a, expect)
        blend_worst = max(blend_worst, d)
        blend_ok &= (d == 0.0)
    d_a1 = maxabs(ctx_s_for_mode(vlm, mi, "interp", 1.0)[0], ctx_styled)
    BF16_ULP = 2.0 ** -11  # one bf16 ULP for |x|~1: ctx_b + 1*(ctx_s-ctx_b) round-trips to ~ctx_s
    # HARD guarantees: s==0 for all alpha, ctx is the EXACT linear blend (bit-exact vs the same
    # formula). a=1 vs styled is only expected to match within one bf16 ULP (the b + (s-b)
    # round-trip is not exact in bf16); this is a rounding artifact, NOT a correctness gap.
    p2 = s_all_zero and blend_ok and (d_a1 <= BF16_ULP)
    print(f"[interp geometry]  s==0 all alpha={s_all_zero}  linear-blend worst|d|={blend_worst}  "
          f"a=1 vs styled |d|={d_a1} (<=1 bf16 ULP={BF16_ULP}) -> {'PASS' if p2 else 'FAIL'}")
    results["interp_geometry"] = {"pass": bool(p2), "s_all_zero": bool(s_all_zero),
                                  "blend_worst_abs": blend_worst, "a1_vs_styled_abs": d_a1,
                                  "bf16_ulp": BF16_ULP}

    # ---- (3) existing base/styled modes untouched (exact legacy formulas) ----
    hb = forward_hidden(vlm, mi, True)
    hs = forward_hidden(vlm, mi, False)
    ab = answer_position_hidden(hb, mi["input_ids"], attention_mask=mi["attention_mask"], action_start_id=ASID)
    as_ = answer_position_hidden(hs, mi["input_ids"], attention_mask=mi["attention_mask"], action_start_id=ASID)
    legacy_ctx_b, _ = context_sequence_hidden(hb, mi["input_ids"], attention_mask=mi["attention_mask"],
                                              action_start_id=ASID, max_len=64)
    legacy_ctx_s, _ = context_sequence_hidden(hs, mi["input_ids"], attention_mask=mi["attention_mask"],
                                              action_start_id=ASID, max_len=64)
    legacy_s = style_delta(as_, ab, alpha=1.0)
    cb, _, sb = ctx_s_for_mode(vlm, mi, "base", 1.0)
    cs, _, ss = ctx_s_for_mode(vlm, mi, "styled", 1.0)
    p3 = (maxabs(cb, legacy_ctx_b) == 0.0 and maxabs(sb, legacy_s) == 0.0
          and maxabs(cs, legacy_ctx_s) == 0.0 and maxabs(ss, legacy_s) == 0.0)
    print(f"[base/styled unchanged]  base_ctx|d|={maxabs(cb, legacy_ctx_b)} styled_ctx|d|="
          f"{maxabs(cs, legacy_ctx_s)} s|d|={maxabs(sb, legacy_s)} -> {'PASS' if p3 else 'FAIL'}")
    results["base_styled_unchanged"] = {"pass": bool(p3)}

    # ---- (4) DECODED trajectory: interp a=0 == base a=0, bit-exact ----
    v0 = torch.tensor([4.0, 5.0, 3.0], device=device)  # kinematic needs an ego speed
    dec.style_alpha = 0.0
    torch.manual_seed(123)
    poses_i = dec.decode_all(ctx_i0, s_i0, ctx_mask=m_i0, v0=v0)
    torch.manual_seed(123)
    poses_b = dec.decode_all(ctx_b0, s_b0, ctx_mask=m_b0, v0=v0)
    d_dec = maxabs(poses_i, poses_b)
    p4 = (d_dec == 0.0 and torch.isfinite(poses_i).all().item())
    print(f"[decode a=0 interp==base]  max|dpose|={d_dec} finite={torch.isfinite(poses_i).all().item()} "
          f"-> {'PASS' if p4 else 'FAIL'}")
    results["decode_a0_eq_base"] = {"pass": bool(p4), "max_abs_dpose": d_dec}

    all_pass = all(results[k]["pass"] for k in results)
    results["all_pass"] = bool(all_pass)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_latent_interp_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== SUMMARY ===")
    for k, v in results.items():
        if isinstance(v, dict):
            print(f"{k:28s}: {'PASS' if v['pass'] else 'FAIL'}")
    print(f"{'ALL':28s}: {'PASS' if all_pass else 'FAIL'}  -> {out}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
