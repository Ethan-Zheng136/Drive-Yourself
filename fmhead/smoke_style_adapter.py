"""smoke_style_adapter.py -- prove the opt-in style adapter does NOT damage the 0.91 head.

CPU / tiny-tensor, NO training, NO VLM. Loads the real full-FT v2 head (fm_decoder.pt) into
BOTH a plain FMHead and a StyleFMHead and checks:

  1. BIT-IDENTICAL s=0: plain(x,t,ctx,s=0) == style(x,t,ctx,s=0) (the 0.91-preservation proof).
     Also: a freshly-built (zero-gated) style adapter leaves output UNCHANGED even for s!=0.
  2. Controllability: with a NON-zero (simulated-trained) adapter, alpha>0 changes the output,
     alpha scales it, and alpha=0 (s=0) is byte-identical to base (zero-gate holds).
  3. Grad: after freeze_base_keep_style(), ONLY the adapter params get grad; the loaded 0.91
     head params are frozen (requires_grad=False, grad None).
  4. dtype-safe (fp32 + bf16), shapes correct.

Run:  python smoke_style_adapter.py
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autovla_fmhead import FMHeadDecoder, autovla_fmhead_config  # noqa: E402

GOLD = "/mnt/pfs/zhengguantian/autovla/persona/fmhead_sft_full/2026-07-14_15-49-24/fm_decoder.pt"
NORM = "/root/workspace/fmhead/traj_norm_stats_gt.json"
DEV = torch.device("cpu")


def build(use_adapter, dtype=torch.float32):
    cfg = autovla_fmhead_config(hidden_size=256, depth=4, style_dropout_prob=0.2,
                                use_style_adapter=use_adapter)
    dec = FMHeadDecoder(cfg, num_samples=8, num_steps=10, cfg_weight=1.0).to(DEV).to(dtype)
    dec.load_normalizer(NORM)
    ck = torch.load(GOLD, map_location="cpu")
    sd = ck["fm_decoder"] if "fm_decoder" in ck else ck
    msg = dec.load_state_dict(sd, strict=False)
    base_missing = [k for k in msg.missing_keys if "style_adapter" not in k]
    dec.eval()
    return dec, base_missing, msg


def main():
    torch.manual_seed(0)
    B, Tctx, C = 3, 20, 2048
    x_t = torch.randn(B, 10, 2)
    t = torch.rand(B)
    ctx = torch.randn(B, Tctx, C)
    mask = torch.ones(B, Tctx)
    s0 = torch.zeros(B, C)
    s_delta = torch.randn(B, C) * 5.0            # simulated persona LoRA anchor delta
    results = {}

    plain, bm_p, _ = build(False)
    style, bm_s, msg_s = build(True)
    assert not bm_p and not bm_s, f"0.91 base keys missing: plain={bm_p} style={bm_s}"
    adapter_missing = [k for k in msg_s.missing_keys if "style_adapter" in k]
    print(f"[load] plain base_missing=0  style base_missing=0  style adapter keys (zero-init, expected missing)={len(adapter_missing)}")

    # --- 1. BIT-IDENTICAL s=0 ---
    with torch.no_grad():
        vp = plain.fm_head(x_t, t, ctx, s0, ctx_mask=mask)
        vs = style.fm_head(x_t, t, ctx, s0, ctx_mask=mask)
        # zero-gated adapter must not change output even for s != 0 at init
        vs_delta_init = style.fm_head(x_t, t, ctx, s_delta, ctx_mask=mask)
    d0 = (vp - vs).abs().max().item()
    d_init = (vp - vs_delta_init).abs().max().item()
    r1 = d0 == 0.0 and d_init == 0.0
    print(f"[1:bit-identical] max|plain(s0)-style(s0)|={d0:.2e}  "
          f"max|plain(s0)-style(s_delta, zero-gate init)|={d_init:.2e} -> {'PASS' if r1 else 'FAIL'}")
    results["1_bit_identical_s0"] = r1

    # --- 2. controllability with a simulated-trained (non-zero) adapter ---
    ad = style.fm_head.style_adapter
    with torch.no_grad():
        for h in ad.block_heads:
            h.weight.normal_(0, 0.02)
        ad.final_head.weight.normal_(0, 0.02)
        ad.block_gates.fill_(0.5); ad.final_gate.fill_(0.5)
    with torch.no_grad():
        base = plain.fm_head(x_t, t, ctx, s0, ctx_mask=mask)
        a0 = style.fm_head(x_t, t, ctx, 0.0 * s_delta, ctx_mask=mask)   # alpha=0 -> s=0
        a5 = style.fm_head(x_t, t, ctx, 0.5 * s_delta, ctx_mask=mask)
        a10 = style.fm_head(x_t, t, ctx, 1.0 * s_delta, ctx_mask=mask)
    zero_gate_a0 = (base - a0).abs().max().item()
    chg5 = (base - a5).abs().mean().item()
    chg10 = (base - a10).abs().mean().item()
    step = (a5 - a10).abs().mean().item()
    r2 = zero_gate_a0 == 0.0 and chg10 > 1e-4 and chg10 > chg5 > 0 and step > 1e-5
    print(f"[2:controllability] zero-gate@alpha0={zero_gate_a0:.2e}  |Δ|@0.5={chg5:.4f}  "
          f"|Δ|@1.0={chg10:.4f}  |Δ(0.5,1.0)|={step:.4f} -> {'PASS' if r2 else 'FAIL'}")
    results["2_controllability"] = r2

    # --- 3. grad: freeze base, only adapter trains (a few steps so gate+heads both move) ---
    style2, _, _ = build(True)
    n_train, n_frozen = style2.freeze_base_keep_style()
    adapter_ids = {id(p) for p in style2.fm_head.style_adapter.parameters()}
    opt = torch.optim.Adam(style2.style_adapter_parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        tau = style2._normalize(torch.randn(B, 10, 2))
        style2.fm_head.flow_matching_loss(tau, ctx, s_delta, ctx_mask=mask, style_dropout_prob=0.0).backward()
        opt.step()
    base_grads = [p.grad is not None and p.grad.abs().sum() > 0
                  for n, p in style2.named_parameters() if id(p) not in adapter_ids]
    adapter_grad = sum(p.numel() for p in style2.fm_head.style_adapter.parameters()
                       if p.grad is not None and p.grad.abs().sum() > 0)
    base_frozen = all(not p.requires_grad for n, p in style2.named_parameters() if id(p) not in adapter_ids)
    r3 = base_frozen and not any(base_grads) and adapter_grad > 0 and n_train < 6e6 and n_frozen > 5e6
    print(f"[3:grad] trainable={n_train/1e3:.1f}K frozen={n_frozen/1e6:.2f}M  base_frozen={base_frozen}  "
          f"base_with_grad={sum(base_grads)}  adapter_grad_elems={adapter_grad} -> {'PASS' if r3 else 'FAIL'}")
    results["3_grad_freeze"] = r3

    # --- 4. dtype-safe ---
    dt_ok = {}
    for dt in (torch.float32, torch.bfloat16):
        d, _, _ = build(True, dtype=dt)
        with torch.no_grad():
            v = d.fm_head(x_t.to(dt), t.to(dt), ctx.to(dt), s_delta.to(dt), ctx_mask=mask.to(dt))
        ok = tuple(v.shape) == (B, 10, 2) and torch.isfinite(v.float()).all()
        dt_ok[str(dt)] = bool(ok)
        print(f"[4:dtype {str(dt).split('.')[-1]}] out={tuple(v.shape)} finite={bool(torch.isfinite(v.float()).all())} -> {'PASS' if ok else 'FAIL'}")
    results["4_dtype"] = all(dt_ok.values())

    # --- 5. train-loop: flow loss to a DDv2-like target DECREASES; s=0 stays bit-identical ---
    style3, _, _ = build(True)
    style3.freeze_base_keep_style()
    plain3, _, _ = build(False)
    with torch.no_grad():
        base_s0_before = plain3.fm_head(x_t, t, ctx, s0, ctx_mask=mask)
        # synthetic "DDv2 teacher" target in NORMALIZED space = a fixed offset from the
        # frozen head's own s=0 velocity field target, so the adapter has something to learn.
    tau_ddv2 = style3._normalize(torch.randn(B, 10, 2) + 3.0)   # a distinct target
    opt = torch.optim.Adam(style3.style_adapter_parameters(), lr=2e-3)
    losses = []
    for _ in range(80):
        opt.zero_grad()
        loss = style3.fm_head.flow_matching_loss(tau_ddv2, ctx, s_delta, ctx_mask=mask, style_dropout_prob=0.0)
        loss.backward(); opt.step(); losses.append(float(loss))
    loss_down = losses[-1] < 0.7 * losses[0]
    with torch.no_grad():
        s0_after = style3.fm_head(x_t, t, ctx, s0, ctx_mask=mask)     # trained adapter, s=0
    s0_identical_after = (base_s0_before - s0_after).abs().max().item()
    r5 = loss_down and s0_identical_after == 0.0
    print(f"[5:train-loop] loss {losses[0]:.4f}->{losses[-1]:.4f} (down={loss_down})  "
          f"s=0 bit-identical AFTER training: max|Δ|={s0_identical_after:.2e} -> {'PASS' if r5 else 'FAIL'}")
    results["5_train_loop"] = r5

    print("\n=== style-adapter non-regression summary ===")
    for k, v in results.items():
        print(f"  {k:22s}: {'PASS' if v else 'FAIL'}")
    ok = all(results.values())
    print(f"  {'ALL':22s}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
