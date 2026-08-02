"""FM head v2 smoke tests (multi-token cross-attention conditioning).

Runs CPU / tiny-tensor checks that do NOT need the real VLM:

  1. shapes: context (B,T_ctx,2048) -> cross-attn -> velocity (B,10,2); flow loss fp32.
  2. dtype-safety: fp32 head AND bf16 head both run forward/loss/sample.
  3. ctx_mask: padded context tokens are ignored (masked positions can't change output).
  4. train==infer causal consistency: context_sequence_hidden is BIT-IDENTICAL whether
     or not GT action tokens are appended after the anchor (cos ~= 1.0) -- using a tiny
     causal transformer as a VLM stand-in (real Qwen check is a separate, GPU script).
  5. grad flow: only-FMHead-trains path; grads reach the "LM" in the full-FT path.

Run:  python smoke_v2.py
"""
import sys, os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fm_head import FMHead, FMHeadConfig                       # noqa: E402
from autovla_fmhead import context_sequence_hidden, answer_position_hidden  # noqa: E402

DEV = torch.device("cpu")
ASID = 151665                                                  # AUTOVLA_ACTION_START_ID


def _head(dtype=torch.float32, d_ctx=2048):
    cfg = FMHeadConfig(d_style=d_ctx, d_ctx=d_ctx, hidden_size=128, depth=3, num_heads=8)
    return FMHead(cfg).to(DEV).to(dtype), cfg


def test_shapes():
    head, cfg = _head()
    B, Tctx = 4, 17
    x_t = torch.randn(B, cfg.horizon, cfg.traj_dim)
    t = torch.rand(B)
    ctx = torch.randn(B, Tctx, cfg.d_ctx)
    ctx_mask = torch.ones(B, Tctx)
    s = torch.randn(B, cfg.d_style)
    v = head(x_t, t, ctx, s, ctx_mask=ctx_mask)
    loss = head.flow_matching_loss(torch.randn(B, cfg.horizon, cfg.traj_dim), ctx, s, ctx_mask=ctx_mask)
    trajs = head.sample(ctx, s, ctx_mask=ctx_mask, num_samples=6, num_steps=30, cfg_weight=2.0)
    ok = (tuple(v.shape) == (B, cfg.horizon, cfg.traj_dim)
          and loss.dtype == torch.float32 and torch.isfinite(loss)
          and tuple(trajs.shape) == (B, 6, cfg.horizon, cfg.traj_dim))
    print(f"[1:shapes] v={tuple(v.shape)} loss={loss.item():.3f}({loss.dtype}) "
          f"trajs={tuple(trajs.shape)} -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_dtype():
    oks = []
    for dt in (torch.float32, torch.bfloat16):
        head, cfg = _head(dtype=dt)
        B, Tctx = 3, 9
        ctx = torch.randn(B, Tctx, cfg.d_ctx)
        s = torch.randn(B, cfg.d_style)
        mask = torch.ones(B, Tctx)
        loss = head.flow_matching_loss(torch.randn(B, cfg.horizon, cfg.traj_dim), ctx, s, ctx_mask=mask)
        trajs = head.sample(ctx, s, ctx_mask=mask, num_samples=4, num_steps=10)
        ok = torch.isfinite(loss) and loss.dtype == torch.float32 and torch.isfinite(trajs).all()
        oks.append(bool(ok))
        print(f"[2:dtype {str(dt).split('.')[-1]}] loss={loss.item():.3f} "
              f"trajs_finite={bool(torch.isfinite(trajs).all())} -> {'PASS' if ok else 'FAIL'}")
    return all(oks)


def test_mask():
    """Masked (padding) context tokens must not change the velocity output."""
    head, cfg = _head()
    head.eval()
    B, Treal, Tpad = 2, 6, 5
    x_t = torch.randn(B, cfg.horizon, cfg.traj_dim)
    t = torch.rand(B)
    s = torch.randn(B, cfg.d_style)
    real = torch.randn(B, Treal, cfg.d_ctx)
    mask_real = torch.ones(B, Treal)
    with torch.no_grad():
        v_ref = head(x_t, t, real, s, ctx_mask=mask_real)
        # append garbage padded tokens with mask=0
        pad = torch.randn(B, Tpad, cfg.d_ctx) * 100.0
        ctx2 = torch.cat([real, pad], dim=1)
        mask2 = torch.cat([mask_real, torch.zeros(B, Tpad)], dim=1)
        v_pad = head(x_t, t, ctx2, s, ctx_mask=mask2)
    max_diff = (v_ref - v_pad).abs().max().item()
    ok = max_diff < 1e-5
    print(f"[3:mask] max|v_ref - v_padded|={max_diff:.2e} -> {'PASS' if ok else 'FAIL'}")
    return ok


class TinyCausalLM(nn.Module):
    """Minimal causal transformer used as a VLM stand-in to verify the context
    sequence is causally independent of future (action) tokens."""
    def __init__(self, vocab=152000, d=64, layers=2, heads=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, 512, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=128, batch_first=True,
                                         activation="gelu", norm_first=True)
        self.blocks = nn.TransformerEncoder(enc, layers)
        self.proj = nn.Linear(d, 2048)

    def forward(self, input_ids, attention_mask=None):
        S = input_ids.shape[1]
        h = self.emb(input_ids) + self.pos[:, :S]
        causal = torch.triu(torch.full((S, S), float("-inf")), diagonal=1)
        h = self.blocks(h, mask=causal)
        return self.proj(h)


def test_train_infer_consistency():
    """context_sequence_hidden must be bit-identical whether or not GT action tokens
    are appended after the generation-start anchor (causal invariance)."""
    torch.manual_seed(0)
    lm = TinyCausalLM().eval()
    B, P = 2, 20                                        # P prompt tokens
    prompt = torch.randint(0, 151000, (B, P))          # all < ASID (prompt-only tokens)
    # inference view: prompt only (no action tokens); anchor = last prompt token
    ids_infer = prompt
    mask_infer = torch.ones(B, P)
    # train view: teacher-forced GT action tokens appended after the prompt
    actions = torch.randint(ASID, ASID + 100, (B, 6))
    ids_train = torch.cat([prompt, actions], dim=1)
    mask_train = torch.ones(B, P + 6)
    with torch.no_grad():
        h_infer = lm(ids_infer)
        h_train = lm(ids_train)
    ctx_i, m_i = context_sequence_hidden(h_infer, ids_infer, attention_mask=mask_infer,
                                         action_start_id=ASID, max_len=64)
    ctx_t, m_t = context_sequence_hidden(h_train, ids_train, attention_mask=mask_train,
                                         action_start_id=ASID, max_len=64)
    same_shape = ctx_i.shape == ctx_t.shape and bool((m_i == m_t).all())
    max_diff = (ctx_i - ctx_t).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(ctx_i.flatten(1), ctx_t.flatten(1)).min().item()
    # anchor vector consistency too
    a_i = answer_position_hidden(h_infer, ids_infer, attention_mask=mask_infer, action_start_id=ASID)
    a_t = answer_position_hidden(h_train, ids_train, attention_mask=mask_train, action_start_id=ASID)
    a_cos = torch.nn.functional.cosine_similarity(a_i, a_t).min().item()
    ok = same_shape and max_diff < 1e-4 and cos > 0.9999 and a_cos > 0.9999
    print(f"[4:train==infer] ctx shape={tuple(ctx_i.shape)} same_mask={same_shape} "
          f"max|Δ|={max_diff:.2e} min_cos(ctx)={cos:.6f} min_cos(anchor)={a_cos:.6f} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def test_grad_flow():
    """Frozen path: only FMHead params get grad. Full-FT path: grad reaches the LM.

    NOTE: the head uses AdaLN-Zero init (all gates = 0), so at step 0 the cross-attn
    branch is gated to zero and d(loss)/d(ctx) is exactly 0 -> grad would NOT reach the
    LM. That is correct behaviour, not a bug. We first warm up the head a few steps so
    the gates leave zero (as they do in real training), then verify LM grad flows.
    """
    torch.manual_seed(0)
    lm = TinyCausalLM()
    head, cfg = _head()
    B, P = 2, 15
    prompt = torch.randint(0, 151000, (B, P))
    actions = torch.randint(ASID, ASID + 100, (B, 5))
    ids = torch.cat([prompt, actions], dim=1)
    mask = torch.ones(B, P + 5)
    gt = torch.randn(B, cfg.horizon, 2)
    s = torch.zeros(B, cfg.d_style)

    # warm up the head (frozen LM) so AdaLN-Zero gates move off zero
    with torch.no_grad():
        h0 = lm(ids, mask)
    ctx0, cm0 = context_sequence_hidden(h0, ids, attention_mask=mask, action_start_id=ASID)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    for _ in range(30):
        opt.zero_grad()
        head.flow_matching_loss(gt, ctx0, s, ctx_mask=cm0).backward()
        opt.step()

    # --- frozen path: LM under no_grad, only head trains ---
    head.zero_grad()
    with torch.no_grad():
        h = lm(ids, mask)
    ctx, cmask = context_sequence_hidden(h, ids, attention_mask=mask, action_start_id=ASID)
    loss = head.flow_matching_loss(gt, ctx, s, ctx_mask=cmask)
    loss.backward()
    head_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    lm_has_grad_frozen = any(p.grad is not None for p in lm.parameters())
    ok_frozen = head_has_grad and not lm_has_grad_frozen

    # --- full-FT path: grad reaches the LM through the context ---
    head.zero_grad(); lm.zero_grad()
    h = lm(ids, mask)                                    # grad-enabled
    ctx, cmask = context_sequence_hidden(h, ids, attention_mask=mask, action_start_id=ASID)
    loss = head.flow_matching_loss(gt, ctx, s, ctx_mask=cmask)
    loss.backward()
    lm_has_grad_full = any(p.grad is not None and p.grad.abs().sum() > 0 for p in lm.parameters())
    ok = ok_frozen and lm_has_grad_full
    print(f"[5:grad] frozen(head_grad={head_has_grad}, lm_grad={lm_has_grad_frozen}) "
          f"fullft(lm_grad={lm_has_grad_full}) -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    torch.manual_seed(0)
    results = {
        "shapes": test_shapes(),
        "dtype": test_dtype(),
        "mask": test_mask(),
        "train==infer": test_train_infer_consistency(),
        "grad": test_grad_flow(),
    }
    print("\n=== FM head v2 smoke summary ===")
    for k, v in results.items():
        print(f"  {k:14s}: {'PASS' if v else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)
