"""Smoke test for fmhead_scorer.FMHeadScorer (Hydra MemoryEffTransformer, ctx_mask, v1/v2
metric sets) -- shapes, masked cross-attn, selection, distill loss, backward, timing.
Run with the project interpreter (CPU is fine; tiny tensors)."""

import time
import torch

from fmhead_scorer import (
    FMHeadScorer, V1_METRICS, V2_EXTRA_METRICS, V1_WEIGHTS, V2_WEIGHTS,
)

torch.manual_seed(0)

B, N_kv, ENV = 2, 64, 2048   # batch, #ctx tokens (ctx_max_len), VLA hidden
N, T, Dp = 16, 10, 3         # candidates, poses, pose dim

# ---- v1 scorer (6 metrics + imi) ----
scorer = FMHeadScorer(env_in_dim=ENV, d_model=256, num_poses=T, pose_dim=Dp)
n_params = sum(p.numel() for p in scorer.parameters())
print(f"[smoke] FMHeadScorer(v1) params={n_params/1e6:.2f}M  heads={list(scorer.heads.keys())}")

env = torch.randn(B, N_kv, ENV)
env_mask = torch.ones(B, N_kv, dtype=torch.bool)
env_mask[0, 40:] = False          # scene 0 has 40 valid ctx tokens, rest padding
traj = torch.randn(B, N, T, Dp)

out = scorer(env, traj, env_mask=env_mask, weights=V1_WEIGHTS)
assert out["scores"].shape == (B, N), out["scores"].shape
assert out["selected_index"].shape == (B,)
assert out["trajectory"].shape == (B, T, Dp), out["trajectory"].shape
for m in V1_METRICS + ["imi"]:
    assert out[m].shape == (B, N), (m, out[m].shape)
assert torch.isfinite(out["scores"]).all(), "non-finite scores"
print(f"[smoke] v1 forward OK  scores{tuple(out['scores'].shape)} sel={out['selected_index'].tolist()} "
      f"traj{tuple(out['trajectory'].shape)}")

for b in range(B):
    k = int(out["selected_index"][b])
    assert torch.allclose(out["trajectory"][b], traj[b, k]), "selection mismatch"
print("[smoke] selected trajectory == argmax candidate  OK")

# ---- v2 scorer (v1 + LK + TLC), v2 weights, no mask ----
scorer2 = FMHeadScorer(env_in_dim=ENV, num_poses=T, pose_dim=Dp,
                       metrics=V1_METRICS + V2_EXTRA_METRICS)
out2 = scorer2(env, traj, env_mask=None, weights=V2_WEIGHTS)
assert out2["scores"].shape == (B, N) and torch.isfinite(out2["scores"]).all()
print(f"[smoke] v2 forward OK  heads={list(scorer2.heads.keys())} sel={out2['selected_index'].tolist()}")

# ---- distillation loss (v1 PDM labels + imi soft target) + backward ----
metric_targets = {m: torch.rand(B, N) for m in V1_METRICS}     # simulated PDM labels in [0,1]
imi_target = torch.rand(B, N).softmax(dim=-1)
losses = scorer.distill_loss(out, metric_targets, imi_target=imi_target)
assert losses["total"].dim() == 0
losses["total"].backward()
grad_ok = all(p.grad is not None for p in scorer.parameters() if p.requires_grad)
print(f"[smoke] distill_loss total={losses['total'].item():.4f} backward grads_ok={grad_ok}")
print("        per-metric=" + ", ".join(f"{k}:{v.item():.3f}" for k, v in losses.items() if k != "total"))

# ---- v2-cache-missing path: LK/TLC absent from targets -> silently skipped, still trains ----
losses2 = scorer2.distill_loss(out2, metric_targets)   # only v1 targets given
assert "lane_keeping" not in losses2 and "total" in losses2
print("[smoke] partial-target (v2 heads unlabeled) skipped cleanly  OK")

# ---- timing: score 16 candidates (replaces 16 PDM simulations) ----
scorer.eval()
with torch.no_grad():
    for _ in range(3):
        scorer(env, traj, env_mask=env_mask)     # warmup
    t0 = time.perf_counter(); iters = 30
    for _ in range(iters):
        scorer(env, traj, env_mask=env_mask)
    dt = (time.perf_counter() - t0) / iters * 1000
print(f"[smoke] fwd latency (B={B}, N={N} cands, CPU): {dt:.2f} ms/call -> ~{dt/B:.2f} ms/scene")

print("[smoke] ALL PASS")
