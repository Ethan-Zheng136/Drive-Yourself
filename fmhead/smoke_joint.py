"""CPU smoke for the JOINT pieces that don't need the 3B LM:
  (a) score_candidates on a REAL navtrain12k token  -> in-loop pdm_score wiring
  (b) bf16 scorer + fp32 CANDIDATES distill_loss + mixed add + backward -> FSDP dtype path
  (c) extract_scorer round-trip -> deployable scorer_final.pt loads back
  (d) token unwrap + masked-loss uniform param participation -> FSDP no-deadlock path
"""
import os, sys, tempfile
import numpy as np
import torch

FM = "/root/workspace/fmhead"
sys.path.insert(0, FM)
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
os.environ.setdefault("NUPLAN_MAPS_ROOT", "/root/workspace/closed_loop/data/navsim/maps")
os.environ.setdefault("OPENSCENE_DATA_ROOT", "/root/workspace/closed_loop/data/navsim")

from fmhead_scorer import FMHeadScorer, V1_METRICS, V1_WEIGHTS

CACHE = "/mnt/pfs/zhengguantian/autovla/exp/metric_cache_navtrain12k"
ok = True

# (a) real-token pdm_score labels ------------------------------------------------
print("== (a) score_candidates on a real navtrain12k token ==")
from label_candidates_pdm import build_pdm_tools, score_candidates, METRIC_KEYS
tools = build_pdm_tools(CACHE)
tok = sorted(tools["loader"].metric_cache_paths.keys())[0]
N, T = 4, 10
t = np.arange(1, T + 1) * 0.5
cands = np.zeros((N, T, 3), dtype=np.float32)
for i, v in enumerate([2.0, 5.0, 8.0, 11.0]):      # different constant speeds -> different scores
    cands[i, :, 0] = v * t                          # forward x(t)=v*t
labels = score_candidates(cands, tok, tools, interval=0.5)
assert labels is not None, "token had no metric cache"
for m in METRIC_KEYS + ["score"]:
    arr = labels[m]
    assert arr.shape == (N,) and np.all(np.isfinite(arr)) and arr.min() >= 0 and arr.max() <= 1.0001, m
print(f"  token={tok[:12]} score(N)={np.round(labels['score'],3)}  -> real per-metric labels OK")

# (b) EXACT training_step dtype path: bf16 scorer, fp32 candidates (as decode_all returns),
#     bf16 grad-carrying ctx (as the VLM emits under FSDP), fp32 flow-loss added to bf16
#     scorer loss, then backward. This reproduces the real run that crashed at pose_embed.
print("== (b) REAL dtype path: bf16 scorer + fp32 candidates + bf16 ctx + mixed add + backward ==")
for scorer_dtype in (torch.bfloat16, torch.float32):
    sc = FMHeadScorer(env_in_dim=2048, d_model=64, n_decoder_layers=1, num_poses=T, metrics=V1_METRICS)
    sc = sc.to(scorer_dtype)
    ctx = torch.randn(1, 12, 2048, dtype=scorer_dtype, requires_grad=True)   # VLM ctx (grad, #1)
    cand_t = torch.randn(1, N, T, 3, dtype=torch.float32)                    # decode_all -> FP32
    mask = torch.ones(1, 12, dtype=torch.bool)
    targets = {m: torch.as_tensor(labels[m], dtype=torch.float32).unsqueeze(0) for m in V1_METRICS}
    out = sc(ctx, cand_t[:, :, :T, :], env_mask=mask, weights=V1_WEIGHTS)    # <- crashed here before
    sloss = sc.distill_loss(out, targets)["total"]
    flow = torch.tensor(0.5, dtype=torch.float32, requires_grad=True)        # fp32 flow loss
    total = flow + 0.5 * sloss                                               # fp32 + bf16 -> promote
    total.backward()
    assert torch.isfinite(total) and ctx.grad is not None and torch.isfinite(ctx.grad).all()
    print(f"  scorer={scorer_dtype} cands=fp32 -> sloss={float(sloss):.4f} total dtype={total.dtype} "
          f"sel={out['selected_index'].tolist()} grad OK")

# (c) extract_scorer round-trip ---------------------------------------------------
print("== (c) extract_scorer round-trip ==")
with tempfile.TemporaryDirectory() as d:
    # dims MUST match config/fmhead_fm3_kin_joint.yaml (d_model=256, layers=3) since the
    # extractor reconstructs the scorer config from that yaml.
    full = FMHeadScorer(env_in_dim=2048, d_model=256, n_decoder_layers=3, num_poses=T, metrics=V1_METRICS)
    lit_sd = {f"scorer.{k}": v.to(torch.bfloat16) for k, v in full.state_dict().items()}
    lit_sd["autovla.dummy"] = torch.zeros(1)     # noise the extractor must ignore
    ckpt_p = os.path.join(d, "epoch=1-loss=0.80.ckpt")
    torch.save({"state_dict": lit_sd}, ckpt_p)
    out_p = os.path.join(d, "scorer_final.pt")
    import subprocess
    r = subprocess.run([sys.executable, f"{FM}/extract_scorer.py", "--ckpt", ckpt_p,
                        "--config", f"{FM}/config/fmhead_fm3_kin_joint.yaml", "--out", out_p],
                       capture_output=True, text=True)
    print("  " + r.stdout.strip().replace("\n", "\n  "))
    if r.returncode != 0:
        print(r.stderr); ok = False
    s = torch.load(out_p, map_location="cpu")
    c = s["config"]
    m = FMHeadScorer(env_in_dim=c["env_in_dim"], d_model=c["d_model"],
                     n_decoder_layers=c["n_decoder_layers"], num_poses=c["num_poses"], metrics=c["metrics"])
    m.load_state_dict(s["scorer"])   # must load cleanly (fp32)
    print(f"  extracted scorer reloads into agent format OK (config={c})")

# (d) training_step control flow: batched-token unwrap + FSDP uniform participation ----
#     When a token has no metric cache, the scorer must STILL run and produce a zero-grad loss
#     for every scorer param (else FSDP reduce-scatter desyncs across ranks -> hang).
print("== (d) token unwrap + masked-loss uniform participation ==")
# batched token (default collate) -> unwrap to scalar, exactly as training_step does
batched_tok = [tok]
tok0 = batched_tok[0] if isinstance(batched_tok, (list, tuple)) else batched_tok
assert isinstance(tok0, str) and score_candidates(cands, tok0, tools, interval=0.5) is not None
# a token guaranteed absent -> labels None path -> masked loss, all params get a (zero) grad
sc = FMHeadScorer(env_in_dim=2048, d_model=64, n_decoder_layers=1, num_poses=T, metrics=V1_METRICS)
sc = sc.to(torch.bfloat16)
ctx = torch.randn(1, 12, 2048, dtype=torch.bfloat16, requires_grad=True)
cand_t = torch.randn(1, N, T, 3, dtype=torch.float32)
mask = torch.ones(1, 12, dtype=torch.bool)
assert score_candidates(cands, "definitely_not_a_real_token", tools) is None

def _grad_set(loss_fn):
    """param names that receive grad under a given loss builder -> FSDP participation set."""
    sc.zero_grad(set_to_none=True)
    out = sc(ctx, cand_t, env_mask=mask, weights=V1_WEIGHTS)
    loss_fn(out).backward()
    return {n for n, p in sc.named_parameters() if p.requires_grad and p.grad is not None}, out

tgt = {m: torch.as_tensor(labels[m], dtype=torch.float32).unsqueeze(0) for m in V1_METRICS}
normal_set, _ = _grad_set(lambda o: sc.distill_loss(o, tgt)["total"])          # labels-present path
masked_set, mo = _grad_set(lambda o: sum(o[m].sum() for m in V1_METRICS if m in o) * 0.0)  # masked
assert masked_set == normal_set, (
    f"masked path grads a DIFFERENT param set than normal -> FSDP deadlock.\n"
    f"  only-normal={sorted(normal_set - masked_set)[:4]}\n"
    f"  only-masked={sorted(masked_set - normal_set)[:4]}")
sc.zero_grad(set_to_none=True)
o2 = sc(ctx, cand_t, env_mask=mask, weights=V1_WEIGHTS)      # fresh graph for magnitude check
(sum(o2[m].sum() for m in V1_METRICS if m in o2) * 0.0).backward()
assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in sc.parameters()), "masked!=0-grad"
print(f"  unwrap('{tok0[:10]}') OK; masked grad-set == normal grad-set ({len(masked_set)} params), "
      f"masked loss is zero-grad -> FSDP-uniform OK")

print("\nALL", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
