# FM Head v2 — Multi-Token Cross-Attention Conditioning

Status: **implemented + smoke-tested (CPU / text-only).** Ready to train once trainval
sensors are restored (Phase C extraction). No change to launch commands.

This note documents the v2 conditioning rewrite agreed in `fm_rewrite_plan.md`: stop
conditioning the flow head on a single pooled `c`, and instead condition it on a
**sequence** of VLM hidden states via **multi-token cross-attention** (GoalFlow-style
joint attention). This is the main fix for the weak-conditioning problem (H1a).

---

## 1. What the context sequence is

At inference the only VLM hidden states that exist are those over the **prompt span**
(there are no ground-truth action tokens yet). v2 conditions on exactly that span:

- `context_sequence_hidden(hidden_last, input_ids, attention_mask, action_start_id, max_len)`
  (in `autovla_fmhead.py`) returns:
  - `ctx`  — `(B, T_ctx, 2048)`: the VLM **last-layer** hidden states over prompt
    positions `0 .. anchor`,
  - `ctx_mask` — `(B, T_ctx)`, `1 = valid`, `0 = right-pad`.
- **anchor** = the generation-start token, identical in train and inference:
  - train (teacher-forced, action tokens present): `anchor = (first action token) - 1`;
  - inference (no action tokens): `anchor = last attended token`.
- We keep **only positions `<= anchor`** — never the GT action tokens. By causal
  attention those hidden states are independent of anything that comes after the anchor,
  so `ctx` is **bit-identical** whether or not action tokens are appended → train == infer.
- The last `max_len` (default **64**) attended prompt tokens are kept (the answer-region
  context, which at the real vision node sits after thousands of vision tokens); rows are
  right-padded to the batch max and masked.

**Style** stays a single vector: `s = alpha * (anchor_styled - anchor_base)` computed from
the same anchor position (persona weight-space axis). Style OFF ⇒ `s = 0`.

So: **content = the prompt-span sequence (cross-attn); style = the anchor-delta vector (AdaLN).**

## 2. How cross-attention is wired

`fm_head.py`:

- New `CrossAttention(dim, num_heads)`: trajectory tokens are the **query**; the projected
  context tokens are **key/value**. `ctx_mask` becomes an SDPA boolean attn-mask so padded
  positions are ignored.
- `DiTBlock` now has **three** sub-layers with **9-way AdaLN-Zero** modulation
  (shift/scale/gate for self-attn, cross-attn, MLP):
  1. self-attn over the 10 trajectory tokens (unchanged),
  2. **cross-attn**: traj tokens attend to the context sequence (new),
  3. MLP.
- `FMHead.ctx_proj = Linear(d_ctx=2048 → hidden=256) + LayerNorm` projects the VLM width to
  head width once; the projected memory is shared by all blocks.
- The AdaLN conditioning vector is now `t_emb + s_emb` (flow-time + **style only**). Scene
  content is no longer summed into the AdaLN vector — it enters exclusively through
  cross-attention. Style-flip + CFG on `s` are preserved.
- AdaLN-Zero init: the cross-attn gate starts at 0 (identity-ish velocity field for stable
  early training). Consequence: at step 0, `d(loss)/d(ctx) = 0` and grad does not yet reach
  the LM — this is expected and disappears after the gates move off zero (verified).

### Dims (real AutoVLA / Qwen2.5-VL-3B)

| tensor | shape | note |
|---|---|---|
| `ctx` | `(B, T_ctx≤64, 2048)` | prompt-span VLM hidden sequence |
| `ctx_mask` | `(B, T_ctx)` | 1 valid / 0 pad |
| `s` | `(B, 2048)` | anchor-delta style axis (0 when style off) |
| `ctx_proj(ctx)` | `(B, T_ctx, 256)` | head width memory |
| trajectory tokens | `(B, 10, 256)` | 10 waypoints @ 0.5 s |
| velocity out | `(B, 10, 2)` | (x_forward, y_left) |

Head size: **2.29 M** params (was ~2.4 M; dropped the pooled `c_proj`, added `ctx_proj` +
per-block cross-attn).

## 3. What changed vs the single-`c` version

| aspect | v1 (single pooled c) | v2 (multi-token) |
|---|---|---|
| content signal | 1 vector `answer_position_hidden` (B,2048) | **sequence** `(B,T_ctx,2048)` prompt span |
| injection | summed into AdaLN vector | **cross-attention** memory in every DiT block |
| AdaLN vector | `t + c + s` | `t + s` (style only) |
| style axis | anchor delta `s` | anchor delta `s` (**unchanged**) |
| CFG | on `s` | on `s` (**unchanged**) |
| loss / target | rectified flow `z1-z0` | **unchanged** |
| normalization | z-score `traj_norm_stats_gt` | **unchanged** |
| frame | (x_fwd, y_left), 10-pose @0.5 s | **unchanged** |
| bf16 handling | `_param_dtype` casts | **unchanged** |
| inference steps | 10 Euler | **30 Euler** (GoalFlow uses 100) |
| train==infer | anchor causal-identical | **sequence causal-identical (cos=1.0)** |

Kept verified-correct: rectified-flow loss/target, z-score normalization, frame/axis/units,
bf16 self-consistency, CFG on style, robust `append_heading`.

## 4. Backward compatibility

`FMHead.forward` / `sample` / `flow_matching_loss` accept a pooled `(B, 2048)` context too
(auto-unsqueezed to a length-1 sequence). So the existing GPU/integration scripts
(`test_integration.py`, `smoke_all.py`, `real_gpu_smoke.py`) still run green via that
fallback; the true multi-token path is exercised by `train_fmhead.py`, `fmhead_sft.py`,
`eval_feasibility.py`, and `fmhead_navsim_agent.py`, which all build the SAME context
sequence through the single shared helper `context_sequence_hidden`.

## 5. Smoke results (this box: CPU / text-only)

`python smoke_v2.py` — all PASS:

| test | result |
|---|---|
| 1 shapes: `(B,T_ctx,2048)`→cross-attn→`(B,10,2)`; loss fp32 | PASS |
| 2 dtype-safety: fp32 **and** bf16 head forward/loss/sample finite | PASS |
| 3 ctx_mask: padded context tokens can't change output (`max|Δ|=0`) | PASS |
| 4 **train==infer**: ctx bit-identical w/ vs w/o action tokens, `min_cos=1.000000` (ctx & anchor) | PASS |
| 5 grad flow: frozen⇒only head grads; full-FT⇒grad reaches LM (after gates leave 0) | PASS |

`python train_smoke.py` — all PASS (mechanism preserved under cross-attn conditioning):

| test | result |
|---|---|
| multimodality: recovers both left/right modes, balanced (0.61/0.39) | PASS |
| style control: style-flip rate **1.000** | PASS |
| CFG: `w=0` spreads (frac 0.39), `w≥1` sharpens to consistent mode (1.00) | PASS |

### Train == infer causal-consistency check (the key guarantee)

`smoke_v2.py::test_train_infer_consistency` uses a real causal transformer as a VLM
stand-in: it runs the **prompt only** (inference view) and the **prompt + GT action tokens**
(train view), extracts the context via the shared helper for both, and confirms
`max|Δ| = 0`, `min_cos = 1.000000` for the whole sequence **and** the anchor vector. This is
the same causal-invariance guarantee the single-token v1 had, now for the full sequence.
(The real-Qwen version of this check runs through the standard consumers once GPU/data
returns; the mechanism is identical.)

## 6. Launch commands (UNCHANGED once trainval sensors are back)

Full-FT SFT (feasibility regime, style OFF):

```bash
cd /root/workspace/fmhead
bash run_fmhead_sft.sh config/fmhead-sft-full.yaml
```

Frozen-VLM GT feasibility train:

```bash
cd /root/workspace/fmhead
python train_fmhead.py --config config/fmhead_gt_feasibility.yaml
```

PDMS eval (30-step sampling, SELECT=pdm|oracle|medoid all still work):

```bash
cd /root/workspace/fmhead
bash run_pdms_fmhead.sh          # SELECT env: medoid | pdm | oracle
```

Feasibility L2 eval:

```bash
cd /root/workspace/fmhead
bash run_eval_feasibility.sh config/fmhead_gt_feasibility.yaml
```

The only config change is `fmhead.num_steps: 10 → 30` in
`config/fmhead_gt_feasibility.yaml` and `config/fmhead-sft-full.yaml`.

## 7. Deferred (clean hooks left)

- **Goal-point token** (GoalFlow's 8th-pose goal): NOT added in v1. Clean hook: append a
  goal embedding as an extra context token (with its own mask entry) inside
  `context_sequence_hidden` / the head's `ctx_proj`. Deferred to **v2.1** (needs a goal
  scorer / candidate vocabulary to be worthwhile).
- **Learned scorer / real GoalFlow-style selection**: `decode_all` already returns all N
  candidates; PDM/oracle selection works in the agent. A learned imitation+DAC scorer to
  replace medoid is future work (v2.2).
- **ctx_max_len tuning**: fixed at 64 (last-K prompt tokens). At the real vision node the
  prompt is dominated by vision tokens; if broader scene grounding helps, raise the cap or
  add strided pooling over the full attended prompt. Left as a config knob.

---

## 8. Real-VLM magnitude / dim / grad audit (`v2_scale_check.py`)

Ran on the **real Qwen2.5-VL-3B** (text-only path, 1 MIG slice), B=3 prompts, S=49 tokens.
Full table + verdicts in `v2_scale_check.json`. **ALL 7 items PASS**; nothing off by an
order of magnitude.

| tensor | shape | dtype | mean | std | min | max | norm |
|---|---|---|---|---|---|---|---|
| vlm.hidden_last | (3,49,2048) | bf16 | -0.033 | 4.29 | -153 | 111 | 2353 |
| **ctx** | (3,49,2048) | **bf16** | -0.036 | 4.19 | -153 | 111 | 2296 |
| ctx per-token L2 norm | — | — | **mean 193.2** | — | 137.5 | 232.0 | — |
| anchor (style src) | (3,2048) | bf16 | 0.039 | 4.90 | -68.5 | 106.5 | 384 |
| **s (s=0 regime)** | (3,2048) | bf16 | **0** | **0** | **0** | **0** | **0** |
| ctx_proj(ctx) | (3,49,256) | fp32 | ~0 | **0.976** | -3.59 | 3.52 | 189 |
| x_embedder(x_t) | (3,10,256) | fp32 | 0.0003 | 0.113 | -0.52 | 0.54 | 9.9 |
| t_embedder(t) | (3,256) | fp32 | -0.018 | 0.417 | -1.28 | 1.02 | 11.6 |
| cvec (t+s) | (3,256) | fp32 | -0.018 | 0.417 | -1.28 | 1.02 | 11.6 |
| AdaLN (all 9 params) | (3,256) | fp32 | **0** | **0** | **0** | **0** | **0** |
| gt_xy (metres) | (3,10,2) | fp32 | 5.04 | 7.91 | -0.17 | **33.5** | 72.3 |
| **z1 (normalized)** | (3,10,2) | fp32 | -0.149 | 0.790 | -2.67 | **0.92** | 6.18 |
| z0 (noise) | (3,10,2) | fp32 | 0.260 | 0.977 | -1.82 | 2.54 | 7.77 |
| u = z1−z0 (target vel) | (3,10,2) | fp32 | -0.409 | 1.30 | -3.01 | 2.49 | 10.5 |
| v (pred vel, init) | (3,10,2) | fp32 | **0** | 0 | 0 | 0 | 0 |
| **flow loss** | scalar | fp32 | **2.18** | — | — | — | — |

Key scale confirmations:
- **ctx magnitude sane**: per-token L2 norm **~193** (bf16), same order as the earlier
  `‖c‖≈221`. The `ctx_proj` **LayerNorm** is essential and verified: it tames the norm-193
  bf16 hidden to **std≈0.98** before cross-attn (no large-activation blow-up).
- **s = exactly 0** in the s=0 regime (no NaN).
- **AdaLN-Zero holds**: all 9 modulation params = 0 at init ⇒ `v = 0` at init (identity flow).
- **Normalization does metres→O(1)**: raw gt is **O(10) m** (max 33.5), z-score `z1` is
  **O(1)** (max |2.67|). Loss = **2.18** — O(1), NOT ~82 (the un-normalized bug) and NOT ~0.
- **Dims** all correct at hidden width 2048 (ctx 2048 → proj 256 → traj/vel 10×2).
- **Grad flow**: frozen ⇒ **7.50 M** head grad elems, all finite, **VLM grads = 0**;
  full-FT ⇒ grad reaches the LM backbone (unfroze last decoder layer 35, all 10 tensors got
  finite grad through `ctx`). No NaN/inf anywhere.
- **Dtype-safe** with real bf16 ctx: fp32 head → loss 1.55 (fp32); bf16 head → loss 1.89 (fp32).

Verdict: v2 magnitudes/dims/grads are all sane on real VLM dims — the earlier scale bug
(un-normalized O(10) m target ⇒ loss ~82) is not present.

---

## 9. Style adapter (opt-in, additive, zero-gated) — Option A

Goal: add persona style **without touching the 0.9088 full-FT v2 head**. The style path is a
SEPARATE, opt-in module; when it is off (or `s=0`) the forward is **byte-identical** to the
current v2 head, so the existing eval/PDMS/L2 commands and the 0.9088 result are untouched.

### What's new (all default-OFF)
- `FMHeadConfig`: `use_style_adapter=False`, `style_adapter_hidden=256`,
  `style_adapter_mode="adaln"` (`"cross_attn"` = reserved Option-B hook, raises if selected).
- `DiTBlock.forward` / `FinalLayer.forward` / `FMHead.forward` gain an OPTIONAL
  `style_mod=None` / `style_mods=None`. When `None` the modulation is computed exactly as
  before (`mod = adaLN(cvec); mod.chunk(...)`), so the add is skipped — no fp reordering.
- `StyleAdapter` (new module): `s_H (2048) → per-block (9·C) + final (2·C)` additive AdaLN
  deltas. Bias-free trunk (`Linear→SiLU`, no LayerNorm) + fixed input scale `1/sqrt(d_style)`
  ⇒ `g(0)=0` exactly AND `g` is magnitude-sensitive so `s=alpha·delta` gives a monotonic
  **dose-response** (an input LayerNorm would be scale-invariant and kill the alpha knob).
  **Only the gates are zero-init** (heads small-random): `delta = gate·head(trunk(s))` is 0 at
  init, yet the gate has a live gradient (head≠0) — avoids the double-zero dead-gradient.
- `StyleFMHead(FMHead)`: content path identical to FMHead; the frozen base **always sees zero
  style** (matching the 0.91 s=0 training), style enters ONLY via the adapter delta, and a
  per-sample active mask (`|s|>0 and not drop_style`) skips the adapter entirely when off.
- `FMHeadDecoder`: builds `StyleFMHead` iff `use_style_adapter`; `freeze_base_keep_style()`
  freezes the loaded 0.91 head + normalizer and trains ONLY the adapter;
  `style_adapter_parameters()` feeds the optimizer.

### Frozen vs trainable (style phase)
- **Frozen (8.55 M):** the entire loaded 0.9088 head (x_embedder, temporal_pos, t/s/ctx proj,
  all DiT blocks, final layer) + normalizer buffers.
- **Trainable (~3.01 M):** only `StyleAdapter` (trunk + per-block/final heads + gates).

### Bit-identical s=0 proof (`smoke_style_adapter.py`, CPU, loads the real `fm_decoder.pt`)
| check | numbers | result |
|---|---|---|
| plain(s=0) vs style(s=0) | `max\|Δ\| = 0.00e+00` | exact |
| plain(s=0) vs style(s!=0) at zero-gate init | `max\|Δ\| = 0.00e+00` | exact |
| controllability (simulated-trained adapter) | zero-gate@a0=0.0, `\|Δ\|@0.5=0.0060`, `\|Δ\|@1.0=0.0122` (monotonic) | PASS |
| grad freeze | trainable 3.01M / frozen 8.55M, base grads=0, adapter grads>0 | PASS |
| dtype fp32 + bf16 | shapes (B,10,2), finite | PASS |

Also: `python fm_head.py` self-test and `smoke_v2.py` produce the **same numbers as before**
(plain path untouched); the eval/PDMS default (`use_style_adapter=false`) loads the 0.91 ckpt
with `strict=True`.

### Option B hook (not wired)
`style_adapter_mode="cross_attn"` is reserved for a second zero-gated cross-attention to a
multi-token style delta; it currently raises `NotImplementedError` (clean hook only).

### (Future) train command — DO NOT run yet
```bash
cd /root/workspace/fmhead
python train_fmhead.py --config config/fmhead_style_ddv2.yaml
# freezes the full-FT v2 head, trains ONLY the ~3.01M style adapter; teacher=DDv2,
# style-dropout 0.2 (CFG), init from fm_decoder.pt, base_ckpt = full-FT base (content match).
```
Open questions: (1) base for the DDv2 delta — full-FT base (content-matched, but the LoRA is
off-distribution there) vs PDMS_89 (LoRA in-distribution, needs a frozen-base head init);
(2) teacher target needs `teacher_trajectory` per token (join the DDv2 style cache) and ideally
a human(s=0)+DDv2(s=1) mix for a true dose-response.

### First-try training prep (DDv2, prepared — NOT launched)
Data (verified): `nocot_ddv2_teacher12k` (gt_trajectory = DDv2 teacher, (10,3)) and
`nocot_navtrain12k` (human GT) each have 11988 tokens, **100% token overlap**; DDv2 differs
from human by ~1.6 m mean-xy (so the teacher field is genuine, not a human copy). For the
additive-FROZEN design we train ON the DDv2 set (`style_target=gt` -> target = DDv2); the
s=0 anchor is the frozen 0.91 head (no human join needed at train time).

Config `config/fmhead_style_ddv2.yaml`: base_ckpt = full-FT base, init_fmhead_ckpt =
full-FT `fm_decoder.pt` (frozen), lora_adapter_path = DDv2 `lora_final` (frozen), style_alpha=1,
style_dropout=0.0 (uncond = frozen head), ckpt_dir = `.../fmhead_ckpts_style_ddv2` (NOT the GOLD
or the 0.91 run dir). Only the ~3.01M adapter trains; VLM+LoRA (3.91B) and the 8.55M head frozen.

Real-loop smoke (text-only, `--no_sft_base`, GPU): loads Qwen + DDv2 LoRA (style ON) + the 0.91
head, freezes VLM+LoRA (requires_grad=0) and the head, trains only 3.01M adapter, computes the
DDv2 flow loss, steps, and saves+reloads a ckpt to the safe dir. Logged loss 2.90->2.37 over 20
steps; the controlled head-level smoke (`smoke_style_adapter.py` test 5) shows a clean
103.96->17.03 decrease AND s=0 byte-identical AFTER training.

Launch (cluster; NOT run here):
```bash
cd /root/workspace/fmhead
# single GPU:
python train_fmhead.py --config config/fmhead_style_ddv2.yaml
# 4-GPU DDP:
torchrun --standalone --nproc_per_node=4 train_fmhead.py --config config/fmhead_style_ddv2.yaml
```
