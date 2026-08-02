# FMHead — Conditional Rectified Flow-Matching Trajectory Decoder Head

A **standalone**, well-documented conditional **rectified flow-matching** head that
generates a future ego trajectory `τ ∈ ℝ^(T×2)` conditioned on a scene/content
vector `c` and a style vector `s`. It is designed to later **drop-in replace
AutoVLA's token→codebook trajectory decoder** — *without modifying any existing
repo code*. This directory contains only the new module + smoke tests; nothing in
AutoVLA or the reference repos was touched.

```
/root/workspace/fmhead/
├── fm_head.py               # the module: FMHead + FMHeadConfig + DiT/AdaLN blocks
├── synthetic.py             # tiny synthetic bimodal trajectory dataset (mechanism smoke tests)
├── train_smoke.py           # the 4 mechanism smoke tests (shape / multimodality / style / CFG)
├── autovla_fmhead.py        # REAL AutoVLA integration (extractor + FMHeadDecoder + subclass)
├── test_integration.py      # integration smoke test w/ mock Qwen VLM (correct 2048-dim hidden)
├── smoke_results.json        # results of train_smoke.py
├── integration_results.json  # results of test_integration.py
├── smoke_run.log / integration_run.log
└── README.md                # this file
```

> **Storage policy:** all *code* lives under `/root/workspace/fmhead/`. These smoke
> tests produce no large artifacts; if real training checkpoints/datasets are ever
> produced they should go to `/mnt/pfs/zhengguantian/…` instead.

---

## 1. Design

### 1.1 What it models

A **conditional velocity field** `v_θ(x_t, t, c, s)` defining a rectified-flow ODE
whose t=0→t=1 transport maps standard Gaussian noise `ε` to a data trajectory `τ`.

- **Trajectory** `τ`: shape `(B, T=8, 2)` — 8 waypoints @ 0.5 s, `(x, y)` in the
  ego frame (our canonical grid).
- **Content condition** `c`: shape `(B, D_cond)`, default `D_cond=2048` to match an
  AutoVLA answer-position hidden state (`h_base`).
- **Style condition** `s`: shape `(B, D_style)`, default `D_style=2048`. Intended
  default = the **activation-space style delta** `s = h_styled − h_base`. Also
  supports a small explicit style vector (e.g. 6-D kinematic) — just set `d_style`.

### 1.2 Architecture (DiT + AdaLN-Zero)

```
x_t (B,T,2) ──Linear──> tokens (B,T,C) + temporal-sincos-pos
                                   │
t ─ sinusoidal ─ MLP ─┐            │  cvec = t_emb + c_emb + s_emb   (B,C)
c ─ MLP(D_cond→C) ────┤──(sum)──►  │  drives AdaLN-Zero modulation
s ─ MLP(D_style→C) ───┘            ▼
              ┌─────────────  depth × DiTBlock  ─────────────┐
              │  self-attn over T tokens  (AdaLN shift/scale/gate)
              │  MLP                        (AdaLN shift/scale/gate)
              └───────────────────────────────────────────────┘
                                   │
                          FinalLayer (AdaLN) ──Linear──> v (B,T,2)
```

- **Injection:** `c` and `s` are each projected to model width `C` by a small MLP,
  summed with the timestep embedding into a single conditioning vector `cvec`,
  which drives **AdaLN-Zero** modulation (shift/scale/gate) inside every DiT block
  and the final layer. This is the standard DiT conditioning path.
- **Classifier-free guidance (on STYLE):** a learnable **null-style** embedding is
  substituted for `s_emb` when style is "dropped". During training we drop style
  with prob `style_dropout_prob` (default 0.2); at sampling we run the field twice
  (with and without style) and combine (see §1.4).
- **AdaLN-Zero init:** the last linear of every AdaLN modulation and the whole final
  projection are zero-initialized, so the head starts as a near-identity velocity
  field → stable early training.

### 1.3 Flow-matching training

For each sample: `t ~ U(0,1)`, `ε ~ N(0,I)`, and

```
x_t = (1 − t)·ε + t·τ            # linear/rectified interpolation
u   = τ − ε                       # constant target velocity
loss = MSE( v_θ(x_t, t, c, s), u )
```

(`FMHead.flow_matching_loss(τ, c, s)`.)

### 1.4 Sampling

Draw `N` noises `ε ~ N(0,I)` and Euler-integrate the ODE from `t=0` to `t=1` over
`K` steps (`K` configurable, ~5–10):

```
x ← ε
for i in range(K):
    v = v_θ(x, t=i/K, c, s)
    x ← x + v·(1/K)
return x                          # (B, N, T, 2), all N samples kept (multimodality)
```

**CFG at sampling** (guidance weight `w`), applied on the style condition:

```
v = v_uncond + w · (v_cond − v_uncond)
```

where `v_uncond` uses the null-style embedding. `w=1` ⇒ plain conditional; `w=0` ⇒
unconditional; `w>1` ⇒ sharpen toward the style-consistent mode.

`FMHead.select_by_score(trajs, scorer)` is a **placeholder** for later PDM-style
selection among the `N` samples (defaults to a cheap medoid heuristic if no scorer
is given).

---

## 2. Interface contract

```python
from fm_head import FMHead, FMHeadConfig

head = FMHead(FMHeadConfig(d_cond=2048, d_style=2048, horizon=8))

# --- velocity forward ---
v = head(x_t, t, c, s)                 # x_t:(B,T,2) t:(B,) c:(B,2048) s:(B,2048) -> v:(B,T,2)

# --- training loss (scalar) ---
loss = head.flow_matching_loss(tau, c, s)          # tau:(B,T,2)

# --- sampling: N multimodal candidates per scene ---
trajs = head.sample(c, s, num_samples=N, num_steps=K, cfg_weight=w)   # -> (B,N,T,2)

# --- selection placeholder (PDM later) ---
best = FMHead.select_by_score(trajs, scorer=None)  # -> (B,T,2)
```

| Symbol | Shape | Meaning |
|---|---|---|
| `x_t` | `(B, T, 2)` | noised/interpolated trajectory at flow-time `t` |
| `t`   | `(B,)` or scalar | flow-time in `[0,1]` (0=noise, 1=data) |
| `c`   | `(B, D_cond)` | scene/content condition (`h_base`) |
| `s`   | `(B, D_style)` | style condition (`h_styled − h_base`, or explicit kinematic) |
| `tau` | `(B, T, 2)` | ground-truth trajectory (training target) |
| output `v` | `(B, T, 2)` | predicted velocity |
| `sample()` | `(B, N, T, 2)` | N candidate trajectories |

**Coordinate/scale note:** the head is scale-agnostic but, like all flow/diffusion
heads, trains best on standardized targets. The smoke tests z-score trajectories
into a well-conditioned space (see `synthetic.py::Normalizer`); a real integration
should apply AutoVLA's trajectory normalization (or a fixed per-dim scale) before
`flow_matching_loss` and invert it after `sample`.

---

## 3. How it would plug into AutoVLA (description only — NOT done here)

AutoVLA (`models/autovla.py`) currently produces a trajectory by autoregressively
generating **action tokens** and decoding them through a **codebook**:

- `predict(self, input_features)` → `vlm.generate(...)` → keep tokens `>= action_start_id`
  → `action_tokenizer.decode_token_ids_to_trajectory(actions_tokens)` → `trajectory`.
- `training_step(...)` supervises those action-token logits (cross-entropy /
  differentiable codebook decode against the GT trajectory).

FMHead replaces the **token→codebook decode** with a **continuous generative decode**
off the LLM hidden state. The key insight: everything FMHead needs is already
present — the hidden state at the answer/action position is exactly our `c = h_base`.

**Plug-in without editing AutoVLA** (do it in a *new* subclass / wrapper module):

1. **Get the condition `c`.** Run the existing VLM forward with
   `output_hidden_states=True` and read the last-layer hidden state at the
   answer/action-token position (the same position whose logits currently feed the
   codebook). That `(B, 2048)` vector is `c`. For style, run a second (styled)
   context/prompt or persona-LoRA pass to get `h_styled` and set `s = h_styled − h_base`
   (or pass an explicit kinematic style vector).

2. **Inference — wrap `predict`.** In a subclass `AutoVLA_FM(AutoVLA)` override a new
   method (or a thin wrapper function) that, instead of calling
   `decode_token_ids_to_trajectory`, calls:
   ```python
   c, s = extract_hidden_states(vlm_outputs)          # (1,2048),(1,2048)
   trajs = fm_head.sample(c, s, num_samples=N, num_steps=K, cfg_weight=w)  # (1,N,8,2)
   trajectory = FMHead.select_by_score(trajs, pdm_scorer)  # (8,2) after de-normalize
   ```
   The base `predict()` stays untouched; the wrapper reuses `get_prompt`, the vlm,
   and the tokenizer, and only swaps the final decode.

3. **Training — add an auxiliary/replacement loss.** In a subclass
   `training_step`, after the normal forward, extract `c` (and `s`) from the hidden
   states and add `fm_head.flow_matching_loss(gt_traj_norm, c, s)` to the loss dict.
   No AutoVLA line is modified; the new `training_step` calls `super().training_step`
   (or reuses its pieces) and augments the returned loss. The FMHead is an
   `nn.Module` registered as a child of the wrapper, so it trains jointly (or the
   base can be frozen and only FMHead + a small adapter trained).

4. **Determinism / eval parity.** Keep AutoVLA's trajectory normalization for `τ`
   so `sample()` outputs land in metres after inverting it; feed the `N` candidates
   to the existing PDM/NAVSIM scorer via `select_by_score`.

This is a **description**; no AutoVLA code was changed.

---

## 4. Reference implementations studied (not modified)

Under `/root/workspace/closed_loop/navsim_candidates_survey/repos/`:

| Repo | File(s) | Conventions borrowed |
|---|---|---|
| **TrajDiff** | `navsim/agents/trajdiff/modules/DiT.py`, `.../planner_head.py` | `TimestepEmbedder` (sinusoidal + MLP), `modulate(x,shift,scale)`, `TrajDiTBlockV2/V3` **AdaLN-Zero** DiT blocks (6-way shift/scale/gate chunk), `FinalLayer`, frozen **temporal sin-cos** pos-embed, xavier + **zero-init** of AdaLN & final proj |
| **GoalFlow** | `navsim/agents/goalflow/goalflow_model_traj.py` | **rectified flow** `get_train_tuple` (`z_t=t·z1+(1−t)·z0`, `target=z1−z0`), **Euler** integration `x += v·(1/steps)`, **classifier-free guidance** via train-time condition **dropout** (`force_dropout`/`navi_dropout`) + cond/uncond velocity combination at sampling |
| **DiffusionDrive** | `navsim/agents/diffusiondrive/*` | multi-sample / anchored multimodality (draw many, keep all, score-select) — reflected in our `sample(num_samples=N)` + `select_by_score` placeholder |

All borrowed as *conventions*; `fm_head.py` is an original, minimal, dependency-light
rewrite (only `torch`; smoke tests additionally use `numpy` + `scikit-learn`).

---

## 5. Smoke-test results

Run: `OMP_NUM_THREADS=8 python train_smoke.py` (device: **CUDA**, torch 2.4.0,
total wall time ≈ **33 s**). All four tests **PASS**. Raw numbers in
`smoke_results.json`; captured log in `smoke_run.log`.

### Test 1 — shape ✅ PASS
`v_θ(x_t, t, c, s)` on random `(B=5)` inputs → output `(5, 8, 2)` as required.

### Test 2 — overfit / multimodality ✅ PASS
Single fixed scene `c₀` with a **bimodal** target (go-left arc ↔ go-right arc,
chosen 50/50, style uninformative). Trained 1200 steps.

- **Flow loss** dropped **1.133 → 0.361** (final/init ratio **0.319**, ~68% drop).
- Sampling **N=256** trajectories and KMeans-2 on the endpoints recovered **both**
  modes:
  - cluster centres `(m)`: `[16.0, −4.06]` and `[16.0, +4.16]`
  - canonical ends `(m)`: `[16.0, +4.0]` (left), `[16.0, −4.0]` (right)
  - centre→canonical distances: **0.06 m** and **0.16 m** (both < 1.5 m)
  - mode fractions: **0.582 / 0.418** (balanced — both modes present)

  ⇒ the head models a genuine multimodal distribution (a plain regressor would
  collapse to the mean `≈[16, 0]`).

### Test 3 — style control ✅ PASS
Same scene, but the mode is now determined by the **style** (`s=+ → left`,
`s=− → right`). Trained 1500 steps (with style-dropout for CFG).

- Conditioning on `s=+`: **frac_left = 1.000** (wants high)
- Conditioning on `s=−`: **frac_right = 1.000** (wants high)
- **style-flip rate = 1.000** ⇒ style fully controls which mode is sampled.

### Test 4 — CFG ✅ PASS
Guidance weight `w` on the style-consistent mode (`s=+` → left). N=256 samples.

| `w` | frac on style-consistent mode | mean endpoint dist to that mode (m) |
|---|---|---|
| 0.0 (uncond) | 0.434 | 4.203 |
| 1.0 (cond)   | 1.000 | **0.176** |
| 2.0          | 1.000 | 0.272 |
| 4.0          | 1.000 | 0.454 |

- Going **unconditional → conditional** is the dominant CFG effect: samples snap
  from a ~50/50 mixture (mean endpoint 4.20 m away, i.e. spread across both modes)
  onto the style-consistent mode (0.18 m). The discrete fraction saturates at 1.0.
- **Honest nuance:** pushing `w` beyond 1 slightly **overshoots** the data mode
  (dist 0.18 → 0.27 → 0.45 m). This is the well-known CFG over-guidance /
  off-manifold effect; for this cleanly-separated toy task `w≈1` is already optimal.
  Pass criterion (`w=4` vs `w=0`: fraction up **and** distance down) holds.

### Summary

| Test | Result | Headline numbers |
|---|---|---|
| 1 shape | **PASS** | out `(5,8,2)` |
| 2 multimodality | **PASS** | loss 1.133→0.361 (0.32×); both modes @ 0.06/0.16 m; fracs 0.58/0.42 |
| 3 style control | **PASS** | style-flip rate 1.000 |
| 4 CFG | **PASS** | uncond→cond dist 4.20 m→0.18 m; over-guidance overshoots (expected) |

---

## 6. Design decisions & uncertainties

- **Conditioning by summation into one AdaLN vector** (`t_emb+c_emb+s_emb`) rather
  than cross-attention to condition tokens. Simpler and sufficient for global scene
  + style vectors; if AutoVLA later exposes a *sequence* of context tokens, swapping
  in a cross-attention DiT block (as in TrajDiff's `TrajDiTBlockV2`) is a drop-in.
- **CFG on style only** (content is always kept). This matches the intended use:
  controllable *style* while the scene remains fully conditioned.
- **Rectified (linear) flow** with constant target velocity `u=τ−ε` — the simplest,
  most robust flow-matching variant; enables few-step Euler sampling (K≈5–10).
- **Normalization matters.** Flow/diffusion heads want standardized targets; the
  smoke tests z-score. A real AutoVLA integration must reuse AutoVLA's own
  trajectory normalization (uncertainty: exact stats/scale used by AutoVLA's
  codebook grid — to be matched at integration time).
- **`select_by_score` is a stub.** Real deployment should wire the NAVSIM/PDM
  scorer (as in GoalFlow/DiffusionDrive) to pick among the N candidates.
- **Not yet validated on real data** — these are synthetic smoke tests proving the
  mechanism (multimodality, style control, CFG), not planning accuracy.

## 7. Reproduce

```bash
cd /root/workspace/fmhead
OMP_NUM_THREADS=8 /root/workspace/miniconda3/envs/autovla/bin/python fm_head.py           # module self-test
OMP_NUM_THREADS=8 /root/workspace/miniconda3/envs/autovla/bin/python train_smoke.py       # 4 mechanism smoke tests (~32s GPU)
OMP_NUM_THREADS=8 /root/workspace/miniconda3/envs/autovla/bin/python test_integration.py  # AutoVLA integration test (~22s GPU)
```

---

## 8. REAL AutoVLA integration (source-verified)

This section replaces the earlier description-only plan with **verified dims and
real, importable integration code** (`autovla_fmhead.py`). AutoVLA repo root:
`/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA/`. **No AutoVLA
file was modified.**

### 8.1 Real dims/shapes found (with source citations)

| Quantity | Value | Source (file:line) |
|---|---|---|
| VLM backbone | `Qwen2_5_VLForConditionalGeneration`, `torch_dtype=bfloat16` | `models/autovla.py:13`, `models/autovla.py:1072-1076` |
| **Hidden size (⇒ `c`,`s` dim)** | **2048**, dtype **bfloat16** | `Qwen2.5-VL-3B-Instruct/config.json` (`hidden_size:2048`); path from `config/training/*.yaml → model.pretrained_model_path` |
| **Trajectory horizon** | **num_poses = 10** @ `interval_length = 0.5 s` | `config/training/*.yaml → model.trajectory.{num_poses,interval_length}`; used `models/autovla.py:36-39` |
| Trajectory columns (decoder output) | `(T, 3)` = `(x_forward, y_left, heading)` | `models/action_tokenizer.py:95-124` (`rollout`), `models/autovla.py:1115`, `navsim/.../autovla_agent.py:481` |
| GT trajectory (train target) | `batch["gt_trajectory"]` `(B, G, ≥2)` metres, x-fwd/y-left; L2 uses `[...,:2]` | `models/autovla.py:757-808` (`_expected_trajectory_l2`) |
| Action sub-vocab | `[action_start_id, action_start_id+n_bins)`, `action_start_id=151665`, codebook `(n_bins,6,4,2)` | `models/action_tokenizer.py:43-55`, `config → model.tokens.action_start_id` |
| Loss/logits interface | `AutoVLA.forward(batch) → CausalLMOutputWithPast(.loss,.logits)`; labels `batch['labels']` | `models/autovla.py:1295-1306`, `:865-896` |
| **Style axis** | styled = LoRA-on forward; **base = `vlm.disable_adapter()` forward** | `models/autovla.py:939-942` (KL uses exactly this base/styled toggle) |
| Inference decode site | `AutoVLA.predict()` → codebook decode → `poses`; agent wraps `Trajectory(poses[:num_poses])` | `models/autovla.py:1088-1117`, `navsim/.../autovla_agent.py:473-481` |

**Scene condition `c`:** the last-layer VLM hidden state at the **answer/action
region**. In teacher-forced training we mean-pool the hidden states at action-token
positions (`labels ∈ [151665, 151665+n_bins)`, shift-aligned exactly as
`_expected_trajectory_l2`); at generation (no labels) we take the last attended
token's hidden state. Extracted via `vlm(..., output_hidden_states=True)`.
`c` dtype = bfloat16, dim 2048.

**Style vector `s`:** `s = alpha·(h_styled − h_base)`, where `h_styled` is the
LoRA-on hidden and `h_base` is the same forward under `vlm.disable_adapter()`.
Dim 2048.

### 8.2 What changed in FMHead to match

| Change | Before | After | Why |
|---|---|---|---|
| `FMHeadConfig.horizon` default | 8 | **10** | AutoVLA `num_poses=10` (the original 8 was the persona-doc grid, not the real config) |
| `d_cond` / `d_style` defaults | 2048 / 2048 | 2048 / 2048 (**already matched** Qwen2.5-VL-3B) | verified, kept |
| dtype handling | fp32-only | **cast `c,s,x_t,t` to param dtype inside `forward`** | accept AutoVLA **bf16** hidden states directly |
| heading | none (xy only) | added `FMHead.append_heading(xy)→(…,3)` | AutoVLA consumes navsim poses `(T,3)`; heading derived as `atan2(Δy,Δx)` |
| normalization | none | `FMHeadDecoder.{fit,set}_normalizer` + normalize in loss / denormalize in decode | AutoVLA trajectories reach ~30 m; flow matching needs standardized targets (raw-metre loss was ~82 and grad-starved; normalized ≈0.9 and healthy) |

### 8.3 Integration entry points (`autovla_fmhead.py`)

- `autovla_fmhead_config()` → `FMHeadConfig` with the exact real dims (2048/2048/10/2).
- `action_region_hidden(hidden_last, labels, attention_mask, action_start_id, n_bins)` → `c` `(B,2048)`.
- `style_delta(h_styled, h_base, alpha)` → `s` `(B,2048)`.
- `FMHeadDecoder` — `nn.Module` drop-in for the codebook decode:
  - `training_loss(gt_trajectory, c, s)` → scalar (normalizes `gt[...,:10,:2]`, calls `flow_matching_loss`).
  - `decode(c, s, scorer)` → navsim poses `(B,10,3)` (`sample` → denormalize → `select_by_score` → `append_heading`).
  - `fit_normalizer(gt)` / `set_normalizer(mean,std)` — **must be fit on real data before training.**
- `build_autovla_fmhead_subclass()` → `AutoVLA_FMHead(AutoVLA)` (guarded import) that:
  - `predict()` — runs the VLM forward for hidden states (styled + `disable_adapter` base), builds `c,s`, calls `fm_decoder.decode`, returns `(poses(10,3), "")` — **replacing the codebook decode without editing `AutoVLA.predict`.**
  - `fm_training_loss(batch)` — returns the FM loss to add inside a subclassed `training_step` (base untouched).

### 8.4 Integration smoke-test results (`test_integration.py`, mock Qwen VLM, CUDA)

The heavy Qwen stack can't load here, so a `MockQwenVLM` emits **2048-dim bf16**
hidden states with a working `disable_adapter()` toggle. All tensors flow through
the real extraction + decode + train paths. **ALL PASS:**

| Test | Result | Key numbers |
|---|---|---|
| A dims match real AutoVLA | PASS | `d_cond=d_style=2048`, `horizon=10`, `traj_dim=2` |
| B hidden→`c`,`s` extraction | PASS | `c=(3,2048)` bf16, `s=(3,2048)` bf16, `|s|_mean=0.013` |
| C decode→navsim poses | PASS | `poses=(3,10,3)`, all finite (xy+heading) |
| D train-loss backward | PASS | normalized loss ≈0.9; grads reach **62/64** params (2 = frozen temporal-pos + null-style) |
| E tiny overfit thru interface | PASS | loss **0.905→0.267**; style-flip `s+→left=1.000`, `s−→right=1.000` |
| F subclass import guarded | PASS | raises `ModuleNotFoundError` outside AutoVLA env (expected) |

### 8.5 Design evaluation & go/no-go

**Sound:**
- *Mode-averaging avoidance* — generative flow head keeps N samples (Test 2/E recover both modes; a regressor/argmax-codebook cannot). ✅
- *Style axis* — `s = h_styled − h_base` is exactly AutoVLA's own base/styled toggle direction; CFG-on-style gives controllability (Test 3/E flip-rate 1.0). ✅
- *Safety selection* — `select_by_score` is the hook for the existing NAVSIM/PDM scorer (`models/utils/score.PDM_Reward`) to pick among candidates. ✅ (currently a medoid stub — must wire the real scorer).
- *Dims/dtype* — 2048 bf16 in, `(10,3)` metres out, verified end-to-end. ✅

**Risks / must-do before real training:**
1. **Normalizer must be fit on real `gt_trajectory` stats** (`fit_normalizer` over a data pass, or `set_normalizer` with precomputed mean/std). Un-fit → ill-conditioned (raw-metre loss ~82).
2. **`c` extraction position** — mean-pool over action positions is a reasonable default; validate against using the single answer/EOS-position hidden state on real data (may affect quality).
3. **2× VLM forward at inference** (styled + base for `s`). Acceptable but note the cost; can cache the base forward or precompute a fixed style vector.
4. **Heading** is derived (finite-difference), not modeled. Fine for NAVSIM xy-based PDM; if a task needs precise yaw, set `traj_dim=3` and train on `gt_trajectory[...,:3]`.
5. **Real scorer wiring** — replace the `select_by_score` medoid stub with `PDM_Reward`.

**GO / NO-GO:** **GO for a training experiment.** Tensor plumbing, dims (2048), dtype
(bf16), horizon (10), output contract (`(10,3)` metres), style axis, multimodality
and controllability are all verified end-to-end against real AutoVLA source. The
only blocking pre-req is fitting the trajectory normalizer on real data (now done,
§9); the scorer wiring can follow after the head trains. No AutoVLA files were modified.

---

## 9. REAL data + REAL model verification (GPU)

Extra deliverables:
`traj_norm_stats.json` (fitted normalizer), `real_gpu_smoke.py` + `real_gpu_results.json`
+ `real_gpu_run.log`.

### 9.1 Real `gt_trajectory` stats (where the data came from)

- **Source:** `config/training/youdrive-ddv2-persona-lora-big-l2.yaml → data.train.json_dataset_path`
  = `/mnt/pfs/zhengguantian/autovla/persona/nocot_ddv2_teacher12k` (the DDv2-teacher
  navtrain SFT set). Each JSON stores `gt_trajectory` as `(10, 3)` = (x_forward,
  y_left, heading) in metres. Verified the loader passes it through unmodified:
  `gt_pos_raw = poses[:, :2]` (`navsim/.../autovla_agent.py:69`, `dataset_utils/sft_dataset.py:134,343`).
- **Computed** per-(waypoint,dim) mean/std over `gt_trajectory[..., :2]` across
  **11,988** samples (0 bad) → saved to `traj_norm_stats.json`:

| waypoint (0.5 s each) | mean x (m) | mean y (m) | std x (m) | std y (m) |
|---|---|---|---|---|
| 1 | 2.25 | 0.02 | 1.71 | 0.10 |
| 5 | 11.75 | 0.39 | 7.67 | 1.93 |
| 10 (end) | 22.86 | 1.31 | 15.15 | 6.67 |

Global mean `(12.76, 0.56) m`, std `(11.51, 3.55) m`. (x-forward grows to ~23 m
over 5 s; y-left is near-zero-mean with growing spread — confirms these targets
MUST be standardized before flow matching.)

Load into the head with `FMHeadDecoder.load_normalizer("traj_norm_stats.json")`.

### 9.2 Did we run the real VLM? YES (LM path), with one env caveat

`real_gpu_smoke.py` **loads the real Qwen2.5-VL-3B-Instruct (bf16) on the A100**,
attaches a PEFT LoRA (AutoVLA's target modules; `lora_B` randomized so the adapter
actually perturbs activations), and extracts a **genuine** hidden state:

- `c` (styled, LoRA-on) = `(1, 2048)` **bfloat16**, ‖c‖≈221.5
- `s = h_styled − h_base` (base via `model.disable_adapter()`) = `(1, 2048)` bf16,
  mean|s|≈0.30 (**non-zero real base-vs-LoRA delta**)

**Caveat (env, not code):** this container has NVML permission-restricted
(`nvidia-smi` → "Insufficient Permissions"). The Qwen **vision encoder**'s large
activation (8131 tokens from 3×4 images) trips the CUDA caching allocator's
NVML free-memory query (`NVML_SUCCESS INTERNAL ASSERT ... CUDACachingAllocator.cpp:838`),
even with `expandable_segments`. So the genuine hidden state was taken from a
**text-only** LM forward (`FMHEAD_TEXT_ONLY=1`): same model, same dtype, same
2048-dim answer-position hidden — only the visual tokens are absent. The FMHead
interface (dims/dtype/extraction/decode/loss/grad) is identical with or without
vision; on a normal training node (NVML accessible) the with-vision path runs
unchanged. Small CUDA allocs are fine here (the FMHead itself trains on GPU in the
same script).

### 9.3 Final real-GPU smoke results (`real_gpu_results.json`)

Feeding the **real** `c, s` through `FMHeadDecoder` (normalizer loaded from real stats):

- **decode** → navsim poses `(1, 10, 3)`, all finite. ✅
- **overfit to the REAL DDv2 teacher trajectory** (fixed real `c,s`, 400 steps):
  loss **1.290 → 0.380**; sampled endpoint `(33.33, 5.64) m` vs GT `(33.51, 5.55) m`
  → **endpoint L2 = 0.20 m**. ✅ (the head reproduces the real teacher waypoint
  from the real Qwen hidden state)
- **style controllability** (real `c` as content, real style direction ±): 
  `s+ → frac_left = 1.000`, `s− → frac_right = 1.000`. ✅
- Result: **PASS, `ran_real_vlm=true`**.

### 9.4 Bugs found & fixed during this pass

1. **No trajectory normalization** → flow-matching loss on raw metres was ~82 and
   grad-starved (only the zero-init final layer learned on step 1). *Fix:* added
   `FMHeadDecoder.{fit,set,load}_normalizer`, normalize in `training_loss`,
   denormalize in `decode`; fitted from real stats. Loss now ~1.0 and healthy
   (62/64 params get grads within a few steps).
2. **bf16 hidden states** vs fp32 head → *Fix:* cast `c,s,x_t,t` to the head's
   param dtype at the `forward` boundary (already in `fm_head.py`); verified the
   real bf16 `c,s` flow through without error.
3. **Wrong horizon default (8)** → real AutoVLA `num_poses=10`. *Fix:* default 10.
4. **NVML/vision allocator assert** (env) → *Fix/workaround:* `FMHEAD_TEXT_ONLY=1`
   + `PYTORCH_NVML_BASED_CUDA_CHECK=0` to obtain the genuine LM hidden state here;
   documented as an env limitation, not a code defect.
5. **PEFT LoRA identity at init** (`lora_B` zero-init → `s=0`) → *Fix (test only):*
   randomize `lora_B` so the base-vs-LoRA delta is genuinely non-zero.

### 9.5 GO + exact next step to launch training

**GO.** Verified end-to-end on the real model + real data: dims 2048/bf16, horizon
10, output `(10,3)` metres, normalizer fitted from 11,988 real trajectories, and
the head reproduces a real teacher trajectory (endpoint L2 0.20 m) and is
style-controllable — all without modifying AutoVLA.

**To launch training** (new script under `fmhead/`, mirrors `tools/run_sft_lora.py`
but swaps the codebook-CE loss for the FM loss; reuses the same config for data/LoRA):

```bash
cd /root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA
export PYTHONPATH="$PWD:$PWD/navsim:/root/workspace/fmhead:${PYTHONPATH:-}"
# training script (to add) builds SFTAutoVLA's dataset + LoRA, then optimizes
#   FMHeadDecoder(load_normalizer('/root/workspace/fmhead/traj_norm_stats.json'))
#   .training_loss(batch['gt_trajectory'], c, s)   # c,s from AutoVLA_FMHead._forward_hidden
# checkpoints/logs -> /mnt/pfs/zhengguantian/... (PFS), code stays in fmhead/
OMP_NUM_THREADS=8 python /root/workspace/fmhead/train_fmhead.py \
  --config training/youdrive-ddv2-persona-lora-big-l2
```

The wrapper is thin: `AutoVLA_FMHead = build_autovla_fmhead_subclass()`; per batch
call `model.fm_training_loss(batch)` (adds FM loss) or run FM-only. **Note:** real
training exercises the **with-vision** forward, which needs a node with NVML access
(the standard 8-GPU training box) — this smoke box could not run the vision tower.
Remaining non-blocking follow-ups: wire the real `PDM_Reward` into
`select_by_score`, and validate mean-pool-over-action-positions vs last-token `c`
extraction on real data.

---

## 10. Decoder-replacement TRAINING flow (frozen VLM + LoRA → train FMHead)

New deliverables: `train_fmhead.py`, `config/fmhead_ddv2.yaml`,
`train_smoke_results.json`, `train_smoke_run.log`.

### 10.1 What is frozen vs trained

- **Frozen / reused (no grad):** the entire AutoVLA pre-decoder stack —
  base Qwen2.5-VL-3B (bf16) **+** the trained persona LoRA
  (`.../lora_ckpts/ddv2_big_final/lora_final`, r=64/α=128). **3.91 B params, all
  `requires_grad=False`**, run under `torch.no_grad()`.
- **Trained:** only `FMHeadDecoder` — **7,361,538 params** (FMHead DiT + c/s
  projectors; the normalizer is fixed buffers). It is the only thing in the
  optimizer.

The persona LoRA is what defines the **style axis**: `h_styled` = LoRA-on forward,
`h_base` = `vlm.disable_adapter()` forward, `s = alpha·(h_styled − h_base)`.

### 10.2 Training-loop structure (`train_fmhead.py`, mirrors `run_sft_lora.py`)

1. `build_frozen_vlm(cfg)` — `AutoVLA(av_config)` (loads Qwen bf16 + processor +
   action_tokenizer), optionally load the SFT/PDMS base ckpt (`sft_model_path`),
   `PeftModel.from_pretrained(vlm, lora_adapter_path)`, then freeze all params.
2. `build_decoder(cfg)` — `FMHeadDecoder(autovla_fmhead_config(...))`,
   `load_normalizer(traj_norm_stats.json)`.
3. Data — the REAL path reuses AutoVLA's `SFTDataset` + `DataCollator` unchanged
   (`make_real_dataloader`); a text-only fallback (`make_textonly_batches`) is used
   only on this NVML/vision-restricted box.
4. Per step (VLM under `no_grad`):
   ```
   c, s = extract_c_s(av, batch, alpha)          # c=(B,2048) bf16 action-region hidden
                                                 # s=alpha*(styled-base) via disable_adapter()
   loss = decoder.training_loss(batch['gt_trajectory'], c, s)   # normalizes [...,:2], FM MSE
   (loss/accum).backward(); clip; opt.step(); sched.step()
   ```
   AdamW on FMHead params only, linear warmup, `grad_clip=1.0`, bf16-safe (c,s bf16
   cast to fp32 at the head boundary).
5. Checkpoints → **PFS** `ckpt_dir/<timestamp>/fmhead_{stepN,final}.pt`
   (`{fm_decoder, config, step, fm_config}`; the code stays in `fmhead/`).

### 10.3 Config knobs (`config/fmhead_ddv2.yaml`)

`autovla_config` (which AutoVLA yaml to reuse for data/model/tokens),
`lora_adapter_path`, `load_sft_base`, `freeze_vlm`, `style_alpha`, `c_reduce`
(`mean`|`last`), `fmhead.{hidden_size,depth,num_heads,horizon=10,traj_dim=2,
style_dropout_prob,num_samples,num_steps,cfg_weight}`, `normalizer_path`,
`train.{lr,weight_decay,batch_size,accumulate_grad_batches,max_steps,warmup_steps,
grad_clip,log_every,ckpt_every}`, `ckpt_dir` (PFS).

### 10.4 Real training smoke — results (`train_smoke_results.json`)

Ran on GPU through the frozen VLM → FMHead (text-only VLM forward on this box; see
§9.2 for the NVML/vision caveat — the real vision path is coded and used on a
normal node). **ALL PASS:**

| Check | Result |
|---|---|
| loop runs end-to-end (frozen VLM → FMHead) | PASS |
| **VLM frozen** — grads on VLM after backward | **0** → PASS |
| FMHead trainable params | **7,361,538** |
| **loss decreases** (mean first10 → last10) | **1.30 → 0.99** (ratio 0.76) → PASS |
| checkpoint save + reload | PASS (28 MB, `.pt` on PFS) |

(The smoke uses a short warmup / no grad-accum / 2 scenes so the decrease is
visible in ~250 steps; the production config keeps warmup=200, accum=4. Loss is
noisy because flow-matching samples random `t, ε` each step, but the trend is
clearly down — consistent with §9.3 where a single-scene overfit reached 0.20 m
endpoint error.)

### 10.5 Exact command to launch full training (NVML-enabled node)

```bash
cd /root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA
export PYTHONPATH="$PWD:$PWD/navsim:/root/workspace/fmhead:${PYTHONPATH:-}"
OMP_NUM_THREADS=8 python /root/workspace/fmhead/train_fmhead.py \
  --config /root/workspace/fmhead/config/fmhead_ddv2.yaml
# -> checkpoints under /mnt/pfs/zhengguantian/autovla/persona/fmhead_ckpts/<timestamp>/
```

On this box only (vision blocked), reproduce the smoke with:
```bash
OMP_NUM_THREADS=8 FMHEAD_TEXT_ONLY=1 PYTORCH_NVML_BASED_CUDA_CHECK=0 \
  python /root/workspace/fmhead/train_fmhead.py \
  --config /root/workspace/fmhead/config/fmhead_ddv2.yaml --smoke --no_sft_base
```

### 10.6 Risks / non-blocking follow-ups

- **Vision forward needs NVML-enabled node** (this box can't run the vision tower;
  code path is correct and used via `make_real_dataloader`).
- **`select_by_score` is still a medoid stub** — wire the real `PDM_Reward`
  (`models/utils/score.py`) for safety-aware candidate selection.
- **`c`-extraction position** — default mean-pool over action-token positions
  (`c_reduce: mean`); validate vs `last` on real data.
- **Joint vs frozen** — default trains only FMHead ("the back part"); if the frozen
  `c` proves too lossy, unfreezing the LoRA (small) is a one-line config change
  (`freeze_vlm: false` + add LoRA params to the optimizer).
- **Throughput** — the frozen VLM forward dominates step time (~2 forwards/step for
  styled+base `s`); cache the base forward or precompute a fixed style vector to
  halve it.

---

## 11. Multi-GPU / multi-node (2×8 = 16 GPU) DDP

`train_fmhead.py` supports `torchrun`-based DDP; `launch_2x8.sh` is the launcher.

### 11.1 What is (and isn't) parallelized

- **Per rank**: each rank builds its **own** frozen Qwen2.5-VL-3B (bf16, ~6–8 GB) +
  persona LoRA on `cuda:LOCAL_RANK`, and its own tiny FMHead. The VLM runs under
  `no_grad` and is **never** placed in DDP.
- **DDP wraps ONLY the FMHead decoder** (`DistributedDataParallel(decoder, …)`).
  Training calls `ddp_decoder(gt, c, s)` — `FMHeadDecoder.forward` is an alias of
  `training_loss`, so DDP installs its all-reduce hooks correctly (calling
  `.training_loss` directly would bypass the reducer).
- `broadcast_buffers=False` (the normalizer buffers are identical per rank, loaded
  from `traj_norm_stats.json`). `find_unused_parameters=False` (tested OK — the
  CFG null-style / style-dropout path via `torch.where` keeps `null_style` in the
  graph with a real zero grad, so nothing is "unused"; override with
  `FMHEAD_DDP_FIND_UNUSED=1` if a future change needs it).
- **Data sharding**: `DistributedSampler(shuffle=True, drop_last=True)` over the real
  `SFTDataset`; `sampler.set_epoch(epoch)` each epoch. 16 ranks see disjoint scenes.
- **Rank-0 only**: logging + checkpointing (barriers around save); same PFS layout,
  run dir suffixed `_x{world_size}` and broadcast so all ranks agree.
- **Backward compatible**: without torchrun (`WORLD_SIZE` unset) it runs the exact
  single-GPU path (`ddp=False`, no sampler, no init_process_group).

### 11.2 Effective batch math

`effective_batch = world_size × per_rank_batch × accumulate_grad_batches`
= `16 × 1 × 4` = **64 scenes / optimizer step** for the default config on 2×8.
(Per-rank `batch_size=1`; the VLM forward dominates memory so keep it at 1.)

### 11.3 Launch commands + required env

**2 nodes × 8 GPU** — set on BOTH nodes (only `NODE_RANK` differs):

| env | meaning |
|---|---|
| `MASTER_ADDR` | IP/host of node 0 (rank-0 node), reachable from node 1 |
| `MASTER_PORT` | free TCP port on `MASTER_ADDR` (e.g. `29500`) |
| `NODE_RANK`   | `0` on node 0, `1` on node 1 |
| (opt) `NPROC`=8, `NNODES`=2, `CONFIG`, `EXTRA`, `NCCL_SOCKET_IFNAME` | |

```bash
# node 0 (e.g. 10.0.0.1):
MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 NODE_RANK=0 bash /root/workspace/fmhead/launch_2x8.sh
# node 1:
MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 NODE_RANK=1 bash /root/workspace/fmhead/launch_2x8.sh
```

**Single node, 8 GPU** (no MASTER_*/NODE_RANK needed):

```bash
cd /root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA
export PYTHONPATH="$PWD:$PWD/navsim:/root/workspace/fmhead:${PYTHONPATH:-}"
OMP_NUM_THREADS=8 torchrun --standalone --nproc_per_node=8 \
  /root/workspace/fmhead/train_fmhead.py --config /root/workspace/fmhead/config/fmhead_ddv2.yaml
```

### 11.4 Memory & expected scaling

- **Per GPU**: frozen Qwen2.5-VL-3B bf16 ≈ 6 GB weights (+persona LoRA) + the real
  vision-token activations for one scene + 7.36 M FMHead (fp32, ~30 MB) + Adam
  state. Comfortably fits an 80 GB A100; even the SFT/PDMS base (`load_sft_base`)
  adds only transient host RAM at load. No optimizer state for the VLM (frozen).
- **Scaling**: gradients are all-reduced over only **7.36 M** params (~30 MB) once
  per optimizer step — negligible vs the VLM forward that dominates step time.
  So throughput scales near-linearly with GPU count (≈16× the single-GPU
  scenes/s on 2×8, minus a small NCCL all-reduce + data-load overhead).

### 11.5 Tested vs not-tested (on this box)

This box exposes **one MIG 3g.40gb slice** (`device_count=1`), so a true
multi-GPU NCCL / multi-node run could not be executed here. What was verified by
running **2 ranks on the shared GPU with the `gloo` backend** (text-only smoke,
`torchrun --standalone --nproc_per_node=2`, `--smoke --no_sft_base`):

| Check | Result |
|---|---|
| DDP init (`init_process_group`) + clean shutdown (`destroy_process_group`) | PASS |
| VLM frozen (grads on VLM = **0**), only **7,361,538** FMHead params train | PASS |
| grads **all-reduce** → **loss decreases** (1.70 → 0.94, ratio 0.55) | PASS |
| **FMHead params in sync across ranks** — signatures `[2855.726533701253, 2855.726533701253]` identical despite **disjoint** per-rank scenes | PASS |
| rank-0-only checkpoint save (`…_smoke_x2/fmhead_final.pt`) + reload | PASS |
| single-GPU (no-torchrun) backward-compat path still runs | PASS |
| `find_unused_parameters=False` sufficient (no DDP unused-param error) | PASS |

Identical-to-double-precision param signatures across ranks that saw **different
data** is the decisive proof the all-reduce is wiring gradients correctly.

**Could NOT verify here** (needs a real ≥2-GPU / 2-node node): NCCL backend,
inter-node rendezvous over `MASTER_ADDR:MASTER_PORT`, and the with-vision forward
(NVML-restricted). The code paths are standard `torchrun`/NCCL + the already-tested
`make_real_dataloader`; run `launch_2x8.sh` on the training cluster to exercise them.

### 11.6 Remaining risks

- **NCCL NIC selection** on the cluster: may need `NCCL_SOCKET_IFNAME=<iface>` /
  `NCCL_IB_DISABLE=0` for inter-node traffic (commented in `launch_2x8.sh`).
- **Data-loader workers** (`num_workers=2`/rank): tune to avoid CPU oversubscription
  at 8 ranks/node.
- If a future change makes some FMHead param conditionally unused per step, set
  `FMHEAD_DDP_FIND_UNUSED=1`.

---

## 12. Feasibility experiment — normal AutoVLA + human GT, STYLE OFF (s=0)

**Goal (locked scope):** first prove the FMHead decoder is a viable replacement for
the codebook decoder under the SAME base AutoVLA weights, conditioning on scene
content `c` only (`s = 0`, no persona LoRA). Then (later) add the style axis.

Deliverables: `config/fmhead_gt_feasibility.yaml`, `traj_norm_stats_gt.json`,
`eval_feasibility.py`, `fmhead_navsim_agent.py`, `run_pdms_fmhead.sh`, `launch_8gpu.sh`.

### 12.1 Dataset (human GT — found, no build needed)

**`/mnt/pfs/zhengguantian/autovla/persona/nocot_navtrain12k`** — the ORIGINAL human /
log-replay GT navtrain set (`dataset_name: nuplan`), **11,988** scenes, same JSON
schema as the teacher sets → loads via AutoVLA `SFTDataset` unchanged. It is the GT
referenced in the configs as *"the original GT, not DDv2 style"*. Verified it differs
from the DDv2 teacher by ~2.5 m/point (teacher endpoint 33.5 m vs human GT 29.4 m on
a sample). The feasibility config points the (reused) AutoVLA config's dataset here
via a `dataset_path` override (AutoVLA repo untouched).

### 12.2 Normalizer (refit on human GT) — `traj_norm_stats_gt.json`

Per-(waypoint,dim) mean/std over `gt_trajectory[..., :2]`, **11,988** human-GT scenes:
- endpoint (t=10) mean **(20.81, 1.18) m**, std **(13.88, 6.03) m**
- global mean **(11.71, 0.49) m**, std **(10.52, 3.17) m**

(Human GT is less aggressive than the DDv2 teacher: 20.8 m vs 22.9 m endpoint.)

### 12.3 Code changes for the s=0 path (backward compatible)

- `build_frozen_vlm`: when `lora_adapter_path` is `none/null`, **skip** the persona
  LoRA → the style axis is OFF and there's a **single** VLM forward per step.
- `extract_c_s(..., style_off=True)`: one VLM forward → `c`, and `s = zeros_like(c)`
  (no `disable_adapter`, no 2× cost). Styled 2-forward path unchanged for later.
- `main`: `style_off = (no LoRA) or style_mode=='none' or style_alpha==0`; VLM
  frozen assert still holds (`requires_grad=0`). `dataset_path` override hook added.
- `AutoVLA_FMHead` subclass + `fmhead_navsim_agent.py` also honor `style_off` (s=0).

### 12.4 s=0 training smoke (this box, text-only) — PASS

| Check | Result |
|---|---|
| no persona LoRA attached (FROZEN **3.76B**, not 3.91B) + single VLM forward | PASS |
| dataset override → `nocot_navtrain12k` (human GT) | PASS |
| style axis OFF (`s=0`) | PASS |
| VLM frozen (grads=0), FMHead trainable **7,361,538** | PASS |
| loss (mean first10→last10) **1.155 → 0.286** (ratio 0.25) | PASS |
| ckpt save+reload (PFS `fmhead_ckpts_gt/…`) | PASS |
| step time ~**43 ms** (vs ~187 ms styled 2-forward) — single-forward ~2× faster | ✓ |

`eval_feasibility.py` was also run structurally (text-only, smoke ckpt): the
ADE/FDE pipeline runs and the `oracle < select` ordering holds (numbers meaningless
at 250 steps / 2 scenes — accuracy comes from the real run).

### 12.5 Launch the feasibility training run (8-GPU)

```bash
CONFIG=config/fmhead_gt_feasibility.yaml bash /root/workspace/fmhead/launch_8gpu.sh
# or explicitly: EXTRA="--max_steps 20000" CONFIG=config/fmhead_gt_feasibility.yaml bash launch_8gpu.sh
# -> checkpoints under /mnt/pfs/zhengguantian/autovla/persona/fmhead_ckpts_gt/<timestamp>_x8/
```
(2×8 = 16 GPU: use `launch_2x8.sh` with `CONFIG=config/fmhead_gt_feasibility.yaml`.)

### 12.6 How the comparison table is produced (after training)

Same base AutoVLA weights, two decoders:

**(a) decoded-traj vs human-GT L2** (held-out navtrain slice), on a vision node:
```bash
python /root/workspace/fmhead/eval_feasibility.py \
  --config config/fmhead_gt_feasibility.yaml \
  --ckpt /mnt/pfs/.../fmhead_ckpts_gt/<run>/fmhead_final.pt --n 500
# -> table rows: codebook | fmhead_select | fmhead_mean | fmhead_oracle  (ADE/FDE metres)
```

**(b) navtest PDMS** via the existing navsim harness (`run_pdm_score_cot.py`):
```bash
# baseline (codebook):
DECODER=codebook TAG=cb_base bash /root/workspace/fmhead/run_pdms_fmhead.sh
# ours (FMHead, s=0):
DECODER=fmhead TAG=fm_gt FMHEAD_CKPT=/mnt/pfs/.../fmhead_final.pt \
  bash /root/workspace/fmhead/run_pdms_fmhead.sh
```

Target comparison table (filled after the runs):

| decoder (same base weights) | ADE (m) | FDE (m) | navtest PDMS |
|---|---|---|---|
| Codebook (baseline) | … | … | … |
| FMHead (s=0, select) | … | … | … |

### 12.7 Blockers / notes

- **Not a blocker**: the human-GT dataset already exists (`nocot_navtrain12k`) — no
  dumping needed.
- **Needs an NVML-enabled node** for the real with-vision training + PDMS (this box's
  MIG/NVML restriction forced the text-only smoke; code paths are correct).
- PDMS reuses the stock harness + `youdrive/_agg_pdms.py`; the FMHead side only
  overrides the hydra `agent._target_` to `FMHeadAutoVLAAgent` (no repo edits).
- `select_by_score` is still the medoid stub — for the PDMS row, wiring the real
  `PDM_Reward` scorer would let FMHead exploit its N candidates (otherwise it selects
  a central sample, a fair but conservative baseline).

---

## 13. Full-FT SFT contender — FMHead decoder REPLACES codebook CE (joint training)

A **faithful replica of AutoVLA's full-FT SFT (`tools/run_sft.py`)** in which the
codebook/action-token CE decoder is replaced by the FMHead + flow-matching loss.
Distinct from §10/§12 (frozen VLM): here the **VLM LM-backbone and the FMHead train
JOINTLY** — gradients flow into the LM. Independent config/output/launcher.

Files: `fmhead_sft.py`, `config/fmhead-sft-full.yaml`, `launch_sft_full_8gpu.sh`,
`prepare_gt_split.py`, `test_sft_grad.py`, `sft_grad_test_results.json`.

### 13.1 Faithful to `run_sft.py`

Same Lightning `Trainer` + `FSDPStrategy(FULL_SHARD)`, `transformer_auto_wrap_policy`
on `Qwen2_5_VLDecoderLayer`, `MixedPrecision(bf16,bf16,bf16)`,
`BackwardPrefetch.BACKWARD_PRE`, `state_dict_type="full"`, `limit_all_gathers=True`,
`gradient_checkpointing_enable()`, `gradient_clip_val=1.0` (value), `num_nodes=1`,
`devices='auto'`, `ModelCheckpoint(monitor="val_loss", save_weights_only=True)`,
`EarlyStopping`, `CSVLogger`. Reuses `SFTDataset`+`DataCollator`+`SFTAutoVLA`
unchanged. Same hparams as `qwen2.5-vl-3B-mix-sft`: AdamW **lr 2e-5**, wd 0.01,
**5 epochs**, bs 1, **accum 4**, **warmup 500**.

### 13.2 What's frozen vs trained (verified param counts)

| Component | Params | State |
|---|---|---|
| VLM **LM backbone** (`vlm.model`+lm_head) | **3,089,577,984** | **TRAINED** (`train_lm_backbone: true`) |
| VLM **vision tower** (`vlm.visual`) | 668,684,288 | **FROZEN** (`train_vision_backbone: false`, 0 grad) |
| **FMHead** decoder | **7,361,538** | **TRAINED** (jointly) |

Total trainable ≈ **3.097 B** (LM + FMHead); vision 0.67 B frozen.

### 13.3 How loss/grad-flow differs from the frozen trainer (§10)

| | Frozen trainer (`train_fmhead.py`) | Full-FT SFT (`fmhead_sft.py`) |
|---|---|---|
| VLM forward | `torch.no_grad()` | **grad-enabled** (no no_grad) |
| Trains | FMHead only (7.36M) | **LM backbone + FMHead** (3.097B) |
| Grad into LM | none (frozen) | **yes** (via `c` = action-region hidden) |
| Parallelism | DDP over FMHead only | **FSDP FULL_SHARD** over Qwen decoder layers |
| Loss | flow loss (s=0/styled) | flow loss (s=0), codebook CE **removed** |

The decoder swap: `training_step` runs `vlm(..., output_hidden_states=True)` (with
grad), extracts `c` (action-region hidden), sets `s=0`, and returns
`fm_decoder(gt[...,:2], c, s)` (the flow loss). `val_loss` = same flow loss on the
disjoint val split, so `ModelCheckpoint(monitor="val_loss")` works.

### 13.4 Data (human-GT, disjoint split)

`prepare_gt_split.py` symlinks `nocot_navtrain12k` (human GT) into a disjoint
**train (11,732)** / **val (256)** split under
`…/persona/nocot_navtrain_gt_split/{train,val}`. Normalizer = `traj_norm_stats_gt.json`.

### 13.5 Test (this box: 1 MIG slice + NVML/vision restriction → no FSDP multi-GPU)

FSDP multi-GPU can't run on one MIG slice, so I unit-tested the model/loss/
trainable-flags **without FSDP** on the text-only path (`test_sft_grad.py`) — **ALL
PASS**:

| Check | Result |
|---|---|
| (a) FMHead flow loss computes | PASS (init 1.094) |
| (b) grad reaches **LM backbone** | PASS (**434/434** param tensors, Σ\|g\|=4.06e4) |
| (c) grad reaches **FMHead** | PASS (61/64, Σ\|g\|=2.36e3) |
| (d) **vision frozen** | PASS (0 trainable, 0 grad) |
| (e) loss decreases | PASS (1.094 → 0.956) |

Subtlety verified: FMHead's **AdaLN-Zero** init makes the gradient to `c` (hence to
the LM) exactly **0 on the first step** (all `c→loss` paths pass through zero-init
adaLN/final weights); after a few steps those weights are nonzero and grad flows to
the full LM backbone — so (b) is asserted after warmup.

**Confirmed by construction (matches `run_sft.py`), NOT runnable here:** the FSDP
FULL_SHARD wiring, bf16 MixedPrecision, the auto-wrap on `Qwen2_5_VLDecoderLayer`
(FMHead excluded from auto-wrap), and the with-vision forward. These need the real
8-GPU node.

### 13.6 Launch (single-node 8-GPU)

```bash
bash /root/workspace/fmhead/launch_sft_full_8gpu.sh
# or: CONFIG=config/fmhead-sft-full.yaml bash /root/workspace/fmhead/launch_sft_full_8gpu.sh
# -> ckpts under /mnt/pfs/zhengguantian/autovla/persona/fmhead_sft_full/<timestamp>/
```
(The launcher ensures the disjoint GT split exists, sets navsim env, tees a PFS log.)

### 13.7 Risks

- **FSDP + non-sharded submodule**: the auto-wrap policy wraps ONLY
  `Qwen2_5_VLDecoderLayer`, so the FMHead is not its own FSDP unit (as required).
  Under Lightning FSDP the **root** wrapper still flat-shards leftover params
  (FMHead + lm_head + embeddings) — identical to how `run_sft.py` treats
  lm_head/embeddings, and negligible for 7.36M. If TRUE replication (unsharded
  FMHead) is ever needed, use FSDP `ignored_states` for `fm_decoder` + a manual
  grad all-reduce; flagged, not needed for correctness.
- **Checkpointing under FSDP**: `state_dict_type="full"` + `save_weights_only=True`
  gathers the full (unsharded) state on rank 0 including `fm_decoder.*` and the
  trained `vlm.*` — a large (~6 GB) file per top-k; ensure PFS space. Reload for
  eval strips prefixes as usual.
- **Memory**: ~3.097 B trainable → AdamW fp32 moments dominate; FSDP FULL_SHARD
  across 8×A100-80GB shards params/grads/opt (~a few GB/GPU) + gradient
  checkpointing for activations. Comfortable on 8×80GB; would be tight on fewer.
- **lm_head unused**: with CE removed, `lm_head` receives no grad (dead weight in
  the optimizer). Harmless; could be excluded to save a little FSDP overhead.
- Needs the NVML-enabled 8-GPU node for the real with-vision run.
