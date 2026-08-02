# FM head — SOTA study (GoalFlow / GuideFlow), scale/dim audit, and rewrite plan

Study + audit + plan only (no fm_head rewrite yet). Repos cloned to `/root/workspace/refs/`
(`GoalFlow`, `GuideFlow` = adept-thu, code present). Our head = `/root/workspace/fmhead/fm_head.py`.

---

## 1. GoalFlow (released) — exact facts (file:line in `refs/GoalFlow`)

### 1.1 Trajectory representation
- `TrajectorySampling(time_horizon=5.5, interval_length=0.5)` → ~11 future poses @0.5 s
  (`navsim/agents/goalflow/goalflow_config.py:46-47`). Ego frame, **x = forward, y = lateral**
  (x_scale 60 m ≫ y_scale 15 m ⇒ x is the long forward axis), heading as 3rd channel.
- The flow operates on a **rotation-augmented** tensor reshaped to `(B, 12, 30)` in `denoise`
  (`goalflow_model_traj.py:450`): 12 pose-tokens × **30 dims = 10 rotations × (x,y,heading)**.

### 1.2 SCALE / NORMALIZATION (top priority) — `normalize_xy_rotation` (`goalflow_model_traj.py:518-539`)
```python
x_scale=60; y_scale=15; heading_scale=math.pi
traj[:,:,0] /= 60;  traj[:,:,1] /= 15;  traj[:,:,2] /= pi;  traj[:,:,2] = atanh(...)   # L520-526
# then rotation augmentation: for i in range(times=10): rotate (x,y) by 2*pi*i/10 -> concat  L528-538
```
`denormalize_xy_rotation` (L541-558) inverts: rotate-back, take first 3, `x*=60; y*=15; heading=tanh*pi`.
- **NOT a z-score.** GoalFlow uses **FIXED per-axis divisors** (x/60, y/15) + a **bounded tanh/atanh
  heading** + **rotation feature-expansion** (×10 → 30-D). The divisors bring x∈[−1,1]≈, y∈[−1,1]≈;
  they are constant across all waypoints (no per-waypoint stats).
- Contrast OURS: per-(waypoint,dim) **z-score** from `traj_norm_stats_gt.json` (endpoint mean
  (20.8,1.18), std (13.9,6.03)); traj_dim=2 raw (no rotation aug); heading appended post-hoc.

### 1.3 Flow-matching formulation
- Rectified/linear interpolant, **velocity target** (`goalflow_model_traj.py:745-746`):
  `z_t = t*z1 + (1-t)*z0 ; target = z1 - z0` (z1=data, z0=noise). **Identical to ours.**
- Training draws `get_train_tuple(z0=noise, z1=normal_trajs)` (L268); loss = MSE(v, target).
- **Sampling: Euler, `infer_steps=100`** (`goalflow_config.py:36`), `trajs += v*(step/infer_steps)`
  (L315/336/373/389). A time-shift schedule `t_shifted` with `alpha=3.0` (L307-310). **We use 10 steps.**
- **CFG**: `denoise(..., force_dropout, navi_dropout)` zeroes all cond / the goal token (L458-462);
  sampling blends cond/uncond with `cond_weight` (L361-364). CFG is on the **goal**.

### 1.4 Conditioning (how features enter) — `denoise` (`goalflow_model_traj.py:444-499`)
- **Multi-token joint self-attention**, NOT AdaLN and NOT a single pooled vector:
  `all_features = cat([state_features (scene+goal tokens), trajectory_features (12 tokens)])` (L460);
  per-token `sigma` (time) embedding concatenated + projected (L466-476); `global_attention_layers`
  (ParallelAttentionLayer) run self-attention over the whole `[cond ; traj]` sequence (L488-494) with
  a local temporal mask (`dists>1`, L482) and relative temporal + type embeddings.
- **Goal point is the key conditioner:** `navi = gt_trajs[:,7:8,:2]` (the 8th pose) →
  `pos2posemb2d` → appended as the last state token (L178-180); `navi_dropout` zeros it (CFG).

### 1.5 Goal construction + final SELECTION
- Goal from a **vocabulary of 8192 candidate points** scored by
  `0.1*log softmax(im_score) + theta*log sigmoid(dac_score) + ep_point_weight*goal_distance`
  (`goalflow_model_traj.py:196-233`, `theta=3.0`, `topk`); top-k goal points averaged → `navi`.
- Trajectory selection: **`anchor_size=10`** candidate trajectories sampled (`goalflow_config.py:38`);
  final = mean of the anchors, or the anchor nearest the goal (`use_nearest` branch). So selection is
  **goal-scored**, imitation+drivable-area-aware — not a blind medoid.

---

## 2. GuideFlow — code availability + style/constraint mechanism

- **Code IS released** (`refs/GuideFlow`, adept-thu, 1247 files): `navsim_{train,test}/agents/
  flowdrive_{unet,DIT}` — incl. `unet/unet.py:EnergyCrossAttention`/`UNetModelEnergy` (L1856/1907),
  `meanflowv1.py:guidance_fn` (CFG with `kappa`, L172-179), `energy_fusion(emb, y)` (L2158).
- **Method (paper + code):** Constraint-Guided Flow Matching. Core = enforce constraints *inside*
  generation: **(1) CVF** constrain the velocity field (correct predicted v toward a constraint-adhering
  field), **(2) CF** constrain the flow states (correct deviating paths), **(3) RFE** refine flow by an
  **EBM** (unify flow-matching + energy model to "discover" constraint-satisfying modes). NavHard EPDMS 43.0.
- **STYLE CONTROL (the novelty threat):** *"parameterizes driving aggressiveness as a control signal"* by
  **conditioning on an environmental REWARD** ("R" in Fig.2) — a scalar/reward signal injected as a
  conditioning input, enabling aggressive↔conservative switching at inference. It is **reward-scalar
  conditioning + CFG**, computed from environment; **not** identity/persona-specific, **not** weight-space,
  **not** language-grounded, and it rides on a **BEV/perception (TransFuser/UNet) backbone — no VLM/VLA**.

**Threat & our differentiation:** GuideFlow does own "explicit style knob for a flow planner." But ours is
distinct: (a) **weight-space persona axis** `s = α·(h_styled − h_base)` from a *trained persona LoRA*
(driver-identity-specific, continuous, from the model's own weights) vs their *environmental-reward scalar*;
(b) **language-grounded VLM backbone** (the `c` is a Qwen2.5-VL answer-position hidden) vs their BEV/UNet;
(c) CFG on the persona axis. Their strongest transferable ideas are **constraint-guided sampling (CVF/CF)**
and **EBM refinement** — orthogonal to our style story and safe to borrow for the *selection/safety* side.

---

## 3. SCALE / DIM AUDIT of our head — verdict: **scale/frame is NOT the bug**

Decisive evidence and concrete checks:

| Check | Ours | GoalFlow | Verdict |
|---|---|---|---|
| Frame | ego, **x_forward, y_left** | ego, x_forward (/60), y_lateral (/15) | **Same frame** — no axis swap |
| Poses / dt | 10 @ 0.5 s | ~11 @ 0.5 s | Compatible (navtest PDM slices num_poses) |
| Norm method | per-(waypoint,dim) **z-score** (mean/std from 11,988 GT) | fixed x/60, y/15, atanh(head) + rot-aug | Different but **both valid & invertible** |
| Denorm applied before PDM? | yes (`decode` denormalizes) | yes (`denormalize_xy_rotation`) | OK |
| **Round-trip correctness** | **eval ADE = 0.66 m** (better than codebook 1.07) | — | **PROVES normalize/denormalize/frame/axis/scale are correct** |

- The single strongest fact: **our L2 eval reaches ADE 0.66 m to human GT using the exact same
  `_normalize`/`_denormalize` + frame the agent uses.** If the axis order, units, per-waypoint scale, or
  denorm were wrong, ADE would be many metres (10–20 m), not 0.66 m. A ~0.28 PDMS with **ego_progress ~0.30
  but ADE 0.66** cannot be a scale/frame error — a mis-scaled trajectory would blow up BOTH ADE and PDMS.
- Therefore the 0.28 is explained by the **already-fixed c train/inference mismatch** (OOD `c` at PDMS
  time → bad trajectories, while eval used the train-consistent `c`) **+ medoid mode-averaging selection**
  (endpoint-closest-to-mean = the most average candidate; oracle-ADE 0.56 shows good candidates exist).
- **Minor, non-fatal scale nits worth aligning anyway:** (a) we integrate only **10 Euler steps** vs
  GoalFlow's **100** — too-few steps can under-shoot the ODE endpoint (a *mild* systematic under-progress
  that could shave ego_progress; cheap to test 20–50 steps); (b) per-waypoint z-score gives later waypoints
  much larger std (endpoint std 13.9/6.0) so the tail dominates the MSE — a fixed scale (GoalFlow) or a
  single global std would weight waypoints more evenly. Neither is a correctness bug.

**Audit conclusion:** *No scale/frame/axis error.* `inspect_trajectories.py` (already written) will confirm
on the vision node: expect FM/GT endpoint-x ratio ≈ 1 and ADE < ~1 m ⇒ **H1** (geometry fine; it's
selection/conditioning), not H2 (scale/format). Increase infer_steps if the ratio is slightly < 1.

---

## 4. Rewrite plan — a proper FM head that exploits our UNIQUE VLM front

Our edge vs GoalFlow/GuideFlow: a strong **VLA/VLM** produces rich hidden states (the closest prior is
WAM-Flow which converts a VLA backbone; GoalFlow/GuideFlow use BEV/TransFuser only). Use it.

### (a) Richer conditioning — replace the single pooled `c`  [effort M, risk M, HIGH value]
- Today: one pooled 2048-D `c` → summed into one AdaLN vector. GoalFlow shows **multi-token cross-attn**
  over scene tokens works far better.
- Plan: keep the DiT trajectory tokens but add **cross-attention to a SEQUENCE of VLM hidden states**
  (the answer-region tokens, not just the single anchor) — e.g. the last-layer hiddens over the last K
  prompt tokens / a learned pooled set — projected to width and used as cross-attn memory (à la GoalFlow's
  joint self-attn / DiffusionDrive cross-attn). Retain AdaLN(time) for the timestep.
- **Add a goal-point mechanism (GoalFlow's biggest lever):** predict/condition on a goal point (the ~4 s
  waypoint) from the VLM `c` (a small goal head), inject as a conditioning token, and use CFG on it.
  This gives the flow a strong anchor and a natural selection score.

### (b) Normalization / scale  [effort S, risk L]
- Keep the **z-score normalizer** (proven correct by ADE 0.66) as the default. Optionally A/B a
  **GoalFlow-style fixed scale (x/≈40, y/≈8 for our GT ranges) + rotation augmentation (times=10)** —
  rotation aug measurably helps flow trajectory heads and would also give us a native heading channel.
- **Raise sampling steps** to ~20–50 (from 10); cheap, removes any Euler under-shoot. (GoalFlow=100.)

### (c) Real scoring / selection — replace the medoid stub  [effort M, risk M, HIGH value]
- Already implemented `SELECT=pdm` (per-candidate navsim PDM) and `SELECT=oracle` in
  `fmhead_navsim_agent.py`. Next: add a **learned lightweight scorer** (GoalFlow-style: imitation + a
  drivable-area/collision head over the N candidates, argmax) so we don't pay N× PDM sims at deploy.
  Borrow GuideFlow's **CVF/CF constraint-guided sampling** to *steer* candidates safe during integration
  (cheaper than post-hoc rejection).

### (d) Keep our differentiation  [effort S, risk L]
- Preserve the **weight-space persona style axis** `s = α·(h_styled − h_base)` + **CFG on s**, and the
  **language-grounded VLM `c`**. This is orthogonal to GoalFlow (no style) and *distinct* from GuideFlow
  (identity/weight-space + language vs their environmental-reward scalar). Frame the paper story around
  "continuous weight-space persona control on a VLM flow decoder," borrowing only GoalFlow's goal/scoring
  and GuideFlow's constraint-guidance as the safety layer.

### Recommended MINIMAL-but-correct v1 (do first, low risk)
1. **infer_steps 10 → 30** and re-run PDMS with the (already fixed) aligned `c`. (trivial)
2. **`SELECT=pdm`** selection instead of medoid. (done — just run it)
3. Keep z-score norm. Run `inspect_trajectories.py` to confirm **H1**.
   → If PDMS jumps toward codebook (~0.89) with these three, the head is fundamentally sound and (a)/(b)
   become quality upgrades, not fixes. Then add **goal-point conditioning + multi-token cross-attn** (a) as v2.

---

## 5. TL;DR
- **GoalFlow**: 11@0.5s ego (x-fwd/y-lat); **fixed scale x/60,y/15 + atanh head + rotation-aug(×10→30D)**;
  rectified flow `z_t=t·z1+(1-t)·z0, target=z1−z0`, **100 Euler steps**; **multi-token joint self-attn**
  conditioning with a **scored goal point** + **CFG on goal**; selection = 10 anchors, goal-scored (im+dac+dist).
- **GuideFlow**: code released; **style = environmental-reward conditioning** (aggressiveness scalar) + CFG;
  safety via **CVF/CF constraint-guided sampling + EBM refinement**; BEV/UNet backbone, no VLM.
- **Scale audit**: our normalization/frame/axis are **correct** (ADE 0.66 m proves it); 0.28 PDMS = the
  fixed c-mismatch + medoid selection, NOT scale. Only nit: bump Euler steps.
- **Plan**: exploit the VLM via multi-token cross-attn + a goal point (from GoalFlow), real PDM/learned
  selection + GuideFlow constraint-guidance, keep z-score norm, and keep our weight-space persona+language
  differentiation. v1 = more steps + pdm-select + confirm H1 before any rewrite.
