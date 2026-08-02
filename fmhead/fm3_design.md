# fm3 design — kinematically-feasible FMHead trajectory decoder

**Status:** research + design only. No training launched, GOLD heads untouched.
**Goal:** make the flow-matching decoder produce feasible trajectories (no acute-angle
kinks, low jerk) — recover the implicit feasibility constraint that AutoVLA's discrete
action codebook gave us for free, without giving up the continuous VLM-conditioned FM head.

---

## 0. Problem restatement (measured)

| metric (per-scene median) | base AutoVLA (codebook) | DDv2 teacher | **fm2 FMHead (ours)** |
|---|---|---|---|
| long_jerk | **0.89** | 4.16 | **8–10** |
| lat_jerk  | — | — | ~4 |
| peak_acc  | — | — | ~5 m/s² |
| peak_dec  | — | — | ~4.6 m/s² |

navtest PDMS is high (GOLD full-FT pdm-select = **0.9088**; ~0.91–0.94 depending on
selection) **despite** the jerk, because PDMS's comfort term is a single soft/thresholded
sub-score — it is largely blind to kinematic infeasibility. So PDMS ≠ feasibility; we must
fix feasibility structurally, not chase PDMS.

**Root cause (confirmed below):** AutoVLA decoded through a *discrete action codebook* of
2048 real, clustered per-step vehicle-footprint transitions. That codebook is an **implicit
feasibility constraint** (only physically-observed action primitives can be emitted → smooth).
fm2 replaced it with a **free continuous xy regressor** (rectified-flow over raw
(x_forward, y_left) waypoints) — dropping the constraint entirely. Nothing in fm2 bounds
per-step acceleration, curvature, or heading change, so the ODE endpoint can zig-zag.

---

## 1. How the AutoVLA codebook enforces feasibility (confirmed, file:line)

### 1.1 Structure — `models/action_tokenizer.py`
- `code_book = pickle.load(agent_vocab.pkl)['token_all']['veh']`, `torch.tensor` shape
  **`(n_bins=2048, 6, 4, 2)`** (`action_tokenizer.py:46-47`). Verified by inspecting the pkl:
  `veh/ped/cyc` each `(2048, 6, 4, 2) float32`.
- Semantics of the 4 axes: **2048** clustered primitives × **6** intra-step sub-poses ×
  **4** footprint corners × **2** xy. Each primitive is a rigid **vehicle footprint**
  (width 2.0 m, length 4.8 m) tracked over 6 sub-steps of one 0.5 s pose interval, expressed
  in the **local frame** of the current pose. (Sample[0] = the 4 corners at (±2.4, ±1.0), a
  4.8×2.0 box, barely moving over 6 sub-steps ⇒ the "hold still" primitive.)

### 1.2 Rollout / heading derivation — `decode_token_ids_to_trajectory` + `rollout` (`action_tokenizer.py:65-124`)
- Each generated action token id indexes one primitive `(6,4,2)`.
- `rollout` composes primitives autoregressively: for each step it transforms the local
  footprint into global frame at the current pose (`transform_to_global`, `:105`), then
  - **next position** = centroid of the last sub-step's 4 corners (`pos_a_next = token_traj_global[:,-1].mean(dim=1)`, `:113`);
  - **next heading** = `atan2` of (front-corner − rear-corner) of that footprint
    (`diff_xy = corner0 − corner3; head = arctan2(dy,dx)`, `:114-115`).
- So heading is a genuine SE(2) pose read off a rigid body, **not** a noisy `atan2(Δy,Δx)`
  of a slow xy path (the exact degeneracy fm2 has to patch with `append_heading`,
  `fm_head.py:597`).

### 1.3 How it's built (clustered) — `tools/action_token/action_token_cluster.py`
- Collect every consecutive-pose transition from real `gt_trajectory`, expressed in the
  **local frame** of the preceding pose (`transform_to_local`, wrap heading) (`:181-199`).
- **K-disk clustering** (`Kdisk_cluster`, `:104-139`): greedy cover — pick a center, absorb
  all transitions whose mean footprint-corner distance `< tol=0.05 m`, average them into one
  representative primitive, remove, repeat to **`num_cluster=2048`** (`:148,218`). Footprint
  `width_length = (2.0, 4.8)` (`:211`).
- **Net effect:** the codebook is 2048 *real, averaged, physically-realizable* single-step
  ego motions. Per-step Δposition and Δheading are therefore bounded to what real vehicles
  actually do ⇒ composing them can only produce smooth, bounded-curvature, low-jerk paths.
  Feasibility is **implicit in the vocabulary**, not enforced by any loss.

**Takeaway for fm3:** the codebook's constraint = *"every 0.5 s transition must be a real,
bounded local motion, and heading is a body pose."* The continuous analog is either
(a) flow in a **bounded per-step control space** and integrate a kinematic model, or
(b) flow a **small residual around a known-feasible anchor**. Both re-import the constraint.

---

## 2. How peer generative planners stay feasible (crisp table)

| planner | representation | feasibility mechanism (concrete) | source |
|---|---|---|---|
| **AutoVLA** | discrete action tokens | **codebook** of 2048 K-disk-clustered real footprint transitions; heading from footprint; feasible-by-vocabulary | `action_tokenizer.py`, `action_token_cluster.py` |
| **GoalFlow** | 11×(x,y,head) rectified flow | **goal-point** (8th pose) from an 8192 scored vocab conditions the flow (+CFG on goal); **fixed** scale x/60,y/15 + `atanh` heading + **rotation-aug ×10**; **100** Euler steps; 10 anchors goal-scored | `fm_rewrite_plan.md §1`; `goalflow_model_traj.py:518-558, 315-389`; `goalflow_config.py:36,38` |
| **DiffusionDrive v1/v2** | anchored **truncated diffusion** | **plan_anchor** = clustered anchor set (`(20,8,2)` `.npy`, `requires_grad=False`) encoded as diffusion queries; DDIM `prediction_type="sample"`, ~10 steps → refine a **residual around a feasible anchor**; fixed norm x/50,y/20 | `diffusiondrivev2_model_rl.py:682-772` (`plan_anchor = nn.Parameter(np.load(...), requires_grad=False)`, `DDIMScheduler(..., prediction_type="sample")`) |
| **GuideFlow** | constraint-guided flow | **CVF/CF** correct the velocity field / flow states toward constraint-satisfying manifolds during sampling; **EBM** refinement | `fm_rewrite_plan.md §2`; `refs/GuideFlow .../meanflowv1.py:172-179`, `unet.py:1856-2158` |
| **TransFuser** | direct MLP/GRU regression | no explicit kinematic constraint; feasibility comes from **imitation of smooth GT** + short 4 s horizon + BEV grounding (weakest constraint, but a stable backbone) | `DiffusionDriveV2/navsim/agents/transfuser/*` |

**Pattern:** every strong generative planner constrains the output either by
**(i) anchoring on a small clustered library of feasible modes** (AutoVLA codebook,
DiffusionDrive plan_anchor, GoalFlow goal vocab) and/or **(ii) refining only a small
residual** (truncated diffusion) and/or **(iii) steering the sampler toward a feasibility
manifold** (GuideFlow CVF/CF). None of them regress free xy from noise the way fm2 does.

---

## 3. Current fm2 head internals (what to change)

- **Representation:** `FMHead` flows over raw normalized xy, `traj_dim=2`, `horizon=10`
  (`fm_head.py:63-64`). Content via multi-token cross-attn over the VLM prompt span; style
  via AdaLN (`fm_head_v2_notes.md`).
- **Loss:** pure rectified-flow MSE on the velocity field, `u = tau − eps`, `x_t=(1-t)eps+t·tau`
  (`fm_head.py:466-498`). **No smoothness / kinematic / anchor term.**
- **Sampling:** plain **Euler**, `num_steps=30` default (configs set 30; GoalFlow uses 100)
  (`fm_head.py:501-563`, `fmhead_gt_feasibility.yaml:38`).
- **Normalization:** per-(waypoint,dim) **z-score** from `traj_norm_stats_gt.json` (endpoint
  mean (20.8, 1.18), std (13.9, 6.03)) — later waypoints have huge std so the MSE tail
  dominates; nothing couples adjacent waypoints, so kinks are unpenalized.
- **Heading:** not modeled; bolted on post-hoc via `append_heading` atan2 forward-fill
  (`fm_head.py:597-636`) — a symptom of decoding in xy instead of pose/control space.
- **Target:** human GT `gt_trajectory[...,:2]` (feasibility regime, s=0).

Nothing in this stack bounds acceleration, curvature, or heading rate ⇒ the observed
long_jerk 8–10.

---

## 4. fm3 constraint options (pros / cons / sketch / expected effect)

### (a) Kinematic parametrization — **flow in control space, integrate a unicycle** ✅ primary
Head outputs per-step controls `u_t = (a_t, ω_t)` (long accel + yaw-rate), or `(a_t, κ_t)`
(accel + curvature), for t=1..10. A differentiable integrator maps them to xy:
```
v_t = clamp(v_{t-1} + a_t·dt,  0, v_max)          # dt=0.5, v0 from ego status / first GT step
ψ_t = ψ_{t-1} + ω_t·dt            (or ω_t = v_t·κ_t)
x_t = x_{t-1} + v_t·cos ψ_t·dt ;  y_t = y_{t-1} + v_t·sin ψ_t·dt
```
Bound `a_t = A·tanh(·)` (A≈4), `ω_t = W·tanh(·)` (W≈0.5 rad/s) so accel & yaw-rate are
physically capped.
- **Pros:** feasible **by construction** — bounded accel/yaw-rate ⇒ bounded curvature,
  no acute kinks; jerk is just Δa (small, low-D, easy to keep smooth). Heading is a native
  state ⇒ **drops the atan2 `append_heading` hack** and fixes the slow-speed heading
  degeneracy that hurt PDM. This is the exact continuous analog of the codebook's
  bounded-local-transition constraint. Strongest, cleanest research story.
- **Cons:** flow now operates in control space → need a **new normalizer** on (a, ω)
  derived from GT (invert GT xy→controls once), and the FM loss geometry changes → highest
  retrain risk of the four; a few edge cases (v0 estimate, reverse) to handle.
- **Sketch:** add `param="kinematic"` + `integrate_unicycle(u, v0, dt)` to `FMHead`; flow
  target = GT-derived controls (fit `traj_norm_stats_ctrl.json`); `decode()` integrates then
  appends the native heading; content/cross-attn/style paths unchanged.
- **Expected effect:** long_jerk from 8–10 → **~1–2** (near base 0.89), acute kinks removed;
  small ADE cost possible if v0/controls are noisy (mitigated by the regularizer + more epochs).

### (b) Anchored residual — **flow a small residual over a smooth anchor** ✅ recommended fallback
Head predicts a bounded residual `Δ` added to a known-feasible anchor `τ_anchor`; final
`τ = τ_anchor + Δ`. Anchor choices, best→simplest:
1. **AutoVLA codebook rollout** (we still have the 2048-primitive codebook + argmax decode):
   the anchor is already smooth & feasible → re-imports the codebook prior directly.
2. **Constant-curvature / constant-accel arc** fit from ego speed + first GT step (cheap,
   always feasible).
3. **DiffusionDrive-style clustered anchor set** (K-means of GT, `(K,10,2)`).
- **Pros:** proven (DiffusionDrive/GoalFlow); keeps xy target + z-score normalizer (low code
  churn); if `Δ` is bounded/penalized it can't inject kinks; naturally multimodal if K>1.
- **Cons:** feasibility only as good as `‖Δ‖` control; needs an anchor source at inference
  (codebook decode adds a VLM generate() call, or precompute an anchor head); not
  feasible-by-construction (a large residual can still kink).
- **Sketch:** `param="residual"`; add `anchor` to the batch (codebook rollout, cached), flow
  target = `(GT − anchor)` normalized; bound `Δ` via tanh·scale; decode = anchor + denorm(Δ).
- **Expected effect:** long_jerk → **~2–4** (anchor-dependent); very low regression risk on
  ADE/PDMS since it starts from a good anchor.

### (c) Soft smoothness regularizer — add jerk/curvature penalty to the flow loss ➕ combine
Add `λ_j·mean(Δ²a) + λ_c·mean(Δ²ψ)` on the **clean-trajectory estimate**
`x̂₁ = x_t + (1−t)·v_θ` (the FM one-step data prediction), computed each training step.
- **Pros:** cheap, orthogonal, works with (a) or (b) or even plain xy; directly targets the
  measured quantity (jerk). No inference change.
- **Cons:** a *bias*, not a guarantee — too-large λ hurts ADE/progress; penalty on `x̂₁` is
  noisy at small t (weight it by t, or only apply for t>0.5).
- **Sketch:** in `flow_matching_loss`, reconstruct `x̂₁`, finite-difference to accel/heading,
  add the penalty; expose `jerk_weight`, `curv_weight` in config.
- **Expected effect:** ~30–50 % jerk reduction on its own; as an add-on to (a)/(b) it mops up
  residual roughness. **Always include.**

### (d) Cheap fixes — more ODE steps, better integrator, output smoothing ➕ always
- **num_steps 30→100** (match GoalFlow) — removes Euler discretization zig-zag; free at train
  time, ~3× sampling cost at eval only.
- **Heun/midpoint (RK2)** integrator instead of Euler — halves discretization error for the
  same step count.
- Optional 1-D **Savitzky-Golay / moving-average** smoothing of the decoded xy at the
  `predict()` boundary (belt-and-suspenders; unnecessary if (a) is used).
- **Pros:** trivial, no retrain needed to test (d) alone. **Cons:** (d) alone cannot fix
  jerk 8–10 — the field itself is jerky, not just under-integrated; it's a multiplier, not a
  cure.
- **Expected effect:** ~10–25 % jerk reduction alone; meaningful only combined with (a)/(b).

---

## 5. Recommendation

**Primary: (a) kinematic unicycle parametrization + (c) light jerk/curvature regularizer +
(d) 100 steps / RK2.**

Why (a) over (b): it is **feasible-by-construction** (the only option that *cannot* emit an
acute kink), it is the crispest continuous generalization of the codebook constraint we are
trying to recover, and it fixes the heading degeneracy for free (removing the `append_heading`
hack that previously hurt PDM). (c)+(d) are cheap insurance and should be on regardless.

**Fallback: (b) anchored-residual on the AutoVLA codebook rollout + (c) + (d)** — choose this
if we want to minimize retrain risk / keep the xy normalizer and GOLD-comparable ADE, since it
starts from an already-feasible anchor and only refines. It is also the most directly
peer-validated (DiffusionDrive/GoalFlow).

Both keep the VLM cross-attn conditioning and the (off) style axis intact, so they compose
with the existing v2 head and the persona story.

---

## 6. fm3 no-style feasibility retrain plan (DRAFT — do NOT launch)

Same recipe as the **fm2 no-style feasibility GOLD head** (full-FT SFT: VLM LM-backbone +
FMHead trained jointly, vision frozen, style OFF/s=0, human-GT navtrain, FMHead flow loss —
i.e. `config/fmhead-sft-full.yaml` + `fmhead_sft.py` + `launch_sft_full_8gpu.sh`, the recipe
that produced `fullft_fm2_head_pdm0.9088.pt`), **plus the chosen constraint and more epochs**.

- **Constraint added:** `param: kinematic` (primary) with `jerk_weight`/`curv_weight`
  (option c) and `num_steps: 100` (option d). A `param: residual` variant is provided
  commented-out for the fallback.
- **Epochs:** 10 → **15** (user suggestion; val_loss was still dropping — GOLD best was
  epoch 7, so 15 gives headroom for the harder control-space target).
- **Base VLM ckpt to load:** `sft_model_path =
  /mnt/pfs/zhengguantian/autovla/ckpts/AutoVLA/AutoVLA_PDMS_89.ckpt` (same warm-start the
  GOLD run used); Qwen weights from
  `/mnt/pfs/zhengguantian/autovla/ckpts/Qwen2.5-VL-3B-Instruct`.
- **Target data:** human GT navtrain, disjoint split
  `/mnt/pfs/zhengguantian/autovla/persona/nocot_navtrain_gt_split/{train,val}` (GT = REAL
  human trajectories; NOT the DDv2 teacher).
- **Output dir (new, PFS, distinct from GOLD & all existing runs):**
  `/mnt/pfs/zhengguantian/autovla/persona/fmhead_fm3_feasibility`.
- **#GPUs:** 8 (single node, FSDP FULL_SHARD bf16), identical to the GOLD launcher.
- **GOLD safety:** new config, new `ckpt_dir`; `fmhead_GOLD/` and both 0.9088 / 0.864 heads
  are never read for writing. This plan only *warm-starts from* `AutoVLA_PDMS_89.ckpt`
  (a base VLM, not a GOLD head).

**Code changes still required before launch** (design-only here — NOT written):
1. `fm_head.py`: add `param ∈ {xy, kinematic, residual}`, `integrate_unicycle()`, and the
   `jerk_weight`/`curv_weight` penalty inside `flow_matching_loss` (on `x̂₁`).
2. `autovla_fmhead.py`/`fmhead_sft.py`: build the control-space target (invert GT xy→controls)
   and fit `traj_norm_stats_ctrl.json`; `decode()` integrates + appends native heading.
3. New normalizer file for control space.

### Draft launch command (single node, 8-GPU — **DO NOT RUN**)
```bash
cd /root/workspace/fmhead
CONFIG=config/fmhead_fm3_feasibility.yaml \
  bash /root/workspace/fmhead/launch_sft_full_8gpu.sh
# uses PY=/root/workspace/miniconda3/envs/autovla/bin/python and sets
# PYTHONPATH=AutoVLA:AutoVLA/navsim:fmhead internally; checkpoints ->
# /mnt/pfs/zhengguantian/autovla/persona/fmhead_fm3_feasibility/<timestamp>/
```
(2×8 = 16 GPU: use `launch_2x8.sh`-style multinode with the same CONFIG.)

### Post-train verification (mirror the fm2 gates)
1. `eval_feasibility.py` — ADE/FDE vs human GT (expect ADE within ~10–15 % of GOLD 0.587
   oracle / 1.0 mean; feasibility fix should not blow up ADE).
2. Jerk audit (same script that produced the 8–10 numbers) — **target long_jerk ≤ 2**,
   lat_jerk ≤ 1.5, peak_acc ≤ 4, peak_dec ≤ 4.
3. `run_pdms_fmhead.sh` (SELECT=pdm, 100 steps) — PDMS should stay ≈ GOLD (feasibility fix
   should not cost PDMS; comfort sub-score should rise).

---

## Deferred ideas / paper notes (2026-07-18)

### Idea: ensemble of specialized fm-decoder heads (recorded, not sexy, try later)
Instead of ONE controllable style axis, train several specialized FMHead decoders
(e.g. an "aggressive" head, a "smooth/conservative" head, ...) and ensemble / select
among them at deploy. Could count as a contribution (a small zoo of feasible style
heads + a selector), but it is less elegant than a single continuous, user-addressable
style axis. Park it as a fallback / ablation; revisit if the conditional-axis route stalls.

### Findings that redirect the style work (from the fm3-kin experiments)
- **fm3-kin ep6 = new base**: navtest PDMS ~0.915 (> GOLD 0.9088), long_jerk ~0.5 (vs fm2 8–10),
  0% acute kinks. Kinematic (control-space accel/yaw-rate + caps) is the feasibility fix. Use ep6
  (ep10/14 overtrain: PDMS 0.895/0.888).
- **Two OPPOSITE style-injection failures (both give a dead alpha knob):**
  1. Frozen 0.91 head + tiny zero-gated adapter -> gates ~0.05, style TOO WEAK (nothing injected).
  2. Unfrozen head trained on DDv2 ("fire") -> DDv2 baked UNCONDITIONALLY into the head weights:
     alpha=0 already DDv2 (peak_acc 0.36->2.74, long_jerk 0.52->4.42, thw 3.37->2.49), alpha 0->1
     nearly flat, L2->DDv2 goes 15->18 (wrong way), head becomes DDv2-specific (breaks platform),
     PDMS drops 0.915 -> ~0.89. Capacity IS there; it went to the wrong place.
- **Conclusion / route:** move style OUT of head weights and the (weak, codebook-regime) LoRA delta,
  INTO an explicit switchable conditioning input z (per-teacher embedding OR target style-metric
  vector) + CFG/style-dropout so alpha/w is a real interpolatable knob and z=null returns to a
  neutral, shareable base. z-route also fixes multi-teacher sharing (one head, many z) and user
  addressability ("you"). Dropout alone is 治标; the strong+aligned SIGNAL (explicit z) is 治本.
