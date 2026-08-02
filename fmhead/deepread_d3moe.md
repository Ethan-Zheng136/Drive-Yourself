# D3-MoE — full annotated walkthrough (`/root/workspace/D3-MoE.py`, 1847 lines)

Reproducible-level read of the whole file, then a "how to port to FMHead" section.
**Study only — neither `D3-MoE.py` nor our code was modified.** All line refs → `D3-MoE.py`.
Our side = `/root/workspace/fmhead/` (`fm_head.py`, `autovla_fmhead.py`).

> One-line mental model: **a TransFuser/VoVNet perception backbone feeds a
> style-conditioned Mixture-of-DiT-Experts that runs DDIM (x0-prediction) diffusion
> to denoise a lon/lat-decomposed 8-waypoint trajectory; a supervised "functional
> router" picks which kinematic expert to use; the whole batch is run 3× (one per
> discrete driving style) and inference returns the "normal" branch.**

---

## 0. Module call graph (who calls who)

```
TransfuserModel.forward(features)                                     L1427
├─ _prepare_features(features)                                        L1122
│   ├─ V299FeatureExtractor.forward(camera, return_dict=True)         L112   → {p3,p6} seqs + main(p6) seq
│   │   └─ build_fcos_vovnet_fpn_backbone_p6 (VoVNet V2-99+FPN)       L65
│   └─ ego_state_encoder(status_feature)                             L680   → ego_feature
├─ stylistic_experts[0..2] : StyleExpert(image_global, ego)          L217   → style_feature ×3  → style_features_3x
├─ gating_network : StyleAwareGatingNetwork.forward(imgs,ego,style)  L286   → lat/lon indices,weights,logits, scene_summary
│   ├─ state_fusion_transformer([cls,ego,style])                     L299
│   ├─ vision_extractor_p3 / _p6 (cross-attn into image tokens)      L304
│   └─ lateral/longitudinal_decision_network (MLP → logits)          L324
├─ [TRAIN] fetch 3 GT variants from all_augmented_data[token]        L1478
│   ├─ _compute_functional_router_loss(...)                          L1383  (router KL loss)
│   │   └─ get_soft_functional_labels_batch(traj, style_idx)         L1182
│   │       └─ _extract_trajectory_features(traj)                    L1132  + KMeans-center buffers
│   ├─ norm_components(gt_lon, gt_lat)                               L161
│   ├─ scheduler.add_noise(gt_norm, noise, t)                        L1569
│   └─ _get_x0_prediction_batch_hard_routing(...)                    L1316  → x0 preds (lon,lat) + deep-sup lists
│       ├─ lateral_experts[i]  : StyleAwareDiT_Expert.forward        L626   (lat first)
│       └─ longitudinal_experts[i] : forward(lateral_hidden=...)     L626   (lon conditioned on lat)
│           └─ StyleAwareCrossAttentionDiTBlock.forward ×3           L526
└─ [INFER] DDIM loop over scheduler.timesteps                        L1741
    └─ _get_x0_prediction_batch_hard_routing(...) each step          L1745
    → denorm_components → multi_trajectory (B,3,8,3) → pick idx 1    L1785,L1833,L1836
```

---

## 1. End-to-end data flow (one forward pass)

**Inputs** (`_prepare_features`, L1122–1130):
- `features["camera_feature_t"]` → `(B, 3, H, W)` camera image (L1123).
- `features["status_feature"]` → `(B, ego_state_dim)` ego status (L1124).
- `features["token"]` → list of scene tokens, used only to fetch augmented GT (L1468, L1639).
- (optional) `features["experts_list_and_style_index"]` → manual expert-index override (L1455).

**Encode** (L1126–1130):
```python
image_features_dict, main_image_sequence = self.feature_extractor(camera_feature_t, return_dict=True)
ego_feature = self.ego_state_encoder(ego_state)
```
- `image_features_dict = {'p3': (B,N3,C), 'p6': (B,N6,C)}`, `main_image_sequence = p6 (B,N6,C)`, `C=context_dim`.
- `ego_feature = (B, C)`.

**Triple for the 3 styles** (L1432–1446): every conditioning tensor is repeated 3× along batch
(`image_sequence_3x`, `ego_feature_3x`, `image_features_dict_3x`); 3 style vectors are produced and
concatenated (`style_features_3x = (3B, style_feature_dim)`, L1439–1446).

**Route** (L1448–1454): `gating_network` returns, for the tripled batch, `lateral_indices_3x`,
`lateral_weights_3x`, `longitudinal_indices_3x`, `longitudinal_weights_3x`, `lateral_logits_3x`,
`longitudinal_logits_3x`, `final_feature_vec`.

**Diffuse** (train: single step at random `t`; infer: DDIM loop): denoise the **normalized lon
`(3B,8,1)` + lat `(3B,8,2)`** component tensors via the routed experts → x0 predictions.

**Output** (inference, L1833–1836):
```python
multi_trajectory = multi_trajectory_3x.view(3, batch_size, -1, 3).transpose(0, 1).contiguous()  # (B,3,8,3)
output["multimodal_trajectories"] = multi_trajectory
output["trajectory"] = multi_trajectory[:, 1, ...]   # (B,8,3) = the NORMAL style
```
Final trajectory shape `(B, 8, 3)` = `(x, y, heading)` per waypoint (denormalized metres).

---

## 2. Backbone & feature encoding (`V299FeatureExtractor`, L45–159)

- Builds **VoVNet V2-99 + FPN** with `build_fcos_vovnet_fpn_backbone_p6` (L65); config `V-99-eSE`,
  FPN out 256, features `stage2..5` (L51–59). Optional pretrained load (L67–78).
- Per FPN scale, `_create_scale_processor` (L87–97): `Conv2d(256→C,1)` projection, a conv
  `pos_encoding` added residually, a `Conv3x3+GroupNorm(8)+GELU` "enhancement" added at 0.5 weight,
  then flatten `HW` to a token sequence and `LayerNorm` (`_process_scale_feature`, L99–110):
```python
feat = processor['projection'](feature); feat = feat + processor['pos_encoding'](feat)
feat = feat + 0.5 * processor['enhancement'](feat)
feat_seq = feat.flatten(2).transpose(1,2); feat_seq = processor['norm'](feat_seq)
```
- `forward(..., return_dict=True)` returns `{p3,p6}` sequences + `main=p6` (L120–143).
- **Ego** is encoded separately by `ego_state_encoder` (2-layer MLP `ego_state_dim→C→C`, L680–684).
- These are the **conditioning tokens**: image token sequences (`p3`,`p6`), a pooled `image_global`
  (`mean over tokens`, L1440) for the StyleExpert, and the ego vector.
- **This is a trained perception stack** (VoVNet), not a frozen VLM.

---

## 3. MoE structure (exact)

**Counts** (`TransfuserModel.__init__`, L686–713):
```python
self.num_lateral_experts     = len(config.lateral_experts)        # L686
self.num_longitudinal_experts = len(config.longitudinal_experts)  # L687
self.lateral_experts      = ModuleList([StyleAwareDiT_Expert(config, "lateral")      for _ in range(num_lateral_experts)])       # L708
self.longitudinal_experts = ModuleList([StyleAwareDiT_Expert(config, "longitudinal") for _ in range(num_longitudinal_experts)])  # L711
self.stylistic_experts    = ModuleList([StyleExpert(context_dim*2, style_feature_dim) for _ in range(3)])                        # L703
```
- The literal counts come from `config.lateral_experts` / `config.longitudinal_experts` (not in this
  file). **Verified indirectly = 5 lateral, 3 longitudinal**: the router's KMeans uses `n_clusters=5`
  for lateral (L936) and `n_clusters=3` for longitudinal (L995), and the router MLPs output
  `num_lateral_experts` / `num_longitudinal_experts` logits (L264, L276). So the design is
  **5 lateral × 3 longitudinal DiT experts + 3 stylistic MLP experts**.
- **Each trajectory expert is a DiT** (`StyleAwareDiT_Expert`, L559), 3 blocks (L592), NOT an MLP.
  Lateral expert outputs `dim=2 [strafe, heading]` (L604), longitudinal outputs `dim=1 [forward]` (L606).
- **Stylistic experts are MLPs** (`StyleExpert`, L203) producing the per-style feature vector.

**Router / gating** (`StyleAwareGatingNetwork`, L221–341). Inputs: `image_features_dict`,
`ego_feature`, `style_features` (L286). Architecture:
1. Build `[cls, ego, style]` tokens, fuse with a 1-layer Transformer encoder (L290–300).
2. Cross-attend that fused state into `p3` and `p6` image tokens via two `TransformerDecoder`s
   (`vision_extractor_p3/_p6`, depth `config.gating_transformer_depth`, L304–314).
3. Flatten & concat the extracted features (`total_feature_dim = feature_dim*6`, L253, L319–322).
4. Two MLP "decision networks" → `lateral_logits (…,5)`, `longitudinal_logits (…,3)` (L324–325).

**Routing math** (L327–334):
```python
lateral_probs = F.softmax(lateral_logits, dim=-1)
lateral_weights, lateral_indices = torch.topk(lateral_probs, self.k_lateral, dim=-1)
lateral_weights = lateral_weights / (lateral_weights.sum(-1, keepdim=True) + 1e-8)   # (same for longitudinal)
```
Top-k **weights are computed**, but the expert dispatch (§below) uses **only index 0 → effectively
top-1 HARD routing**. There is no soft weighted mixture of experts at dispatch.

**Combine / dispatch** (`_get_x0_prediction_batch_hard_routing`, L1316–1381): pre-allocate zero
buffers, then for each expert run **only the samples routed to it** (mask on `indices[:,0]`):
```python
for i in range(self.num_lateral_experts):
    mask = (lateral_indices[:, 0] == i)          # L1331  (top-1 hard routing)
    if mask.sum() == 0: continue
    hidden_feat_list, explicit_feat_list = self.lateral_experts[i](**inputs)   # L1345
    for j in range(num_dit_layers):
        lat_hidden_output[j][mask] = hidden_feat_list[j]; lat_explicit_output[j][mask] = explicit_feat_list[j]
```
Longitudinal experts are dispatched the same way (L1351–1372) **but receive the lateral experts'
per-layer hidden features** for the routed samples (`lateral_features_for_lon`, L1356) as
cross-attention memory. So **outputs are not summed** — each sample takes exactly one lateral + one
longitudinal expert's explicit output (`final_lat = lat_explicit_output[-1]`, `final_lon =
lon_explicit_output[-1]`, L1374–1381).

---

## 4. Supervised "functional router" (the key mode-selection piece)

The router is **not** trained by RL or load-balancing — it is **supervised to predict which kinematic
cluster the GT trajectory belongs to**, per style.

**Step A — cluster centers (offline, cached)** (`_compute_data_driven_centers`, L831–1074):
- For each style (conservative/normal/aggressive) and each GT trajectory, compute two 2-D feature
  vectors via `_compute_trajectory_stats` (L1076–1120):
  - **lateral feature** = `(net_heading_change, signed_max_lateral_deviation)` (L1117);
  - **longitudinal feature** = `(mean_speed, mean_acc)` (L1118) (speeds = `Δpos/0.5`, outliers clipped).
- `StandardScaler` z-scores each, then **KMeans** with percentile-seeded init:
  `n_clusters=5` for lateral (L935–942), `n_clusters=3` for longitudinal (L994–1001), centers sorted
  by feature-0. Per-cluster stds are estimated from the nearest 30%/40% of points (L954–965, L1013–1024).
- Centers + stds + scaler mean/scale are stored as **buffers** `lateral_centers_{k}`,
  `lateral_stds_{k}`, `..._scaler_mean/scale_{k}` for `k∈{0,1,2}` styles (L1056–1072), cached to
  `aggressive_conservative.pkl` with a data fingerprint (L727–798).

**Step B — soft target labels** (`get_soft_functional_labels_batch`, L1182–1260):
```python
lateral_features, longitudinal_features = self._extract_trajectory_features(trajectories)  # L1187 (batched)
# per style: z-score with that style's scaler buffers, then Gaussian kernel to each center:
diff = style_lateral_features_scaled - center; normalized_diff = diff / std
dist_squared = (normalized_diff**2).sum(-1); logit = -dist_squared / (2*temperature)   # L1226-1231
style_lateral_dist = softmax(stack(lateral_logits), -1)                                 # L1235
# label smoothing 0.1 over 5 (lateral) / 3 (longitudinal) buckets:
style_lateral_dist = 0.9*style_lateral_dist + 0.1/5                                      # L1254
```
So the label for each sample is a **soft distribution over the 5 (lat) / 3 (lon) experts**, peaked at
the cluster whose center is closest (in that style's scaled kinematic space) to the sample's own
kinematics.

**Step C — router loss** (`_compute_functional_router_loss`, L1383–1425): **KL divergence** between
the router's predicted expert distribution and those soft labels:
```python
pred_lateral_log_probs = F.log_softmax(valid_lateral_logits, -1)                        # L1410
lateral_router_loss = F.kl_div(pred_lateral_log_probs, target_lateral_dist, reduction='batchmean')  # L1413
```
(same for longitudinal, L1419). Router accuracy vs argmax label is logged (L1536–1558). **No
load-balancing / entropy / switch-transformer aux loss** anywhere — balancing is implicit in the data.

> **This is the single most important thing to understand:** "mode selection" = a KL-supervised
> classifier that maps scene→(lateral bucket, longitudinal bucket), where the buckets are KMeans
> clusters of GT trajectory kinematics computed **per driving style**. The diffusion expert then just
> refines the trajectory within the chosen bucket.

---

## 5. The 3 discrete style branches

**In the data** (`augmented_trajectories.pkl`, L718): each token stores up to 3 GT variants —
`final_conservative_trajectory`, `dataset_trajectory` (= "normal", the real log), and
`pure_acceleration_trajectory` (= "aggressive"), the last gated by generation/filter flags +
`is_trajectory_anomalous` (L860–871, L1500). Fetched per batch at train (L1478–1520) and val
(L1650–1687); tripled into `gt_trajectory_3x = cat([conservative, normal, aggressive])` (L1516) with a
`valid_mask` marking which tokens actually had that variant (L1522).

**In the network:** there is **no explicit "style id" input** — instead the batch is **run 3×** and a
per-style **StyleExpert** MLP produces a distinct `style_feature` for each branch:
```python
for style_idx in range(3):
    style_feature = self.stylistic_experts[style_idx](image_global, ego_feature)   # L1442-1443
style_features_3x = torch.cat(style_features_list, dim=0)                           # L1446
```
Style then enters conditioning in **three places**:
1. **Router** — `style_token` fused into the gating state sequence (L297–300), so routing is
   style-dependent.
2. **DiT AdaLN** — `style_global = style_global_projection(style_features)` (L640) is concatenated with
   the timestep embedding to drive AdaLN modulation (L530): `adaln_in = cat([timestep_cond, style_global])`.
3. **DiT cross-attn** — `style_sequence` is concatenated with the ego sequence into `unified_cond`
   (L637–638) and consumed by `attn_ca_unified` (L544).

**Selection train vs infer:** at train, each of the 3 branches is supervised against its own GT variant
(per-style losses, L1617–1630). At inference, all 3 are produced but **only "normal" (index 1) is
returned** as `output["trajectory"]` (L1836); the other two live in `multimodal_trajectories`. The
`experts_list_and_style_index` hook (L1455–1457) can force a branch's expert indices. **Style is a
discrete 3-way category — no continuous knob, no guidance weight.**

---

## 6. Diffusion head (DDIM, x0)

**Scheduler** (L715–716):
```python
self.scheduler = DDIMScheduler(num_train_timesteps=config.num_train_timesteps,
                               beta_schedule="squaredcos_cap_v2", prediction_type="sample")
```
- `prediction_type="sample"` ⇒ the network predicts **x0 (the clean normalized trajectory)** directly,
  not ε and not velocity.
- Train timesteps = `config.num_train_timesteps`; inference uses `config.num_diffusion_steps_inference`
  DDIM steps (`self.scheduler.set_timesteps(...)`, L1734). Cosine (`squaredcos_cap_v2`) β schedule.

**What it denoises:** the **normalized lon `(3B,8,1)` and lat `(3B,8,2)` component tensors**, over the
8 waypoints — i.e. the *full* (component-split) trajectory in normalized space, **not** offsets and
**not** anchor residuals. Two independent DDIM chains (lon, lat), stepped separately (L1758–1764):
```python
noisy_lon_norm = self.scheduler.step(pred_x0_lon, t, noisy_lon_norm).prev_sample   # L1758
noisy_lat_norm = self.scheduler.step(pred_x0_lat, t, noisy_lat_norm).prev_sample   # L1762
```
Inference starts from pure Gaussian noise (`torch.randn(3B,8,1)` / `(3B,8,2)`, L1738–1739). **A single
noise draw per style branch — no N-sample multimodality.**

**Per-block conditioning** (`StyleAwareCrossAttentionDiTBlock.forward`, L526–557) — 4 injection paths:
1. AdaLN produces 9 params from `[timestep_cond, style_global]` (L530–532), **AdaLN-Zero init**
   (`nn.init.zeros_` on the last adaLN linear, L523–524).
2. **Self-attn** over the 8 traj tokens (RoPE) modulated by AdaLN shift/scale/gate `s_sa,c_sa,g_sa` (L534–537).
3. **Cross-attn → image tokens** `image_cond` (residual, *no* AdaLN gate) (L539–541).
4. **Cross-attn → unified `[ego, style]`** modulated by `s_uni,c_uni,g_uni` (L543–546).
5. **Cross-attn → lateral hidden** (longitudinal experts only) `attn_ca_lateral` (L548–551).
6. **SwiGLU MLP** (`silu(gate)*fc`) modulated by `s_mlp,c_mlp,g_mlp` (L553–555).

**Deep supervision:** every block has its own `explicit_output_head` (Linear→GELU→Linear→**Tanh**,
L619–624) producing an explicit x0 estimate; `forward` returns the list over all 3 blocks
(`explicit_output_list`, L642–656). The loss supervises **all 3 layers** with progressively increasing
weight (`compute_deep_supervision_loss`, L1262–1291):
```python
weight = 0.3 + 0.7 * (idx / (num_layers - 1))     # L1270
layer_loss = F.l1_loss(pred[valid_mask], gt[valid_mask])
total_loss = sum(weighted_losses) / num_layers
```

---

## 7. Trajectory param & normalization

- **lon(1)+lat(2) split**: `gt_lon = gt[...,:1]` (forward_dist), `gt_lat = gt[...,1:]` (strafe_dist,
  heading_change) (L1560). Full traj `(…,8,3)` = `(forward, strafe, heading)`? — actually the raw GT is
  `(x, y, heading)`; here index 0 is treated as the longitudinal component and 1:3 as lateral
  (strafe, heading). T = **8 waypoints** (pos_embed `(1,8,hidden)`, L611).
- **Fixed min-max normalization** (`norm_components`, L161–180): 
  `forward_dist_norm = 2*(forward - (-1))/(60-(-1)) - 1` (range `[-1,60]→[-1,1]`);
  `strafe_norm = strafe/17.723`; `heading_norm = heading/1.382`. Denorm inverts these (L182–201).
  Hand-crafted constants, **not** data-driven z-score.
- **lon conditioned on lat (verified):** in the dispatch, lateral experts run first and their per-layer
  hidden features are passed to the longitudinal experts (`lateral_hidden_features_list`, L1356→L629),
  which consume them via `attn_ca_lateral` only when `output_type=="longitudinal"` (L646–650). So
  **speed/longitudinal decoding attends to the already-decoded path shape.**

---

## 8. Full loss stack (train path L1466–1635; val adds heading)

| Loss term | Formula (verbatim ref) | Line |
|---|---|---|
| `lateral_router_loss` | `KL(log_softmax(lat_logits) ‖ soft_lat_labels)` | L1413, set L1533 |
| `longitudinal_router_loss` | `KL(log_softmax(lon_logits) ‖ soft_lon_labels)` | L1419, set L1534 |
| `final_lon_loss` | `L1(pred_lon_norm[valid], gt_lon_norm[valid])` (x0) | L1584 |
| `final_lat_loss` | `L1(pred_lat_norm[valid], gt_lat_norm[valid])` (x0) | L1585 |
| `deep_lon_loss`, `deep_lat_loss` | deep supervision L1 over 3 layers, progressive weights | L1587–1593 |
| `longitudinal_component_loss` | `final_lon_loss + deep_lon_loss` | L1595 |
| `lateral_component_loss` | `final_lat_loss + deep_lat_loss` | L1596 |
| `trajectory_l2_loss` | `MSE(recon_denorm[valid], gt[valid])` | L1601 |
| `trajectory_l1_loss` | `L1(recon_denorm[valid], gt[valid])` | L1606 |
| `trajectory_point_loss` | `mean ‖recon_xy − gt_xy‖₂` (ADE) | L1611–1615 |
| `style_specific_losses` | per-style L1 (logging only) | L1617–1630 |
| `heading_loss` (val only) | `L1(recon[...,2], gt[...,2])` | L1806 |

**Which perception/aux losses exist vs NOT — explicit answer for the user:**
- **Present:** trajectory reconstruction (x0 L1 on components, denorm L2/L1, ADE point loss, heading L1)
  + **router KL** (the only "mode/classification" aux) + deep supervision.
- **ABSENT:** there is **no BEV/map segmentation loss, no agent-detection/box loss, no
  collision/off-road/rule penalty, no PDM/PDMS term, and no explicit load-balancing loss**. The model
  is trajectory-reconstruction + supervised-routing only. (Safety/quality is handled *offline* by the
  `is_trajectory_anomalous` data filter, not by a training loss.)
- The individual `output[...]` terms are returned unweighted; the actual scalarization (per-term
  weights) happens in the training wrapper outside this file.

---

## 9. Inference / selection

- `self.scheduler.set_timesteps(config.num_diffusion_steps_inference)` (L1734); DDIM loop over
  `scheduler.timesteps` (L1741), each step calls `_get_x0_prediction_batch_hard_routing` and
  `scheduler.step(pred_x0, t, noisy)` for lon & lat (L1745–1764).
- Denormalize (`denorm_components`, L1785), stack to `multi_trajectory_3x (3B,8,3)`, reshape to
  `(B,3,8,3)` (L1833), expose as `multimodal_trajectories`, and **return the normal branch**:
  `output["trajectory"] = multi_trajectory[:, 1, ...]` (L1836).
- **`is_trajectory_anomalous` (L17–43) is a DATA-CURATION filter, not an inference selector** — it is
  only used to gate which augmented "aggressive" GTs enter the center computation (L865, L896) and the
  per-batch GT fetch (L1500, L1673). The runtime output is *not* anomaly-filtered here.
- **No learned scorer / PDM ranking of the 3 modes in this file** — "selection" = hard-pick index 1.

---

## 10. 怎么用 / Porting to FMHead (frozen-VLM `c` + rectified flow)

Our FMHead differs fundamentally (keep this explicit): **frozen Qwen2.5-VL-3B hidden `c` as content
condition**, **rectified flow-matching** (`u=τ−ε`, few-step Euler), **continuous weight-space style
axis** `s=α·(h_styled−h_base)` with **CFG weight w**, and a 7.36M drop-in decoder for AutoVLA's
codebook. The items below borrow *mechanisms*, not D3's discrete-style/MoE identity.

### ① Learned selection / functional-router head → replace `select_by_score` medoid stub  — HIGH
Port D3's §4 idea onto our N flow samples. Precompute KMeans centers on GT kinematics (heading-change,
lateral-dev, speed, acc) once from `traj_norm_stats_gt`-style data; add a tiny head that scores samples.
```python
# offline (once): centers = KMeans(k).fit(kinematic_feats(gt_trajs))   # store as buffer
# train: label = softmax(-||scaled_feat(gt) - centers||^2)  ; add KL(head(c) ‖ label)
# infer: pick the sample whose kinematic bucket == argmax head(c)  (replaces medoid)
def select_by_router(self, trajs, c):                     # trajs (B,N,T,2)
    feats = kinematic_feats(trajs)                        # (B,N,F)  heading/lat-dev/speed/acc
    tgt_bucket = self.router_head(c).argmax(-1)           # (B,)
    samp_bucket = assign_to_centers(feats, self.centers)  # (B,N)
    ok = (samp_bucket == tgt_bucket[:,None])
    return pick_first_true_else_medoid(trajs, ok)
```
Effort **medium** (kinematic feats + KMeans buffers + KL term + head); Benefit **high** (trained
selector, no RL/PDM; stepping-stone to PDM); Risk **medium** (must NOT leak style when style OFF; keep
router conditioned on `c` only).

### ② `is_trajectory_anomalous` filter on flow samples — cheap, HIGH-ROI
Port L17–43 verbatim (pure geometry) and apply as a **post-sampling filter** inside `sample()` /
`decode()` before selection:
```python
def _filter_anomalous(self, trajs):                       # trajs (B,N,T,3) metres
    keep = ~batched_is_anomalous(trajs)                   # bounds + >65°/step heading
    return trajs, keep   # if all filtered for a scene, fall back to unfiltered
```
Effort **low** (~25 lines, no training); Benefit **medium-high** (drops degenerate flow draws; cleaner
PDM input); Risk **low** (geometry-only; always keep a fallback).

### ③ lon/lat decomposition (+ lon←lat coupling) — MEDIUM (control-friendly)
Split our `τ` head into forward vs (strafe, heading) streams (D3 L161–201, L626–656); optionally let the
longitudinal flow cross-attend the lateral hidden. Rationale: our **continuous style axis** becomes more
interpretable — aggressiveness ≈ longitudinal magnitude, path preference ≈ lateral.
```python
# two small output heads off the shared DiT trunk: v_lat (…,2), v_lon (…,1)
# lon block cross-attends lat hidden (as D3): attn_ca_lateral(x_lon, lat_hidden)
```
Effort **medium**; Benefit **medium** (decoupled/interpretable `s`); Risk **medium** (more parts; erodes
our "tiny single head" simplicity — keep behind a config flag).

### ④ MoE experts as an *option/complement* to N-sample multimodality — LOWER
Replace/augment stochastic N-sampling with K flow-velocity experts + a (supervised) gate on `c`. Gives
labelable, structured modes.
```python
# K small velocity heads share the DiT trunk; gate(c) -> top-1 expert (or soft mix)
# multimodality = experts, not (only) noise; select via ① router head
```
Effort **high**; Benefit **medium-high** (explicit modes); Risk **medium** (routing complexity vs our
minimalism). Revisit only if N-sample modes prove hard to select among. **Skip D3's load-balancing** —
like D3, a supervised router removes the need.

**Do NOT borrow** (keeps differentiation): the 3 discrete data-augmentation styles + StyleExpert branches
(our continuous `s` + CFG is strictly more general), and the VoVNet perception (we reuse a frozen VLM).

---

## 11. Novelty / differentiation (unchanged, restated)
Overlap = "style-aware generative multimodal AD planner (DiT+AdaLN+diffusion)". We stay differentiated by:
(1) **continuous weight-space→activation-space style axis** `s=α(h_styled−h_base)` + **CFG weight w** vs
D3's **3 discrete data-aug buckets** (which also *need* 3 GT variants per scene); (2) **frozen language-
grounded VLM backbone + drop-in codebook-decoder replacement** vs a bespoke VoVNet planner;
(3) **rectified flow + CFG** vs DDIM x0. Borrow items ①②③ are mechanism-level and do not adopt D3's
discrete-style/MoE core, so they don't dilute the contribution.
