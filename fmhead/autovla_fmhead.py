"""autovla_fmhead.py -- REAL integration of FMHead into AutoVLA.

Replaces AutoVLA's discrete action-token -> codebook trajectory decoder with the
continuous conditional flow-matching FMHead, WITHOUT modifying any AutoVLA file.
Everything here is additive: a hidden-state extractor, a decoder wrapper module,
and a thin AutoVLA subclass that overrides `predict` and exposes a training loss.

Verified against real AutoVLA source (paths under
`/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA/`):

  * VLM = Qwen2_5_VLForConditionalGeneration, torch_dtype=bfloat16
        models/autovla.py:13, models/autovla.py:1072-1076
  * hidden_size = 2048  (Qwen2.5-VL-3B-Instruct config.json)
        config/training/*.yaml: pretrained_model_path .../Qwen2.5-VL-3B-Instruct
  * trajectory: num_poses=10, interval_length=0.5 s
        config/training/*.yaml: model.trajectory.{num_poses,interval_length}
  * action-token sub-vocab: [action_start_id, action_start_id + n_bins),
        action_start_id=151665 ; codebook (n_bins, 6, 4, 2)
        models/action_tokenizer.py:43-55 ; config .../tokens.action_start_id
  * discrete decode -> navsim poses (T, 3) = (x_forward, y_left, heading)
        models/action_tokenizer.py:95-124 (rollout) ; predict() [0,1:]
        models/autovla.py:1115 ; navsim .../autovla_agent.py:481 Trajectory(poses[:num_poses])
  * gt_trajectory (B, G, >=2) x-forward/y-left metres, L2 target uses [..., :2]
        models/autovla.py:757-808 (_expected_trajectory_l2)
  * style axis: styled = LoRA-on forward ; base = vlm.disable_adapter() forward
        models/autovla.py:939-942 (KL uses disable_adapter to get the base logits)

The heavy Qwen VLM is NOT loaded here; the AutoVLA subclass import is guarded so
this module imports cleanly in any env, and the tensor plumbing is proven by
`test_integration.py` with a mock VLM emitting hidden states of the CORRECT dim.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from fm_head import FMHead, FMHeadConfig, StyleFMHead


# =============================================================================
# Real AutoVLA constants (verified from source; see module docstring)
# =============================================================================
AUTOVLA_HIDDEN_SIZE = 2048          # Qwen2.5-VL-3B-Instruct hidden_size
AUTOVLA_NUM_POSES = 10              # model.trajectory.num_poses
AUTOVLA_INTERVAL_S = 0.5           # model.trajectory.interval_length
AUTOVLA_ACTION_START_ID = 151665    # model.tokens.action_start_id
AUTOVLA_VLM_DTYPE = torch.bfloat16  # from_pretrained(torch_dtype=torch.bfloat16)


def autovla_fmhead_config(hidden_size: int = 256, depth: int = 4,
                          style_dropout_prob: float = 0.2,
                          use_style_adapter: bool = False,
                          style_adapter_hidden: int = 256,
                          style_adapter_mode: str = "adaln",
                          constraint: Optional[dict] = None,
                          use_z_encoder: bool = False,
                          z_encoder_dims: Tuple[int, int] = (512, 256),
                          z_norm_delta: bool = False) -> FMHeadConfig:
    """FMHeadConfig whose dims exactly match real AutoVLA (Qwen2.5-VL-3B).

    ``use_style_adapter`` defaults False -> plain v2 head (byte-identical). Set True to
    build the opt-in StyleFMHead (frozen base + zero-gated style adapter).

    ``constraint`` (fm3, ADDITIVE): optional dict of feasibility knobs. None or
    {"mode": "none"} -> byte-identical fm2 head. Recognized keys: mode
    (none|soft|kinematic), jerk_weight, curv_weight, smooth_t_min, accel_max,
    yawrate_max, v_max, dt.
    """
    kn = dict(constraint or {})
    mode = kn.get("mode", "none")
    cfg = FMHeadConfig(
        horizon=AUTOVLA_NUM_POSES,          # 10
        traj_dim=2,                         # (x_forward, y_left) or (a_long, yaw_rate)
        d_cond=AUTOVLA_HIDDEN_SIZE,         # 2048 (legacy pooled; unused when use_context)
        d_style=AUTOVLA_HIDDEN_SIZE,        # 2048
        use_context=True, d_ctx=AUTOVLA_HIDDEN_SIZE, ctx_max_len=64,   # v2 multi-token context
        hidden_size=hidden_size, depth=depth,
        num_heads=8, mlp_ratio=4.0, cond_hidden=512,
        style_dropout_prob=style_dropout_prob,
        use_style_adapter=use_style_adapter, style_adapter_hidden=style_adapter_hidden,
        style_adapter_mode=style_adapter_mode,
        use_z_encoder=use_z_encoder, z_encoder_dims=tuple(z_encoder_dims),
        z_norm_delta=z_norm_delta,
        constraint_mode=mode,
        jerk_weight=float(kn.get("jerk_weight", 0.0)),
        curv_weight=float(kn.get("curv_weight", 0.0)),
        smooth_t_min=float(kn.get("smooth_t_min", 0.0)),
        accel_max=float(kn.get("accel_max", 5.0)),
        yawrate_max=float(kn.get("yawrate_max", 0.8)),
        v_max=float(kn.get("v_max", 25.0)),
        kin_dt=float(kn.get("dt", AUTOVLA_INTERVAL_S)),
    )
    return cfg


# =============================================================================
# 1. Hidden-state extraction (scene condition c and style delta s)
# =============================================================================
def action_region_hidden(
    hidden_last: torch.Tensor,           # (B, S, H) last-layer hidden states
    labels: Optional[torch.Tensor] = None,   # (B, S) label ids (teacher-forced path)
    attention_mask: Optional[torch.Tensor] = None,  # (B, S)
    action_start_id: int = AUTOVLA_ACTION_START_ID,
    n_bins: Optional[int] = None,
    reduce: str = "mean",
) -> torch.Tensor:
    """Extract the (B, H) scene/content condition ``c`` from VLM hidden states.

    Uses the hidden states at the ANSWER/ACTION region:
      * teacher-forced (training): positions where ``labels`` fall in the action
        sub-vocabulary ``[action_start_id, action_start_id + n_bins)`` are the
        action-token positions (shift-aligned: the hidden at position p produces
        the token at p+1, so we read hidden at those p's -> use labels[:,1:]).
      * generation/no-labels: fall back to the last non-padded token hidden state
        (the answer position right before the trajectory is emitted).

    ``reduce`` in {"mean","last"} pools over the action positions.
    Returns dtype = hidden_last.dtype (bf16 in real AutoVLA).
    """
    B, S, H = hidden_last.shape
    if labels is not None:
        # shift-align: hidden[:, :-1] predicts labels[:, 1:]
        shifted_labels = labels[:, 1:]
        h_shift = hidden_last[:, :-1, :]
        upper = action_start_id + n_bins if n_bins is not None else torch.iinfo(labels.dtype).max
        mask = (shifted_labels >= action_start_id) & (shifted_labels < upper)  # (B, S-1)
        out = h_shift.new_zeros(B, H)
        for b in range(B):
            idx = torch.nonzero(mask[b], as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                # no action tokens in this row -> last valid token
                out[b] = _last_valid_hidden(hidden_last[b], attention_mask[b] if attention_mask is not None else None)
            elif reduce == "last":
                out[b] = h_shift[b, idx[-1]]
            else:
                out[b] = h_shift[b, idx].mean(dim=0)
        return out
    # no labels: last non-pad token per row
    return torch.stack([
        _last_valid_hidden(hidden_last[b], attention_mask[b] if attention_mask is not None else None)
        for b in range(B)
    ], dim=0)


def _last_valid_hidden(hidden_bsh: torch.Tensor, mask_s: Optional[torch.Tensor]) -> torch.Tensor:
    """(S, H), (S,) -> (H,) hidden at the last attended token."""
    if mask_s is None:
        return hidden_bsh[-1]
    nz = torch.nonzero(mask_s, as_tuple=False)
    if nz.numel() == 0:
        return hidden_bsh[-1]
    return hidden_bsh[nz[-1, 0]]


def answer_position_hidden(
    hidden_last: torch.Tensor,           # (B, S, H) last-layer hidden states
    input_ids: torch.Tensor,             # (B, S) token ids (MUST be the actual input ids)
    attention_mask: Optional[torch.Tensor] = None,  # (B, S)
    action_start_id: int = AUTOVLA_ACTION_START_ID,
) -> torch.Tensor:
    """Extract c at the **generation-start / answer position** -- IDENTICAL in train & infer.

    Rationale (the fix for the train/infer c mismatch): the codebook/FMHead trajectory is
    generated *autoregressively starting from the assistant-generation-start token* (the
    token right before the first action token). By causal attention, the hidden state at
    that anchor depends ONLY on the prompt (tokens <= anchor) and is therefore invariant to
    whether GT action tokens follow it. So we read c at exactly that anchor in every path:

      * TRAINING / eval (teacher-forced, input_ids contain action tokens >= action_start_id):
        anchor = (index of the FIRST action token) - 1  == the last prompt/header token.
      * INFERENCE (agent, get_prompt with add_generation_prompt=True, NO action tokens):
        anchor = last attended (non-pad) token == the same assistant-generation-start token.

    Both read the same semantic position with the same value -> train c == infer c.
    Returns (B, H), dtype = hidden_last.dtype.
    """
    B, S, H = hidden_last.shape
    out = hidden_last.new_zeros(B, H)
    for b in range(B):
        ids = input_ids[b]
        act = torch.nonzero(ids >= action_start_id, as_tuple=False)
        if act.numel() > 0:
            anchor = int(act[0, 0]) - 1                      # token right before first action
            anchor = max(anchor, 0)
        elif attention_mask is not None:
            nz = torch.nonzero(attention_mask[b], as_tuple=False)
            anchor = int(nz[-1, 0]) if nz.numel() > 0 else S - 1
        else:
            anchor = S - 1
        out[b] = hidden_last[b, anchor]
    return out


def context_sequence_hidden(
    hidden_last: torch.Tensor,           # (B, S, H) last-layer hidden states
    input_ids: torch.Tensor,             # (B, S) token ids
    attention_mask: Optional[torch.Tensor] = None,  # (B, S)
    action_start_id: int = AUTOVLA_ACTION_START_ID,
    max_len: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """v2 conditioning: return the VLM hidden states over the PROMPT span as a SEQUENCE.

    Returns (ctx, ctx_mask): ctx=(B, T_ctx, H), ctx_mask=(B, T_ctx) with 1=valid.

    The prompt span is tokens 0 .. anchor, where anchor = the generation-start token
    (identical in train & inference, exactly like `answer_position_hidden`):
      * train (teacher-forced, action tokens present): anchor = (first action tok) - 1;
      * inference (no action tokens): anchor = last attended token.
    We keep ONLY positions <= anchor (never the GT action tokens), so by causal attention
    the context is bit-identical whether or not action tokens follow -> train == infer.
    The last ``max_len`` attended prompt tokens are kept (the answer-region context);
    rows are right-padded to the batch max and masked.
    """
    B, S, H = hidden_last.shape
    seqs, lens = [], []
    for b in range(B):
        ids = input_ids[b]
        act = torch.nonzero(ids >= action_start_id, as_tuple=False)
        if act.numel() > 0:
            anchor = max(int(act[0, 0]) - 1, 0)
        elif attention_mask is not None:
            nz = torch.nonzero(attention_mask[b], as_tuple=False)
            anchor = int(nz[-1, 0]) if nz.numel() > 0 else S - 1
        else:
            anchor = S - 1
        if attention_mask is not None:
            pos = [j for j in range(anchor + 1) if bool(attention_mask[b, j])]
        else:
            pos = list(range(anchor + 1))
        pos = pos[-max_len:] if len(pos) > max_len else pos
        seqs.append(hidden_last[b, pos])          # (L_b, H)
        lens.append(len(pos))
    Lmax = max(lens)
    ctx = hidden_last.new_zeros(B, Lmax, H)
    ctx_mask = hidden_last.new_zeros(B, Lmax)
    for b, (sq, L) in enumerate(zip(seqs, lens)):
        ctx[b, :L] = sq
        ctx_mask[b, :L] = 1.0
    return ctx, ctx_mask


def style_delta(h_styled: torch.Tensor, h_base: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Activation-space style vector s = alpha * (h_styled - h_base), shape (B, H).

    ``h_styled`` = LoRA-adapted forward hidden; ``h_base`` = base forward hidden
    (obtained via ``vlm.disable_adapter()``). See AutoVLA.training_step KL block
    (models/autovla.py:939-942) for the base/styled toggle pattern.
    """
    return alpha * (h_styled - h_base)


HiddenFn = Callable[[bool], torch.Tensor]
"""A callable(disable_adapter: bool) -> (B, H) action-region hidden state.

Injected by the caller so the same code path works with the real AutoVLA VLM
(toggling LoRA) or with a mock in tests. ``disable_adapter=True`` must return the
BASE (LoRA-off) hidden state; ``False`` the styled (LoRA-on) hidden state."""


# =============================================================================
# 2. Decoder wrapper module (drop-in for the codebook decode)
# =============================================================================
class FMHeadDecoder(nn.Module):
    """Continuous flow-matching trajectory decoder that consumes AutoVLA hidden states.

    Inference : decode(c, s) -> navsim poses (B, num_poses, 3) via
                FMHead.sample + select_by_score + append_heading.
    Training  : training_loss(gt_trajectory, c, s) -> scalar flow-matching MSE,
                using gt_trajectory[..., :num_poses, :2] as the target tau.
    """

    def __init__(self, config: Optional[FMHeadConfig] = None,
                 num_samples: int = 16, num_steps: int = 30, cfg_weight: float = 2.0,
                 style_alpha: float = 1.0, content_source: str = "styled"):
        super().__init__()
        self.config = config or autovla_fmhead_config()
        # content_source (ADDITIVE): which VLM forward provides the CROSS-ATTN CONTENT (ctx).
        #   "styled" (DEFAULT) -> LEGACY inference behavior (content from the LoRA-ON forward),
        #                         byte-identical to before.
        #   "base"   -> content from the LoRA-OFF (neutral) forward -> the body stays LoRA-free
        #               and alpha=0/z=null returns the clean neutral base. The style delta
        #               s = alpha*(styled_anchor - base_anchor) is computed the SAME either way.
        #   "interp" -> LATENT CONTENT INTERPOLATION (inference-only, NO training / NO new module):
        #               feed the cross-attn CONTENT as c(alpha)=ctx_base + alpha*(ctx_styled-ctx_base)
        #               and force the AdaLN style s=0 (NO z-encoder, NO adapter, NO CFG style path).
        #               alpha is the ONLY knob: alpha=0 -> c=ctx_base (== content_source="base" with
        #               s=0, bit-exact clean base); alpha up -> content latent shifts toward persona.
        assert content_source in ("styled", "base", "interp"), content_source
        self.content_source = content_source
        assert self.config.d_style == AUTOVLA_HIDDEN_SIZE, self.config.d_style
        assert self.config.d_ctx == AUTOVLA_HIDDEN_SIZE, self.config.d_ctx
        assert self.config.horizon == AUTOVLA_NUM_POSES, self.config.horizon
        # opt-in style adapter: build StyleFMHead (frozen base + zero-gated adapter) only when
        # config.use_style_adapter is True; default False -> plain FMHead (byte-identical v2).
        self.fm_head = (StyleFMHead if getattr(self.config, "use_style_adapter", False)
                        else FMHead)(self.config)
        self.num_samples = num_samples
        self.num_steps = num_steps
        self.cfg_weight = cfg_weight
        self.style_alpha = style_alpha

        # Per-(waypoint, dim) trajectory normalization. Flow matching mixes the
        # target with unit-variance Gaussian noise, so targets MUST be standardized
        # (AutoVLA trajectories reach ~30 m over the 5 s / 10-pose horizon). Identity
        # by default; call `fit_normalizer`/`set_normalizer` on real data before
        # training. Buffers so they persist in the checkpoint and move with .to().
        T, D = self.config.horizon, self.config.traj_dim
        self.register_buffer("traj_mean", torch.zeros(T, D))
        self.register_buffer("traj_std", torch.ones(T, D))

    # -- normalization -----------------------------------------------------
    @torch.no_grad()
    def fit_normalizer(self, gt_trajectory: torch.Tensor, momentum: Optional[float] = None):
        """Fit per-(waypoint,dim) mean/std from a batch of gt trajectories (B,>=T,>=2)."""
        T, D = self.config.horizon, self.config.traj_dim
        tau = gt_trajectory[:, :T, :D].float()
        mean = tau.mean(dim=0)
        std = tau.std(dim=0).clamp_min(0.5)
        if momentum is None:
            self.traj_mean.copy_(mean)
            self.traj_std.copy_(std)
        else:
            self.traj_mean.mul_(1 - momentum).add_(momentum * mean)
            self.traj_std.mul_(1 - momentum).add_(momentum * std)

    @torch.no_grad()
    def set_normalizer(self, mean: torch.Tensor, std: torch.Tensor):
        self.traj_mean.copy_(mean.to(self.traj_mean))
        self.traj_std.copy_(std.to(self.traj_std).clamp_min(1e-3))

    @torch.no_grad()
    def load_normalizer(self, json_path: str):
        """Load per-(waypoint,dim) mean/std from a stats JSON (see traj_norm_stats.json)."""
        import json
        with open(json_path) as f:
            st = json.load(f)
        mean = torch.tensor(st["per_waypoint_mean"], dtype=torch.float32)
        std = torch.tensor(st["per_waypoint_std"], dtype=torch.float32)
        assert mean.shape == (self.config.horizon, self.config.traj_dim), mean.shape
        self.set_normalizer(mean, std)
        return st

    def _normalize(self, tau: torch.Tensor) -> torch.Tensor:
        return (tau - self.traj_mean.to(tau)) / self.traj_std.to(tau)

    def _denormalize(self, tau: torch.Tensor) -> torch.Tensor:
        return tau * self.traj_std.to(tau) + self.traj_mean.to(tau)

    # -- style-adapter freezing -------------------------------------------
    def freeze_base_keep_style(self):
        """STYLE phase: freeze the loaded 0.91 head + normalizer; train ONLY the style
        adapter. Returns (n_trainable, n_frozen). Requires use_style_adapter=True."""
        adapter = getattr(self.fm_head, "style_adapter", None)
        assert adapter is not None, "freeze_base_keep_style needs a StyleFMHead (use_style_adapter=True)"
        adapter_ids = {id(p) for p in adapter.parameters()}
        n_train, n_frozen = 0, 0
        for p in self.parameters():
            if id(p) in adapter_ids:
                p.requires_grad_(True); n_train += p.numel()
            else:
                p.requires_grad_(False); n_frozen += p.numel()
        return n_train, n_frozen

    def style_adapter_parameters(self):
        """Iterator over ONLY the trainable style-adapter params (for the optimizer)."""
        adapter = getattr(self.fm_head, "style_adapter", None)
        return iter(()) if adapter is None else adapter.parameters()

    def freeze_all_but_z_path(self):
        """Z-PATH STYLE phase (train_z_path_only): train ONLY the learned z-encoder (its MLP +
        the zero-init output gate). FREEZE everything else in the head -- INCLUDING ``s_proj``
        (the AdaLN style projection) -- plus the DiT blocks, self/cross-attn, embedders, final
        layer, temporal_pos, ctx_proj, null_style, and the normalizer buffers. The VLM + persona
        LoRA are frozen OUTSIDE this module.

        Rationale: the content/feasibility BODY *and* the final style projection stay exactly the
        loaded neutral fm3-kin base, so nothing but the z-encoder can move. Combined with the
        z-encoder's ZERO-init output gate (z_norm_delta) this GUARANTEES alpha=0 / z=null is
        BIT-EXACT the clean 0.915 fm3-kin base: the z-encoder emits 0 at init -> ``s_proj``
        receives 0 -> ``s_proj(0)`` == the base's own neutral AdaLN point (s_proj unchanged).
        Style then grows smoothly as ONLY the z-encoder learns. Returns (n_trainable, n_frozen).
        Requires use_z_encoder=True."""
        z_enc = getattr(self.fm_head, "z_encoder", None)
        assert z_enc is not None, "freeze_all_but_z_path needs a head with use_z_encoder=True"
        train_ids = {id(p) for p in z_enc.parameters()}
        n_train, n_frozen = 0, 0
        for p in self.parameters():
            if id(p) in train_ids:
                p.requires_grad_(True); n_train += p.numel()
            else:
                p.requires_grad_(False); n_frozen += p.numel()
        return n_train, n_frozen

    def z_path_parameters(self):
        """Iterator over ONLY the trainable z-path params (the z-encoder) for the optimizer."""
        z_enc = getattr(self.fm_head, "z_encoder", None)
        if z_enc is None:
            return iter(())
        return z_enc.parameters()

    def unfreeze_head_and_style(self):
        """STYLE phase, option (1) "don't over-freeze": UNFREEZE the FM head's OWN params
        (DiT blocks + embedders + cross-attn + final layer) AND the style adapter (if any),
        training them TOGETHER. The VLM + persona LoRA remain frozen OUTSIDE this module
        (train_fmhead.py never enables their grads). Returns (n_trainable, n_frozen).

        The only intentionally-frozen param is the fixed sin-cos ``temporal_pos`` when
        ``learn_temporal_pos`` is False (matching how the head was pre-trained). This is
        ADDITIVE: it does NOT alter :meth:`freeze_base_keep_style` (adapter-only), so the
        old GOLD style path is untouched."""
        learn_tpos = bool(getattr(self.config, "learn_temporal_pos", False))
        n_train, n_frozen = 0, 0
        for name, p in self.named_parameters():
            if name.endswith("temporal_pos") and not learn_tpos:
                p.requires_grad_(False); n_frozen += p.numel()
            else:
                p.requires_grad_(True); n_train += p.numel()
        return n_train, n_frozen

    # -- training ----------------------------------------------------------
    def forward(self, gt_trajectory: torch.Tensor, ctx: torch.Tensor, s: torch.Tensor,
                ctx_mask: Optional[torch.Tensor] = None,
                style_dropout_prob: Optional[float] = None) -> torch.Tensor:
        """Alias of :meth:`training_loss` so the module can be wrapped in
        ``DistributedDataParallel`` (DDP installs its grad-sync hooks around the
        wrapped module's ``forward``; training must call ``ddp_decoder(gt, ctx, s)``).
        ``ctx`` = VLM context sequence (B, T_ctx, d_ctx)."""
        return self.training_loss(gt_trajectory, ctx, s, ctx_mask=ctx_mask,
                                  style_dropout_prob=style_dropout_prob)

    def _estimate_v0(self, gt_xy: torch.Tensor) -> torch.Tensor:
        """Initial speed v0 (B,) from the first GT step: |wp0|/dt (v0_source=first_gt_step)."""
        return torch.linalg.norm(gt_xy[:, 0, :], dim=-1) / float(self.config.kin_dt)

    def training_loss(self, gt_trajectory: torch.Tensor, ctx: torch.Tensor, s: torch.Tensor,
                      ctx_mask: Optional[torch.Tensor] = None,
                      style_dropout_prob: Optional[float] = None) -> torch.Tensor:
        """gt_trajectory: (B, G>=num_poses, >=2) metres -> scalar flow-matching loss.
        ctx: (B, T_ctx, d_ctx) VLM context sequence (cross-attn content).

        fm3 modes (ADDITIVE; default "none" -> the original fm2 xy path, unchanged):
          * soft      : xy flow + jerk/curvature smoothness penalty on x_hat1.
          * kinematic : invert GT xy -> unicycle controls (a_long, yaw_rate) with the
            per-sample initial speed, normalize with the CONTROL stats, and flow-match in
            control space (+ optional control-smoothness penalty). Feasibility is enforced
            by integration + caps at decode time.
        """
        T = self.config.horizon
        assert gt_trajectory.shape[1] >= T, f"gt has {gt_trajectory.shape[1]} poses, need >= {T}"
        dt = self.fm_head.x_embedder.weight.dtype
        mode = self.config.constraint_mode
        # thread alpha as the z-encoder's EXTERNAL output gain (norm_delta path); s already
        # carries alpha as a direction (normalized away), so the gain here restores the dose.
        self.fm_head.style_gain = float(self.style_alpha)

        if mode == "kinematic":
            gt_xy = gt_trajectory[:, :T, :2].float()               # metres
            v0 = self._estimate_v0(gt_xy)                          # (B,)
            controls = FMHead.xy_to_controls(gt_xy, v0, dt=self.config.kin_dt)  # (B,T,2)
            tau = self._normalize(controls.to(dt))                 # control-space z-score
            return self.fm_head.flow_matching_loss(
                tau, ctx, s, ctx_mask=ctx_mask, style_dropout_prob=style_dropout_prob,
                smooth_space="control", traj_mean=self.traj_mean, traj_std=self.traj_std)

        # xy path (none | soft): identical to fm2 when jerk/curv weights are 0.
        tau = gt_trajectory[:, :T, :2].to(dt)
        tau = self._normalize(tau)
        return self.fm_head.flow_matching_loss(
            tau, ctx, s, ctx_mask=ctx_mask, style_dropout_prob=style_dropout_prob,
            smooth_space="xy", traj_mean=self.traj_mean, traj_std=self.traj_std)

    # -- inference ---------------------------------------------------------
    def _samples_to_poses(self, samples: torch.Tensor,
                          v0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, N, T, 2) normalized head samples -> navsim poses (B, N, T, 3) [x,y,heading] metres.

        none|soft : denormalize xy + append_heading (atan2 forward-fill) -- the fm2 path.
        kinematic : denormalize -> physical controls -> integrate_unicycle (with feasibility
        caps) -> native integrated heading (drops the atan2 hack).
        """
        phys = self._denormalize(samples)              # (B, N, T, 2)
        if self.config.constraint_mode != "kinematic":
            return FMHead.append_heading(phys)         # (B, N, T, 3)
        B, N, T, _ = phys.shape
        if v0 is None:
            v0 = phys.new_zeros(B)                      # deploy fallback (no ego speed given)
        v0e = v0.view(B, 1).expand(B, N).reshape(B * N)
        xy, psi = FMHead.integrate_unicycle(
            phys.reshape(B * N, T, 2), v0e, dt=self.config.kin_dt,
            accel_max=self.config.accel_max, yawrate_max=self.config.yawrate_max,
            v_max=self.config.v_max, clamp=True)
        poses = torch.cat([xy, psi.unsqueeze(-1)], dim=-1)   # native heading column
        return poses.view(B, N, T, 3)

    @torch.no_grad()
    def decode(self, ctx: torch.Tensor, s: torch.Tensor,
               ctx_mask: Optional[torch.Tensor] = None,
               scorer: Optional[Callable] = None,
               return_candidates: bool = False,
               v0: Optional[torch.Tensor] = None):
        """ctx:(B,T_ctx,d_ctx), s:(B,d_style) -> navsim poses (B, num_poses, 3) [x,y,heading].

        ``v0`` (B,) initial ego speed -- used ONLY in kinematic mode for unicycle
        integration (ignored by the none/soft xy paths)."""
        self.fm_head.style_gain = float(self.style_alpha)   # external alpha gain (z norm path)
        samples = self.fm_head.sample(
            ctx, s, ctx_mask=ctx_mask, num_samples=self.num_samples,
            num_steps=self.num_steps, cfg_weight=self.cfg_weight,
        )                                              # (B, N, T, 2) normalized
        cand = self._samples_to_poses(samples, v0=v0)  # (B, N, T, 3) metres [x,y,heading]
        B = cand.shape[0]
        if scorer is None:
            endpoints = cand[:, :, -1, :2]                        # (B, N, 2)
            mean_ep = endpoints.mean(dim=1, keepdim=True)
            scores = -torch.linalg.norm(endpoints - mean_ep, dim=-1)  # (B, N) medoid
        else:
            scores = scorer(cand[..., :2])
        best_idx = scores.argmax(dim=1)                          # (B,)
        poses = cand[torch.arange(B, device=cand.device), best_idx]   # (B, T, 3)
        if return_candidates:
            return poses, self._denormalize(samples)
        return poses

    @torch.no_grad()
    def decode_all(self, ctx: torch.Tensor, s: torch.Tensor,
                   ctx_mask: Optional[torch.Tensor] = None,
                   v0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return ALL N candidate navsim poses in metres: (B, N, num_poses, 3) [x,y,heading]."""
        self.fm_head.style_gain = float(self.style_alpha)   # external alpha gain (z norm path)
        samples = self.fm_head.sample(ctx, s, ctx_mask=ctx_mask, num_samples=self.num_samples,
                                      num_steps=self.num_steps, cfg_weight=self.cfg_weight)
        return self._samples_to_poses(samples, v0=v0)  # (B, N, T, 3)

    # -- convenience: build c and s from a HiddenFn ------------------------
    def conditions_from_hidden_fn(self, hidden_fn: HiddenFn) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (c, s) given a callable(disable_adapter)->hidden. c = styled hidden,
        s = alpha*(styled - base). If no adapter is available, base==styled -> s=0."""
        h_styled = hidden_fn(False)
        try:
            h_base = hidden_fn(True)
        except Exception:
            h_base = h_styled
        c = h_styled
        s = style_delta(h_styled, h_base, alpha=self.style_alpha)
        return c, s


# =============================================================================
# 3. Thin AutoVLA subclass (real integration entry points; guarded import)
# =============================================================================
def build_autovla_fmhead_subclass():
    """Return an `AutoVLA_FMHead(AutoVLA)` subclass, or raise if AutoVLA unavailable.

    Kept behind a factory so importing this module never requires the heavy Qwen
    stack. Call this inside the AutoVLA training/eval process (where `models`
    is importable and the checkpoint can load).
    """
    from models.autovla import AutoVLA  # noqa: WPS433 (guarded, intentional)

    class AutoVLA_FMHead(AutoVLA):
        """AutoVLA with the codebook decode replaced by FMHeadDecoder.

        NOTE: this subclass adds behavior only; it never edits AutoVLA. It reuses
        `get_prompt`, `self.vlm`, `self.processor`, `self.action_tokenizer`.
        """

        def __init__(self, config, inference=False, device="cpu",
                     fm_decoder: Optional[FMHeadDecoder] = None, style_alpha: float = 1.0,
                     style_off: bool = False):
            super().__init__(config, inference=inference, device=device)
            self.fm_decoder = fm_decoder or FMHeadDecoder(style_alpha=style_alpha)
            self.fm_decoder.to(device)
            # style_off=True: no persona LoRA / s=0 -> single VLM forward, no disable_adapter.
            self.style_off = style_off

        # -- hidden-state helpers ------------------------------------------
        def _forward_ctx(self, model_inputs: Dict, disable_adapter: bool):
            """One VLM forward -> (ctx, ctx_mask, anchor) where:
              ctx (B,T_ctx,2048) = prompt-span hidden SEQUENCE (v2 content, cross-attn),
              ctx_mask (B,T_ctx), anchor (B,2048) = generation-start hidden (for style delta).
            All computed from positions <= anchor -> identical in train & inference."""
            _cx = self.vlm.disable_adapter() if (disable_adapter and hasattr(self.vlm, "disable_adapter")) \
                else contextlib.nullcontext()
            keep = {k: v for k, v in model_inputs.items()
                    if k not in ("gt_trajectory", "gt_action", "has_cot",
                                 "input_features", "token")}
            with _cx:
                out = self.vlm(**keep, output_hidden_states=True)
            h = out.hidden_states[-1]                                 # (B, S, 2048)
            ctx, ctx_mask = context_sequence_hidden(
                h, keep["input_ids"], attention_mask=keep.get("attention_mask"),
                action_start_id=self.action_start_id, max_len=self.fm_decoder.config.ctx_max_len)
            anchor = answer_position_hidden(h, keep["input_ids"],
                                            attention_mask=keep.get("attention_mask"),
                                            action_start_id=self.action_start_id)
            return ctx, ctx_mask, anchor

        def _ctx_and_style(self, model_inputs):
            # styled forward -> styled anchor (for the delta) + styled content ctx.
            ctx_s, mask_s, anchor_styled = self._forward_ctx(model_inputs, disable_adapter=False)
            if self.style_off:
                return ctx_s, mask_s, torch.zeros_like(anchor_styled)  # style axis OFF (s=0)
            try:
                ctx_b, mask_b, anchor_base = self._forward_ctx(model_inputs, disable_adapter=True)
            except Exception:
                ctx_b, mask_b, anchor_base = ctx_s, mask_s, anchor_styled
            s = style_delta(anchor_styled, anchor_base, alpha=self.fm_decoder.style_alpha)
            # content_source (ADDITIVE): "styled" (DEFAULT) -> LEGACY (content from LoRA-ON
            # forward), byte-identical. "base" -> content from the LoRA-OFF forward so the body
            # stays LoRA-free / neutral. The style delta s is identical either way.
            cs = getattr(self.fm_decoder, "content_source", "styled")
            if cs == "interp":
                # LATENT CONTENT INTERPOLATION: c(alpha)=ctx_base + alpha*(ctx_styled-ctx_base),
                # AdaLN style s=0. ctx_b/ctx_s share input_ids+mask -> identical shape & mask, so
                # the interpolation is elementwise. alpha=0 -> ctx_b, s=0 == content_source="base".
                alpha = float(self.fm_decoder.style_alpha)
                ctx_i = ctx_b + alpha * (ctx_s - ctx_b)
                return ctx_i, mask_b, torch.zeros_like(anchor_base)
            if cs == "base":
                return ctx_b, mask_b, s
            return ctx_s, mask_s, s

        # -- inference: override predict -----------------------------------
        @torch.no_grad()
        def predict(self, input_features, scorer=None):
            """FM-decode replacement for the codebook decode. Returns (poses(10,3), cot='')."""
            inputs = self.get_prompt(input_features)
            model_inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                            for k, v in inputs.items()}
            ctx, ctx_mask, s = self._ctx_and_style(model_inputs)
            poses = self.fm_decoder.decode(ctx, s, ctx_mask=ctx_mask, scorer=scorer)  # (B,10,3)
            return poses[0].float().cpu(), ""

        # -- training: auxiliary FM loss -----------------------------------
        def fm_training_loss(self, batch, style_dropout_prob=None) -> torch.Tensor:
            """Compute FMHead flow-matching loss from a training batch (LoRA-on hidden)."""
            ctx, ctx_mask, s = self._ctx_and_style({k: v for k, v in batch.items()})
            return self.fm_decoder.training_loss(batch["gt_trajectory"], ctx, s,
                                                 ctx_mask=ctx_mask, style_dropout_prob=style_dropout_prob)

    return AutoVLA_FMHead


__all__ = [
    "AUTOVLA_HIDDEN_SIZE", "AUTOVLA_NUM_POSES", "AUTOVLA_INTERVAL_S",
    "AUTOVLA_ACTION_START_ID", "AUTOVLA_VLM_DTYPE",
    "autovla_fmhead_config", "action_region_hidden", "answer_position_hidden",
    "context_sequence_hidden", "style_delta",
    "FMHeadDecoder", "build_autovla_fmhead_subclass",
]
