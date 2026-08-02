"""fm_head.py -- Conditional Rectified Flow-Matching Trajectory Decoder Head ("FMHead").

A STANDALONE, dependency-light module that generates a future ego trajectory
`tau` of shape (B, T, 2) by integrating a learned velocity field, conditioned on:

  * a scene/content condition  c  (e.g. an AutoVLA answer-position hidden state),
  * a style condition          s  (e.g. an activation-space delta `h_styled - h_base`,
                                    or a small explicit kinematic style vector).

It is designed to later DROP-IN REPLACE AutoVLA's token -> codebook trajectory
decoder without modifying AutoVLA. See README.md for the plug-in contract.

--------------------------------------------------------------------------------
Design lineage (conventions borrowed, code written fresh & minimal)
--------------------------------------------------------------------------------
* DiT block / AdaLN-Zero timestep+condition modulation, sinusoidal `TimestepEmbedder`,
  `modulate(x, shift, scale)`, zero-init of the AdaLN + final projection, temporal
  sin-cos position embedding, xavier init:
      TrajDiff  (navsim/agents/trajdiff/modules/DiT.py -- TrajDiTBlockV2/V3, FinalLayer)
* Rectified flow-matching parameterization  x_t = (1-t)*eps + t*tau ,  u = tau - eps ,
  Euler ODE integration  x <- x + v * dt , and classifier-free guidance via
  condition-dropout at train + convex/linear combination of cond/uncond velocities
  at sampling:
      GoalFlow  (navsim/agents/goalflow/goalflow_model_traj.py -- get_train_tuple,
                 denoise(..., force_dropout/navi_dropout), flow sampling loop)
* Anchored / multi-sample multimodality (draw N noises, return all N; later PDM
  select-by-score) conceptually follows:
      DiffusionDrive (navsim/agents/diffusiondrive/*)

The three reference repos live under
`/root/workspace/closed_loop/navsim_candidates_survey/repos/` and were NOT modified.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Config
# =============================================================================
@dataclass
class FMHeadConfig:
    """Configuration for :class:`FMHead`.

    All dims are configurable so the head can match different AutoVLA hidden
    widths (``d_cond`` / ``d_style`` default to 2048 to match an AutoVLA
    answer-position hidden state).
    """

    # --- trajectory geometry ---
    # Defaults match REAL AutoVLA (config model.trajectory): num_poses=10 @ 0.5 s.
    # traj_dim=2 models the continuous (x_forward, y_left) waypoints; this is exactly
    # AutoVLA's `gt_trajectory[..., :2]` (its differentiable-L2 target). The discrete
    # decoder additionally emits a heading column -> navsim consumes (num_poses, 3);
    # use `FMHead.append_heading(xy)` at the predict() boundary to produce that column.
    horizon: int = 10           # T waypoints (AutoVLA num_poses)
    traj_dim: int = 2           # (x, y) per waypoint

    # --- conditioning dims ---
    # Defaults match Qwen2.5-VL-3B-Instruct hidden_size = 2048 (config.json).
    d_cond: int = 2048          # (legacy) pooled scene/content width (compat; unused when use_context)
    d_style: int = 2048         # style condition width (h_styled - h_base, or small explicit)

    # --- v2: multi-token context conditioning (the H1a fix) ---
    # Instead of a single pooled `c`, the head cross-attends to a SEQUENCE of VLM hidden
    # states over the prompt span (up to the generation-start anchor). d_ctx = VLM hidden.
    use_context: bool = True    # v2 default: cross-attn over the context sequence
    d_ctx: int = 2048           # VLM hidden width of the context tokens
    ctx_max_len: int = 64       # cap on # context tokens fed to cross-attn (last-K of prompt)

    # --- style adapter (opt-in; OFF by default -> plain v2 head, byte-identical) ---
    use_style_adapter: bool = False       # True -> build StyleFMHead (frozen base + adapter)
    style_adapter_hidden: int = 256       # width of the style adapter trunk
    style_adapter_mode: str = "adaln"     # "adaln" (Option A, wired) | "cross_attn" (Option B hook)

    # --- z-encoder (opt-in; OFF by default -> byte-identical style path) ---
    # A learned MLP that purifies/aligns/amplifies the raw LoRA task-vector delta BEFORE it
    # enters the existing AdaLN style path (s_proj). z = enc(alpha*delta). When OFF (default)
    # the style path is EXACTLY the pre-existing `s_proj(s)` -> every existing ckpt/decode is
    # byte-identical. See :class:`ZEncoder`. Independent of the (older) style adapter; intended
    # for the plain FMHead style path (use_style_adapter should stay False when this is True).
    use_z_encoder: bool = False
    z_encoder_dims: Tuple[int, int] = (512, 256)   # hidden widths d_style -> h1 -> h2 -> d_style
    # z-encoder STABILITY fixes (ADDITIVE; default False -> the ORIGINAL z-encoder path,
    # byte-identical). When True the z-encoder (i) L2-normalizes the raw delta to a unit
    # DIRECTION before the MLP (scene-invariant input magnitude -- tames a large ||delta||),
    # (ii) gates its output by a ZERO-init scalar so the injected AdaLN modulation is EXACTLY
    # 0 at init (forward == the clean base, then grows smoothly -- restores the old adapter's
    # zero-gate warm-up the plain z-encoder dropped), and (iii) applies alpha as an EXTERNAL
    # gain on the encoder OUTPUT (outside the norm) so alpha stays a strictly-monotonic linear
    # dose knob and the modulation magnitude is bounded regardless of ||delta||.
    z_norm_delta: bool = False

    # --- transformer backbone width ---
    hidden_size: int = 256      # model width the DiT operates at
    depth: int = 4              # number of DiT blocks
    num_heads: int = 8
    mlp_ratio: float = 4.0

    # --- conditioning MLP ---
    cond_hidden: int = 512      # hidden width of the c / s projection MLPs

    # --- classifier-free guidance (on STYLE) ---
    style_dropout_prob: float = 0.2   # prob. of dropping style during training

    # --- misc ---
    learn_temporal_pos: bool = False  # False -> frozen sin-cos temporal pos-embed

    # --- fm3 feasibility constraint (ADDITIVE; mode="none" -> byte-identical to fm2) ---
    # none      : the original fm2 free-xy rectified flow (default, unchanged).
    # soft      : variant A ("fm3-lite") -- xy flow + a smoothness (jerk/curvature) penalty.
    # kinematic : variant B ("fm3-kin")  -- flow in (a_long, yaw_rate) control space and
    #             integrate a unicycle -> feasible-by-construction xy + native heading.
    constraint_mode: str = "none"     # {none | soft | kinematic}
    # soft / optional-kinematic smoothness regularizer (option c). 0 -> penalty OFF.
    jerk_weight: float = 0.0          # lambda on jerk (3rd diff of xy / 1st diff of accel)
    curv_weight: float = 0.0          # lambda on curvature/yaw-rate smoothness
    smooth_t_min: float = 0.0         # only apply the penalty for flow-time t > this
    # kinematic unicycle caps (feasible-by-construction bounds; used at decode integration)
    accel_max: float = 5.0            # m/s^2  |a_long| cap
    yawrate_max: float = 0.8          # rad/s  |yaw-rate| cap
    v_max: float = 25.0               # m/s    speed clamp
    kin_dt: float = 0.5               # s      integration step (== interval_length)

    def __post_init__(self):
        assert self.hidden_size % self.num_heads == 0, \
            "hidden_size must be divisible by num_heads"
        assert self.constraint_mode in ("none", "soft", "kinematic"), self.constraint_mode


# =============================================================================
# Building blocks
# =============================================================================
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation. ``x`` is (B, T, C); ``shift``/``scale`` are (B, C)."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal embedding of a (possibly fractional) scalar timestep.

    ``t`` is a 1-D tensor of shape (B,) with values in [0, 1]. Returns (B, dim).
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepEmbedder(nn.Module):
    """Embeds a scalar flow-time t in [0,1] into a hidden-size vector."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # timestep_embedding computes the sinusoid in float32 internally; cast it to
        # the MLP's weight dtype so a bf16 head (FSDP MixedPrecision) doesn't hit a
        # Float-vs-BFloat16 matmul mismatch.
        emb = timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class CondProjector(nn.Module):
    """Projects a raw condition vector (D_in) to model width via a small MLP."""

    def __init__(self, d_in: int, hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention over the T trajectory tokens (batch_first)."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)  # (B, H, N, hd)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class CrossAttention(nn.Module):
    """Multi-head cross-attention: trajectory tokens (query) attend to context tokens.

    ``ctx`` is (B, T_ctx, C); ``ctx_mask`` is (B, T_ctx) with 1 = valid (attend), 0 = pad.
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor,
                ctx_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        M = ctx.shape[1]
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(ctx).reshape(B, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        attn_mask = None
        if ctx_mask is not None:
            # (B, 1, 1, M) bool: True = attend (SDPA masks the False positions with -inf)
            attn_mask = ctx_mask.bool().view(B, 1, 1, M).expand(B, self.num_heads, N, M)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # (B, H, N, hd)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class DiTBlock(nn.Module):
    """DiT block with AdaLN-Zero conditioning + cross-attention to the context sequence.

    ``cvec`` (B, C) = timestep_emb + style_emb drives 9 modulation params (shift/scale/gate
    for self-attn, cross-attn, and MLP). Content is injected via cross-attention to the VLM
    context tokens (v2), not baked into ``cvec`` -- so the AdaLN path carries the style axis
    and the cross-attn carries the (multi-token) scene content.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0,
                 use_context: bool = True):
        super().__init__()
        self.use_context = use_context
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(hidden_size, num_heads)
        if use_context:
            self.norm_ca = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.cross_attn = CrossAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, cvec: torch.Tensor,
                ctx: Optional[torch.Tensor] = None,
                ctx_mask: Optional[torch.Tensor] = None,
                style_mod: Optional[torch.Tensor] = None) -> torch.Tensor:
        # style_mod is an OPT-IN additive delta (B, 9*C) to the AdaLN params from the
        # zero-gated style adapter. When None (default / style-off) the modulation is
        # computed EXACTLY as before -> the s=0 / no-adapter forward is byte-identical.
        mod = self.adaLN_modulation(cvec)
        if style_mod is not None:
            mod = mod + style_mod
        (shift_msa, scale_msa, gate_msa, shift_ca, scale_ca, gate_ca,
         shift_mlp, scale_mlp, gate_mlp) = mod.chunk(9, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        if self.use_context and ctx is not None:
            x = x + gate_ca.unsqueeze(1) * self.cross_attn(
                modulate(self.norm_ca(x), shift_ca, scale_ca), ctx, ctx_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """AdaLN final layer mapping hidden tokens back to trajectory space (traj_dim)."""

    def __init__(self, hidden_size: int, out_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, cvec: torch.Tensor,
                style_mod: Optional[torch.Tensor] = None) -> torch.Tensor:
        mod = self.adaLN_modulation(cvec)
        if style_mod is not None:              # opt-in style delta; None -> byte-identical
            mod = mod + style_mod
        shift, scale = mod.chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


def _sincos_1d(embed_dim: int, length: int) -> torch.Tensor:
    """Return frozen (length, embed_dim) 1-D sin-cos positional embedding."""
    assert embed_dim % 2 == 0
    pos = torch.arange(length, dtype=torch.float32)
    omega = torch.arange(embed_dim // 2, dtype=torch.float32) / (embed_dim / 2.0)
    omega = 1.0 / (10000 ** omega)
    out = pos[:, None] * omega[None, :]
    return torch.cat([out.sin(), out.cos()], dim=1)


# =============================================================================
# Z-ENCODER (opt-in) -- learned purify/align/amplify of the raw style delta
# =============================================================================
class ZEncoder(nn.Module):
    """Learned encoder for the style task-vector: z = enc(alpha * delta).

    Keeps LoRA as the style SOURCE (the raw activation-space delta s = alpha*(h_styled -
    h_base) is preserved) and inserts a small learned MLP that purifies / aligns / amplifies
    it before it feeds the FMHead's existing AdaLN style path (``s_proj``).

    Design (mirrors the StyleAdapter rationale):
      * bias-free Linear + SiLU, NO LayerNorm. SiLU(0)=0 and every layer is bias-free, so
        ``enc(0) == 0`` EXACTLY -> alpha=0 / s=0 leaves the style path at its zero point (the
        neutral base), and the mapping stays magnitude-sensitive so ``alpha`` gives a
        MONOTONIC dose-response (an input LayerNorm would be scale-invariant and destroy the
        alpha knob -- the exact bug the old adapter's fixed ``in_scale`` avoided).
      * a fixed input scale ``in_scale = 1/sqrt(d_style)`` (NOT a norm) keeps the large VLM
        hidden delta (||s|| ~ O(20+)) in a sane pre-activation range while preserving linearity
        in ||s||.
      * maps d_style -> h1 -> h2 -> d_out; d_out defaults to d_style so ``z`` drops straight
        into the existing ``s_proj`` with NO shape change (the loaded head's s_proj is reused).
    """

    def __init__(self, d_style: int, dims: Tuple[int, int] = (512, 256),
                 d_out: Optional[int] = None, norm_delta: bool = False):
        super().__init__()
        d_out = d_out or d_style
        h1, h2 = dims
        self.norm_delta = bool(norm_delta)
        self.in_scale = 1.0 / (d_style ** 0.5)   # used ONLY in the legacy (norm_delta=False) path
        self.net = nn.Sequential(
            nn.Linear(d_style, h1, bias=False), nn.SiLU(),
            nn.Linear(h1, h2, bias=False), nn.SiLU(),
            nn.Linear(h2, d_out, bias=False),
        )
        # STABILITY fix (2) -- the key fix, only in the norm_delta path (legacy stays identical):
        # ZERO-INIT the z-encoder's OWN FINAL Linear (AdaLN-zero / ControlNet "zero-conv" trick;
        # done in FMHead._init_weights AFTER the generic xavier apply). At init the final layer
        # emits EXACTLY 0 -> injected modulation is 0 -> forward == clean base (GATE2). Unlike a
        # separate zero scalar gate (which zeros the gradient to the WHOLE encoder -- the classic
        # double-zero dead-gradient trap), the final Linear's gradient is proportional to its
        # NONZERO input (h2 from xavier-init earlier layers), so it TRAINS from step 0 and the
        # earlier layers come alive as soon as its weights move. No separate scalar gate.

    def forward(self, s: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
        x = s.to(self.net[0].weight.dtype)
        if self.norm_delta:
            # fix (1): L2-normalize the delta to a UNIT DIRECTION (scene-invariant magnitude).
            # eps guard keeps normalize(0)==0 exactly so s=0 / alpha=0 stays the neutral base.
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
            # fix (1): alpha as EXTERNAL gain OUTSIDE the norm (strictly-monotonic linear knob);
            # fix (2): net's zero-init final Linear makes this exactly 0 at init.
            return float(gain) * self.net(x)
        # legacy path (byte-identical to the original z-encoder): fixed in_scale, no gain.
        return self.net(x * self.in_scale)


# =============================================================================
# The FMHead
# =============================================================================
class FMHead(nn.Module):
    """Conditional rectified flow-matching trajectory decoder head.

    Interface contract
    -------------------
    velocity forward:
        v = self.forward(x_t, t, c, s)           -> (B, T, traj_dim)
    training:
        loss = self.flow_matching_loss(tau, c, s)  (scalar)
    sampling:
        trajs = self.sample(c, s, num_samples=N, num_steps=K, cfg_weight=w)
                                                 -> (B, N, T, traj_dim)
    selection (placeholder for later PDM scoring):
        best = FMHead.select_by_score(trajs, scorer)  -> (B, T, traj_dim)

    Shapes
    ------
    c : (B, d_cond)      scene/content condition (e.g. h_base)
    s : (B, d_style)     style condition (e.g. h_styled - h_base, or explicit kinematic)
    tau/x_t : (B, T, traj_dim)
    t : (B,)  flow time in [0, 1]  (0 = pure noise eps, 1 = data tau)
    """

    def __init__(self, config: Optional[FMHeadConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = FMHeadConfig(**kwargs)
        self.config = config
        C = config.hidden_size

        # trajectory token embedder: (x, y) -> hidden
        self.x_embedder = nn.Linear(config.traj_dim, C)

        # temporal position embedding over the T waypoint tokens
        self.temporal_pos = nn.Parameter(
            torch.zeros(1, config.horizon, C), requires_grad=config.learn_temporal_pos
        )

        # flow-time embedder + style projector (content now via cross-attn on context)
        self.t_embedder = TimestepEmbedder(C)
        self.s_proj = CondProjector(config.d_style, config.cond_hidden, C)

        # opt-in learned z-encoder that purifies the raw style delta before s_proj.
        # OFF by default (None) -> the style path is byte-identical to the pre-existing head.
        self.z_encoder = (ZEncoder(config.d_style, dims=tuple(config.z_encoder_dims),
                                   d_out=config.d_style,
                                   norm_delta=bool(getattr(config, "z_norm_delta", False)))
                          if getattr(config, "use_z_encoder", False) else None)
        # EXTERNAL style gain (alpha) applied to the z-encoder output when z_norm_delta is on
        # (fix (1): alpha is a linear knob OUTSIDE the input norm). Plain float, NOT a param
        # (never trained / saved). Set per-call by FMHeadDecoder from its style_alpha; default
        # 1.0 (train-time full dose). Ignored by the legacy z-path and the no-z-encoder path.
        self.style_gain: float = 1.0

        # v2: project the VLM context-token sequence (d_ctx) to head width for cross-attn
        if config.use_context:
            self.ctx_proj = nn.Sequential(
                nn.Linear(config.d_ctx, C), nn.LayerNorm(C))
        # legacy pooled-c path (only built/used if NOT use_context)
        if not config.use_context:
            self.c_proj = CondProjector(config.d_cond, config.cond_hidden, C)

        # learnable "null" style embedding used for classifier-free guidance
        # (substituted for the style embedding when style is dropped).
        self.null_style = nn.Parameter(torch.zeros(1, C))

        self.blocks = nn.ModuleList(
            [DiTBlock(C, config.num_heads, config.mlp_ratio, use_context=config.use_context)
             for _ in range(config.depth)]
        )
        self.final_layer = FinalLayer(C, config.traj_dim)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_basic)

        if not self.config.learn_temporal_pos:
            self.temporal_pos.data.copy_(
                _sincos_1d(self.config.hidden_size, self.config.horizon).unsqueeze(0)
            )
        nn.init.normal_(self.null_style, std=0.02)

        # zero-init AdaLN modulations & final projection (AdaLN-Zero) so the head
        # starts as an identity-ish velocity field -> stable early training.
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # z-encoder norm_delta path: ZERO-init its FINAL Linear (ControlNet "zero-conv") so the
        # injected style modulation is EXACTLY 0 at init (forward == clean base) WHILE the final
        # layer still receives a live gradient (proportional to its nonzero input) -> it trains
        # from step 0, unlike a separate zero scalar gate which kills the whole encoder's grad.
        # The earlier z-encoder layers keep their xavier init (nonzero) so they come alive as
        # soon as the final weights move. Legacy z-path (norm_delta=False) is left untouched.
        if self.z_encoder is not None and getattr(self.z_encoder, "norm_delta", False):
            nn.init.constant_(self.z_encoder.net[-1].weight, 0)   # bias-free -> weight only

    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self.x_embedder.weight.device

    def _param_dtype(self) -> torch.dtype:
        """The head's (uniform) parameter dtype. Used to keep every forward
        dtype-self-consistent regardless of fp32 vs bf16 (FSDP MixedPrecision)."""
        return self.x_embedder.weight.dtype

    # ------------------------------------------------------------------
    def _cond_vector(
        self,
        t: torch.Tensor,
        s: torch.Tensor,
        drop_style: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build the (B, C) AdaLN conditioning vector = t_emb + s_emb (STYLE axis).

        v2: scene content is injected via cross-attention to the context sequence,
        NOT into this vector. The AdaLN path now carries only the flow-time and the
        style axis (so style-flip + CFG still work). ``drop_style`` (B,) bool replaces
        the style embedding with the learnable null-style (CFG dropout / uncond branch).

        Dtype policy: cast to the head's own param dtype so it works fp32 or bf16.
        """
        dt = self._param_dtype()
        t_emb = self.t_embedder(t.to(dt))
        s_in = s.to(dt)
        # opt-in: purify the raw style delta with the learned z-encoder before s_proj.
        # z_encoder(0)==0 exactly, so s=0 (alpha=0 / CFG-null input) is unchanged. When the
        # z-encoder is OFF (None, default) this line is skipped -> byte-identical style path.
        if self.z_encoder is not None:
            if getattr(self.z_encoder, "norm_delta", False):
                # norm_delta path: alpha (self.style_gain) is the EXTERNAL output gain and the
                # zero-conv final Linear emits 0 at init -> s_in==0 -> s_proj(0) == the base's
                # neutral AdaLN point, so alpha=0/z=null is bit-exact the clean base.
                s_in = self.z_encoder(s_in, gain=float(getattr(self, "style_gain", 1.0)))
            else:
                s_in = self.z_encoder(s_in)
        s_emb = self.s_proj(s_in)
        if drop_style is not None:
            null = self.null_style.to(dt).expand_as(s_emb)
            s_emb = torch.where(drop_style.view(-1, 1), null, s_emb)
        return t_emb + s_emb

    def _project_ctx(self, ctx: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Project the VLM context sequence (B, T_ctx, d_ctx) -> (B, T_ctx, C).
        Accepts a pooled (B, d_ctx) vector too (treated as a length-1 context)."""
        if ctx is None or not self.config.use_context:
            return None
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(1)
        return self.ctx_proj(ctx.to(self._param_dtype()))

    # ------------------------------------------------------------------
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        ctx: torch.Tensor,
        s: torch.Tensor,
        ctx_mask: Optional[torch.Tensor] = None,
        drop_style: Optional[torch.Tensor] = None,
        style_mods: Optional[List[torch.Tensor]] = None,
        final_style_mod: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict the velocity field v_theta(x_t, t, ctx, s) -> (B, T, traj_dim).

        ``ctx`` is the VLM context sequence (B, T_ctx, d_ctx) [or a pooled (B, d_ctx)];
        content enters via cross-attention. ``s`` is the style axis (AdaLN + CFG).

        ``style_mods`` (list of per-block (B, 9*C) deltas) and ``final_style_mod``
        ((B, 2*C)) are the OPT-IN additive AdaLN deltas from the style adapter. Both
        default to None -> every block/final call is byte-identical to the pre-adapter
        v2 head (the 0.91 s=0 forward is unchanged).
        """
        if t.dim() == 0:
            t = t.expand(x_t.shape[0])
        dt = self._param_dtype()
        cvec = self._cond_vector(t, s, drop_style=drop_style)
        ctx_kv = self._project_ctx(ctx)
        h = self.x_embedder(x_t.to(dt)) + self.temporal_pos.to(dt)
        for i, block in enumerate(self.blocks):
            h = block(h, cvec, ctx_kv, ctx_mask,
                      style_mod=(style_mods[i] if style_mods is not None else None))
        return self.final_layer(h, cvec, style_mod=final_style_mod)

    # ------------------------------------------------------------------
    def flow_matching_loss(
        self,
        tau: torch.Tensor,
        ctx: torch.Tensor,
        s: torch.Tensor,
        ctx_mask: Optional[torch.Tensor] = None,
        style_dropout_prob: Optional[float] = None,
        return_parts: bool = False,
        smooth_space: str = "xy",
        traj_mean: Optional[torch.Tensor] = None,
        traj_std: Optional[torch.Tensor] = None,
    ):
        """Rectified flow-matching MSE loss.

        Sample t~U(0,1), eps~N(0,I); build x_t = (1-t)*eps + t*tau; target
        velocity u = tau - eps; predict v = v_theta(x_t, t, ctx, s); return
        MSE(v, u). ``ctx`` = VLM context sequence (cross-attn content). With prob
        ``style_dropout_prob`` the style is dropped (null-style) for CFG.

        fm3 (ADDITIVE): if ``config.jerk_weight`` or ``config.curv_weight`` > 0 a
        smoothness penalty is added on the one-step clean-trajectory estimate
        ``x_hat1 = x_t + (1-t)*v`` (denormalized to physical units via ``traj_mean``/
        ``traj_std`` when provided). ``smooth_space`` selects the physical meaning of
        ``x_hat1``: "xy" -> jerk = 3rd diff of position, curv = 2nd diff of heading;
        "control" -> jerk = 1st diff of a_long, curv = 1st diff of yaw-rate. With both
        weights 0 (the fm2 default) NONE of this runs -> byte-identical to before.
        """
        if style_dropout_prob is None:
            style_dropout_prob = self.config.style_dropout_prob
        B = tau.shape[0]
        device = tau.device
        dt = self._param_dtype()
        tau = tau.to(dt)                       # match head dtype (bf16 under FSDP, fp32 otherwise)
        t = torch.rand(B, device=device, dtype=dt)
        eps = torch.randn_like(tau)            # inherits tau's dtype -> dt
        t_b = t.view(B, 1, 1)
        x_t = (1.0 - t_b) * eps + t_b * tau
        u = tau - eps
        drop_style = torch.rand(B, device=device) < style_dropout_prob
        v = self.forward(x_t, t, ctx, s, ctx_mask=ctx_mask, drop_style=drop_style)
        fm_loss = F.mse_loss(v.float(), u.float())   # compute the loss in fp32 for stable numerics

        jw, cw = self.config.jerk_weight, self.config.curv_weight
        smooth_loss = x_t.new_zeros(())
        if jw > 0.0 or cw > 0.0:
            # one-step data prediction x_hat1 (rectified-flow: tau = x_t + (1-t)*u).
            # x_hat1 is a NOISY estimate at small t, so we (i) use an L1 (robust) penalty and
            # (ii) t-weight it (contribution ~ t): the clean end (t->1, x_hat1->GT) dominates
            # and the noisy early samples (t < smooth_t_min) are zeroed. This keeps the flow
            # loss stable while still biasing the field toward smooth clean-trajectory outputs.
            x_hat1 = (x_t + (1.0 - t_b) * v).float()
            if traj_mean is not None and traj_std is not None:
                x_hat1 = x_hat1 * traj_std.to(x_hat1) + traj_mean.to(x_hat1)   # -> physical units
            jerk_pen, curv_pen = self._smoothness_penalty(
                x_hat1, smooth_space, dt=self.config.kin_dt)  # (B,), (B,) physical units (L1)
            tf = t.float()
            w = tf * (tf > self.config.smooth_t_min).float()   # t-weight + early-noise floor
            denom = w.sum().clamp_min(1.0)
            smooth_loss = ((jw * jerk_pen + cw * curv_pen) * w).sum() / denom
        loss = fm_loss + smooth_loss

        if return_parts:
            return loss, {"t": t.detach(), "drop_frac": drop_style.float().mean().detach(),
                          "fm_loss": fm_loss.detach(), "smooth_loss": smooth_loss.detach()}
        return loss

    # ------------------------------------------------------------------
    @staticmethod
    def _wrap_angle(a: torch.Tensor) -> torch.Tensor:
        """Wrap radians to (-pi, pi]."""
        return torch.atan2(torch.sin(a), torch.cos(a))

    @classmethod
    def _smoothness_penalty(cls, phys: torch.Tensor, space: str,
                            dt: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-sample (jerk_pen, curv_pen) from a physical (B, T, 2) trajectory estimate.

        Penalties are ROBUST L1 (mean-absolute) in PHYSICAL units (scaled by ``dt``) so the
        config weights are meaningful and outlier-x_hat1 samples can't blow up training:
        jerk in m/s^3, curvature in rad/s^2 (control) or rad/s (xy heading-rate).

        space="xy"      : phys = (x_forward, y_left) metres.
            jerk = d^3(position)/dt^3 (origin (0,0) prepended so pose-0 uses the real forward
            motion); curv = d(heading)/dt (yaw-rate of the segment heading).
        space="control" : phys = (a_long [m/s^2], yaw_rate [rad/s]).
            jerk = d(a_long)/dt (the accel-jerk); curv = d(yaw_rate)/dt.
        Returns two (B,) tensors (mean-absolute, L1).
        """
        B = phys.shape[0]
        if space == "control":
            a = phys[..., 0]                                    # (B, T)
            w = phys[..., 1]
            jerk = (a[:, 1:] - a[:, :-1]) / dt                 # (B, T-1) m/s^3
            curv = (w[:, 1:] - w[:, :-1]) / dt                 # (B, T-1) rad/s^2
            jerk_pen = jerk.abs().mean(dim=1) if jerk.shape[1] > 0 else phys.new_zeros(B)
            curv_pen = curv.abs().mean(dim=1) if curv.shape[1] > 0 else phys.new_zeros(B)
            return jerk_pen, curv_pen
        # xy space
        origin = phys.new_zeros(B, 1, 2)
        P = torch.cat([origin, phys], dim=1)                   # (B, T+1, 2)
        vel = (P[:, 1:] - P[:, :-1]) / dt                      # (B, T, 2) m/s
        acc = (vel[:, 1:] - vel[:, :-1]) / dt                  # (B, T-1, 2) m/s^2
        jerk = (acc[:, 1:] - acc[:, :-1]) / dt                 # (B, T-2, 2) m/s^3
        jerk_pen = jerk.abs().mean(dim=(1, 2)) if jerk.shape[1] > 0 else phys.new_zeros(B)
        psi = torch.atan2(vel[..., 1], vel[..., 0])            # (B, T) segment heading
        dpsi = cls._wrap_angle(psi[:, 1:] - psi[:, :-1]) / dt  # (B, T-1) rad/s
        curv_pen = dpsi.abs().mean(dim=1) if dpsi.shape[1] > 0 else phys.new_zeros(B)
        return jerk_pen, curv_pen

    # ------------------------------------------------------------------
    # Unicycle (variant B "fm3-kin"): bounded control space <-> xy, feasible by construction.
    # ------------------------------------------------------------------
    @classmethod
    def xy_to_controls(cls, xy: torch.Tensor, v0: torch.Tensor, dt: float = 0.5) -> torch.Tensor:
        """Invert an xy trajectory to unicycle controls (a_long, yaw_rate).

        ``xy`` (B, T, 2) metres in the ego frame (origin (0,0), heading 0 at t=0);
        ``v0`` (B,) the initial speed. Returns controls (B, T, 2) = (a_long, yaw_rate).
        This is the EXACT inverse of :meth:`integrate_unicycle` (clamp off) given v0, so
        a zero flow-loss reproduces the GT xy. Stationary segments hold the last heading.
        """
        B, T, _ = xy.shape
        origin = xy.new_zeros(B, 1, 2)
        P = torch.cat([origin, xy], dim=1)                    # (B, T+1, 2)
        d = P[:, 1:] - P[:, :-1]                               # (B, T, 2) per-step displacement
        seg = torch.linalg.norm(d, dim=-1)                    # (B, T)
        v = seg / dt                                          # (B, T) speed during each step
        raw_psi = torch.atan2(d[..., 1], d[..., 0])           # (B, T) (ill-defined when seg~0)
        # forward-fill heading over near-stationary segments (hold last well-defined psi)
        moving = seg > (0.05 * dt)                            # ~ >0.05 m/s
        psi = torch.zeros_like(raw_psi)
        prev = xy.new_zeros(B)
        for tt in range(T):
            cur = torch.where(moving[:, tt], raw_psi[:, tt], prev)
            psi[:, tt] = cur
            prev = cur
        # controls: a_t=(v_t-v_{t-1})/dt (v_{-1}=v0); omega_t=wrap(psi_t-psi_{t-1})/dt (psi_{-1}=0)
        v_prev = torch.cat([v0.view(B, 1), v[:, :-1]], dim=1)  # (B, T)
        a = (v - v_prev) / dt
        psi_prev = torch.cat([xy.new_zeros(B, 1), psi[:, :-1]], dim=1)
        omega = cls._wrap_angle(psi - psi_prev) / dt
        return torch.stack([a, omega], dim=-1)                # (B, T, 2)

    @classmethod
    def integrate_unicycle(cls, controls: torch.Tensor, v0: torch.Tensor, dt: float = 0.5,
                           accel_max: Optional[float] = None, yawrate_max: Optional[float] = None,
                           v_max: Optional[float] = None,
                           clamp: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrate unicycle controls (a_long, yaw_rate) -> (xy, heading).

        ``controls`` (..., T, 2); ``v0`` (...,) initial speed. When ``clamp`` (decode
        default) the controls/speed are hard-capped to the feasibility envelope
        (accel_max, yawrate_max, v_max) -> bounded per-step accel & yaw-rate => bounded
        curvature => acute kinks IMPOSSIBLE (the by-construction guarantee). Returns
        (xy (..., T, 2), heading (..., T)) with heading the NATIVE integrated yaw
        (drops the atan2 append_heading hack).
        """
        a = controls[..., 0]
        omega = controls[..., 1]
        if clamp:
            if accel_max is not None:
                a = a.clamp(-accel_max, accel_max)
            if yawrate_max is not None:
                omega = omega.clamp(-yawrate_max, yawrate_max)
        v = v0.unsqueeze(-1) + torch.cumsum(a, dim=-1) * dt   # (..., T) speed after each step
        if clamp:
            v = v.clamp(0.0, v_max if v_max is not None else float("inf"))
        psi = torch.cumsum(omega, dim=-1) * dt                # (..., T) yaw after each step (psi_{-1}=0)
        dx = v * torch.cos(psi) * dt
        dy = v * torch.sin(psi) * dt
        x = torch.cumsum(dx, dim=-1)
        y = torch.cumsum(dy, dim=-1)
        xy = torch.stack([x, y], dim=-1)                      # (..., T, 2)
        return xy, psi

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        ctx: torch.Tensor,
        s: torch.Tensor,
        ctx_mask: Optional[torch.Tensor] = None,
        num_samples: int = 8,
        num_steps: int = 30,
        cfg_weight: float = 1.0,
        return_trajectory_over_time: bool = False,
    ):
        """Draw ``num_samples`` trajectories per scene by Euler-integrating the ODE.

        Integrate  dx/dt = v_theta(x, t, ctx, s)  from t=0 (x=eps~N(0,I)) to t=1
        over ``num_steps`` uniform Euler steps (default 30; GoalFlow uses 100).

        ``ctx`` = VLM context sequence (B, T_ctx, d_ctx) [or pooled (B, d_ctx)].
        Classifier-free guidance on STYLE:
            v = v(uncond_style) + w * (v(cond_style) - v(uncond_style)).

        Returns
        -------
        trajs : (B, N, T, traj_dim)
        (if return_trajectory_over_time: also a list of (B,N,T,traj_dim) snapshots)
        """
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(1)
        B = ctx.shape[0]
        N = num_samples
        cfg = self.config
        device = self.device
        pdt = self._param_dtype()               # head dtype (bf16 under FSDP, else fp32)

        # expand conditions to (B*N, ...)
        Tc = ctx.shape[1]
        ctx_e = ctx.unsqueeze(1).expand(B, N, Tc, ctx.shape[-1]).reshape(B * N, Tc, ctx.shape[-1])
        mask_e = None
        if ctx_mask is not None:
            mask_e = ctx_mask.unsqueeze(1).expand(B, N, Tc).reshape(B * N, Tc)
        s_e = s.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)

        x = torch.randn(B * N, cfg.horizon, cfg.traj_dim, device=device, dtype=pdt)
        step_dt = 1.0 / num_steps
        use_cfg = abs(cfg_weight - 1.0) > 1e-6
        snapshots: List[torch.Tensor] = []

        for i in range(num_steps):
            t_val = torch.full((B * N,), i * step_dt, device=device, dtype=pdt)
            v_cond = self.forward(x, t_val, ctx_e, s_e, ctx_mask=mask_e)
            if use_cfg:
                drop = torch.ones(B * N, dtype=torch.bool, device=device)
                v_uncond = self.forward(x, t_val, ctx_e, s_e, ctx_mask=mask_e, drop_style=drop)
                v = v_uncond + cfg_weight * (v_cond - v_uncond)
            else:
                v = v_cond
            x = x + v * step_dt
            if return_trajectory_over_time:
                snapshots.append(x.view(B, N, cfg.horizon, cfg.traj_dim).clone())

        trajs = x.view(B, N, cfg.horizon, cfg.traj_dim)
        if return_trajectory_over_time:
            return trajs, snapshots
        return trajs

    # ------------------------------------------------------------------
    @staticmethod
    def select_by_score(
        trajs: torch.Tensor,
        scorer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Placeholder for later PDM-style selection among the N samples.

        Parameters
        ----------
        trajs  : (B, N, T, traj_dim) candidate trajectories.
        scorer : callable mapping (B, N, T, traj_dim) -> (B, N) scores
                 (higher = better). If ``None``, defaults to selecting the
                 sample closest to the per-scene mean endpoint (a cheap,
                 dependency-free medoid heuristic).

        Returns
        -------
        best : (B, T, traj_dim) the argmax-scoring trajectory per scene.
        """
        B, N = trajs.shape[:2]
        if scorer is None:
            endpoints = trajs[:, :, -1, :]                       # (B, N, 2)
            mean_ep = endpoints.mean(dim=1, keepdim=True)        # (B, 1, 2)
            scores = -torch.linalg.norm(endpoints - mean_ep, dim=-1)  # (B, N)
        else:
            scores = scorer(trajs)
        best_idx = scores.argmax(dim=1)                          # (B,)
        return trajs[torch.arange(B, device=trajs.device), best_idx]

    # ------------------------------------------------------------------
    @staticmethod
    def append_heading(xy: torch.Tensor, min_step: float = 0.5) -> torch.Tensor:
        """Append a heading column to an (..., T, 2) xy trajectory -> (..., T, 3).

        Heading = absolute yaw (radians, CCW+, ego frame x-forward/y-left), matching
        the convention AutoVLA's codebook `rollout` emits and navsim's PDM expects.

        ROBUST derivation (fixes the PDMS-collapse bug): a naive per-waypoint
        `atan2(Δy, Δx)` is DEGENERATE when the vehicle is slow/stopped -- the tiny,
        noisy xy deltas make heading swing arbitrarily (observed up to ~±pi, i.e.
        pointing backward), which orients the ego footprint wrong and wrecks
        collision / drivable-area / direction-compliance / progress in PDM even
        though the xy path (and xy-only L2) looks fine. We therefore:
          * anchor the first segment at the current ego origin (0,0) so pose-0's
            heading uses the real forward motion origin->wp0 (not wp0->wp1);
          * treat any segment shorter than ``min_step`` metres (≈ <1 m/s at dt=0.5)
            as undefined and HOLD the last well-defined heading (forward-fill);
          * if the whole trajectory is (near-)stationary, heading = 0 everywhere
            (matches the stored heading of standing-still scenes).
        This yields a smooth, well-defined heading like the codebook path.
        """
        assert xy.shape[-1] == 2, xy.shape
        lead = xy.shape[:-2]
        # origin-anchored deltas: prepend the ego current pose (0,0) as pose -1
        origin = xy.new_zeros(lead + (1, 2))
        xyo = torch.cat([origin, xy], dim=-2)                    # (..., T+1, 2)
        d = xyo[..., 1:, :] - xyo[..., :-1, :]                   # (..., T, 2)
        step = torch.linalg.norm(d, dim=-1)                      # (..., T)
        raw = torch.atan2(d[..., 1], d[..., 0])                  # (..., T)
        valid = step > min_step
        T = xy.shape[-2]
        # forward-fill heading over valid segments; leading invalids -> 0 (forward)
        heading = torch.zeros_like(raw)
        prev = raw.new_zeros(lead)
        have = torch.zeros(lead, dtype=torch.bool, device=xy.device)
        for t in range(T):
            vt = valid[..., t]
            prev = torch.where(vt, raw[..., t], prev)
            have = have | vt
            heading[..., t] = torch.where(vt, raw[..., t], prev)
        return torch.cat([xy, heading.unsqueeze(-1)], dim=-1)     # (..., T, 3)


# =============================================================================
# STYLE ADAPTER (opt-in, additive, zero-gated) -- Option A
# =============================================================================
class StyleAdapter(nn.Module):
    """Maps a persona style vector s_H (d_style) to ADDITIVE, zero-gated deltas on each
    DiT block's 9-way AdaLN params and the final layer's 2-way AdaLN params (Option A).

    Design guarantees:
      * bias-free trunk + heads, SiLU (SiLU(0)=0) and NO LayerNorm => g(0) == 0 EXACTLY
        (a zero style vector -> zero delta, always) AND g is magnitude-sensitive so
        s = alpha*delta gives a MONOTONIC dose-response in alpha (an input LayerNorm would
        be scale-invariant and destroy the alpha knob).
      * only the per-block/final GATES are ZERO-initialized (delta = gate*head(trunk(s)));
        the heads are small-random. So at init the delta is EXACTLY 0 (gate=0) -> the
        wrapped head == the frozen head, BUT the gate still receives gradient (head!=0)
        so the adapter can train (avoids the double-zero dead-gradient of zeroing both).
    A fixed input scale (1/sqrt(d_style)) keeps the large VLM-hidden delta (||s||~O(20+))
    in a sane range without a scale-invariant norm. Separate module: the frozen head never
    sees these params.
    """

    def __init__(self, d_style: int, hidden_size: int, depth: int,
                 style_hidden: int = 256, use_context: bool = True):
        super().__init__()
        self.depth = depth
        self.hidden_size = hidden_size
        self.in_scale = 1.0 / (d_style ** 0.5)          # fixed, magnitude-preserving
        # bias-free trunk: Linear(no bias) -> SiLU  => trunk(0)=0 exactly; homogeneous-ish in ||s||
        self.trunk = nn.Sequential(
            nn.Linear(d_style, style_hidden, bias=False),
            nn.SiLU(),
        )
        self.block_heads = nn.ModuleList(
            [nn.Linear(style_hidden, 9 * hidden_size, bias=False) for _ in range(depth)])
        self.final_head = nn.Linear(style_hidden, 2 * hidden_size, bias=False)
        # per-block + final scalar gates: ZERO-init (delta=0 at init). Heads: small-random so
        # the gate has a non-zero gradient path (escapes the double-zero dead start).
        self.block_gates = nn.Parameter(torch.zeros(depth))
        self.final_gate = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.trunk[0].weight, std=0.02)
        for h in self.block_heads:
            nn.init.normal_(h.weight, std=0.02)
        nn.init.normal_(self.final_head.weight, std=0.02)

    def forward(self, s: torch.Tensor):
        """s: (B, d_style) -> (list[depth] of (B, 9*C), (B, 2*C))."""
        s = s.to(self.trunk[0].weight.dtype)     # VLM delta may be bf16; match adapter dtype
        f = self.trunk(s * self.in_scale)
        mods = [self.block_gates[i] * self.block_heads[i](f) for i in range(self.depth)]
        final_mod = self.final_gate * self.final_head(f)
        return mods, final_mod


class StyleFMHead(FMHead):
    """FMHead + opt-in zero-gated StyleAdapter. Content path is IDENTICAL to FMHead; the
    frozen base always sees ZERO style (matching the s=0 training of the 0.91 head), and
    the persona style enters ONLY through the additive, per-sample-masked adapter deltas.

    s=0 (or dropped / no active sample) -> adapter is skipped entirely -> forward is
    byte-identical to the plain FMHead(s=0). This is the 0.91-preservation guarantee.
    """

    def __init__(self, config: Optional[FMHeadConfig] = None, **kwargs):
        super().__init__(config, **kwargs)
        c = self.config
        if getattr(c, "style_adapter_mode", "adaln") == "cross_attn":
            # OPTION B hook (a second zero-gated cross-attn to a multi-token style delta).
            # Left intentionally unwired; enable + implement in a future revision.
            raise NotImplementedError(
                "style_adapter_mode='cross_attn' (Option B) is a reserved hook; only "
                "'adaln' (Option A) is wired.")
        self.style_adapter = StyleAdapter(
            d_style=c.d_style, hidden_size=c.hidden_size, depth=c.depth,
            style_hidden=getattr(c, "style_adapter_hidden", 256), use_context=c.use_context)

    def forward(self, x_t, t, ctx, s, ctx_mask=None, drop_style=None,
                style_mods=None, final_style_mod=None):
        # active sample = nonzero style AND not CFG-dropped
        active = (s.abs().sum(dim=-1) > 0)
        if drop_style is not None:
            active = active & (~drop_style.view(-1))
        if not bool(active.any()):
            # style-off for the whole batch -> base sees ZERO style, adapter skipped
            # (style_mods=None) -> byte-identical to plain FMHead at s=0.
            return super().forward(x_t, t, ctx, torch.zeros_like(s), ctx_mask=ctx_mask,
                                   drop_style=drop_style)
        mods, fmod = self.style_adapter(s)
        m = active.view(-1, 1).to(mods[0].dtype)         # per-sample zero-mask
        mods = [mm * m for mm in mods]
        fmod = fmod * m
        # base content path ALWAYS sees zero style (as during 0.91 training); style is the
        # additive adapter delta only.
        return super().forward(x_t, t, ctx, torch.zeros_like(s), ctx_mask=ctx_mask,
                               drop_style=drop_style, style_mods=mods, final_style_mod=fmod)


# =============================================================================
# Self-test (shape check) -- run `python fm_head.py`
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = FMHeadConfig(d_style=2048, d_ctx=2048, hidden_size=128, depth=2)
    head = FMHead(cfg)
    B, Tctx = 4, 12
    x_t = torch.randn(B, cfg.horizon, cfg.traj_dim)
    t = torch.rand(B)
    ctx = torch.randn(B, Tctx, cfg.d_ctx)                  # multi-token VLM context
    ctx_mask = torch.ones(B, Tctx)
    s = torch.randn(B, cfg.d_style)
    v = head(x_t, t, ctx, s, ctx_mask=ctx_mask)
    assert v.shape == (B, cfg.horizon, cfg.traj_dim), v.shape
    loss = head.flow_matching_loss(torch.randn(B, cfg.horizon, cfg.traj_dim), ctx, s, ctx_mask=ctx_mask)
    trajs = head.sample(ctx, s, ctx_mask=ctx_mask, num_samples=6, num_steps=5, cfg_weight=2.0)
    assert trajs.shape == (B, 6, cfg.horizon, cfg.traj_dim), trajs.shape
    best = FMHead.select_by_score(trajs)
    assert best.shape == (B, cfg.horizon, cfg.traj_dim), best.shape
    # pooled (B, d_ctx) context should also work (length-1)
    v2 = head(x_t, t, torch.randn(B, cfg.d_ctx), s)
    assert v2.shape == (B, cfg.horizon, cfg.traj_dim)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[fm_head v2 self-test] OK  v={tuple(v.shape)} loss={loss.item():.4f} "
          f"trajs={tuple(trajs.shape)} best={tuple(best.shape)} params={n_params/1e6:.2f}M")

    # z-encoder opt-in: (i) enc(0)==0 exactly so s=0 leaves the style path unchanged (byte-
    # identical to bypassing the z-encoder); (ii) s!=0 changes the velocity (axis alive).
    head_z = FMHead(FMHeadConfig(d_style=2048, d_ctx=2048, hidden_size=128, depth=2,
                                 use_z_encoder=True))
    head_z.eval()
    s0 = torch.zeros(B, cfg.d_style)
    assert head_z.z_encoder(s0).abs().max().item() == 0.0, "enc(0) must be exactly 0"
    with torch.no_grad():
        z_s0 = head_z(x_t, t, ctx, s0, ctx_mask=ctx_mask)          # through z-encoder
        saved = head_z.z_encoder; head_z.z_encoder = None
        plain_s0 = head_z(x_t, t, ctx, s0, ctx_mask=ctx_mask)      # z-encoder bypassed
        head_z.z_encoder = saved
        # NOTE: the head is AdaLN-Zero at init (zero-init modulation + final layer) so the full
        # forward is ~0 for ANY style until trained; the meaningful at-init check is that the
        # style signal actually REACHES the AdaLN conditioning vector (the axis comes alive
        # after training -- exercised in the GPU smoke). So compare _cond_vector, not forward.
        sd_ = torch.randn(B, cfg.d_style) * 5.0
        cvec_s0 = head_z._cond_vector(t, s0)
        cvec_sd = head_z._cond_vector(t, sd_)
    assert torch.allclose(z_s0, plain_s0, atol=1e-6), (z_s0 - plain_s0).abs().max().item()
    assert (cvec_s0 - cvec_sd).abs().mean().item() > 1e-4, "z-encoder style axis dead (cvec)"
    nz = sum(p.numel() for p in head_z.z_encoder.parameters())
    print(f"[fm_head z-enc self-test] OK  enc(0)==0 & s=0 bypass-identical "
          f"(max|Δ|={(z_s0-plain_s0).abs().max().item():.1e})  "
          f"cvec axis-alive |Δ(s0,sδ)|={(cvec_s0-cvec_sd).abs().mean().item():.4f}  "
          f"z_params={nz/1e6:.2f}M")
