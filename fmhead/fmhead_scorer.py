"""fmhead_scorer.py -- learned, DEPLOYABLE trajectory scorer for FMHead candidates.

Distills the NAVSIM PDM/EPDMS simulator (the privileged "teacher" run offline via
mode=pdm/epdms on the FMHead's OWN sampled candidates) into a small network that predicts
per-metric sub-scores from INFERENCE-AVAILABLE features only (the VLA context sequence +
the candidate poses). At inference it REPLACES the simulator: no metric cache, no
ground-truth future, one batched forward -> legal and ~1-2 orders faster than mode=pdm.

Architecture faithfully mirrors Hydra-MDP's HydraTrajHead (GTRS repo,
gtrs/navsim/agents/gtrs_dense/hydra_model.py), adapted for CONTINUOUS trajectories
(no fixed vocabulary -- we score the FMHead's actual N samples):
  pose_embed(T*3) -> MemoryEffTransformer self-attn over the N candidates (Hydra's exact
  block, vendored in hydra_attn.py) -> nn.TransformerDecoder cross-attn to the VLA context
  tokens (ctx, with ctx_mask) -> (+ optional ego status) -> per-metric MLP heads.
Selection = log-linear aggregation (argmax over candidates), structured like the NAVSIM
PDMS/EPDMS score so argmax == max predicted PDMS/EPDMS.

Supervision (see label_candidates_pdm.py): pdm_score() returns PDMResults with
{no_at_fault_collisions, drivable_area_compliance, ego_progress,
time_to_collision_within_bound, comfort, driving_direction_compliance}. lane_keeping /
traffic_light_compliance are v2-EPDMS only. Loss = per-metric BCE-with-logits + optional
imitation cross-entropy (Hydra weights) -- knowledge distillation of the simulator.

Standalone (torch + hydra_attn only) so it inserts into the FMHead agent/trainer without
pulling navsim/gtrs internals.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hydra_attn import MemoryEffTransformer


# v1 PDM sub-metrics -- EXACTLY the fields pdm_score() -> PDMResults exposes.
V1_METRICS: List[str] = [
    "no_at_fault_collisions",          # NC   (multiplicative)
    "drivable_area_compliance",        # DAC  (multiplicative)
    "driving_direction_compliance",    # DDC  (multiplicative)
    "ego_progress",                    # EP   (weighted)
    "time_to_collision_within_bound",  # TTC  (weighted)
    "comfort",                         # C    (weighted)
]
# v2 EPDMS-only metrics (need a navtrain v2 metric cache to label).
V2_EXTRA_METRICS: List[str] = ["lane_keeping", "traffic_light_compliance"]

# Log-linear selection weights structured like the NAVSIM score:
#   multiplicative metrics contribute log(m); the weighted group is log(sum w_i * m_i).
# v1 (== PDMS): NC*DAC*DDC * (5*EP + 5*TTC + 2*C).  argmax == max predicted PDMS.
V1_WEIGHTS: Dict[str, float] = {
    "no_at_fault_collisions": 1.0,
    "drivable_area_compliance": 1.0,
    "driving_direction_compliance": 1.0,
    "_group_weight": 1.0,
    "_group": {"ego_progress": 5.0, "time_to_collision_within_bound": 5.0, "comfort": 2.0},
}
# v2 (== EPDMS-style): add TLC as multiplicative and LK into the weighted group.
V2_WEIGHTS: Dict[str, float] = {
    "no_at_fault_collisions": 1.0,
    "drivable_area_compliance": 1.0,
    "driving_direction_compliance": 1.0,
    "traffic_light_compliance": 1.0,
    "_group_weight": 1.0,
    "_group": {
        "ego_progress": 5.0, "time_to_collision_within_bound": 5.0,
        "lane_keeping": 2.0, "comfort": 2.0,
    },
}


class FMHeadScorer(nn.Module):
    """Learned scorer over N continuous FMHead trajectory candidates.

    forward inputs:
      env_tokens : (B, N_kv, env_in_dim)  VLA context sequence `ctx` (e.g. 2048-d).
      trajectories: (B, N, T, pose_dim)   continuous candidates in metres, (x,y,heading).
      env_mask   : (B, N_kv) bool or None  True=valid ctx token (== ctx_mask); padding ignored.
      ego_status : (B, ego_dim) or None    optional explicit ego state; added if given.
    forward outputs (dict):
      {metric: (B, N)} raw logits, 'scores': (B, N), 'selected_index': (B,),
      'trajectory': (B, T, pose_dim).
    """

    def __init__(
        self,
        env_in_dim: int = 2048,        # VLA (Qwen2.5-VL-3B) hidden size
        d_model: int = 256,
        d_ffn: int = 1024,
        nhead: int = 8,
        n_decoder_layers: int = 3,     # cross-attn TransformerDecoder layers (Hydra: 3)
        use_self_attn: bool = True,    # MemoryEffTransformer self-attn over candidates
        self_attn_memory_efficient: bool = False,  # chunked+ckpt branch; off is faster for small N
        num_poses: int = 10,           # FMHead horizon (navtest consumes first 8)
        pose_dim: int = 3,             # (x, y, heading)
        metrics: Optional[List[str]] = None,
        include_imi: bool = True,
        ego_dim: Optional[int] = None, # explicit ego status dim; None to disable
        dropout: float = 0.0,
    ):
        super().__init__()
        self.metrics = list(metrics) if metrics is not None else list(V1_METRICS)
        self.include_imi = include_imi
        self.num_poses = num_poses
        self.pose_dim = pose_dim
        self.d_model = d_model

        # env token projection: env_in_dim -> d_model (Hydra uses a 1x1 conv downscale)
        self.env_proj = nn.Sequential(nn.Linear(env_in_dim, d_model), nn.LayerNorm(d_model))

        # trajectory pose embedding: (T*pose_dim) -> d_model  (Hydra pos_embed)
        self.pose_embed = nn.Sequential(
            nn.Linear(num_poses * pose_dim, d_ffn), nn.ReLU(), nn.Linear(d_ffn, d_model)
        )

        # self-attention across the N candidate tokens -- Hydra-MDP's EXACT block (vendored).
        self.traj_encoder = (
            MemoryEffTransformer(d_model=d_model, nhead=nhead, dim_feedforward=d_ffn,
                                 dropout=dropout, memory_efficient=self_attn_memory_efficient)
            if use_self_attn else None
        )

        # cross-attention: candidate tokens (query) attend to ctx tokens (key/value).
        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward=d_ffn, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(dec_layer, n_decoder_layers)

        # optional explicit ego-status encoding (added to fused tokens, Hydra style).
        self.status_encoding = nn.Linear(ego_dim, d_model) if ego_dim is not None else None

        # per-metric heads: 2-layer MLP; imi is a deeper 3-layer MLP (Hydra).
        self.heads = nn.ModuleDict()
        for m in self.metrics:
            self.heads[m] = nn.Sequential(
                nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, 1)
            )
        if include_imi:
            self.heads["imi"] = nn.Sequential(
                nn.Linear(d_model, d_ffn), nn.ReLU(),
                nn.Linear(d_ffn, d_ffn), nn.ReLU(),
                nn.Linear(d_ffn, 1),
            )

    def forward(
        self,
        env_tokens: torch.Tensor,               # (B, N_kv, env_in_dim)
        trajectories: torch.Tensor,             # (B, N, T, pose_dim)
        env_mask: Optional[torch.Tensor] = None,     # (B, N_kv) bool, True=valid
        ego_status: Optional[torch.Tensor] = None,   # (B, ego_dim)
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        B, N, T, D = trajectories.shape
        assert T == self.num_poses and D == self.pose_dim, (
            f"trajectory (T={T},D={D}) != (num_poses={self.num_poses},pose_dim={self.pose_dim})"
        )

        # Cast ALL float inputs to the module's parameter dtype so callers may pass fp32 OR bf16
        # (e.g. under FSDP MixedPrecision the scorer is bf16, but decode_all returns fp32 candidates
        # and the VLM ctx is bf16). Without this, F.linear raises "mat1 and mat2 must have the same
        # dtype". Mirrors fm_head.py's per-input dtype casting.
        p_dtype = next(self.parameters()).dtype
        env_tokens = env_tokens.to(p_dtype)
        trajectories = trajectories.to(p_dtype)
        if ego_status is not None:
            ego_status = ego_status.to(p_dtype)

        env = self.env_proj(env_tokens)                          # (B, N_kv, d_model)
        traj_emb = self.pose_embed(trajectories.reshape(B, N, T * D))  # (B, N, d_model)
        if self.traj_encoder is not None:
            traj_emb = self.traj_encoder(traj_emb)               # self-attn over N candidates

        # nn.TransformerDecoder: memory_key_padding_mask True == IGNORE, so invert ctx_mask.
        key_pad = (~env_mask.bool()) if env_mask is not None else None
        tr_out = self.transformer(traj_emb, env, memory_key_padding_mask=key_pad)  # (B, N, d_model)
        if self.status_encoding is not None and ego_status is not None:
            tr_out = tr_out + self.status_encoding(ego_status).unsqueeze(1)

        result: Dict[str, torch.Tensor] = {}
        for m, head in self.heads.items():
            result[m] = head(tr_out).squeeze(-1)                 # (B, N) raw logits

        scores = self.aggregate(result, weights)
        result["scores"] = scores
        sel = scores.argmax(dim=1)                               # (B,)
        result["selected_index"] = sel
        result["trajectory"] = trajectories[torch.arange(B, device=trajectories.device), sel]
        return result

    def aggregate(
        self, logits: Dict[str, torch.Tensor], weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """Log-linear aggregation of per-metric logits into one score per candidate
        (additive in log-space = multiplicative in prob-space; mirrors NAVSIM PDMS/EPDMS)."""
        w = weights if weights is not None else V1_WEIGHTS
        base = logits[self.metrics[0]]
        score = torch.zeros_like(base)
        if w.get("imi") and "imi" in logits:
            score = score + w["imi"] * logits["imi"].softmax(dim=-1).clamp_min(1e-8).log()
        for k, wk in w.items():
            if k in ("imi", "_group", "_group_weight") or k not in logits:
                continue
            score = score + wk * logits[k].sigmoid().clamp_min(1e-8).log()
        grp = w.get("_group")
        if grp:
            inner = torch.zeros_like(score)
            for k, wk in grp.items():
                if k in logits:
                    inner = inner + wk * logits[k].sigmoid()
            score = score + w.get("_group_weight", 1.0) * inner.clamp_min(1e-8).log()
        return score

    def distill_loss(
        self,
        logits: Dict[str, torch.Tensor],          # forward() output (per-metric (B,N))
        metric_targets: Dict[str, torch.Tensor],  # {metric: (B,N)} PDM/EPDMS labels in [0,1]
        imi_target: Optional[torch.Tensor] = None, # (B,N) soft distribution (sums to 1) or None
        metric_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Knowledge-distillation loss (Hydra-MDP): per-metric BCE-with-logits + optional
        imitation cross-entropy. Returns per-metric losses + 'total'. Metrics absent from
        `metric_targets` (e.g. v2-only LK/TLC before a v2 cache exists) are skipped."""
        mw = metric_weights or {
            "no_at_fault_collisions": 3.0, "drivable_area_compliance": 3.0,
            "time_to_collision_within_bound": 4.0, "ego_progress": 2.0,
            "driving_direction_compliance": 1.0, "comfort": 2.0,
            "lane_keeping": 2.0, "traffic_light_compliance": 3.0,
        }
        losses: Dict[str, torch.Tensor] = {}
        base = logits[self.metrics[0]]
        total = base.new_zeros(())
        for m, tgt in metric_targets.items():
            if m not in logits:
                continue
            l = F.binary_cross_entropy_with_logits(logits[m], tgt.to(logits[m].dtype).clamp(0, 1))
            losses[m] = l
            total = total + mw.get(m, 1.0) * l
        if imi_target is not None and "imi" in logits:
            imi_l = F.cross_entropy(logits["imi"], imi_target.to(logits["imi"].dtype))
            losses["imi"] = imi_l
            total = total + imi_l
        losses["total"] = total
        return losses
