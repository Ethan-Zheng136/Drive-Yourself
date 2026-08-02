"""fmhead_navsim_agent.py -- navsim agent that decodes with FMHead (for navtest PDMS).

Subclasses AutoVLA's `AutoVLAAgent` (no repo edits) and swaps the trajectory decode from
the codebook/token decoder to the trained FMHead decoder. Supports three candidate-
selection modes (the FMHead samples N candidates per scene):

  * medoid (default): select_by_score's cheap "closest-to-mean-endpoint" pick. This is
    mode-averaging and is the confirmed cause of low PDMS vs oracle-ADE.
  * pdm   : score each of the N candidates with the REAL navsim PDM metric
    (models.utils.score.PDM_Reward.rl_pdm_score, same sub-metrics as run_pdm_score) and
    take the argmax -> safety-aware selection usable at inference.
  * oracle: (eval-only diagnostic, uses GT) pick the candidate with min ADE to the scene
    GT future trajectory -> UPPER BOUND on what selection can achieve.
  * dump_all: (ADDITIVE, no selection change) write ALL N candidate trajectories per token to
    `fmhead_candidate_dump_dir/{token}.npy` (shape (N, num_poses, 3)) and RETURN the medoid pick
    as a valid placeholder. Enables OFFLINE re-selection under any scorer (v1-PDM / v2-EPDMS /
    oracle) from a single inference pass -- so we can measure the selection headroom and build a
    v2-aware-selected submission WITHOUT re-running the VLM. Respects the current style config
    (style_off / content_source / alpha), so it dumps candidates for base OR any styled run.
  * dump_scorer: (ADDITIVE) like dump_all but ALSO saves the VLA context `ctx`(+`ctx_mask`) the
    learned scorer conditions on, as `{token}.npz` (cands, ctx, ctx_mask). This is the training
    data for the deployable learned scorer (fmhead_scorer.FMHeadScorer): candidates are labelled
    offline by pdm_score (label_candidates_pdm.py), and the scorer is trained on the SAME
    (ctx, candidate) pairs it sees at inference (mode=learned) -- no fixed vocabulary needed.

pdm/oracle/dump_all/dump_scorer need the scene (token / GT), so those modes set
`requires_scene = True` and the harness calls `compute_trajectory(agent_input, scene)`.
mode=learned needs NO scene (deployable: scores candidates from ctx+poses via a trained
FMHeadScorer, no metric cache / no GT future). Wire via hydra `agent._target_`
(see run_pdms_fmhead.sh); PYTHONPATH must include /root/workspace/fmhead. Style axis OFF.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

FMHEAD_DIR = os.path.dirname(os.path.abspath(__file__))
if FMHEAD_DIR not in sys.path:
    sys.path.insert(0, FMHEAD_DIR)

from navsim.agents.autovla_agent import AutoVLAAgent  # noqa: E402
from navsim.common.dataclasses import Trajectory  # noqa: E402
from autovla_fmhead import (  # noqa: E402
    FMHeadDecoder, answer_position_hidden, context_sequence_hidden, style_delta,
    autovla_fmhead_config,
)
from fmhead_scorer import FMHeadScorer, V1_WEIGHTS, V2_WEIGHTS  # noqa: E402


class FMHeadAutoVLAAgent(AutoVLAAgent):
    """AutoVLAAgent whose trajectory decode uses the FMHead (medoid | pdm | oracle select)."""

    def __init__(self,
                 fmhead_ckpt_path: str,
                 fmhead_normalizer_path: str = None,
                 fmhead_style_off: bool = True,
                 fmhead_style_alpha: float = 0.0,
                 fmhead_c_reduce: str = "mean",
                 fmhead_hidden_size: int = 256,
                 fmhead_depth: int = 4,
                 fmhead_num_samples: int = 16,
                 fmhead_num_steps: int = 10,
                 fmhead_cfg_weight: float = 1.0,
                 fmhead_select_mode: str = "medoid",       # medoid | pdm | oracle | dump_all | dump_scorer | learned
                 fmhead_metric_cache_path: str = None,     # required for pdm
                 fmhead_candidate_dump_dir: str = None,    # required for dump_all/dump_scorer
                 fmhead_scorer_ckpt_path: str = None,      # required for learned (train_scorer.py output)
                 fmhead_scorer_weights: str = "v1",        # learned aggregation preset: v1 | v2
                 fmhead_use_style_adapter: bool = False,   # opt-in: load a style-adapter head
                 fmhead_constraint: dict = None,           # fm3: {mode:none|soft|kinematic, ...}
                 fmhead_content_source: str = "base",      # base (this agent's legacy) | styled
                 **kwargs):
        super().__init__(**kwargs)
        self._fm_ckpt = fmhead_ckpt_path
        self._fm_norm = fmhead_normalizer_path
        self._fm_style_off = fmhead_style_off
        self._fm_alpha = fmhead_style_alpha
        self._fm_c_reduce = fmhead_c_reduce
        self._fm_select = fmhead_select_mode
        self._fm_metric_cache = fmhead_metric_cache_path
        self._fm_cand_dump_dir = fmhead_candidate_dump_dir
        self._fm_scorer_ckpt = fmhead_scorer_ckpt_path
        self._fm_scorer_weights = fmhead_scorer_weights
        self._fm_use_style_adapter = fmhead_use_style_adapter
        # content_source (ADDITIVE): this agent has ALWAYS read content from the BASE forward,
        # so default "base" preserves its behavior byte-identically; "styled" opts into content
        # from the LoRA-ON forward. (The style delta s is identical either way.) "interp" is the
        # inference-only LATENT CONTENT INTERPOLATION mode: content c(alpha)=ctx_base +
        # alpha*(ctx_styled-ctx_base) with AdaLN style s=0 (NO adapter, NO z-encoder, NO CFG
        # style path). alpha=0 -> ctx_base, s=0 == "base" at alpha=0 (bit-exact clean base).
        assert fmhead_content_source in ("base", "styled", "interp"), fmhead_content_source
        self._fm_content_source = fmhead_content_source
        # fm3 constraint: None / {"mode":"none"} -> byte-identical fm2 xy decode.
        self._fm_constraint = dict(fmhead_constraint) if fmhead_constraint else None
        self._fm_hp = dict(hidden_size=fmhead_hidden_size, depth=fmhead_depth,
                           num_samples=fmhead_num_samples, num_steps=fmhead_num_steps,
                           cfg_weight=fmhead_cfg_weight)
        # pdm/oracle/dump_all/dump_scorer need the scene (token / GT) -> tell the harness to pass it.
        self.requires_scene = fmhead_select_mode in ("pdm", "oracle", "dump_all", "dump_scorer")

    def initialize(self) -> None:
        super().initialize()  # loads base AutoVLA weights (+ optional PERSONA_ADAPTERS)
        device = self.autovla.device
        ck = torch.load(self._fm_ckpt, map_location=device)
        # Decide use_style_adapter from the explicit flag OR by DETECTING style_adapter keys in
        # the ckpt (robust: a style ckpt is loaded WITH the adapter submodule so its
        # fm_head.style_adapter.* keys match; a plain 0.91 ckpt has none -> plain head). This
        # makes the load STRICT in both cases (no silently-ignored adapter weights).
        sd = ck["fm_decoder"]
        ckpt_has_adapter = any(k.startswith("fm_head.style_adapter") for k in sd)
        use_style_adapter = bool(self._fm_use_style_adapter or ckpt_has_adapter)
        # DETECT the opt-in z-encoder from the ckpt keys (a z-path ckpt must be built WITH the
        # z_encoder submodule so its keys load strictly; a plain/adapter ckpt has none). Infer
        # the hidden dims from the stored weight shapes so any dims load correctly.
        ckpt_has_z = any(k.startswith("fm_head.z_encoder") for k in sd)
        z_dims = (512, 256)
        if ckpt_has_z:
            h1 = sd["fm_head.z_encoder.net.0.weight"].shape[0]
            h2 = sd["fm_head.z_encoder.net.2.weight"].shape[0]
            z_dims = (int(h1), int(h2))
        # DETECT the norm_delta stability variant. The zero-conv variant is STRUCTURALLY
        # identical to the legacy z-path (same net.{0,2,4} keys, no extra params), so it can't be
        # inferred from the weight keys -- read it from the saved config (train_fmhead.save_ckpt
        # stores fm_config=vars(config) and the raw cfg). Falls back to False (legacy) if absent.
        _fmcfg = ck.get("fm_config") if isinstance(ck, dict) else None
        _rawcfg = ck.get("config") if isinstance(ck, dict) else None
        ckpt_z_norm = bool(ckpt_has_z and (
            (isinstance(_fmcfg, dict) and _fmcfg.get("z_norm_delta", False))
            or (isinstance(_rawcfg, dict)
                and isinstance(_rawcfg.get("fmhead"), dict)
                and _rawcfg["fmhead"].get("z_norm_delta", False))))
        fm_cfg = autovla_fmhead_config(hidden_size=self._fm_hp["hidden_size"],
                                       depth=self._fm_hp["depth"], style_dropout_prob=0.0,
                                       use_style_adapter=use_style_adapter,
                                       use_z_encoder=ckpt_has_z, z_encoder_dims=z_dims,
                                       z_norm_delta=ckpt_z_norm,
                                       constraint=self._fm_constraint)
        dec = FMHeadDecoder(fm_cfg, num_samples=self._fm_hp["num_samples"],
                            num_steps=self._fm_hp["num_steps"],
                            cfg_weight=self._fm_hp["cfg_weight"],
                            style_alpha=self._fm_alpha,
                            content_source=self._fm_content_source).to(device)
        if self._fm_norm:
            dec.load_normalizer(self._fm_norm)
        # STRICT load in BOTH cases: adapter built => style keys match exactly; plain => 0.91
        # path is byte-identical to before. A key mismatch is a hard error (never silent).
        dec.load_state_dict(ck["fm_decoder"], strict=True)
        dec.eval()
        self._fm_use_style_adapter = use_style_adapter   # reflect what was actually built
        self._fm_use_z_encoder = ckpt_has_z
        self._fm_use_z_norm_delta = ckpt_z_norm
        self._fm_decoder = dec
        print(f"[fmhead-agent] head loaded (use_style_adapter={use_style_adapter}, "
              f"ckpt_has_adapter={ckpt_has_adapter}, use_z_encoder={ckpt_has_z}, "
              f"z_norm_delta={ckpt_z_norm}, content_source={self._fm_content_source}, "
              f"strict=True)", flush=True)

        self._pdm = None
        if self._fm_select == "pdm":
            assert self._fm_metric_cache, "pdm select needs fmhead_metric_cache_path"
            from pathlib import Path
            from models.utils.score import PDM_Reward
            # BUG-1 fix: MetricCacheLoader does pathlib `cache_path / "metadata"`, so it MUST
            # receive a Path, not a str (str/str -> TypeError). Same bug class as GoalFlow.
            self._pdm = PDM_Reward(Path(self._fm_metric_cache))

        # learned: load the trained deployable scorer (train_scorer.py output). It scores the
        # N candidates from ctx + poses ONLY (no metric cache, no GT future) -> legal + fast.
        self._scorer = None
        self._scorer_wts = None
        if self._fm_select == "learned":
            assert self._fm_scorer_ckpt, "learned select needs fmhead_scorer_ckpt_path"
            sc = torch.load(self._fm_scorer_ckpt, map_location=device)
            c = sc["config"]
            self._scorer = FMHeadScorer(
                env_in_dim=c["env_in_dim"], d_model=c["d_model"],
                n_decoder_layers=c.get("n_decoder_layers", 3),
                num_poses=c["num_poses"], metrics=c["metrics"]).to(device)
            self._scorer.load_state_dict(sc["scorer"])
            self._scorer.eval()
            self._scorer_wts = (V2_WEIGHTS if str(self._fm_scorer_weights).lower() == "v2"
                                else V1_WEIGHTS)
            print(f"[fmhead-agent] learned scorer loaded (ckpt={os.path.basename(self._fm_scorer_ckpt)}, "
                  f"weights={self._fm_scorer_weights}, num_poses={c['num_poses']}, metrics={c['metrics']})",
                  flush=True)

        av = self.autovla
        asid = av.action_start_id
        _DROP = ("gt_trajectory", "gt_action", "has_cot", "input_features", "token",
                 "data_path", "text", "image_inputs", "video_inputs")

        ctx_max_len = self._fm_decoder.config.ctx_max_len
        agent_self = self  # read style knobs dynamically so a sweep can flip them at runtime

        @torch.no_grad()
        def _hidden(input_features, disable_adapter):
            inputs = av.get_prompt(input_features)
            mi = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            keep = {k: v for k, v in mi.items() if k not in _DROP}
            cx = av.vlm.disable_adapter() if (disable_adapter and hasattr(av.vlm, "disable_adapter")) \
                else __import__("contextlib").nullcontext()
            with cx:
                out = av.vlm(**keep, output_hidden_states=True, use_cache=False)
            return out.hidden_states[-1], keep

        @torch.no_grad()
        def _extract_ctx_s(input_features):
            """v2 style-injection: (ctx (B,T,2048), ctx_mask, s (B,2048)).

            CONTENT (ctx + base anchor) is read from the BASE VLM (LoRA disabled) so
            scene/safety content is DECOUPLED from style. STYLE s = alpha*(styled_anchor
            - base_anchor), the persona weight-space delta -> AdaLN (+CFG). style_off (no
            LoRA): a SINGLE base forward, s=0 -> the PDMS/feasibility path is unchanged
            (disable_adapter is a nullcontext when no adapter is attached)."""
            hb, keep = _hidden(input_features, disable_adapter=True)          # BASE (content)
            anchor_base = answer_position_hidden(hb, keep["input_ids"],
                                                 attention_mask=keep.get("attention_mask"), action_start_id=asid)
            if agent_self._fm_style_off:
                # content_source irrelevant (no LoRA -> base==styled); single base forward, s=0.
                ctx, ctx_mask = context_sequence_hidden(hb, keep["input_ids"],
                                                        attention_mask=keep.get("attention_mask"),
                                                        action_start_id=asid, max_len=ctx_max_len)
                return ctx, ctx_mask, torch.zeros_like(anchor_base)
            hs, ks = _hidden(input_features, disable_adapter=False)           # STYLED (LoRA on)
            anchor_styled = answer_position_hidden(hs, ks["input_ids"],
                                                   attention_mask=ks.get("attention_mask"), action_start_id=asid)
            s = style_delta(anchor_styled, anchor_base, alpha=agent_self._fm_alpha)  # persona delta
            # thread alpha to the decoder so the z-encoder norm_delta path applies it as the
            # EXTERNAL output gain (s is L2-normalized to a direction, so the baked-in alpha is
            # washed out and must be restored here). Legacy / no-z-encoder paths ignore it.
            agent_self._fm_decoder.style_alpha = float(agent_self._fm_alpha)
            cs = agent_self._fm_content_source
            # content_source (ADDITIVE): "interp" -> LATENT CONTENT INTERPOLATION (inference-only,
            # NO training / NO new module): decode with cross-attn content c(alpha)=ctx_base +
            # alpha*(ctx_styled-ctx_base) and AdaLN style s=0. ctx_b/ctx_s share input_ids+mask ->
            # identical shape & mask, so the interpolation is elementwise. alpha=0 -> ctx_base,
            # s=0 == content_source="base" at alpha=0 (bit-exact clean fm3-kin base).
            if cs == "interp":
                ctx_b, ctx_mask = context_sequence_hidden(hb, keep["input_ids"],
                                                          attention_mask=keep.get("attention_mask"),
                                                          action_start_id=asid, max_len=ctx_max_len)
                ctx_s, _ = context_sequence_hidden(hs, ks["input_ids"],
                                                   attention_mask=ks.get("attention_mask"),
                                                   action_start_id=asid, max_len=ctx_max_len)
                alpha = float(agent_self._fm_alpha)
                ctx = ctx_b + alpha * (ctx_s - ctx_b)
                return ctx, ctx_mask, torch.zeros_like(anchor_base)
            # "base" (DEFAULT/legacy) -> content from the LoRA-OFF forward (body LoRA-free);
            # "styled" -> content from the LoRA-ON forward.
            h_content, keep_c = (hs, ks) if cs == "styled" else (hb, keep)
            ctx, ctx_mask = context_sequence_hidden(h_content, keep_c["input_ids"],
                                                    attention_mask=keep_c.get("attention_mask"),
                                                    action_start_id=asid, max_len=ctx_max_len)
            return ctx, ctx_mask, s

        def _v0_from_features(input_features):
            """Initial ego speed (1,) tensor for kinematic unicycle integration.
            Uses vehicle_velocity (ego status). None outside kinematic mode."""
            if self._fm_decoder.config.constraint_mode != "kinematic":
                return None
            vel = input_features.get("vehicle_velocity")
            if vel is None:
                return torch.zeros(1, device=device)
            v = torch.as_tensor(np.asarray(vel, dtype=np.float32).reshape(-1), device=device)
            speed = float(torch.linalg.norm(v)) if v.numel() > 1 else float(v.abs().item())
            return torch.tensor([speed], device=device)

        self._v0_from_features = _v0_from_features

        @torch.no_grad()
        def fm_predict(input_features):   # medoid single-trajectory (kept for predict())
            ctx, ctx_mask, s = _extract_ctx_s(input_features)
            v0 = _v0_from_features(input_features)
            return self._fm_decoder.decode(ctx, s, ctx_mask=ctx_mask, v0=v0)[0].float().cpu(), ""

        self._extract_ctx_s = _extract_ctx_s
        av.predict = fm_predict
        print(f"[fmhead-agent] FMHead decode active (mode={self._fm_select}, "
              f"style_off={self._fm_style_off}, N={self._fm_hp['num_samples']}, "
              f"ckpt={os.path.basename(self._fm_ckpt)})", flush=True)

    # ----------------------------------------------------------------------
    def _build_features(self, scene_data):
        features = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(scene_data))
        if self.sensor_data_path:
            features.update({"sensor_data_path": self.sensor_data_path})
        return features

    @torch.no_grad()
    def compute_trajectory(self, agent_input, scene=None):
        """FMHead decode + (medoid|pdm|oracle) selection over N candidates."""
        self.autovla.eval()
        num_poses = self._trajectory_sampling.num_poses
        features = self._build_features(agent_input)

        # learned: DEPLOYABLE selection. Decode N candidates, then score them with the trained
        # scorer using ONLY (ctx, poses) -- no metric cache, no GT future -> legal at inference,
        # and much cheaper than mode=pdm (one scorer forward vs N pdm_score sim rollouts). Handled
        # BEFORE the scene guard because learned needs no scene (requires_scene excludes it).
        if self._fm_select == "learned":
            ctx, ctx_mask, s = self._extract_ctx_s(features)
            v0 = self._v0_from_features(features)
            cands = self._fm_decoder.decode_all(ctx, s, ctx_mask=ctx_mask, v0=v0)  # (B,N,H,3) device
            traj_in = cands[:, :, :num_poses, :].float()                          # (B,N,num_poses,3)
            out = self._scorer(ctx.float(), traj_in, env_mask=ctx_mask, weights=self._scorer_wts)
            best = cands[0, int(out["selected_index"][0])][:num_poses].float().cpu().numpy()
            return Trajectory(best, self._trajectory_sampling), ""

        # medoid: identical to the base path (single decode), no scene needed
        if self._fm_select == "medoid" or scene is None:
            poses, _ = self.autovla.predict(features)
            return Trajectory(poses[:num_poses, :], self._trajectory_sampling), ""

        ctx, ctx_mask, s = self._extract_ctx_s(features)
        v0 = self._v0_from_features(features)
        cands = self._fm_decoder.decode_all(ctx, s, ctx_mask=ctx_mask, v0=v0)[0].float().cpu().numpy()  # (N,T,3) m

        if self._fm_select == "dump_all":
            # ADDITIVE diagnostic: persist ALL N candidates for OFFLINE re-selection, then return
            # the medoid (closest-to-mean endpoint) as a valid placeholder trajectory. No scorer is
            # invoked here, so this path can NEVER trigger the two_frame single-stage scoring bug.
            token = scene.scene_metadata.initial_token
            assert self._fm_cand_dump_dir, "dump_all needs fmhead_candidate_dump_dir"
            os.makedirs(self._fm_cand_dump_dir, exist_ok=True)
            np.save(os.path.join(self._fm_cand_dump_dir, f"{token}.npy"),
                    cands[:, :num_poses, :].astype(np.float32))
            eps = cands[:, num_poses - 1, :2]
            medoid = int(np.argmin(np.linalg.norm(eps - eps.mean(axis=0), axis=-1)))
            return Trajectory(cands[medoid][:num_poses], self._trajectory_sampling), ""

        if self._fm_select == "dump_scorer":
            # ADDITIVE (learned-scorer training data): persist the N candidates TOGETHER with the
            # VLA context sequence `ctx` + `ctx_mask` the scorer conditions on -- so the scorer is
            # trained on EXACTLY the (ctx, candidate) pairs it will see at inference (mode=learned),
            # no fixed vocabulary, no train/inference distribution mismatch. Returns the medoid pick
            # as a valid placeholder (no scorer invoked -> cannot trigger the two_frame scoring bug).
            token = scene.scene_metadata.initial_token
            assert self._fm_cand_dump_dir, "dump_scorer needs fmhead_candidate_dump_dir"
            os.makedirs(self._fm_cand_dump_dir, exist_ok=True)
            ctx_np = ctx[0].float().cpu().numpy().astype(np.float16)          # (T_ctx, 2048)
            if ctx_mask is not None:
                mask_np = ctx_mask[0].cpu().numpy().astype(bool)             # (T_ctx,)
            else:
                mask_np = np.ones(ctx_np.shape[0], dtype=bool)
            np.savez(os.path.join(self._fm_cand_dump_dir, f"{token}.npz"),
                     cands=cands[:, :num_poses, :].astype(np.float32),       # (N, num_poses, 3)
                     ctx=ctx_np, ctx_mask=mask_np)
            eps = cands[:, num_poses - 1, :2]
            medoid = int(np.argmin(np.linalg.norm(eps - eps.mean(axis=0), axis=-1)))
            return Trajectory(cands[medoid][:num_poses], self._trajectory_sampling), ""

        if self._fm_select == "pdm":
            # metric-cache key = the CURRENT-frame token, stored as SceneMetadata.initial_token
            # (== scene_dict_list[num_history_frames-1]["token"], the exact key the harness and
            # MetricCacheLoader use). SceneMetadata has NO `.token` attribute -> the old
            # `scene.scene_metadata.token` raised AttributeError on EVERY scene, so the pdm
            # score dataframe had no valid rows and run_pdm_score_cot crashed at df['score'].mean().
            token = scene.scene_metadata.initial_token
            scores = []
            for k in range(cands.shape[0]):
                try:
                    tr = Trajectory(cands[k][:num_poses], self._trajectory_sampling)
                    scores.append(float(self._pdm.rl_pdm_score(tr, token)))
                except Exception as e:  # noqa: BLE001 - a failed candidate must not win
                    print(f"[fmhead-agent] pdm score failed cand={k} token={token}: {e!r}", flush=True)
                    scores.append(-1.0)
            best = cands[int(np.argmax(scores))]
        elif self._fm_select == "oracle":
            gt = np.asarray(scene.get_future_trajectory(num_poses).poses)     # (num_poses, 3)
            L = min(gt.shape[0], cands.shape[1])
            ades = [float(np.linalg.norm(cands[k][:L, :2] - gt[:L, :2], axis=-1).mean())
                    for k in range(cands.shape[0])]
            best = cands[int(np.argmin(ades))]
        else:
            raise ValueError(self._fm_select)

        return Trajectory(best[:num_poses], self._trajectory_sampling), ""
