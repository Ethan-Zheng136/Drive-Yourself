import torch
import os
import math
import contextlib
from tqdm import tqdm
from typing import Dict, Any
import pytorch_lightning as pl
from pathlib import Path
import torch.nn.functional as F
import numpy as np
from typing import List
from torch.distributed.fsdp import StateDictType
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from models.action_tokenizer import ActionTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from models.utils.score import PDM_Reward, TrajectorySampling, Trajectory


class GRPOAutoVLA(pl.LightningModule):
    def __init__(self, config: dict, inference=False):
        super().__init__()
        self.cfg = config
        self.use_cot = config['model']['use_cot']
        self.save_hyperparameters()

        # GRPO accumulates the policy gradient over K same-scene rollouts in micro-batches and
        # backpropagates each micro-batch incrementally (K full VLM forwards held simultaneously
        # would OOM). That requires MANUAL optimization: we own zero_grad / manual_backward /
        # grad-clip / step in training_step. (Lightning automatic optimization can only backward
        # the single returned loss, which would force all K graphs to be retained at once.)
        self.automatic_optimization = False

        # Load trajectory sampling from config or use default
        traj_conf = config['model']['trajectory']
        self.trajectory_sampling = TrajectorySampling(
            num_poses=traj_conf['num_poses'],
            interval_length=traj_conf['interval_length']
        )
        
        # Load token configs
        token_conf = config['model']['tokens']
        self.action_start_id = token_conf['action_start_id']
        self.assistant_id = torch.tensor(token_conf['assistant_id'])

        # Training model (wrapped by Lightning FSDPStrategy)
        self.autovla = AutoVLA(config)

        self.autovla.train()
        self._train_vision_backbone = config['model']['train_vision_backbone']
        self._train_llm_backbone = config['model']['train_lm_backbone']

        # --- YouDrive composable / no-merge ("de-merge") Stage-2 GRPO ---
        # When model.stage1_adapter_path is set, the policy = raw base + FROZEN always-on stage1
        # + TRAINABLE stage2 (wired in run_rft.py). The KL reference (the "styled policy") is then
        # base+stage1 with stage2 DISABLED, computed on the SAME model by toggling active adapters
        # (no separate ~7.5GB reference_model). When unset (null) => legacy behavior below.
        self._demerge = bool(config['model'].get('stage1_adapter_path'))
        self._policy_adapters = ["stage1", "stage2"]   # both active during policy forward
        self._ref_adapters = ["stage1"]                # stage2 off => styled reference

        # online reference model — only needed for the KL term. KL is abandoned (kl_beta=0),
        # so skip building/loading it to save ~15GB/GPU and load time. In de-merge mode the
        # reference is the stage1-only forward, so no separate model is built either.
        self.reference_model = None
        _kl_beta = float(config.get('rl', {}).get('kl_beta', 0.0))
        if (not inference) and _kl_beta > 0 and not self._demerge:
            self.reference_model = AutoVLA(config, inference=True)
            _ref_path = config.get('rl', {}).get('reference_model_path') or config['model']['sft_model_path']
            state_dict = torch.load(_ref_path)["state_dict"]
            state_dict = {k.replace("autovla.", "").replace("drivevla.", ""): v for k, v in state_dict.items()}
            self.reference_model.load_state_dict(state_dict, strict=False)
            self.reference_model.eval()
            print(f"Using online reference model from {_ref_path}")

        # sample generation config
        sample_conf = config['training']['sample']
        # max_new_tokens caps the ROLLOUT decode length. A valid completion is only the
        # action tokens (num_poses, e.g. 10) plus a little formatting margin + EOS; without
        # this cap a rollout that never emits EOS decodes hundreds of wasted tokens (x K per
        # step). Default 64 (>> 10 poses, safe margin); overridable via config
        # training.sample.max_new_tokens or env MAX_NEW_TOKENS.
        _max_new_tokens = int(sample_conf.get('max_new_tokens', 64))
        _env_mnt = os.environ.get("MAX_NEW_TOKENS")
        if _env_mnt not in (None, ""):
            _max_new_tokens = int(_env_mnt)
        self._sample_generation_temperature = {
            "max_length": sample_conf['max_length'],
            "max_new_tokens": _max_new_tokens,
            "temperature": sample_conf['temperature'],
            "top_k": sample_conf['top_k'],
            "top_p": sample_conf['top_p'],
        }

        # reward function
        self.train_critic = PDM_Reward(Path(config['data']['train']['metric_cache_path']))
        self.val_critic = PDM_Reward(Path(config['data']['val']['metric_cache_path']))

        # --- YouDrive: multi-component reward  R = w_pdms*PDMS + w_style*style_sim ---
        _rew = config.get('rl', {}).get('reward', {})
        self._use_pdms = _rew.get('pdms', {}).get('use', True)        # safety reward
        self._pdms_w = float(_rew.get('pdms', {}).get('weight', 1.0))
        _style = _rew.get('style', {})
        self._use_style = _style.get('use', False)                    # DDv2 style reward
        self._style_w = float(_style.get('weight', 1.0))
        # Which profile sub-metrics the style similarity is scored over. PRIMARY path: an explicit
        # LIST of feature keys taken directly from the style cache, e.g. [long_jerk, lat_jerk].
        # The reward then averages the z-normalized per-feature distances over exactly that list, so
        # the GRPO style reward matches whatever eval sub-metric set we calibrate against.
        # Backward-compat STRING aliases are still accepted:
        #   "full" -> all cache feats (legacy full kinematic+social profile).
        #   "kin5" -> the old discriminative kinematic subset (_STYLE_KIN5 below).
        # Data (8 drivers) showed STYLE is a ~1-D kinematic axis: long/lat jerk discriminate drivers,
        # v_avg + ALL social/spacing do NOT; averaging the full profile diluted the signal so
        # base ~= teacher (non-discriminative) and GRPO collapsed style back to base. An explicit
        # short list (e.g. [long_jerk, lat_jerk]) keeps the reward sharp.
        _feat_cfg = _style.get('features', 'kin5')
        if isinstance(_feat_cfg, (list, tuple)):
            self._style_feat_mode = 'list'
        else:
            self._style_feat_mode = str(_feat_cfg).lower()
        self._style_cache = None
        if self._use_style:
            import pickle as _pk
            _cp = _style.get('cache_path', '/mnt/pfs/zhengguantian/autovla/persona/style_cache_navtrain12k.pkl')
            _blob = _pk.load(open(_cp, 'rb'))
            self._style_cache = _blob['cache']; self._style_norm = _blob['norm']; self._style_feats = _blob['feats']
            # The actual subset scored by _style_reward. Explicit list = primary path; strings = aliases.
            if isinstance(_feat_cfg, (list, tuple)):
                self._style_score_feats = [f for f in _feat_cfg if f in self._style_feats]
                assert self._style_score_feats, \
                    f"style features {list(_feat_cfg)}: none present in cache feats {self._style_feats}"
            elif self._style_feat_mode == 'full':
                self._style_score_feats = list(self._style_feats)
            else:
                self._style_score_feats = [f for f in self._STYLE_KIN5 if f in self._style_feats]
                assert self._style_score_feats, \
                    f"kin5 style: none of {self._STYLE_KIN5} present in cache feats {self._style_feats}"
            print(f"[youdrive] style reward ON  w={self._style_w}  mode={self._style_feat_mode} "
                  f"score_feats={self._style_score_feats}  cache={_cp} "
                  f"({len(self._style_cache)} tokens, {len(self._style_feats)} cache feats)", flush=True)
        print(f"[youdrive] reward: pdms={self._use_pdms}(w={self._pdms_w}) style={self._use_style}(w={self._style_w})", flush=True)

        # --- YouDrive: Lagrangian CONSTRAINED reward (maximize style s.t. PDMS >= floor) ---
        # When constraint.use=true the additive (w_pdms*PDMS + w_style*style) reward is REPLACED by
        #   r = style_sim - lambda * relu(pdms_target - pdms)
        # lambda is a NON-trainable dual variable (register_buffer => requires_grad=False, never added
        # to the optimizer, lives on the right device, synced across ranks by deriving its update from
        # the all_gather'd GLOBAL mean PDMS). use=false => exact additive fall-back (backward compat).
        _con = config.get('rl', {}).get('constraint', {})
        self._use_constraint = bool(_con.get('use', False))
        self._pdms_target = float(_con.get('pdms_target', 0.85))
        self._lambda_lr = float(_con.get('lambda_lr', 0.5))
        self._lambda_max = float(_con.get('lambda_max', 50.0))
        _lambda_init = float(_con.get('lambda_init', 1.0))
        self.register_buffer("constraint_lambda", torch.tensor(_lambda_init, dtype=torch.float32))
        self._last_pdms = None   # per-rollout PDMS stashed by reward_function for the dual-ascent update
        if self._use_constraint:
            assert self._use_pdms and self._use_style, \
                "constraint.use=true requires reward.pdms.use=true AND reward.style.use=true"
            print(f"[youdrive] CONSTRAINED reward ON: maximize style s.t. PDMS>={self._pdms_target} "
                  f"(lambda_init={_lambda_init}, lambda_lr={self._lambda_lr}, lambda_max={self._lambda_max})", flush=True)

        # sliding window for training reward
        if not inference:
            self.window_size = config['rl']['reward'].get("sliding_window_size", 100)
            self.register_buffer("training_reward_buffer", torch.zeros(self.window_size))
            self.register_buffer("sliding_idx",   torch.zeros(1, dtype=torch.long))
            self.register_buffer("window_count",  torch.zeros(1, dtype=torch.long))

    def _freeze_stage1(self):
        """Re-disable grads on the frozen stage1 adapter (de-merge mode only).

        PEFT's set_adapter() flips requires_grad=True on EVERY active adapter, so after we
        restore the policy adapter set (["stage1","stage2"]) the frozen stage1 would become
        trainable again. Re-freezing here guarantees the invariant: only stage2 is trainable."""
        for _n, _p in self.autovla.vlm.named_parameters():
            if "lora_" in _n and ".stage1." in _n:
                _p.requires_grad = False

    def _grad_sync_ctx(self, is_last_backward):
        """No-sync context for DDP gradient accumulation across the K micro-batch backwards.

        We backward each micro-batch separately to bound memory; under DDP every backward would
        otherwise trigger its own all-reduce. We suppress the all-reduce on every backward EXCEPT
        the final one (which then syncs the accumulated gradient once). On a single process (smoke)
        or when no DDP wrapper exposes no_sync(), this is a no-op."""
        if is_last_backward:
            return contextlib.nullcontext()
        ddp_module = getattr(getattr(self, "trainer", None), "model", None)
        if ddp_module is not None and hasattr(ddp_module, "no_sync"):
            return ddp_module.no_sync()
        return contextlib.nullcontext()

    def training_step(self, batch):
        """Joint objective:  L = ce_weight * CE(DDv2 target tokens)  +  pdms_weight * GRPO(PDMS).
        CE (the SFT imitation loss) is the optional style backbone; the PDMS policy-gradient is the
        safety/constraint signal.

        REAL (per-prompt) GRPO: for the SAME scene (gen_data) we draw K=group_size rollouts, score
        each, and normalize the advantage WITHIN that K-group, LOCALLY on the rank (no cross-rank
        all_gather for the advantage). Each rank still processes its own scene; the group is the K
        samples of THAT scene. all_gather is used ONLY for the GLOBAL mean PDMS (lambda dual-ascent)
        and logging means.

        MANUAL optimization: we own zero_grad / micro-batch manual_backward / grad-clip / step here.
        The K policy forwards are processed in micro-batches and back-propagated incrementally so we
        never hold all K VLM graphs at once (would OOM)."""
        self.autovla.train()
        device = next(self.parameters()).device
        opt = self.optimizers()
        opt.zero_grad()

        # Split the two input formats: tokenized teacher-forced (CE) vs raw features (rollout).
        gen_data = {"input_features": batch.pop("input_features"), "token": batch.pop("token")}

        ce_use = self.cfg["rl"].get("ce", {}).get("use", False)
        ce_weight = float(self.cfg["rl"].get("ce", {}).get("weight", 1.0))
        pdms_use = self._use_pdms
        pdms_weight = float(self.cfg["rl"]["reward"].get("pdms", {}).get("weight", 1.0))
        # Constrained mode: the reward already encodes the safety weighting via lambda, so don't
        # double-scale the policy gradient by reward.pdms.weight (use 1.0). The rollout still runs
        # because pdms_use is true. (Additive mode keeps the configured weight => backward compat.)
        if self._use_constraint:
            pdms_weight = 1.0
        kl_beta = float(self.cfg["rl"].get("kl_beta", 0.0))
        scale = self.cfg["rl"]["reward"].get("scale", 1.0)

        # GRPO group size K (same-scene rollouts). Env RL_GROUP_SIZE overrides the config for smoke.
        group_size = int(self.cfg["rl"].get("group_size", 8))
        _env_K = os.environ.get("RL_GROUP_SIZE")
        if _env_K not in (None, ""):
            group_size = int(_env_K)
        if group_size < 1:
            group_size = 1
        # micro-batch size for the logps forward+backward (1 or 2 typical; reduce to 1 on OOM).
        logp_chunk = int(os.environ.get("LOGP_CHUNK", "1"))
        if logp_chunk < 1:
            logp_chunk = 1

        # Count the total number of manual_backward calls so we can sync DDP grads only on the last.
        run_pdms = bool(pdms_use and pdms_weight > 0)
        run_ce = bool(ce_use and ce_weight > 0)
        n_pdms_chunks = ((group_size + logp_chunk - 1) // logp_chunk) if run_pdms else 0
        n_backwards = (1 if run_ce else 0) + n_pdms_chunks
        bwd_i = 0

        total_loss_val = 0.0

        # ---- CE: DDv2-style imitation backbone (teacher forced on DDv2 action tokens) ----
        if run_ce:
            ce_out = self.autovla(dict(batch))  # forward pops keys; copy keeps batch intact
            ce_loss = ce_out.loss
            with self._grad_sync_ctx(bwd_i == n_backwards - 1):
                self.manual_backward(ce_weight * ce_loss)
            bwd_i += 1
            total_loss_val += ce_weight * float(ce_loss.detach())
            self.log("ce_loss", ce_loss.item(), sync_dist=True, prog_bar=True, on_step=True, on_epoch=False)

        # ---- PDMS: REAL GRPO over K same-scene rollouts ----
        if run_pdms:
            with torch.no_grad():
                # K rollouts of the SAME scene (single shared vision/prompt forward, K decodes).
                samples = self.generate_sample(gen_data, model=self.autovla, device=device,
                                               num_samples=group_size)
                rewards, pdms_vals = [], []
                for s in samples:
                    r = (self.reward_function(s) * scale).reshape(-1)
                    rewards.append(r)
                    # reward_function stashes this rollout's PDMS in self._last_pdms.
                    pdms_vals.append(self._last_pdms.reshape(-1))
                rewards_t = torch.cat(rewards).float()           # (K,)
                pdms_t = torch.cat(pdms_vals).float()            # (K,)

                # WITHIN-GROUP advantage normalization (local to this rank's scene; NO cross-rank
                # gather). adv_i = (r_i - mean(r_1..r_K)) / (std(r_1..r_K) + 1e-4).
                grp_mean = rewards_t.mean()
                grp_std = torch.nan_to_num(rewards_t.std())
                advantages = (rewards_t - grp_mean) / (grp_std + 1e-4)   # (K,)
                self.log("reward_std", grp_std.item(), sync_dist=False, prog_bar=True,
                         on_step=True, on_epoch=False)

                # --- Lagrangian DUAL ASCENT on lambda (constrained mode only) ---
                # GLOBAL and identical across ranks: derive the global mean PDMS from ALL K x world
                # rollouts (all_gather the rank's K PDMS values, then average). Non-trainable buffer,
                # updated once per optimizer step under no_grad on every rank identically:
                #   lambda <- clip( lambda + lambda_lr * (pdms_target - global_mean_pdms), 0, lambda_max )
                # below target -> lambda grows (push safety); >= target -> lambda shrinks (style breathes).
                if self._use_constraint:
                    global_pdms = self.all_gather(pdms_t).reshape(-1).float().mean()
                    new_lambda = (self.constraint_lambda
                                  + self._lambda_lr * (self._pdms_target - global_pdms)
                                  ).clamp(0.0, self._lambda_max)
                    _lam_prev = float(self.constraint_lambda.item())
                    self.constraint_lambda.fill_(float(new_lambda))
                    if os.environ.get("CONSTRAINT_DEBUG") == "1":
                        _dir = "GROW(safety)" if float(new_lambda) > _lam_prev else ("SHRINK(style)" if float(new_lambda) < _lam_prev else "flat")
                        print(f"[constraint] dual-ascent: batch_pdms={float(global_pdms):.4f} target={self._pdms_target} "
                              f"lambda {_lam_prev:.4f} -> {float(new_lambda):.4f}  [{_dir}]  "
                              f"K={len(samples)} reward_std={float(grp_std):.4f}", flush=True)
                    self.log("lambda", self.constraint_lambda.item(), sync_dist=False,
                             prog_bar=True, on_step=True, on_epoch=False)
                    self.log("batch_pdms", global_pdms.item(), sync_dist=False,
                             prog_bar=True, on_step=True, on_epoch=False)

            # Policy loss accumulated over the K samples in MICRO-BATCHES, each back-propagated
            # incrementally (bounded memory). Objective is the mean over K of the per-sample loss.
            K = len(samples)
            pol_loss_accum = 0.0
            kl_accum, n_kl = 0.0, 0
            for start in range(0, K, logp_chunk):
                chunk = samples[start:start + logp_chunk]
                m = len(chunk)
                input_ids = torch.cat([c["input_ids"] for c in chunk], dim=0)
                attention_mask = torch.cat([c["attention_mask"] for c in chunk], dim=0)
                completion_mask = torch.cat([c["completion_mask"] for c in chunk], dim=0)
                prompt_length = chunk[0]["prompt_length"]
                # pixel_values / grid belong to the single shared prompt; repeat to the m rows.
                pv = chunk[0]["pixel_values_videos"].repeat(m, 1)
                vg = chunk[0]["video_grid_thw"].repeat(m, 1)
                adv_chunk = advantages[start:start + m]

                per_token_logps = self.get_per_token_logps(
                    self.autovla.vlm, input_ids, attention_mask, pv, vg)
                per_token_logps = per_token_logps[:, prompt_length - 1:]

                per_token_loss = -(torch.exp(per_token_logps - per_token_logps.detach())
                                   * adv_chunk.unsqueeze(-1))

                # optional KL leash, per-sample, exactly as before. De-merge mode: KL to the STYLED
                # policy (base+stage1, stage2 off) by toggling active adapters on the SAME model.
                if kl_beta > 0:
                    ref_logps = None
                    with torch.no_grad():
                        if self._demerge:
                            # base_model.set_adapter takes a list (PeftModel.set_adapter does not).
                            self.autovla.vlm.base_model.set_adapter(self._ref_adapters)   # stage1 only (styled ref)
                            ref_logps = self.get_per_token_logps(
                                self.autovla.vlm, input_ids, attention_mask, pv, vg)
                            self.autovla.vlm.base_model.set_adapter(self._policy_adapters)  # restore stage1+stage2
                            self._freeze_stage1()                                  # set_adapter re-enabled stage1 grad
                        elif self.reference_model is not None:
                            ref_logps = self.get_per_token_logps(
                                self.reference_model.vlm, input_ids, attention_mask, pv, vg)
                    if ref_logps is not None:
                        ref_logps = ref_logps[:, prompt_length - 1:]
                        per_token_kl = torch.exp(ref_logps - per_token_logps) - (ref_logps - per_token_logps) - 1
                        per_token_loss = per_token_loss + kl_beta * per_token_kl
                        kl_div = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
                        kl_accum += float(kl_div.detach()) * m
                        n_kl += m

                per_sample_loss = (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)  # (m,)
                # contribution of this micro-batch to the mean over all K samples.
                chunk_loss = per_sample_loss.sum() / K
                with self._grad_sync_ctx(bwd_i == n_backwards - 1):
                    self.manual_backward(pdms_weight * chunk_loss)
                bwd_i += 1
                pol_loss_accum += float(chunk_loss.detach())

            total_loss_val += pdms_weight * pol_loss_accum
            self.log("pol_loss", pol_loss_accum, sync_dist=True, prog_bar=True, on_step=True, on_epoch=False)
            if n_kl > 0:
                self.log("kl", kl_accum / n_kl, sync_dist=True, prog_bar=True, on_step=True, on_epoch=False)
            self.training_buffer_record(rewards_t.mean())

        # ---- grad clip (value, 1.0; matches the old configure_gradient_clipping) + step ----
        params_with_grad = [p for p in self.parameters() if p.grad is not None]
        if params_with_grad:
            torch.nn.utils.clip_grad_value_(params_with_grad, clip_value=1.0)
        opt.step()

        self.log("loss", total_loss_val, sync_dist=True, prog_bar=True)
    
    def training_buffer_record(self, step_reward):
        idx = self.sliding_idx.item()
        self.training_reward_buffer[idx] = step_reward

        new_idx = (idx + 1) % self.window_size
        self.sliding_idx.fill_(new_idx)
        new_count = min(self.window_count.item() + 1, self.window_size)
        self.window_count.fill_(new_count)

        if new_count >= self.window_size:
            sliding_avg = self.training_reward_buffer.mean()
            self.log(
                "avg_train_reward",
                sliding_avg,
                sync_dist=False, 
                prog_bar=True
            )

    def on_after_backward(self):
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.log("grad_norm", total_norm, sync_dist=True)
    
    # ---- DDv2 style reward (kinematic v/a/j long+lat + social), per token ----
    _DT = 0.5
    _CORRIDOR = 1.5
    # The 5 driver-DISCRIMINATIVE kinematic features (the 1-D style axis), as named in
    # build_style_cache.py kin(): longitudinal jerk RMS, lateral jerk RMS, peak long accel,
    # peak long decel, peak lateral accel. These are the only features the kin5 reward scores.
    # Style axis = 2 DECORRELATED kinematic representatives (longitudinal jerk + lateral jerk).
    # The 8-driver correlation matrix showed peak_acc/peak_dec/long_jerk are ~1.0 collinear
    # (one longitudinal axis) and lat_amax/lat_jerk ~1.0 collinear (one lateral axis); jerk has
    # the highest discriminability (CV 111/102). Scoring exactly these two with equal weight ==
    # S = 0.5*long_jerk + 0.5*lat_jerk (the agreed metric), so the GRPO reward matches eval S.
    _STYLE_KIN5 = ["long_jerk", "lat_jerk"]

    @staticmethod
    def _ego_series(xy):
        p = np.array([[0.0, 0.0]] + list(xy), dtype=float)
        v = np.diff(p, axis=0) / GRPOAutoVLA._DT
        return p[1:], np.hypot(v[:, 0], v[:, 1])

    @staticmethod
    def _kin(xy):
        DT = GRPOAutoVLA._DT
        p = np.array([[0.0, 0.0]] + list(xy), float); v = np.diff(p, axis=0) / DT
        vx, vy = v[:, 0], v[:, 1]; sp = np.hypot(vx, vy)
        ax = np.diff(sp) / DT if len(sp) > 1 else np.array([0.])
        lj = np.diff(ax) / DT if len(ax) > 1 else np.array([0.])
        ay = np.diff(vy) / DT if len(vy) > 1 else np.array([0.])
        aj = np.diff(ay) / DT if len(ay) > 1 else np.array([0.])
        return {"v_avg": float(sp.mean()), "v_std": float(sp.std()),
                "peak_acc": float(ax.max()) if ax.size else 0.0, "peak_dec": float(-ax.min()) if ax.size else 0.0,
                "long_jerk": float(np.sqrt((lj ** 2).mean())) if lj.size else 0.0,
                "lat_amax": float(np.abs(ay).max()) if ay.size else 0.0,
                "lat_jerk": float(np.sqrt((aj ** 2).mean())) if aj.size else 0.0}

    @staticmethod
    def _interact(ego_xy, agents):
        pos, sp = GRPOAutoVLA._ego_series(ego_xy)
        K = min(len(pos), len(agents)); CORR = GRPOAutoVLA._CORRIDOR
        thw = ttc = lead_gap = math.inf; dmin = dvru = math.inf
        for k in range(K):
            exy = pos[k]; espeed = max(sp[k], 1e-3)
            axy, avel, vru = agents[k]
            if len(axy) == 0:
                continue
            d = np.hypot(axy[:, 0] - exy[0], axy[:, 1] - exy[1])
            dmin = min(dmin, float(d.min()))
            if vru.any():
                dvru = min(dvru, float(d[vru].min()))
            ahead = (axy[:, 0] > exy[0]) & (np.abs(axy[:, 1] - exy[1]) < CORR)
            if ahead.any():
                gap = axy[ahead, 0] - exy[0]
                j = int(np.argmin(gap)); g = float(gap[j])
                lead_gap = min(lead_gap, g); thw = min(thw, g / espeed)
                close = espeed - float(avel[ahead][j, 0])
                if close > 0.1:
                    ttc = min(ttc, g / close)
        f = lambda v: (None if math.isinf(v) else v)
        return {"thw_min": f(thw), "ttc_min": f(ttc), "lead_gap_min": f(lead_gap),
                "dist_any_min": f(dmin), "dist_vru_min": f(dvru)}

    def _style_reward(self, sample):
        """style_sim in (0,1]: exp(-mean z-normalized |profile(rollout) - profile(DDv2@token)|).
        Scored ONLY over self._style_score_feats -- the explicit sub-metric LIST from config
        (primary path, e.g. [long_jerk, lat_jerk]) or a 'full'/'kin5' string alias. Higher = more
        DDv2-like.

        We always compute the (cheap) kinematic profile via _kin(). The (costlier) social /
        interaction features via _interact() are computed ONLY when the requested sub-metric list
        contains at least one feature that _kin does NOT produce -- derived from _kin's actual
        output keys, so the decision tracks the explicit list rather than a hard-coded mode.

        When CONSTRAINT_DEBUG=1, prints the per-feature |z| components so the score is auditable."""
        tok = sample['token']
        if isinstance(tok, (list, tuple)):
            tok = tok[0]
        entry = self._style_cache.get(tok) if self._style_cache is not None else None
        if entry is None:
            return 0.0
        traj = np.asarray(sample['trajectory'].poses, dtype=float)[:, :2]
        agents = [(np.asarray(a, float), np.asarray(v, float), np.asarray(r, bool)) for a, v, r in entry["agents"]]
        # Always compute the cheap kinematic profile; add the costlier social/interaction features
        # ONLY when a requested sub-metric is not produced by _kin (derived from _kin's own keys).
        prof = dict(self._kin(traj))
        if set(self._style_score_feats) - set(prof.keys()):
            prof.update(self._interact(traj, agents))
        ddv2 = entry["profile"]
        zs = []; _dbg = []
        for fkey in self._style_score_feats:
            m, s = self._style_norm.get(fkey, (0.0, 1.0))
            a = prof.get(fkey); b = ddv2.get(fkey)
            if a is None or b is None:
                continue
            z = abs((a - b) / s)
            zs.append(z)
            if os.environ.get("CONSTRAINT_DEBUG") == "1":
                _dbg.append(f"{fkey}:|z|={z:.3f}(roll={a:.3f},ddv2={b:.3f})")
        if not zs:
            return 0.0
        score = float(math.exp(-float(np.mean(zs))))
        if os.environ.get("CONSTRAINT_DEBUG") == "1":
            print(f"[style:{self._style_feat_mode}] tok={tok} mean|z|={np.mean(zs):.3f} "
                  f"score={score:.4f}  " + "  ".join(_dbg), flush=True)
        return score

    def reward_function(self, sample):
        device = next(self.parameters()).device

        # multi-component reward: R = w_pdms * PDMS(safety) + w_style * style_sim(DDv2 profile)
        pdms = float(self.train_critic.rl_pdm_score(sample['trajectory'], sample['token'])) if self._use_pdms else 0.0
        style = self._style_reward(sample) if self._use_style else 0.0
        if self._use_pdms:
            self.log("pdms_r", pdms, sync_dist=True, prog_bar=True, on_step=True, on_epoch=False)
        if self._use_style:
            self.log("style_r", style, sync_dist=True, prog_bar=True, on_step=True, on_epoch=False)

        # stash this rollout's PDMS so training_step can do the dual-ascent lambda update from the
        # all_gather'd GLOBAL mean PDMS (kept identical across ranks).
        self._last_pdms = torch.tensor(float(pdms), device=device)

        if self._use_constraint:
            # Lagrangian: style is the OBJECTIVE, the PDMS floor is a one-sided barrier.
            #   r = style - lambda * relu(pdms_target - pdms)
            # Only pays the penalty when PDMS dips BELOW the floor; above the floor style is free.
            lam = float(self.constraint_lambda.item())
            violation = max(self._pdms_target - pdms, 0.0)   # relu(target - pdms)
            reward = torch.tensor(style - lam * violation, device=device)
            if os.environ.get("CONSTRAINT_DEBUG") == "1":
                print(f"[constraint] r = style({style:.4f}) - lambda({lam:.4f})*relu({self._pdms_target}-pdms({pdms:.4f})="
                      f"{violation:.4f}) = {float(reward):.4f}", flush=True)
            self.log("violation", violation, sync_dist=True, prog_bar=True, on_step=True, on_epoch=False)
            self.log("lambda", lam, sync_dist=False, prog_bar=True, on_step=True, on_epoch=False)
        else:
            reward = torch.tensor(self._pdms_w * pdms + self._style_w * style).to(device)

        # Add chain-of-thought penalty (if "need cot" is found in the generated text).
        if self.use_cot:
            cot_conf = self.cfg['rl']['cot_penalty']
            cot_penalty_coef = cot_conf['coef']
            center = cot_conf['center']
            cot_penalty_weight = cot_conf['weight']

            cot_penalties = torch.stack([
                torch.sigmoid(torch.tensor(
                    (len(text) - center) * cot_penalty_coef,
                    device=device,
                    dtype=reward.dtype
                ))
                if "complex scenario" in text.lower() else torch.tensor(
                    0.0, device=device, dtype=reward.dtype
                )
                for text in sample['completion_texts']
            ])
            reward = reward - cot_penalty_weight * cot_penalties
        else:
            cot_penalties = torch.tensor(0.0, device=device, dtype=reward.dtype)

        self.log("train_reward", reward, sync_dist=True, prog_bar=True, on_step=True, on_epoch=False)
        self.log("cot_penalty", cot_penalties.mean(), sync_dist=True, prog_bar=True, on_step=True, on_epoch=False)

        return reward
    
    def get_per_token_logps(self, model, input_ids, attention_mask, pixel_values_videos, video_grid_thw):
        # Get the per-token log probabilities for the completions for the model and the reference model
        logits = model(input_ids, attention_mask=attention_mask, 
                       pixel_values_videos=pixel_values_videos, 
                       video_grid_thw=video_grid_thw).logits  # (B, L, V)
        
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it

        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        log_probs = torch.log_softmax(logits, dim=-1)  # (B, L-1, V)
        per_token_logps = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
        return per_token_logps

    def generate_sample(self, data, model, device, num_samples=1):
        """Draw num_samples=K rollouts of the SAME scene and return a LIST of K sample dicts.

        Efficient path: the vision/prompt forward runs ONCE; num_return_sequences=K makes generate
        repeat only the decoding. Each returned sequence becomes its own sample dict (its own
        trajectory / completion_ids / completion_mask / attention_mask / input_ids row). The shared
        prompt's pixel_values_videos / video_grid_thw are attached to every sample (they get repeated
        to the micro-batch width when logps are recomputed in training_step)."""
        # Get the model inputs (single shared prompt; vision encode happens once here / in generate).
        inputs = model.get_prompt(data['input_features'])
        model_inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        # set seed (per-rank diversity); with do_sample + num_return_sequences the K rollouts differ.
        torch.manual_seed(int(str(device).split(':')[-1]))

        # Generate K completions. num_return_sequences=K => one vision/prompt forward, K decodes.
        with torch.no_grad():
            prompt_completion_ids = model.vlm.generate(
                **model_inputs,
                do_sample=True,
                num_return_sequences=num_samples,
                # max_new_tokens caps the rollout decode (takes precedence over max_length in HF):
                # a valid 10-pose completion needs only ~10 action tokens + EOS, so this prevents
                # runaway no-EOS decodes from wasting hundreds of tokens x K per step.
                max_new_tokens=self._sample_generation_temperature['max_new_tokens'],
                max_length=self._sample_generation_temperature['max_length'],
                temperature=self._sample_generation_temperature['temperature'],
                top_k=self._sample_generation_temperature['top_k'],
                top_p=self._sample_generation_temperature['top_p'],
            )

            prompt_length = inputs.input_ids.size(1)
            prompt_mask = model_inputs['attention_mask']                  # (1, P)
            completion_ids = prompt_completion_ids[:, prompt_length:]     # (K, Lc)
            K = prompt_completion_ids.size(0)

            # Per-row completion mask (mask through the first EOS inclusive).
            is_eos = completion_ids == model.processor.tokenizer.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()   # (K, Lc)

            # prompt mask is shared across the K rows -> expand, then concat with each completion mask.
            attention_mask = torch.cat([prompt_mask.expand(K, -1), completion_mask], dim=1)  # (K, L)

            completion_texts = model.processor.batch_decode(completion_ids)

            pv = model_inputs['pixel_values_videos']
            vg = model_inputs['video_grid_thw']

            n_poses = self.trajectory_sampling.num_poses
            samples = []
            for i in range(K):
                # Decode THIS row's trajectory (was hard-coded to row 0 before).
                actions_tokens = completion_ids[i][completion_ids[i] >= self.action_start_id]
                if len(actions_tokens) > n_poses:
                    actions_tokens = actions_tokens[:n_poses]
                elif len(actions_tokens) < n_poses:
                    actions_tokens = torch.cat(
                        [actions_tokens, torch.zeros(n_poses - len(actions_tokens)).to(device)])
                    actions_tokens = actions_tokens.long()
                trajectory = self.autovla.action_tokenizer.decode_token_ids_to_trajectory(
                    actions_tokens.cpu())[0, 1:]
                trajectory = Trajectory(trajectory.cpu().numpy(), self.trajectory_sampling)

                samples.append({
                    'trajectory': trajectory,
                    'token': data['token'],
                    'completion_texts': [completion_texts[i]],
                    'prompt_length': prompt_length,
                    'input_ids': prompt_completion_ids[i:i + 1],
                    'completion_ids': completion_ids[i:i + 1],
                    'attention_mask': attention_mask[i:i + 1],
                    'completion_mask': completion_mask[i:i + 1],
                    'pixel_values_videos': pv,      # shared single-prompt vision tensor
                    'video_grid_thw': vg,
                })

        # clean up
        torch.cuda.empty_cache()

        return samples
    
    def configure_optimizers(self):
        if not self._train_vision_backbone:
            for param in self.autovla.vlm.visual.parameters():
                param.requires_grad = False

        if not self._train_llm_backbone:
            for param in self.autovla.vlm.model.parameters():
                param.requires_grad = False

        params_to_update = []
        for param in self.autovla.vlm.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

        assert len(params_to_update) > 0, 'No parameters to update'

        lr = float(self.cfg['training']['learning_rate'])
        wd = float(self.cfg['training'].get('weight_decay', 0.0))
        optimizer = torch.optim.AdamW(
            params_to_update,
            lr=lr,
            weight_decay=wd
        )

        return optimizer
    
    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        # Filter out parameters with no gradient to avoid empty tensor lists
        params_with_grad = [p for p in self.parameters() if p.grad is not None]
        if params_with_grad:
            torch.nn.utils.clip_grad_value_(params_with_grad, clip_value=gradient_clip_val)

    def on_save_checkpoint(self, checkpoint: dict):
        # only save main model
        sd = checkpoint.get("state_dict", {})
        for k in list(sd):
            if k.startswith("reference_model."):
                sd.pop(k)

class SFTAutoVLA(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.cfg = config
        self.save_hyperparameters()

        self.autovla = AutoVLA(config)
        self.autovla.train()

        self._train_vision_backbone = config['model']['train_vision_backbone']
        self._train_llm_backbone = config['model']['train_lm_backbone']

        # === YouDrive: auxiliary L2 trajectory-regression loss (Stage-1 Style-LoRA) ===
        # L = CE + w_l2 * L2(expected_trajectory, DDv2_gt_trajectory). The expected trajectory
        # is a *differentiable* decode: at each action-token position we softmax the logits over
        # the ACTION sub-vocabulary and take the probability-weighted ("expected") motion
        # primitive, then roll it out with the SAME kinematics as ActionTokenizer.rollout. This
        # gives a dense continuous geometric signal that bypasses the action-token argmax/
        # quantization ceiling. Gated by config['model']['l2']; default OFF => pure CE (the exact
        # original behavior, full backward compat).
        l2conf = config['model'].get('l2', {}) or {}
        self._l2_use = bool(l2conf.get('use', False))
        self._l2_weight = float(l2conf.get('weight', 1.0))
        self._l2_loss_type = str(l2conf.get('loss_type', 'mse'))   # 'mse' | 'smooth_l1'
        self._action_start_id = self.autovla.action_start_id

        # Precompute per-token LOCAL pose primitives from the action codebook (n_bins, 6, 4, 2),
        # matching ActionTokenizer.rollout exactly:
        #   next_pos  = prev_pos + R(prev_head) @ centroid(last-substep over its 4 corners)
        #   next_head = prev_head + atan2(corner0 - corner3) of the last substep (local frame)
        # so a one-hot expected-trajectory reproduces decode_token_ids_to_trajectory bit-for-bit.
        cb = self.autovla.action_tokenizer.code_book.float()        # (n_bins, 6, 4, 2)
        self._n_action_bins = cb.shape[0]
        _local_xy = cb[:, -1, :, :].mean(dim=1)                     # (n_bins, 2)
        _d = cb[:, -1, 0, :] - cb[:, -1, 3, :]                      # (n_bins, 2)
        _local_head = torch.atan2(_d[:, 1], _d[:, 0])               # (n_bins,)
        self.register_buffer("_act_local_xy", _local_xy, persistent=False)
        self.register_buffer("_act_local_head", _local_head, persistent=False)
        if self._l2_use:
            print(f"[youdrive] L2 traj-reg ON  weight={self._l2_weight}  loss={self._l2_loss_type}  "
                  f"action_sub_vocab=[{self._action_start_id}, {self._action_start_id + self._n_action_bins}) "
                  f"({self._n_action_bins} bins)", flush=True)

        # === YouDrive: ego-progress matching loss (match the DDv2 teacher's forward progress) ===
        # progress = net forward displacement of the differentiable expected trajectory;
        # loss = relu(progress_DDv2 - progress_student) (one-sided: only penalize UNDER-progressing
        # relative to DDv2, never penalize going further). This injects the SAFE-COMPATIBLE part of
        # the DDv2 style (assertive forward motion / ego_progress) without the unsafe raw-jerk part.
        # Gated by config['model']['progress']; default OFF.
        progconf = config['model'].get('progress', {}) or {}
        self._prog_use = bool(progconf.get('use', False))
        self._prog_weight = float(progconf.get('weight', 0.1))
        self._prog_one_sided = bool(progconf.get('one_sided', True))
        if self._prog_use:
            print(f"[youdrive] ego-progress loss ON  weight={self._prog_weight}  "
                  f"one_sided={self._prog_one_sided}", flush=True)

    def _expected_trajectory_l2(self, logits, labels, gt_trajectory):
        """Differentiable expected future trajectory + L2 to the raw DDv2 trajectory.

        :param logits: (B, T, V) LM logits from the teacher-forced forward.
        :param labels: (B, T) label ids; action-token positions are labels in
                       [action_start_id, action_start_id + n_action_bins).
        :param gt_trajectory: (B, G, >=2) raw continuous DDv2 trajectory (x forward, y left, m).
        :returns: (l2_loss or None, n_pred_per_sample, (dbg_pred, dbg_gt))
        """
        asid = self._action_start_id
        n_bins = self._n_action_bins
        # teacher forcing: logits[t] predicts the token at labels[t+1].
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        action_mask = (shift_labels >= asid) & (shift_labels < asid + n_bins)   # (B, T-1)

        local_xy = self._act_local_xy.to(device=logits.device, dtype=torch.float32)
        local_head = self._act_local_head.to(device=logits.device, dtype=torch.float32)
        G = gt_trajectory.shape[1]

        per_losses, n_pred = [], []
        dbg_pred = dbg_gt = None
        for b in range(logits.size(0)):
            pos_idx = torch.nonzero(action_mask[b], as_tuple=False).squeeze(-1)  # in-order positions
            if pos_idx.numel() == 0:
                n_pred.append(0)
                continue
            # softmax over the ACTION sub-vocabulary ONLY (not the full ~150k Qwen vocab).
            # float32 keeps the softmax / rollout numerically stable under bf16-true.
            act_logits = shift_logits[b, pos_idx, asid:asid + n_bins].float()    # (T, n_bins)
            probs = torch.softmax(act_logits, dim=-1)
            Tm = min(probs.size(0), G)

            pos = act_logits.new_zeros(2)
            head = act_logits.new_zeros(())
            traj = []
            for t in range(Tm):
                p = probs[t]
                exy = p @ local_xy                 # expected local (dx, dy)   (2,)
                eh = (p * local_head).sum()        # expected local dheading   scalar
                c, s = torch.cos(head), torch.sin(head)
                gx = exy[0] * c - exy[1] * s       # R(head) @ exy  (matches transform_to_global)
                gy = exy[0] * s + exy[1] * c
                pos = pos + torch.stack([gx, gy])
                head = head + eh
                traj.append(pos)
            pred = torch.stack(traj, dim=0)                                     # (Tm, 2)
            gt = gt_trajectory[b, :Tm, :2].to(pred.dtype)                       # (Tm, 2)
            if self._l2_loss_type == 'smooth_l1':
                per_losses.append(F.smooth_l1_loss(pred, gt))
            else:
                per_losses.append(F.mse_loss(pred, gt))
            n_pred.append(Tm)
            if dbg_pred is None:
                dbg_pred, dbg_gt = pred.detach(), gt.detach()

        if not per_losses:
            return None, n_pred, (dbg_pred, dbg_gt)
        return torch.stack(per_losses).mean(), n_pred, (dbg_pred, dbg_gt)

    def _progress_loss(self, logits, labels, gt_trajectory):
        """Differentiable ego-progress matching to the DDv2 teacher.

        Builds the same differentiable expected trajectory as _expected_trajectory_l2, then
        compares NET FORWARD PROGRESS (straight-line displacement from start to the final pose)
        of the student vs the DDv2 gt. one_sided=True -> penalize only when the student
        under-progresses relative to DDv2 (relu); else symmetric |diff|.
        :returns: (progress_loss or None, (pred_prog_dbg, gt_prog_dbg))
        """
        asid = self._action_start_id
        n_bins = self._n_action_bins
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        action_mask = (shift_labels >= asid) & (shift_labels < asid + n_bins)
        local_xy = self._act_local_xy.to(device=logits.device, dtype=torch.float32)
        local_head = self._act_local_head.to(device=logits.device, dtype=torch.float32)
        G = gt_trajectory.shape[1]

        per, dbg = [], None
        for b in range(logits.size(0)):
            pos_idx = torch.nonzero(action_mask[b], as_tuple=False).squeeze(-1)
            if pos_idx.numel() == 0:
                continue
            act_logits = shift_logits[b, pos_idx, asid:asid + n_bins].float()
            probs = torch.softmax(act_logits, dim=-1)
            Tm = min(probs.size(0), G)
            pos = act_logits.new_zeros(2)
            head = act_logits.new_zeros(())
            for t in range(Tm):
                p = probs[t]
                exy = p @ local_xy
                eh = (p * local_head).sum()
                c, s = torch.cos(head), torch.sin(head)
                pos = pos + torch.stack([exy[0] * c - exy[1] * s, exy[0] * s + exy[1] * c])
                head = head + eh
            pred_prog = torch.linalg.vector_norm(pos)                       # net displacement (m)
            gt_prog = torch.linalg.vector_norm(gt_trajectory[b, Tm - 1, :2].to(pred_prog.dtype))
            if self._prog_one_sided:
                per.append(torch.relu(gt_prog - pred_prog))
            else:
                per.append((gt_prog - pred_prog).abs())
            if dbg is None:
                dbg = (float(pred_prog.detach()), float(gt_prog.detach()))

        if not per:
            return None, dbg
        return torch.stack(per).mean(), dbg

    def training_step(self, batch):
        hascot = batch['has_cot']
        gt_trajectory = batch["gt_trajectory"]
        gt_action = batch["gt_action"]
        # AutoVLA.forward pops keys in-place; snapshot before it mutates batch so the
        # base (LoRA-disabled) forward below can be re-run on the same inputs.
        kl_beta = float(os.environ.get("PERSONA_KL_BETA", "0"))
        kl_batch = dict(batch) if kl_beta > 0 else None
        output = self.autovla(batch)
        loss = output.loss

        # === Add additional loss on action tokens ===
        # output.logits shape: (B, T, V), labels shape: (B, T)
        logits = output.logits
        vocab_size = logits.size(-1)
        # Flatten logits and labels for token-wise loss
        labels = batch['labels']
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        logits_flat = shift_logits.view(-1, vocab_size)
        labels_flat = shift_labels.view(-1)
        # Identify action token positions
        action_mask = (labels_flat >= self.autovla.action_start_id)  # shape: (B*T,)
        # Compute token-wise cross-entropy loss
        ce_loss_all = F.cross_entropy(logits_flat, labels_flat, reduction='none')  # shape: (B*T,)
        # Extract loss for action tokens
        action_loss = ce_loss_all[action_mask]
        # Add to total loss with optional weighting factor
        if action_loss.numel() > 0:
            action_loss = action_loss.mean()

        # # add more penalty for CoT reasoning data
        if hascot[0] == True:
            # print("add more penalty for CoT reasoning data")
            loss = loss * 40
            loss = loss + action_loss

        # === YouDrive: auxiliary L2 trajectory-regression loss ===
        # L = CE + w_l2 * L2(expected_trajectory, DDv2_gt_trajectory). Dense continuous geometric
        # signal that bypasses the action-token quantization ceiling. Logs ce_loss / l2_loss
        # separately. OFF by default (config['model']['l2'].use=false) => loss == CE (unchanged).
        if self._l2_use and self._l2_weight > 0:
            self.log("ce_loss", loss.item(), batch_size=gt_action.shape[0],
                     sync_dist=True, prog_bar=True)
            l2_loss, n_pred, (dbg_pred, dbg_gt) = self._expected_trajectory_l2(
                output.logits, batch['labels'], gt_trajectory)
            if l2_loss is not None:
                loss = loss + self._l2_weight * l2_loss
                self.log("l2_loss", l2_loss.item(), batch_size=gt_action.shape[0],
                         sync_dist=True, prog_bar=True)
                if os.environ.get("L2_DEBUG") == "1":
                    print(f"[l2] n_pred_per_sample={n_pred} (gt poses={gt_trajectory.shape[1]})  "
                          f"l2={l2_loss.item():.4f}  w_l2={self._l2_weight}", flush=True)
                    if dbg_pred is not None:
                        for t in range(min(3, dbg_pred.shape[0])):
                            print(f"[l2]   t={t} expected=({dbg_pred[t,0]:+.3f},{dbg_pred[t,1]:+.3f}) "
                                  f"gt=({dbg_gt[t,0]:+.3f},{dbg_gt[t,1]:+.3f})", flush=True)

        # === YouDrive: ego-progress matching loss (safe-compatible style; match DDv2 forward progress) ===
        if self._prog_use and self._prog_weight > 0:
            prog_loss, prog_dbg = self._progress_loss(output.logits, batch['labels'], gt_trajectory)
            if prog_loss is not None:
                loss = loss + self._prog_weight * prog_loss
                self.log("prog_loss", prog_loss.item(), batch_size=gt_action.shape[0],
                         sync_dist=True, prog_bar=True)
                if os.environ.get("PROG_DEBUG") == "1" and prog_dbg is not None:
                    print(f"[prog] pred_disp={prog_dbg[0]:.3f}m  ddv2_disp={prog_dbg[1]:.3f}m  "
                          f"loss={prog_loss.item():.4f}  w={self._prog_weight}", flush=True)

        # === KL anchor to frozen base policy (no-merge constraint) ===
        # Keeps the LoRA-on action distribution close to the LoRA-off base, so persona
        # imitation does not overwrite base's GRPO-acquired PDMS competence. Off by default
        # (PERSONA_KL_BETA=0 reproduces the original pure-imitation training exactly).
        if kl_beta > 0:
            with torch.no_grad():
                with self.autovla.vlm.disable_adapter():
                    base_logits = self.autovla(dict(kl_batch)).logits  # base = LoRA disabled
            shift_on = logits[..., :-1, :]
            shift_base = base_logits[..., :-1, :]
            kl_labels = batch['labels'][..., 1:]
            kl_mask = (kl_labels >= self.autovla.action_start_id)
            if kl_mask.any():
                lp_on = F.log_softmax(shift_on[kl_mask], dim=-1)
                lp_base = F.log_softmax(shift_base[kl_mask], dim=-1)
                kl_anchor = (lp_on.exp() * (lp_on - lp_base)).sum(-1).mean()
                loss = loss + kl_beta * kl_anchor
                self.log("kl_anchor", kl_anchor.item(),
                         batch_size=gt_action.shape[0], sync_dist=True, prog_bar=True)

        self.log("train_loss", loss.item(),
                 batch_size=gt_action.shape[0],
                 sync_dist=True,
                 prog_bar=True)
        
        
        return loss
    
    def validation_step(self, batch):
        gt_trajectory = batch["gt_trajectory"]
        gt_action = batch["gt_action"]

        output = self.autovla(batch)
        loss = output.loss
        self.log("val_loss", loss.item(),
                 batch_size=gt_action.shape[0],
                 sync_dist=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        if not self._train_vision_backbone:
            for param in self.autovla.vlm.visual.parameters():
                param.requires_grad = False

        if not self._train_llm_backbone:
            for param in self.autovla.vlm.model.parameters():
                param.requires_grad = False

        params_to_update = []
        for param in self.autovla.vlm.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

        assert len(params_to_update) > 0, 'No parameters to update'

        optimizer = torch.optim.AdamW(
            params_to_update,
            lr=self.cfg['training']['learning_rate'],
            weight_decay=self.cfg['training'].get('weight_decay', 0.0)
        )
        lr_warmpup_step = self.cfg['training']['lr_warmup_step']
        lr_step_freq = self.cfg['training']['lr_step_frequency']
        lr_step_gamma = self.cfg['training']['lr_step_gamma']

        def lr_update(step, warmup_step, step_size, gamma):
            if step < warmup_step:
                # warm up lr
                lr_scale = 1 - (warmup_step - step) / warmup_step * 0.95
            else:
                n = (step - warmup_step) // step_size
                lr_scale = gamma ** n

            if lr_scale < 1e-2:
                lr_scale = 1e-2
            elif lr_scale > 1:
                lr_scale = 1

            return lr_scale
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: lr_update(
                step,
                lr_warmpup_step,
                lr_step_freq,
                lr_step_gamma,
            )
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    @torch.no_grad()
    def calculate_metrics(self, logits, labels, gt_trajectory):
        # Find start index for ground truth sequence
        gt_start_idx = self.find_assistant_start_idx(labels[0])
        gt_tokens = labels[0, gt_start_idx+1:] # shifted
        pred_tokens = logits[0, gt_start_idx:-1].argmax(dim=-1)

        # Find action tokens in ground truth and predicted sequences
        gt_action_idx = gt_tokens >= self.autovla.action_start_id
        pred_action_idx = pred_tokens >= self.autovla.action_start_id

        if len(pred_tokens[pred_action_idx]) != len(gt_tokens[gt_action_idx]):
            pred_action_idx = gt_action_idx
            
        gt_action_tokens = gt_tokens[gt_action_idx]
        pred_action_tokens = pred_tokens[pred_action_idx]

        # Decode predicted trajectory
        # pred_trajectory = self.autovla.action_tokenizer.decode_token_ids_to_trajectory(pred_action_tokens.cpu())
        # action_acc = (pred_action_tokens == gt_action_tokens).float().mean()
        # traj_mse = torch.norm(pred_trajectory[0, 1:, :2] - gt_trajectory[0].cpu(), dim=-1).mean()
        # traj_mse = traj_mse.to(logits.device)

        # return {
        #     'action_acc': action_acc,
        #     'traj_mse': traj_mse
        # }
    
    @staticmethod
    def find_assistant_start_idx(labels):
        assistant_id = torch.tensor(ASSISTANT_ID).to(labels.device)
        
        for j in range(len(labels) - len(assistant_id) + 1):
            if torch.equal(labels[j:j + len(assistant_id)], assistant_id):
                start_idx = j
                break

        return start_idx


class AutoVLA(torch.nn.Module):
    def __init__(self, config, inference=False, device='cpu'):
        super().__init__()
        self.device = device

        model_path = config['model']['pretrained_model_path']
        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer, 
                                                model_config=config['model'])
        self.vlm.resize_token_embeddings(len(self.processor.tokenizer))

        self.video_conf = config['model']['video']
        self.action_start_id = config['model']['tokens']['action_start_id']

        self.use_cot = config['model']['use_cot']
        self.gen_conf = config['inference']['sample']

    def predict(self, input_features):
        inputs = self.get_prompt(input_features)
        model_inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        outputs = self.vlm.generate(
            **model_inputs,
            max_length=self.gen_conf['max_length'],
            do_sample=True,
            temperature=self.gen_conf['temperature'],
            top_k=self.gen_conf['top_k'],
            top_p=self.gen_conf['top_p'],
        )

        outputs_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]

        outputs_trimmed = outputs_trimmed[0][:-1].cpu() # remove end token
        cot_results = self.processor.decode(outputs_trimmed)
        # if 'Chain-of-Thought is not needed' not in self.processor.decode(outputs_trimmed):
        #     print(self.processor.decode(outputs_trimmed))
        #     print("has cot")
        # else:
        #     print(self.processor.decode(outputs_trimmed))
        #     print("no cot")
        actions_tokens = outputs_trimmed[outputs_trimmed >= self.action_start_id]

        trajectory = self.action_tokenizer.decode_token_ids_to_trajectory(actions_tokens)[0, 1:]

        return trajectory, cot_results
    
    def get_prompt(self, input_features, image_mode="video"):
        # image sensor
        images = input_features['images']

        min_pixels = self.video_conf.get("min_pixels", 28 * 28 * 128)
        max_pixels = self.video_conf.get("max_pixels", 28 * 28 * 128)

        camera_images = {}
        
        # List of camera types to load
        camera_types = ['front_camera', 'front_left_camera', 'front_right_camera']
        
        # When sensor_data_path is set, image paths are relative and need the prefix.
        # When it is null/empty (e.g. nuScenes stores full paths), use them as-is.
        for camera_type in camera_types:
            camera_images[camera_type] = []
            for i in range(4):
                img = images[camera_type][i]
                if input_features['sensor_data_path']:
                    camera_images[camera_type].append(
                        os.path.join(input_features['sensor_data_path'], img))
                else:
                    camera_images[camera_type].append(img)

        # Assign to individual variables for message formatting
        front_camera_1, front_camera_2, front_camera_3, front_camera_4 = camera_images['front_camera']
        front_left_camera_1, front_left_camera_2, front_left_camera_3, front_left_camera_4 = camera_images['front_left_camera']
        front_right_camera_1, front_right_camera_2, front_right_camera_3, front_right_camera_4 = camera_images['front_right_camera']


        # vehicle state
        velocity = input_features["vehicle_velocity"]

        if isinstance(velocity, list) or isinstance(velocity, np.ndarray):
            velocity_x = velocity[0]
            velocity_y = velocity[1]
            velocity = np.sqrt(velocity_x**2 + velocity_y**2)
    
        acceleration = input_features["vehicle_acceleration"]
        if isinstance(acceleration, list) or isinstance(acceleration, np.ndarray):
            acceleration_x = acceleration[0]
            acceleration_y = acceleration[1]
            acceleration = np.sqrt(acceleration_x**2 + acceleration_y**2)

        instruction = input_features["driving_command"].lower()
    
        user_content = [
            {
                "type": "text",
                "text": (
                    "The autonomous vehicle is equipped with three cameras mounted at the front, left, and right, enabling a comprehensive perception of the surrounding environment."
                )
            },
            {
                "type": "text",
                "text": "The first video presents the front view of the vehicle, comprising four sequential frames sampled at 2 Hz."
            },
            {
                "type": "video",
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "video": [
                    f"file://{front_camera_1}",
                    f"file://{front_camera_2}",
                    f"file://{front_camera_3}",
                    f"file://{front_camera_4}",
                ]
            },
            {
                "type": "text",
                "text": "The second video presents the front-left view of the vehicle, comprising four sequential frames sampled at 2 Hz."
            },
            {
                "type": "video",
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "video": [
                    f"file://{front_left_camera_1}",
                    f"file://{front_left_camera_2}",
                    f"file://{front_left_camera_3}",
                    f"file://{front_left_camera_4}",
                ]
            },
            {
                "type": "text",
                "text": "The third video presents the front-right view of the vehicle, comprising four sequential frames sampled at 2 Hz."
            },
            {
                "type": "video",
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "video": [
                    f"file://{front_right_camera_1}",
                    f"file://{front_right_camera_2}",
                    f"file://{front_right_camera_3}",
                    f"file://{front_right_camera_4}",
                ]
            },
            {
                "type": "text",
                "text": (
                    f"The current velocity of the vehicle is {velocity:.3f} m/s, and the current acceleration is {acceleration:.3f} m/s². "
                    f"The driving instruction is: {instruction}. Based on this information, plan the action trajectory for the autonomous vehicle over the next five seconds."
                )
            },
        ]

        if self.use_cot:
            messages = [
                {   
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text":
                            "You are an Advanced Driver Assistance and Full Self-Driving System. "
                            "You will receive visual observations from the ego vehicle’s cameras and dynamic information about the vehicle’s current state. "
                            "Your task is to predict the optimal driving action for the next five seconds.\n\n"
                            "First, carefully analyze the surrounding environment by considering traffic lights, the movements of other vehicles and pedestrians, lane markings, and any other relevant factors.\n\n"
                            "If necessary, use step-by-step reasoning (Chain-of-Thought) to arrive at the best driving action. Otherwise, you may directly predict the final driving action.\n\n"
                            "Structure your reasoning as follows:\n"
                            "1. **Scene Analysis**: Describe the traffic situation, including relevant environmental cues such as traffic lights, lane markings, and the behaviors of surrounding vehicles or pedestrians.\n"
                            "2. **Identification of Critical Objects**: Identify two to three critical road users or obstacles, specifying their relative positions to the ego vehicle.\n"
                            "3. **Prediction of Critical Object Behavior**: Predict the potential movements of the identified critical objects.\n"
                            "4. **Ego Vehicle Intent Reasoning**: Based on the observed environment and current vehicle state, reason about the desired intent of the ego vehicle.\n"
                            "5. **Final Action Decision**: Select one lateral action and one longitudinal action:\n"
                            "- **Lateral actions** (choose exactly one): [move forward, turn left, change lane to left, turn right, change lane to right]\n"
                            "- **Longitudinal actions** (choose exactly one): [stop, deceleration to zero, maintain constant speed, quick deceleration, deceleration, quick acceleration, acceleration]\n\n"
                            "Present the final action clearly after your reasoning steps."
                        }
                    ]
                },

                {
                    "role": "user",
                    "content": user_content
                },


            ]
        else:
            messages = [
                {   
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text":
                            "You are an Advanced Driver Assistance and Full Self-Driving System. "
                            "You will be provided with video observations from the ego vehicle’s surrounding cameras, along with the vehicle’s current dynamic states. "
                            "Your task is to predict the most appropriate driving action for the next five seconds."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": user_content
                },
            ]

        image_inputs, video_inputs = process_vision_info(messages)
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        return inputs
    
    def forward(self, inputs):
        inputs.pop('gt_trajectory')
        inputs.pop('gt_action')
        inputs.pop('has_cot')
        # YouDrive: these are carried only for the GRPO rollout / PDMS scoring path
        # (generate_sample + reward), not VLM inputs. Drop them so Qwen's forward
        # doesn't choke on unexpected kwargs in the plain SFT/CE path.
        inputs.pop('input_features', None)
        inputs.pop('token', None)
        outputs: CausalLMOutputWithPast = self.vlm(**inputs)

        return outputs