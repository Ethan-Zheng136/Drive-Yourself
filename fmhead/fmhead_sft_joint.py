"""fmhead_sft_joint.py -- JOINT training: the fm3-kin full-FT SFT recipe (fmhead_sft.py,
UNCHANGED) with an FMHeadScorer that FIRES in the SAME training step.

WHAT THIS ADDS over fmhead_sft.py (which stays byte-identical and reproducible):
  * an attached FMHeadScorer (small, trainable, shares the LM/backbone features);
  * in each step, AFTER the flow-matching loss, we (1) decode N candidates from the
    CURRENT decoder (no_grad -- sampling is non-differentiable), (2) score every
    candidate with the REAL NAVSIM pdm_score (label_candidates_pdm.score_candidates,
    keyed by the batch `token` -> metric_cache_navtrain12k), (3) add a per-metric
    distillation loss for the scorer. total = flow_loss + lambda * scorer_loss.

WHY IT IS "JOINT" (not the decoupled dump->label->train pipeline): the scorer's
cross-attention reads `ctx` WITH grad, so the LM backbone co-adapts its features to
serve BOTH the generator (flow) and the selector (scorer). Candidates are detached
(there is no differentiable path from a post-hoc argmax selection back to the sampler
without RL -- see the discussion in the design notes).

Trains on navtrain (the SFT dataset), so mode=learned can be evaluated on navtest with
NO train/test leakage.

Config: config/fmhead_fm3_kin_joint.yaml == fmhead_fm3_kin.yaml + a `joint_scorer:` block.
When joint_scorer.enabled is false, training_step falls back to the parent (identical to
fmhead_sft.py). Launch: launch_sft_joint_8gpu.sh (8-GPU FSDP, same as the fm3-kin run).

The scorer is saved INSIDE the Lightning checkpoint under `scorer.*`; pull a standalone
deployable scorer_final.pt (agent mode=learned format) with extract_scorer.py.
"""

from __future__ import annotations

import argparse
import datetime
import functools
import os
import sys

import numpy as np
import torch

AUTOVLA_ROOT = "/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA"
FMHEAD_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (AUTOVLA_ROOT, os.path.join(AUTOVLA_ROOT, "navsim"), FMHEAD_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import fmhead_sft as _base  # noqa: E402  (parent SFT module -- UNCHANGED)
from fmhead_sft import FMHeadSFTAutoVLA, load_config, _DROP  # noqa: E402
from autovla_fmhead import answer_position_hidden, context_sequence_hidden  # noqa: E402
from fmhead_scorer import FMHeadScorer, V1_METRICS, V2_EXTRA_METRICS, V1_WEIGHTS  # noqa: E402


# ---------------------------------------------------------------------------
class FMHeadSFTJoint(FMHeadSFTAutoVLA):
    """fm3-kin SFT + a jointly-fired FMHeadScorer distilled from in-loop pdm_score."""

    def __init__(self, config: dict):
        super().__init__(config)  # builds autovla (3B) + fm_decoder + freeze semantics
        j = dict(config.get("joint_scorer") or {})
        self._joint_enabled = bool(j.get("enabled", False))
        self._joint_lambda = float(j.get("loss_weight", 0.5))
        self._joint_metric_cache = j.get("metric_cache")
        self._joint_interval = float(config["model"]["trajectory"]["interval_length"])
        num_poses = int(config["model"]["trajectory"]["num_poses"])
        self._sc_env_in = int(j.get("env_in_dim", 2048))          # Qwen2.5-VL-3B hidden
        self._sc_dmodel = int(j.get("d_model", 256))
        self._sc_layers = int(j.get("n_decoder_layers", 3))
        self._sc_num_poses = num_poses
        # metric heads: v1 (navtrain12k cache) or v1+v2 extras (needs a v2 cache to label)
        self._sc_metrics = list(V1_METRICS)
        if str(j.get("metrics", "v1")).lower() == "v2":
            self._sc_metrics = list(V1_METRICS) + list(V2_EXTRA_METRICS)

        if self._joint_enabled:
            assert self._joint_metric_cache, "joint_scorer.enabled needs joint_scorer.metric_cache"
            self.scorer = FMHeadScorer(
                env_in_dim=self._sc_env_in, d_model=self._sc_dmodel,
                n_decoder_layers=self._sc_layers, num_poses=num_poses,
                metrics=self._sc_metrics)
            print(f"[joint] scorer attached: env_in={self._sc_env_in} d_model={self._sc_dmodel} "
                  f"layers={self._sc_layers} num_poses={num_poses} metrics={self._sc_metrics} "
                  f"lambda={self._joint_lambda}", flush=True)
        else:
            self.scorer = None
        self._pdm_tools = None  # lazy, per-rank (built on first labelled step)
        self._joint_missing = 0

        # freeze_fm_decoder (variant #3, "scorer-only"): freeze the FMHead so ONLY the scorer
        # trains (base+head both frozen from the ep6 warm-start). variant #2 keeps it trainable.
        self._freeze_head = bool(j.get("freeze_fm_decoder", False))
        if self._freeze_head:
            for p in self.fm_decoder.parameters():
                p.requires_grad_(False)
            print("[joint] fm_decoder FROZEN -> scorer-only training (variant #3)", flush=True)

    def _setup_trainable(self):
        # extend the parent freeze semantics: also freeze the FMHead when freeze_fm_decoder is
        # set. (configure_optimizers re-calls this, so the freeze must be idempotent here too.)
        super()._setup_trainable()
        if getattr(self, "_freeze_head", False):
            for p in self.fm_decoder.parameters():
                p.requires_grad_(False)

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None,
                                    gradient_clip_algorithm=None):
        # FSDP FULL_SHARD + mostly-frozen model (variants #2/#3: only the tiny scorer / head
        # trains): the few trainable params don't shard onto every rank, so on some ranks the
        # local grad list is EMPTY and torch's foreach clip_grad_* asserts
        # (`_group_tensors_by_device_and_dtype([[]])` -> "nested_tensorlist[0].empty()").
        # `value` clipping is a purely LOCAL element-wise clamp (no collective), so skipping it
        # on a rank that owns no trainable grad is correct and cannot desync. Full-FT (#1) has
        # grads on every rank, so it always clips as before.
        has_grad = any(p.grad is not None
                       for grp in optimizer.param_groups for p in grp["params"])
        if not has_grad:
            return
        self.clip_gradients(optimizer, gradient_clip_val=gradient_clip_val,
                            gradient_clip_algorithm=gradient_clip_algorithm)

    # -- reusable NAVSIM PDM stack, built ONCE per rank (heavy import + cache loader) --
    def _tools(self):
        if self._pdm_tools is None:
            from label_candidates_pdm import build_pdm_tools
            self._pdm_tools = build_pdm_tools(self._joint_metric_cache)
        return self._pdm_tools

    # -- flow loss + ctx (mirrors fmhead_sft._fm_loss, but ALSO returns ctx for the scorer) --
    def _flow_and_ctx(self, batch):
        dev = next(self.parameters()).device
        inputs = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        vlm_in = {k: v for k, v in inputs.items() if k not in _DROP}
        # When the LM is frozen (variants #2/#3) run its forward under no_grad: ctx becomes a
        # constant INPUT to the head/scorer (which still train -- their own params carry grad),
        # saving activation memory and avoiding frozen-param gradient-checkpointing errors. When
        # the LM trains (#1 full-FT) ctx MUST carry grad so the backbone co-adapts.
        import contextlib
        lm_ctx = contextlib.nullcontext() if self._train_llm_backbone else torch.no_grad()
        with lm_ctx:
            outputs = self.autovla.vlm(**vlm_in, output_hidden_states=True, use_cache=False)
        h = outputs.hidden_states[-1]                                    # (B,S,2048)
        ctx, ctx_mask = context_sequence_hidden(h, inputs["input_ids"],
                                                attention_mask=inputs.get("attention_mask"),
                                                action_start_id=self._asid,
                                                max_len=self.fm_decoder.config.ctx_max_len)
        anchor = answer_position_hidden(h, inputs["input_ids"],
                                        attention_mask=inputs.get("attention_mask"),
                                        action_start_id=self._asid)
        s = torch.zeros_like(anchor)                                     # style axis OFF (s=0)
        flow = self.fm_decoder(inputs["gt_trajectory"], ctx, s, ctx_mask=ctx_mask)  # fwd == loss
        return flow, ctx, ctx_mask, s, inputs

    def _v0_from_gt(self, gt, dev):
        """Initial ego speed for the kinematic unicycle decode: |first GT displacement|/dt."""
        if self.fm_decoder.config.constraint_mode != "kinematic":
            return None
        g = gt.reshape(-1, gt.shape[-1])
        return (torch.linalg.norm(g[0, :2]) / self._joint_interval).reshape(1).to(dev)

    def training_step(self, batch, batch_idx=0):
        if not self._joint_enabled:
            return super().training_step(batch, batch_idx)

        flow, ctx, ctx_mask, s, inputs = self._flow_and_ctx(batch)
        dev = ctx.device

        # (1) decode N candidates from the CURRENT decoder (no grad: sampling is non-diff).
        with torch.no_grad():
            v0 = self._v0_from_gt(inputs["gt_trajectory"], dev)
            cands = self.fm_decoder.decode_all(ctx.detach(), s.detach(),
                                               ctx_mask=ctx_mask, v0=v0)  # (B,N,H,3)
        num_poses = min(self._sc_num_poses, cands.shape[2])
        cands_np = cands[0, :, :num_poses, :].float().cpu().numpy()       # (N,H,3) navsim metres

        # (2) real NAVSIM pdm_score per candidate -> per-metric distillation targets.
        #     `token` arrives batched (default collate turns per-sample strings into a list); we
        #     only score batch index 0 (cands[0]), so unwrap to that scalar token string. Passing
        #     the raw list would make `token not in cache_paths` always True -> labels always None
        #     -> the scorer would silently never train.
        token = inputs.get("token")
        if isinstance(token, (list, tuple)):
            token = token[0] if len(token) else None
        labels = self._tools_score(cands_np, token) if token is not None else None

        # (3) scorer ALWAYS runs, EVERY step. Under FSDP all ranks must exercise the same params in
        #     backward; if a rank whose token lacks a metric cache skipped the scorer, its reduce-
        #     scatter for the scorer shard would never fire and the collective would deadlock. So
        #     when labels are missing we mask the scorer loss to 0 (params stay in the graph with
        #     zero gradient) and train the generator (flow) only -- no silent divergence.
        #     ctx carries grad in #1 (LM co-adapts) and is a constant input in #2/#3.
        out = self.scorer(ctx, cands[:, :, :num_poses, :], env_mask=ctx_mask, weights=V1_WEIGHTS)
        if labels is None:
            self._joint_missing += 1
            # zero-grad loss that touches EXACTLY the params distill_loss would (each metric head
            # + shared trunk via tr_out); imi head is untouched in both paths, so the grad-param
            # set is identical whether or not a label exists -> FSDP stays uniform across ranks.
            sloss = sum(out[m].sum() for m in self._sc_metrics if m in out) * 0.0
            have = 0.0
        else:
            targets = {m: torch.as_tensor(labels[m], device=dev, dtype=torch.float32).unsqueeze(0)
                       for m in self._sc_metrics if m in labels}
            sloss = self.scorer.distill_loss(out, targets)["total"]
            have = 1.0
        total = flow + self._joint_lambda * sloss
        self.log("train_loss", flow.item(), batch_size=1, sync_dist=True, prog_bar=True)
        self.log("scorer_loss", float(sloss.item()), batch_size=1, sync_dist=True, prog_bar=True)
        self.log("scorer_have_label", have, batch_size=1, sync_dist=True)
        return total

    def _tools_score(self, cands_np, token):
        from label_candidates_pdm import score_candidates
        return score_candidates(cands_np, token, self._tools(), interval=self._joint_interval)

    # val loss stays the pure flow loss (checkpoint monitor == fmhead_sft's val_loss).


# ===========================================================================
# Trainer (mirrors fmhead_sft.build_and_fit, but instantiates FMHeadSFTJoint and
# casts the scorer to bf16 alongside the FMHead so FSDP MixedPrecision is uniform).
# ===========================================================================
def build_and_fit_joint(config_path, seed=42):
    import pytorch_lightning as pl
    from pytorch_lightning import seed_everything
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from pytorch_lightning.strategies import FSDPStrategy
    from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch.utils.data import DataLoader
    from transformers import AutoProcessor
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer
    from dataset_utils.sft_dataset import SFTDataset, DataCollator

    seed_everything(seed)
    os.chdir(AUTOVLA_ROOT)
    config = load_config(config_path)

    processor = AutoProcessor.from_pretrained(config["model"]["pretrained_model_path"], use_fast=True)
    using_cot = config["model"]["use_cot"]
    train_ds = SFTDataset(config["data"]["train"], config["model"], processor, using_cot=using_cot)
    val_ds = SFTDataset(config["data"]["val"], config["model"], processor, using_cot=using_cot)

    model = FMHeadSFTJoint(config)
    base = config["model"].get("sft_model_path")
    if base and str(base).lower() not in ("none", "null", ""):
        print(f"[joint] warm-start base: {base}", flush=True)
        sd = torch.load(base, map_location="cpu")["state_dict"]
        sd = {k.replace("autovla.", "", 1) if k.startswith("autovla.") else k: v for k, v in sd.items()}
        msg = model.autovla.load_state_dict(sd, strict=False)
        print(f"[joint] base loaded (missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)})",
              flush=True)
    # OPTIONAL FMHead warm-start (variants #2/#3): initialise fm_decoder from a trained head
    # (extract_full_ft_ckpt.py's fm_decoder.pt == {"fm_decoder": stripped_state_dict}). Absent
    # (variant #1 full-FT) -> head trains from scratch, exactly like the fm3-kin GOLD run.
    head_init = config.get("fmhead_head_init")
    if head_init and str(head_init).lower() not in ("none", "null", ""):
        print(f"[joint] warm-start head: {head_init}", flush=True)
        hd = torch.load(head_init, map_location="cpu")
        hd = hd["fm_decoder"] if isinstance(hd, dict) and "fm_decoder" in hd else hd
        msg = model.fm_decoder.load_state_dict(hd, strict=False)
        print(f"[joint] head loaded (missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)})",
              flush=True)
    # gradient checkpointing only helps (and only works cleanly) when the LM actually trains.
    # For the frozen-LM variants (#2/#3) its forward runs under no_grad -> skip it.
    if config["model"].get("train_lm_backbone", True):
        model.autovla.vlm.model.gradient_checkpointing_enable()

    data_collator = DataCollator(processor=processor,
                                 ignore_index=config["model"]["tokens"]["ignore_index"],
                                 assistant_id=config["model"]["tokens"]["assistant_id"])
    train_dl = DataLoader(train_ds, batch_size=config["training"]["batch_size"],
                          collate_fn=data_collator, num_workers=config["training"]["num_workers"],
                          shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=config["inference"]["batch_size"],
                        collate_fn=data_collator, num_workers=config["inference"]["num_workers"],
                        shuffle=False)

    # bf16 cast (uniform dtype under FSDP MixedPrecision) -- FMHead AND scorer, matching
    # fmhead_sft.py's default FMHEAD_SFT_BF16=1 path.
    if os.environ.get("FMHEAD_SFT_BF16", "1") == "1":
        model.fm_decoder = model.fm_decoder.to(torch.bfloat16)
        if model.scorer is not None:
            model.scorer = model.scorer.to(torch.bfloat16)
        print("[joint] FMHead + scorer cast to bfloat16", flush=True)

    wrap_policy = functools.partial(transformer_auto_wrap_policy,
                                    transformer_layer_cls={Qwen2_5_VLDecoderLayer})
    save_dir = os.path.join(config["ckpt_dir"], datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(save_dir, exist_ok=True)

    trainer = pl.Trainer(
        num_nodes=1, max_epochs=config["training"]["epochs"], accelerator="gpu", devices="auto",
        accumulate_grad_batches=config["training"]["accumulate_grad_batches"],
        strategy=FSDPStrategy(
            auto_wrap_policy=wrap_policy, cpu_offload=False,
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16,
                                           buffer_dtype=torch.bfloat16),
            sharding_strategy="FULL_SHARD", backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            state_dict_type="full", limit_all_gathers=True, use_orig_params=True),
        callbacks=[
            # save_last=True guarantees a final ckpt even for variant #3 (frozen base+head ->
            # val_loss is CONSTANT, so a val_loss-monitored top_k/EarlyStopping is meaningless).
            # No EarlyStopping here: it would kill #3 immediately on the flat val_loss; all three
            # just run the full `epochs` (time is not the constraint).
            ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=3, dirpath=save_dir,
                            filename="epoch={epoch}-loss={val_loss:.4f}", auto_insert_metric_name=False,
                            save_weights_only=True, every_n_epochs=1, save_last=True),
            LearningRateMonitor(logging_interval="step"),
        ],
        gradient_clip_algorithm="value", gradient_clip_val=1.0,
        logger=CSVLogger(save_dir=save_dir), enable_model_summary=True,
    )
    torch.cuda.empty_cache()
    print(f"[joint] save_dir={save_dir} (scorer saved INSIDE epoch=*.ckpt under scorer.*; "
          f"extract with extract_scorer.py)", flush=True)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = args.config if os.path.isabs(args.config) else os.path.join(FMHEAD_DIR, args.config)
    build_and_fit_joint(cfg, seed=args.seed)
