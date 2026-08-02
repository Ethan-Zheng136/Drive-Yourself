"""fmhead_sft.py -- FAITHFUL replica of AutoVLA full-FT SFT (tools/run_sft.py), with the
codebook/action-token CE decoder REPLACED by the FMHead + flow-matching loss.

"Complete training" contender: the VLM LM-backbone (vlm.model + lm_head) AND the FMHead
are trained JOINTLY -- gradients DO flow into the LM backbone (NOT no_grad, unlike the
frozen feasibility trainer train_fmhead.py). Vision tower frozen. Style axis OFF (s=0).

Faithful to run_sft.py:
  * reuses SFTDataset + DataCollator + SFTAutoVLA machinery unchanged;
  * Lightning Trainer + FSDPStrategy FULL_SHARD, transformer_auto_wrap_policy on
    Qwen2_5_VLDecoderLayer only (so the FMHead is NOT its own FSDP unit), bf16
    MixedPrecision (param/reduce/buffer), BackwardPrefetch.BACKWARD_PRE,
    state_dict_type="full", gradient_checkpointing, grad clip value=1.0,
    num_nodes=1 devices='auto';
  * same optimizer/schedule as qwen2.5-vl-3B-mix-sft (AdamW lr 2e-5, wd 0.01,
    5 epochs, bs 1, accum 4, warmup 500) + ModelCheckpoint(monitor="val_loss").

Only differences vs run_sft.py: the training/validation loss is the FMHead flow
loss on the action-region hidden c (s=0), the codebook-CE path is removed, and the
trainable set additionally includes the FMHead.
"""

from __future__ import annotations

import argparse
import datetime
import functools
import os
import sys

import torch
import yaml

AUTOVLA_ROOT = "/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA"
FMHEAD_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (AUTOVLA_ROOT, os.path.join(AUTOVLA_ROOT, "navsim"), FMHEAD_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.autovla import SFTAutoVLA  # noqa: E402
from autovla_fmhead import (  # noqa: E402
    FMHeadDecoder, answer_position_hidden, context_sequence_hidden, autovla_fmhead_config,
)

torch.set_float32_matmul_precision("high")

# keys carried in the batch that are NOT VLM forward kwargs (mirrors AutoVLA.forward)
_DROP = ("gt_trajectory", "gt_action", "has_cot", "input_features", "token",
         "data_path", "text", "image_inputs", "video_inputs", "labels")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# =============================================================================
# The decoder-swapped SFT LightningModule
# =============================================================================
class FMHeadSFTAutoVLA(SFTAutoVLA):
    """SFTAutoVLA whose training/val loss is the FMHead flow-matching loss (s=0),
    replacing the codebook/action-token CE. LM backbone trained jointly."""

    def __init__(self, config: dict):
        super().__init__(config)  # builds self.autovla (Qwen2.5-VL-3B) + SFT machinery
        fh = config["fmhead"]
        # fm3 (ADDITIVE): pass the optional feasibility constraint block. Absent / mode:none
        # -> byte-identical fm2 head. See fm_head.FMHeadConfig / autovla_fmhead_config.
        fm_cfg = autovla_fmhead_config(hidden_size=fh["hidden_size"], depth=fh["depth"],
                                       style_dropout_prob=fh["style_dropout_prob"],
                                       constraint=fh.get("constraint"))
        fm_cfg.num_heads = fh["num_heads"]; fm_cfg.mlp_ratio = fh["mlp_ratio"]
        fm_cfg.cond_hidden = fh["cond_hidden"]; fm_cfg.horizon = fh["horizon"]
        fm_cfg.traj_dim = fh["traj_dim"]
        print(f"[sft] fmhead constraint_mode={fm_cfg.constraint_mode} "
              f"jerk_w={fm_cfg.jerk_weight} curv_w={fm_cfg.curv_weight} "
              f"accel_max={fm_cfg.accel_max} yawrate_max={fm_cfg.yawrate_max}", flush=True)
        self.fm_decoder = FMHeadDecoder(fm_cfg, num_samples=fh["num_samples"],
                                        num_steps=fh["num_steps"], cfg_weight=fh["cfg_weight"],
                                        style_alpha=config.get("style_alpha", 0.0))
        self.fm_decoder.load_normalizer(config["normalizer_path"])
        self._c_reduce = config.get("c_reduce", "mean")
        self._asid = self.autovla.action_start_id
        self._n_bins = self.autovla.action_tokenizer.n_bins
        self._setup_trainable()

    # -- freeze semantics (mirrors SFTAutoVLA.configure_optimizers) -------------
    def _setup_trainable(self):
        if not self._train_vision_backbone:
            for p in self.autovla.vlm.visual.parameters():
                p.requires_grad_(False)
        if not self._train_llm_backbone:
            for p in self.autovla.vlm.model.parameters():
                p.requires_grad_(False)
        # FMHead params stay trainable (except its frozen sin-cos temporal_pos buffer/param)

    # -- FMHead flow loss on the action-region hidden (WITH grad into LM) -------
    def _fm_loss(self, batch):
        dev = next(self.parameters()).device
        inputs = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        vlm_in = {k: v for k, v in inputs.items() if k not in _DROP}
        # NOTE: no torch.no_grad() -> gradients flow back into vlm.model (LM backbone).
        outputs = self.autovla.vlm(**vlm_in, output_hidden_states=True, use_cache=False)
        h = outputs.hidden_states[-1]                                   # (B, S, 2048), grad-enabled
        # v2: multi-token context SEQUENCE over the prompt span (cross-attn content),
        # identical in train & inference (positions <= generation-start anchor).
        ctx, ctx_mask = context_sequence_hidden(h, inputs["input_ids"],
                                                attention_mask=inputs.get("attention_mask"),
                                                action_start_id=self._asid,
                                                max_len=self.fm_decoder.config.ctx_max_len)
        anchor = answer_position_hidden(h, inputs["input_ids"],
                                        attention_mask=inputs.get("attention_mask"),
                                        action_start_id=self._asid)
        s = torch.zeros_like(anchor)                                    # style axis OFF (feasibility)
        return self.fm_decoder(inputs["gt_trajectory"], ctx, s, ctx_mask=ctx_mask)  # fwd == loss

    def training_step(self, batch, batch_idx=0):
        loss = self._fm_loss(batch)
        self.log("train_loss", loss.item(), batch_size=1, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx=0):
        loss = self._fm_loss(batch)
        self.log("val_loss", loss.item(), batch_size=1, sync_dist=True, prog_bar=True)
        return loss

    # -- optimizer/schedule: mirror SFTAutoVLA but include the FMHead params -----
    def configure_optimizers(self):
        self._setup_trainable()
        params_to_update = [p for p in self.parameters() if p.requires_grad]
        assert params_to_update, "No parameters to update"
        optimizer = torch.optim.AdamW(
            params_to_update, lr=self.cfg["training"]["learning_rate"],
            weight_decay=self.cfg["training"].get("weight_decay", 0.0))
        warmup = self.cfg["training"]["lr_warmup_step"]
        step_freq = self.cfg["training"]["lr_step_frequency"]
        gamma = self.cfg["training"]["lr_step_gamma"]

        def lr_update(step, warmup_step, step_size, g):
            if step < warmup_step:
                lr_scale = 1 - (warmup_step - step) / warmup_step * 0.95
            else:
                n = (step - warmup_step) // step_size
                lr_scale = g ** n
            return min(max(lr_scale, 1e-2), 1.0)

        sched = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda s: lr_update(s, warmup, step_freq, gamma))
        return [optimizer], [{"scheduler": sched, "interval": "step"}]


# =============================================================================
# Trainer (faithful to run_sft.py) + main
# =============================================================================
def build_and_fit(config_path, seed=42):
    import pytorch_lightning as pl
    from pytorch_lightning import seed_everything
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from pytorch_lightning.strategies import FSDPStrategy
    from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch.utils.data import DataLoader
    from transformers import AutoProcessor
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer
    from dataset_utils.sft_dataset import SFTDataset, DataCollator

    seed_everything(seed)
    os.chdir(AUTOVLA_ROOT)  # codebook_cache_path is repo-relative
    config = load_config(config_path)

    processor = AutoProcessor.from_pretrained(config["model"]["pretrained_model_path"], use_fast=True)
    using_cot = config["model"]["use_cot"]
    train_ds = SFTDataset(config["data"]["train"], config["model"], processor, using_cot=using_cot)
    val_ds = SFTDataset(config["data"]["val"], config["model"], processor, using_cot=using_cot)

    model = FMHeadSFTAutoVLA(config)
    # optional warm-start of the LM backbone from the AutoVLA SFT/PDMS base
    base = config["model"].get("sft_model_path")
    if base and str(base).lower() not in ("none", "null", ""):
        print(f"[sft] warm-start base: {base}", flush=True)
        sd = torch.load(base, map_location="cpu")["state_dict"]
        sd = {k.replace("autovla.", "", 1) if k.startswith("autovla.") else k: v for k, v in sd.items()}
        # load into the inner AutoVLA (fm_decoder.* absent from base -> strict=False)
        msg = model.autovla.load_state_dict(sd, strict=False)
        print(f"[sft] base loaded (missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)})",
              flush=True)
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

    # FMHead is NOT a Qwen2_5_VLDecoderLayer -> excluded from auto-wrap (not its own
    # FSDP unit). Only the Qwen decoder layers become sharded FSDP units, exactly as
    # run_sft.py. The root FSDP then flat-shards the leftover params (lm_head/embeds +
    # FMHead). Qwen params are bf16 but the FMHead is fp32 -> with the default
    # use_orig_params=False, FSDP builds one uniform-dtype FlatParameter and dies with
    #   "Must flatten tensors with uniform dtype but got torch.bfloat16 and torch.float32".
    #
    # FIX (Option B, chosen): use_orig_params=True. FSDP then keeps the ORIGINAL params
    # (no single uniform-dtype FlatParameter), so a bf16-VLM + fp32-FMHead mix is allowed;
    # the optimizer also sees original params (fp32 Adam moments for the head). Least
    # invasive, keeps FMHead master weights fp32, and all other run_sft.py settings intact.
    # (MixedPrecision(param_dtype=bf16) still casts params to bf16 for COMPUTE, so numerics
    # match the faithful bf16-mixed run; the fp32 master is only for storage/optimizer.)
    #
    # DTYPE (BUG-B fix): this is a bf16-mixed FSDP run (MixedPrecision(param_dtype=bf16)).
    # With use_orig_params=True the ORIGINAL fp32 FMHead params are exposed to the module
    # (so next(params).dtype reports fp32) while FSDP does the COMPUTE in bf16 -> the head's
    # own per-input dtype casts can't reconcile reported!=compute, giving
    #   RuntimeError: mat1 and mat2 must have the same dtype (Float vs BFloat16).
    # Cast the FMHead to bf16 so reported dtype == compute dtype == the VLM's bf16: the head
    # is then uniformly bf16, fm_head.py casts all its inputs to bf16 (next(params).dtype),
    # and the flow-matching loss is still computed in fp32 (F.mse_loss(v.float(), u.float())).
    # Set FMHEAD_SFT_BF16=0 to keep the head fp32 (relies solely on use_orig_params + the
    # per-input casts; only try this if a node specifically needs an fp32 master head).
    if os.environ.get("FMHEAD_SFT_BF16", "1") == "1":
        model.fm_decoder = model.fm_decoder.to(torch.bfloat16)
        print("[sft] FMHead cast to bfloat16 (uniform dtype under FSDP MixedPrecision)", flush=True)
    else:
        print("[sft] FMHead kept fp32 (relying on use_orig_params + per-input dtype casts)", flush=True)

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
            state_dict_type="full", limit_all_gathers=True,
            use_orig_params=True),  # <-- BUG-2 fix: allow bf16 VLM + fp32 FMHead mixed dtypes
        callbacks=[
            ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=3, dirpath=save_dir,
                            filename="epoch={epoch}-loss={val_loss:.4f}", auto_insert_metric_name=False,
                            save_weights_only=True, every_n_epochs=1),
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
            LearningRateMonitor(logging_interval="step"),
        ],
        gradient_clip_algorithm="value", gradient_clip_val=1.0,
        logger=CSVLogger(save_dir=save_dir), enable_model_summary=True,
    )
    torch.cuda.empty_cache()
    print(f"[sft] save_dir={save_dir}", flush=True)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = args.config if os.path.isabs(args.config) else os.path.join(FMHEAD_DIR, args.config)
    build_and_fit(cfg, seed=args.seed)
