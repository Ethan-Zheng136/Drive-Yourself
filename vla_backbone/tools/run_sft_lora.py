import sys
import os
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "navsim"))

import yaml
import torch
import argparse
import functools
import pytorch_lightning as pl

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import FSDPStrategy

from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import BackwardPrefetch
from torch.utils.data import DataLoader

from dataset_utils.sft_dataset import SFTDataset, DataCollator
from models.autovla import SFTAutoVLA
from transformers import AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer
import datetime

torch.set_float32_matmul_precision('high')


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


class LoRASaveCallback(pl.Callback):
    """Save ONLY the peft LoRA adapter (~tiny) every N optimizer steps + at end. Rank-0 only."""
    def __init__(self, save_dir, every_n_steps=200):
        self.save_dir = save_dir
        self.every = every_n_steps
        self._last = -1
    def _save(self, pl_module, tag):
        import os as _o
        out = _o.path.join(self.save_dir, tag)
        pl_module.autovla.vlm.save_pretrained(out, save_embedding_layers=False)  # pure LoRA A/B only (~15MB)
        print(f"[youdrive] saved LoRA adapter -> {out}", flush=True)
    def on_train_batch_end(self, trainer, pl_module, *a, **k):
        s = trainer.global_step
        if trainer.is_global_zero and s > 0 and s % self.every == 0 and s != self._last:
            self._last = s
            self._save(pl_module, f"lora_step{s}")
    def on_train_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            self._save(pl_module, "lora_final")


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    seed_everything(args.seed)

    # Load configuration
    config = load_config(f"./config/{args.config}.yaml")

    # Model, dataset, and dataloader
    processor = AutoProcessor.from_pretrained(config['model']['pretrained_model_path'], use_fast=True)
    
    # Get using_cot setting from config (default to True if not specified)
    using_cot = config['model']['use_cot']
    

    train_dataset = SFTDataset(config['data']['train'], config['model'], processor, using_cot=using_cot)
        
    # Randomly sample from training set if train_sample_size is specified
    train_sample_size = config['training']['train_sample_size']
    if train_sample_size is not None and len(train_dataset) > train_sample_size:
        indices = torch.randperm(len(train_dataset))[:train_sample_size]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
    else:
        print("no sampling")
        
    val_dataset = SFTDataset(config['data']['val'], config['model'], processor, using_cot=using_cot)

    from peft import get_peft_model, LoraConfig, TaskType
    model = SFTAutoVLA(config)
    # gradient checkpointing conflicts with FSDP+LoRA writeback ("Cannot writeback when shape changes");
    # run_rft.py runs LoRA without it. Re-enable with use_reentrant=False if OOM.
    if int(os.environ.get("GRAD_CKPT", "1")):   # DDP-safe; saves memory
        model.autovla.vlm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # --- load AutoVLA base checkpoint (init), then add a fresh trainable LoRA ---
    base_ckpt = config['model'].get('sft_model_path', None)
    if base_ckpt:
        print(f"[youdrive] loading base ckpt: {base_ckpt}", flush=True)
        sd = torch.load(base_ckpt, map_location="cpu")['state_dict']
        msg = model.load_state_dict(sd, strict=False)
        print(f"[youdrive] base loaded. missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}", flush=True)

    lconf = config['model'].get('lora', {})
    if lconf.get('use', False):
        lora_config = LoraConfig(
            task_type=TaskType[lconf.get('task_type', 'CAUSAL_LM')],
            target_modules=lconf.get('target_modules', ["q_proj", "v_proj", "k_proj", "o_proj"]),
            r=lconf.get('r', 16), lora_alpha=lconf.get('alpha', 32),
            lora_dropout=lconf.get('dropout', 0.05), bias=lconf.get('bias', 'none'),
        )
        model.autovla.vlm = get_peft_model(model.autovla.vlm, lora_config)
        tot = sum(p.numel() for p in model.parameters())
        tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[youdrive] LoRA on. trainable={tr:,} / total={tot:,} ({100*tr/tot:.3f}%)", flush=True)
    model = model.to(torch.bfloat16)   # uniform dtype for FSDP flatten (base + LoRA)
    
    # Create data collator with config parameters
    data_collator = DataCollator(
        processor=processor,
        ignore_index=config['model']['tokens']['ignore_index'],
        assistant_id=config['model']['tokens']['assistant_id']
    )
    
    train_data = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        collate_fn=data_collator,
        num_workers=config['training']['num_workers'],
        shuffle=True,
    )

    val_data = DataLoader(
        val_dataset,
        batch_size=config['inference']['batch_size'],
        collate_fn=data_collator,
        num_workers=config['inference']['num_workers'],
        shuffle=False,
    )    

    # Training
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Qwen2_5_VLDecoderLayer
        },
    )

    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.environ.get("SAVE_DIR", f"/mnt/pfs/zhengguantian/autovla/persona/lora_ckpts/{current_date}")
    os.makedirs(save_dir, exist_ok=True)
    _save_every = int(os.environ.get("SAVE_EVERY", "200"))
    
    _smoke = bool(int(os.environ.get("SMOKE", "0")))
    # torchrun sets WORLD_SIZE=total procs, LOCAL_WORLD_SIZE=procs per node.
    # PL needs num_nodes=#nodes and devices=#gpus-per-node.
    _ws = int(os.environ.get("WORLD_SIZE", "0"))
    _lws = int(os.environ.get("LOCAL_WORLD_SIZE", "0"))
    if _ws > 0 and _lws > 0:
        _num_nodes = max(_ws // _lws, 1)
        _devices = _lws
    else:
        _num_nodes = 1
        _devices = 'auto'
    print(f"[youdrive] num_nodes={_num_nodes} devices={_devices} (WORLD_SIZE={_ws} LOCAL_WORLD_SIZE={_lws})", flush=True)
    trainer = pl.Trainer(
        num_nodes=_num_nodes,
        devices=_devices,
        fast_dev_run=_smoke,            # SMOKE=1 -> 1 train batch then stop
        max_epochs=config['training']['epochs'],
        accelerator="gpu",
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        # LoRA-only (7M trainable) fits per-GPU -> DDP avoids all FSDP+PEFT flat-param/writeback issues.
        strategy="ddp_find_unused_parameters_true",
        precision="bf16-true",
        num_sanity_val_steps=0,
        limit_val_batches=0,   # skip val; we select checkpoints by YDSP style-shift, not val_loss
        callbacks=[
            LoRASaveCallback(save_dir, every_n_steps=_save_every),
            LearningRateMonitor(logging_interval="step"),
        ],
        gradient_clip_algorithm = 'value',
        gradient_clip_val = 1.0,

        logger=CSVLogger(save_dir=f"{save_dir}"),
        enable_model_summary=True,

        # limit_val_batches=0.001
    )
    torch.cuda.empty_cache()
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)