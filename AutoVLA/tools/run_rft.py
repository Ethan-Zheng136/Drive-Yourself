import os
import sys
import math
import yaml
import torch
import argparse
import functools
from peft import get_peft_model, LoraConfig, TaskType

from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import FSDPStrategy

from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import BackwardPrefetch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

from models.autovla import GRPOAutoVLA
from dataset_utils.sft_dataset import SFTDataset, DataCollator
from transformers import AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer
import datetime
import warnings

warnings.filterwarnings("ignore", message=".*weights_only=False.*")


torch.set_float32_matmul_precision('high')

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


class GroupSampler(DistributedSampler):
    """
    Standard data-parallel DistributedSampler: each rank gets a DISJOINT, EVEN
    (equal-length) ~1/num_replicas shard of the dataset.

    GRPO grouping is now LOCAL to each rank (K=group_size rollouts of the SAME
    scene are drawn inside generate_sample), so ranks must see DIFFERENT scenes,
    i.e. ordinary data-parallel sharding. The OLD behavior (every rank gets the
    full index list / the same scene per step) was the throughput bug: 8 GPUs each
    processed all N scenes with zero data-parallel speedup.

    Sharding semantics (identical to torch.utils.data.DistributedSampler):
      - padded_total = ceil(N / num_replicas) * num_replicas
        => self.num_samples = padded_total // num_replicas (per-rank count)
        => self.total_size  = padded_total
      - __iter__ builds the (epoch-seeded, optionally shuffled) length-N index
        list, pads it up to total_size by repeating from the front, then takes the
        strided slice indices[rank : total_size : num_replicas].
    Every rank therefore yields EXACTLY self.num_samples indices. Equal per-rank
    length is MANDATORY under DDP: unequal counts would let the rank(s) that finish
    early hang at the next epoch-boundary all_reduce/all_gather (deadlock).

    set_epoch(epoch) (called by Lightning) updates self.epoch, which seeds the
    shuffle so each epoch differs.

    If the distributed process group is not initialized, falls back to
    single-process mode (num_replicas=1, rank=0 => the full dataset).
    """
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        # Resolve (num_replicas, rank): explicit args win (lets callers / unit tests build any
        # world without a live process group); otherwise read the default process group; otherwise
        # (dist not initialized AND not specified) fall back to single-process (full dataset).
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0

        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)
        # NOTE: do NOT override self.num_samples / self.total_size. The parent already
        # sets num_samples = ceil(N / num_replicas) (per-rank) and total_size = the EVEN
        # padded total. Overriding them to len(dataset) was the bug (every rank saw all N).

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # pad up to the EVEN total by repeating indices from the front.
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # drop the tail so the length is evenly divisible across replicas.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size, (len(indices), self.total_size)

        # DISJOINT, EVEN, strided shard for THIS rank.
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples, (len(indices), self.num_samples)
        return iter(indices)


from pytorch_lightning.callbacks import Callback as _PLCallback


class LoRASaveCallback(_PLCallback):
    """Save ONLY the peft LoRA adapter (~15MB) every N optimizer steps + at end. Rank-0 only."""
    def __init__(self, save_dir, every_n_steps=500):
        self.save_dir = save_dir
        self.every = every_n_steps
        self._last = -1

    def _save(self, pl_module, tag):
        out = os.path.join(self.save_dir, tag)
        vlm = pl_module.autovla.vlm
        # YouDrive de-merge: in the multi-adapter model, save ONLY the trainable "stage2" adapter
        # (never the frozen stage1). PEFT writes it to <out>/stage2/. Legacy single-adapter runs
        # ("default") are unaffected and still save to <out> directly.
        _cfgs = getattr(vlm, "peft_config", {}) or {}
        if "stage2" in _cfgs:
            vlm.save_pretrained(out, save_embedding_layers=False, selected_adapters=["stage2"])
            print(f"[youdrive-grpo] saved STAGE2 LoRA adapter -> {os.path.join(out, 'stage2')}", flush=True)
        else:
            vlm.save_pretrained(out, save_embedding_layers=False)
            print(f"[youdrive-grpo] saved LoRA adapter -> {out}", flush=True)

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
    parser.add_argument("--local-rank", type=int, default=0)

    args = parser.parse_args()
    seed_everything(args.seed)

    # Load configuration
    config = load_config(f"./config/{args.config}.yaml")

    # YouDrive: env overrides for quick ablations (all optional; empty => use config)
    if os.environ.get("DEVICES") not in (None, ""):
        config['training']['devices'] = int(os.environ["DEVICES"])
    _rl = config.setdefault('rl', {})
    if os.environ.get("KL_BETA") not in (None, ""):
        _rl['kl_beta'] = float(os.environ["KL_BETA"])
    if os.environ.get("REFERENCE_MODEL_PATH"):
        _rl['reference_model_path'] = os.environ["REFERENCE_MODEL_PATH"]
    # YouDrive de-merge (no-merge composable LoRA) Stage-2 GRPO knobs (all optional; empty => config).
    # SFT_MODEL_PATH lets the de-merge launch point sft_model_path at the RAW base ckpt without
    # editing the yaml; STAGE1_ADAPTER_PATH switches on the de-merge path (frozen stage1 + trainable stage2).
    if os.environ.get("SFT_MODEL_PATH"):
        config['model']['sft_model_path'] = os.environ["SFT_MODEL_PATH"]
    if os.environ.get("STAGE1_ADAPTER_PATH"):
        config['model']['stage1_adapter_path'] = os.environ["STAGE1_ADAPTER_PATH"]
    if os.environ.get("CE_USE") not in (None, ""):
        _rl.setdefault('ce', {})['use'] = (os.environ["CE_USE"] == "1")
    if os.environ.get("CE_WEIGHT") not in (None, ""):
        _rl.setdefault('ce', {})['weight'] = float(os.environ["CE_WEIGHT"])
    _rew = _rl.setdefault('reward', {})
    if os.environ.get("PDMS_USE") not in (None, ""):
        _rew.setdefault('pdms', {})['use'] = (os.environ["PDMS_USE"] == "1")
    if os.environ.get("PDMS_WEIGHT") not in (None, ""):
        _rew.setdefault('pdms', {})['weight'] = float(os.environ["PDMS_WEIGHT"])
    if os.environ.get("STYLE_USE") not in (None, ""):
        _rew.setdefault('style', {})['use'] = (os.environ["STYLE_USE"] == "1")
    if os.environ.get("STYLE_WEIGHT") not in (None, ""):
        _rew.setdefault('style', {})['weight'] = float(os.environ["STYLE_WEIGHT"])
    # Lagrangian constrained reward knobs (all optional; empty => config).
    _con = _rl.setdefault('constraint', {})
    if os.environ.get("CONSTRAINT_USE") not in (None, ""):
        _con['use'] = (os.environ["CONSTRAINT_USE"] == "1")
    if os.environ.get("PDMS_TARGET") not in (None, ""):
        _con['pdms_target'] = float(os.environ["PDMS_TARGET"])
    if os.environ.get("LAMBDA_INIT") not in (None, ""):
        _con['lambda_init'] = float(os.environ["LAMBDA_INIT"])
    if os.environ.get("LAMBDA_LR") not in (None, ""):
        _con['lambda_lr'] = float(os.environ["LAMBDA_LR"])
    if os.environ.get("LAMBDA_MAX") not in (None, ""):
        _con['lambda_max'] = float(os.environ["LAMBDA_MAX"])
    print(f"[youdrive-grpo] constraint: use={_con.get('use')} pdms_target={_con.get('pdms_target')} "
          f"lambda_init={_con.get('lambda_init')} lambda_lr={_con.get('lambda_lr')} lambda_max={_con.get('lambda_max')}", flush=True)
    print(f"[youdrive-grpo] rl config: kl_beta={_rl.get('kl_beta')} "
          f"pdms={_rew.get('pdms',{}).get('use')} style={_rew.get('style',{}).get('use')} "
          f"ref={_rl.get('reference_model_path') or config['model']['sft_model_path']}", flush=True)

    # Dataset and dataloader.
    # YouDrive: use SFTDataset (gives tokenized DDv2 targets for the CE backbone) and also
    # carry input_features+token (for the PDMS rollout). collate_fn = SFT DataCollator.
    #
    # SHARDING: we pass NO custom sampler and just shuffle=True, exactly like the working SFT
    # path (run_sft_lora.py). Lightning (use_distributed_sampler=True, the default) then injects a
    # correctly-configured DistributedSampler AFTER the DDP process group is initialized, so each
    # rank gets a DISJOINT ~N/world shard of DIFFERENT scenes. (Passing our own DistributedSampler
    # subclass here was the throughput bug: it is constructed BEFORE dist.init, so it reads
    # world_size=1 -> no shard, and because it is already a DistributedSampler subclass Lightning
    # refuses to re-wrap it -> every rank iterated the full dataset -> ~60h/epoch.) GRPO grouping is
    # LOCAL to each rank (K rollouts of the same scene inside generate_sample), so cross-rank
    # sampling is not needed and plain data-parallel sharding is exactly right.
    processor = AutoProcessor.from_pretrained(config['model']['pretrained_model_path'], use_fast=True)
    train_dataset = SFTDataset(config['data']['train'], config['model'], processor,
                               using_cot=config['model']['use_cot'])
    data_collator = DataCollator(
        processor=processor,
        ignore_index=config['model']['tokens']['ignore_index'],
        assistant_id=config['model']['tokens']['assistant_id'],
    )
    train_data = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        shuffle=True,
        collate_fn=data_collator,
    )

    # Model
    model = GRPOAutoVLA(config)
    # model.load_state_dict(
    #     torch.load(config['sft_model_path'])['state_dict'], 
    #     strict=False
    # )

    # TODO: remove this hard coding
    print(f"Loading and remapping checkpoint from: {config['model']['sft_model_path']}")
    full_checkpoint = torch.load(config['model']['sft_model_path'], map_location="cpu")
    sd = full_checkpoint['state_dict']
    
    # Load the state dict
    msg = model.load_state_dict(sd, strict=False)

    # Create a LoRA configuration. Adjust the parameters (r, lora_alpha, lora_dropout) as needed.
    if config['model']['lora'].get("use", False):
        print("Using LoRA mode for GRPO training.")
        lora_conf = config['model']['lora']
        lora_config = LoraConfig(
            task_type=TaskType[lora_conf.get("task_type", "CAUSAL_LM")],
            target_modules=lora_conf.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
            r=lora_conf.get("r", 8),
            lora_alpha=lora_conf.get("alpha", 8),
            lora_dropout=lora_conf.get("dropout", 0.1),
            bias=lora_conf.get("bias", "none")
        )
        _stage1_path = config['model'].get('stage1_adapter_path')
        if _stage1_path:
            # YouDrive de-merge (no-merge composable LoRA): raw base (already loaded above via
            # sft_model_path) + FROZEN always-on stage1 + fresh TRAINABLE stage2. Replaces the
            # init-from-merged-persona_init flow. Save ONLY stage2. KL ref = stage1-only forward.
            from peft import PeftModel
            print(f"[youdrive-grpo] DE-MERGE path: raw base + FROZEN stage1 ({_stage1_path}) + TRAINABLE stage2")
            model.autovla.vlm = PeftModel.from_pretrained(
                model.autovla.vlm, _stage1_path, adapter_name="stage1", is_trainable=False)
            model.autovla.vlm.add_adapter("stage2", lora_config)   # zero-init => starts == base+stage1
            # activate via tuner: PeftModel.set_adapter() rejects lists; base_model does multi-active
            model.autovla.vlm.base_model.set_adapter(["stage1", "stage2"])  # both active (additive)
            model._freeze_stage1()                                 # only stage2 trainable
            _tr = sum(p.numel() for n, p in model.autovla.vlm.named_parameters() if p.requires_grad)
            _s1 = sum(p.numel() for n, p in model.autovla.vlm.named_parameters() if ".stage1." in n)
            _s2 = sum(p.numel() for n, p in model.autovla.vlm.named_parameters() if ".stage2." in n)
            print(f"[youdrive-grpo] de-merge trainable params={_tr}  (stage2={_s2}, stage1[FROZEN]={_s1})")
        else:
            model.autovla.vlm = get_peft_model(model.autovla.vlm, lora_config)
        print("LoRA-enabled model trainable parameters:",
              sum(p.numel() for p in model.autovla.vlm.parameters() if p.requires_grad))
    model = model.to(torch.bfloat16)

    # Training
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Qwen2_5_VLDecoderLayer
        },
    )

    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.environ.get("SAVE_DIR",
        f"/mnt/pfs/zhengguantian/autovla/persona/grpo_ckpts/{current_date}")
    os.makedirs(save_dir, exist_ok=True)
    _save_every = int(os.environ.get("SAVE_EVERY", "500"))
    
    # YouDrive: multi-node support. Cluster torchrun sets WORLD_SIZE/LOCAL_WORLD_SIZE;
    # mirror run_sft_lora.py so 8x8 / 16x8 jobs work without editing the config.
    _ws = int(os.environ.get("WORLD_SIZE", "0"))
    _lws = int(os.environ.get("LOCAL_WORLD_SIZE", "0"))
    if _ws > 0 and _lws > 0:
        _num_nodes = max(_ws // _lws, 1)
        _devices = _lws
    else:
        _num_nodes = 1
        _devices = config['training']['devices']
    print(f"[youdrive-grpo] num_nodes={_num_nodes} devices={_devices} (WORLD_SIZE={_ws} LOCAL_WORLD_SIZE={_lws})", flush=True)

    _max_steps = int(os.environ.get("MAX_STEPS", "-1"))   # YouDrive: SMOKE cap; -1 = unlimited
    trainer = Trainer(
        num_nodes=_num_nodes,
        max_epochs=config['training']['epochs'],
        max_steps=_max_steps,
        accelerator="gpu",
        devices=_devices, 
        num_sanity_val_steps=0,
        # YouDrive: LoRA model (frozen base + small adapter) fits per-GPU -> use DDP like the SFT path.
        # FSDP FULL_SHARD + autoregressive .generate() produced nan/inf in the sampling step (rollout);
        # DDP avoids the sharded-param generation instability. base is frozen, so no memory issue.
        strategy="ddp_find_unused_parameters_true",
        precision="bf16-true",
        # YouDrive: disable Lightning's AUTO ModelCheckpoint. With no explicit ModelCheckpoint in
        # callbacks, Lightning still injects a default one that saves the full ~7.5GB model at epoch
        # end to the logger dir -> this is what hit `Errno 122 Disk quota exceeded` and killed the
        # 9.5h run at the very end. LoRASaveCallback already persists the ~15MB adapter to PFS.
        enable_checkpointing=False,
        callbacks=[
            # YouDrive: save ONLY the LoRA adapter (~15MB) every N steps + at end, to PFS (save_dir).
            LoRASaveCallback(save_dir, every_n_steps=_save_every),
            LearningRateMonitor(logging_interval="step")
        ],
        # YouDrive: logs go to PFS, NEVER local repo `runs/` (local disk is only ~200G, fills fast).
        logger=[CSVLogger(save_dir=save_dir), TensorBoardLogger(save_dir=save_dir)],
        enable_model_summary=True,
        log_every_n_steps=1,
        # NOTE: GRPOAutoVLA uses MANUAL optimization (it micro-batches the K-rollout policy
        # backward), which is incompatible with Lightning's automatic gradient clipping. The
        # equivalent value-clip (clip_value=1.0) is applied manually inside training_step.
        limit_val_batches=0
    )

    trainer.fit(model, train_dataloaders=train_data)
