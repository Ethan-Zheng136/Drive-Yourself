"""train_fmhead.py -- train the FMHead decoder that REPLACES AutoVLA's codebook decoder.

Reuses the frozen AutoVLA pre-decoder stack (base Qwen2.5-VL-3B bf16 + trained
persona LoRA) and trains ONLY the FMHead decoder ("the back part"). Mirrors the
structure of `tools/run_sft_lora.py` (config-driven, same dataset/collator, PFS
checkpoints) but:

  * the VLM (and its LoRA) are FROZEN and run under torch.no_grad() -> no VLM
    activations retained, only FMHead gets gradients/optimizer;
  * per step we extract c = action-region hidden (2048, bf16) and the style delta
    s = alpha*(h_styled - h_base) via `vlm.disable_adapter()`, then optimize
    `FMHeadDecoder(gt_trajectory[..., :2], c, s)` (== training_loss).

Multi-GPU / multi-node (torchrun):
  * Each rank builds its OWN frozen VLM (bf16, ~8GB) on cuda:LOCAL_RANK + the tiny
    FMHead. ONLY the FMHead decoder is wrapped in DistributedDataParallel; the
    frozen VLM stays OUTSIDE DDP and runs under no_grad (never in DDP).
  * DistributedSampler shards the real SFTDataset so the ranks see disjoint scenes.
  * Logging + checkpointing happen on rank 0 only (barriers around save).
  * Running WITHOUT torchrun (WORLD_SIZE unset) falls back to the single-GPU path.

Usage:
  # single GPU (unchanged):
  OMP_NUM_THREADS=8 python train_fmhead.py --config config/fmhead_ddv2.yaml
  # single node, 8 GPU:
  torchrun --standalone --nproc_per_node=8 train_fmhead.py --config config/fmhead_ddv2.yaml
  # 2 nodes x 8 GPU: see launch_2x8.sh
  # smoke on this box (NVML/vision restricted -> text-only VLM forward):
  OMP_NUM_THREADS=8 FMHEAD_TEXT_ONLY=1 torchrun --standalone --nproc_per_node=2 \
      train_fmhead.py --config config/fmhead_ddv2.yaml --smoke --no_sft_base
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import yaml

AUTOVLA_ROOT = "/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA"
FMHEAD_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (AUTOVLA_ROOT, os.path.join(AUTOVLA_ROOT, "navsim"), FMHEAD_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from autovla_fmhead import (  # noqa: E402
    AUTOVLA_HIDDEN_SIZE, AUTOVLA_VLM_DTYPE, FMHeadDecoder, answer_position_hidden,
    context_sequence_hidden, autovla_fmhead_config, style_delta,
)

torch.set_float32_matmul_precision("high")


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


# =============================================================================
# Distributed setup (torchrun); no-op single-GPU fallback
# =============================================================================
class DistCtx:
    """Holds distributed state. When not launched by torchrun, acts single-GPU."""

    def __init__(self):
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.enabled = self.world_size > 1
        # Production: one GPU per rank -> cuda:LOCAL_RANK. On a box with fewer GPUs
        # than ranks (e.g. this single-MIG test box), clamp so multiple ranks share
        # a device -- lets the DDP path be exercised without N physical GPUs.
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            self.dev_index = self.local_rank if self.local_rank < n else (self.local_rank % n)
            self.device = torch.device(f"cuda:{self.dev_index}")
            self.shared_gpu = self.dev_index != self.local_rank
        else:
            self.dev_index = 0
            self.device = torch.device("cpu")
            self.shared_gpu = False
        # nccl for real multi-GPU; gloo allowed for the same-GPU / CPU test path.
        self.backend = os.environ.get(
            "FMHEAD_DDP_BACKEND", "nccl" if torch.cuda.is_available() else "gloo")

    def init(self):
        if self.enabled:
            if torch.cuda.is_available():
                torch.cuda.set_device(self.dev_index)
            dist.init_process_group(backend=self.backend)
            dist.barrier()
        return self

    @property
    def is_main(self):
        return self.rank == 0

    def log(self, *a, **k):
        if self.is_main:
            print(*a, **k, flush=True)

    def barrier(self):
        if self.enabled:
            dist.barrier()

    def cleanup(self):
        if self.enabled and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


# =============================================================================
# Frozen AutoVLA stack (base VLM + persona LoRA)
# =============================================================================
def _lora_configured(cfg):
    """True iff a real persona LoRA path is configured (style axis ON)."""
    lp = cfg.get("lora_adapter_path")
    return bool(lp) and str(lp).strip().lower() not in ("none", "null", "")


def build_frozen_vlm(cfg, device, load_sft_base=True, log=print, base_ckpt_override=None):
    """Construct AutoVLA, (optionally) load the SFT/PDMS base, (optionally) attach the
    persona LoRA, and FREEZE everything. Returns (autovla_module, av_config, lora_on).

    ``base_ckpt_override`` (e.g. the full-FT extracted ``full_ft_base.ckpt``) replaces
    the config's ``sft_model_path`` so the TRAINED VLM is used as the base instead of
    AutoVLA_PDMS_89 -- required to evaluate the full-FT model. None -> config default.

    When ``lora_adapter_path`` is null/none the persona LoRA is NOT attached -> the
    style axis is OFF (s=0), and hidden extraction uses a SINGLE VLM forward (no
    disable_adapter, no 2x cost). This is the "normal AutoVLA + human GT" regime."""
    os.chdir(AUTOVLA_ROOT)  # codebook_cache_path etc. are repo-relative
    av_config = load_yaml(os.path.join(AUTOVLA_ROOT, "config", cfg["autovla_config"] + ".yaml"))

    from models.autovla import AutoVLA
    t0 = time.time()
    av = AutoVLA(av_config, device=device)          # loads Qwen2.5-VL-3B bf16 + processor + action_tokenizer
    log(f"[vlm] AutoVLA(Qwen2.5-VL-3B) built in {time.time()-t0:.1f}s on {device}")

    base_ckpt = base_ckpt_override or av_config["model"].get("sft_model_path")
    if load_sft_base and base_ckpt:
        log(f"[vlm] loading base: {base_ckpt}"
            + ("  (OVERRIDE: full-FT trained VLM)" if base_ckpt_override else "  (config sft_model_path)"))
        sd = torch.load(base_ckpt, map_location="cpu")["state_dict"]
        sd = {k.replace("autovla.", "").replace("drivevla.", ""): v for k, v in sd.items()}
        msg = av.load_state_dict(sd, strict=False)
        log(f"[vlm] base loaded (missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)})")

    lora_on = _lora_configured(cfg)
    if lora_on:
        from peft import PeftModel
        lora_path = cfg["lora_adapter_path"]
        log(f"[vlm] attaching persona LoRA (style axis ON): {lora_path}")
        av.vlm = PeftModel.from_pretrained(av.vlm, lora_path, is_trainable=False)
    else:
        log("[vlm] NO persona LoRA (style axis OFF, s=0) -> single VLM forward per step")

    # FREEZE the whole pre-decoder stack
    n_frozen = 0
    for p in av.parameters():
        p.requires_grad_(False)
        n_frozen += p.numel()
    av.eval()
    av.to(device)
    log(f"[vlm] FROZEN: {n_frozen/1e9:.2f}B params (VLM{'+LoRA' if lora_on else ''}), requires_grad=False")
    return av, av_config, lora_on


# =============================================================================
# c / s extraction (frozen; no grad)
# =============================================================================
@torch.no_grad()
def extract_c_s(av, model_inputs, alpha, reduce="mean", style_off=False, ctx_max_len=64,
                content_source=None):
    """v2: return (ctx, ctx_mask, s).

    ctx (B,T_ctx,2048) = VLM hidden SEQUENCE over the prompt span up to the
    generation-start anchor (multi-token content, cross-attn); ctx_mask (B,T_ctx).
    s (B,2048) = style axis from the anchor vector. Both use only positions <= anchor
    so they are IDENTICAL in training (teacher-forced) and inference (agent).

    style_off=True (no LoRA): a SINGLE VLM forward, s = zeros. style_off=False: two
    forwards (LoRA on/off), s = alpha*(anchor_styled - anchor_base).

    ``content_source`` (ADDITIVE) selects which forward provides the CROSS-ATTN CONTENT:
      * None  -> LEGACY behavior == "base" (content from the LoRA-OFF forward); this is what
                 this trainer has always done, so absent-key runs are byte-identical.
      * "base"   -> content from the LoRA-OFF (neutral) forward -> body stays LoRA-free.
      * "styled" -> content from the LoRA-ON forward.
    The style delta ``s = alpha*(styled_anchor - base_anchor)`` is IDENTICAL for all values."""
    asid = av.action_start_id
    keep = {k: v for k, v in model_inputs.items()
            if k not in ("gt_trajectory", "gt_action", "has_cot", "input_features",
                         "token", "data_path", "text", "image_inputs", "video_inputs")}

    def _hidden(disable):
        cx = av.vlm.disable_adapter() if disable else contextlib.nullcontext()
        with cx:
            out = av.vlm(**keep, output_hidden_states=True, use_cache=False)
        return out.hidden_states[-1]

    cs = content_source if content_source is not None else "base"   # LEGACY default = base

    # base forward: content (legacy) + base anchor. style_off (no LoRA): disable_adapter is a
    # nullcontext -> a SINGLE forward, s=0 (feasibility/PDMS regime unchanged).
    h_base = _hidden(True)                # base hidden (LoRA off)
    anchor_base = answer_position_hidden(h_base, keep["input_ids"],
                                         attention_mask=keep.get("attention_mask"), action_start_id=asid)
    if style_off:
        h_content = h_base                 # content_source irrelevant (base==styled, no LoRA)
        s = torch.zeros_like(anchor_base)  # style axis OFF
    else:
        h_styled = _hidden(False)          # styled (LoRA on) -> +1 forward for the delta
        anchor_styled = answer_position_hidden(h_styled, keep["input_ids"],
                                               attention_mask=keep.get("attention_mask"), action_start_id=asid)
        s = style_delta(anchor_styled, anchor_base, alpha=alpha)
        h_content = h_styled if cs == "styled" else h_base
    ctx, ctx_mask = context_sequence_hidden(h_content, keep["input_ids"],
                                            attention_mask=keep.get("attention_mask"),
                                            action_start_id=asid, max_len=ctx_max_len)
    return ctx, ctx_mask, s


# =============================================================================
# Decoder (trainable)
# =============================================================================
def build_decoder(cfg, device):
    fh = cfg["fmhead"]
    use_style_adapter = bool(fh.get("use_style_adapter", False))
    # fm3 (ADDITIVE): pass the optional feasibility/kinematic constraint block. Absent /
    # {"mode":"none"} -> byte-identical fm2 head (old xy style path unchanged). Present with
    # mode:kinematic -> control-space flow + unicycle decode (required for the fm3-kin base,
    # whose head was trained in control space with the CONTROL normalizer).
    fm_cfg = autovla_fmhead_config(hidden_size=fh["hidden_size"], depth=fh["depth"],
                                   style_dropout_prob=fh["style_dropout_prob"],
                                   use_style_adapter=use_style_adapter,
                                   style_adapter_hidden=fh.get("style_adapter_hidden", 256),
                                   style_adapter_mode=fh.get("style_adapter_mode", "adaln"),
                                   constraint=fh.get("constraint"),
                                   use_z_encoder=bool(fh.get("use_z_encoder", False)),
                                   z_encoder_dims=tuple(fh.get("z_encoder_dims", (512, 256))),
                                   z_norm_delta=bool(fh.get("z_norm_delta", False)))
    fm_cfg.num_heads = fh["num_heads"]; fm_cfg.mlp_ratio = fh["mlp_ratio"]
    fm_cfg.cond_hidden = fh["cond_hidden"]; fm_cfg.horizon = fh["horizon"]
    fm_cfg.traj_dim = fh["traj_dim"]
    dec = FMHeadDecoder(fm_cfg, num_samples=fh["num_samples"], num_steps=fh["num_steps"],
                        cfg_weight=fh["cfg_weight"], style_alpha=cfg["style_alpha"],
                        content_source=(cfg.get("content_source") or "styled")).to(device)
    dec.load_normalizer(cfg["normalizer_path"])
    # STYLE phase: warm-start the head from a strong content ckpt (e.g. full-FT v2
    # fm_decoder.pt) so cross-attn content is already learned and style training only has
    # to shape the AdaLN/style path. strict=False tolerates the normalizer buffers AND the
    # new style-adapter keys (absent in the 0.91 ckpt -> stay zero-init).
    init_ck = cfg.get("init_fmhead_ckpt")
    if init_ck and str(init_ck).strip().lower() not in ("none", "null", ""):
        ck = torch.load(init_ck, map_location=device)
        sd = ck["fm_decoder"] if "fm_decoder" in ck else ck
        msg = dec.load_state_dict(sd, strict=False)
        _is_new = lambda k: ("style_adapter" in k) or ("z_encoder" in k)
        new_missing = [k for k in msg.missing_keys if _is_new(k)]      # adapter + z-encoder (fresh)
        base_missing = [k for k in msg.missing_keys if not _is_new(k)]
        print(f"[fmhead] init from {init_ck} (base_missing={len(base_missing)} "
              f"new_module_missing={len(new_missing)} unexpected={len(msg.unexpected_keys)})", flush=True)
        assert not base_missing, f"base head keys did not all load: {base_missing[:6]}"
    # STYLE-phase trainable set. Two mutually-exclusive modes (both keep VLM+LoRA frozen):
    #   * unfreeze_head=True  (option 1, "don't over-freeze"): train the FM head's OWN DiT
    #     blocks TOGETHER with the style adapter -> the head can re-aim, not just a tiny gate.
    #   * else + use_style_adapter: the ORIGINAL adapter-only path (freeze the loaded head,
    #     train ONLY the zero-gated style adapter). GOLD-safe / untouched by the flag above.
    unfreeze_head = bool(fh.get("unfreeze_head", False))
    train_z_only = bool(fh.get("train_z_path_only", False))
    if train_z_only:
        # Z-PATH mode (takes precedence): train ONLY z_encoder + s_proj; freeze the whole
        # body (DiT/cross-attn/embedders/final/temporal_pos/ctx_proj/null_style) + VLM+LoRA
        # -> alpha=0/z=null == the clean neutral fm3-kin base. Requires use_z_encoder=True.
        n_train, n_frozen = dec.freeze_all_but_z_path()
        print(f"[fmhead] TRAIN-Z-PATH-ONLY: trainable={n_train/1e6:.2f}M (z_encoder ONLY; "
              f"s_proj FROZEN) frozen={n_frozen/1e6:.2f}M (body: DiT/cross-attn/embedders/"
              f"final/s_proj; VLM+LoRA frozen outside)", flush=True)
    elif unfreeze_head:
        n_train, n_frozen = dec.unfreeze_head_and_style()
        print(f"[fmhead] UNFREEZE-HEAD style mode: trainable={n_train/1e6:.2f}M "
              f"(FM head DiT blocks{' + style adapter' if use_style_adapter else ''}; "
              f"temporal_pos frozen={n_frozen/1e3:.1f}K) | VLM+LoRA still frozen", flush=True)
    elif use_style_adapter:
        n_train, n_frozen = dec.freeze_base_keep_style()
        print(f"[fmhead] STYLE ADAPTER on: trainable={n_train/1e3:.1f}K frozen={n_frozen/1e6:.2f}M "
              f"(0.91 head frozen; only the adapter trains)", flush=True)
    return dec


# =============================================================================
# Data
# =============================================================================
def make_real_dataloader(av, av_config, dctx):
    """The REAL vision path: reuse AutoVLA's SFTDataset + DataCollator unchanged.

    With >1 rank a DistributedSampler shards scenes disjointly across ranks.
    Returns (dataloader, sampler_or_None)."""
    from dataset_utils.sft_dataset import SFTDataset, DataCollator
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    ds = SFTDataset(av_config["data"]["train"], av_config["model"], av.processor,
                    using_cot=av_config["model"]["use_cot"])
    collate = DataCollator(processor=av.processor,
                           ignore_index=av_config["model"]["tokens"]["ignore_index"],
                           assistant_id=av_config["model"]["tokens"]["assistant_id"])
    sampler = None
    if dctx.enabled:
        sampler = DistributedSampler(ds, num_replicas=dctx.world_size, rank=dctx.rank,
                                     shuffle=True, drop_last=True)
    loader = DataLoader(ds, batch_size=1, collate_fn=collate, num_workers=2,
                        shuffle=(sampler is None), sampler=sampler)
    return loader, sampler


def make_textonly_batches(av, n, device, dctx, dataset_dir=None):
    """SMOKE fallback (NVML/vision restricted here): build text-only VLM inputs from
    real dataset JSONs. Same fm_step path; only the vision tokens are absent.

    Shards scenes disjointly per rank (files[rank::world_size]) so that WITHOUT
    all-reduce the ranks' grads/params would diverge -- a strict DDP-sync test."""
    dataset_dir = dataset_dir or "/mnt/pfs/zhengguantian/autovla/persona/nocot_ddv2_teacher12k"
    all_files = sorted(glob.glob(os.path.join(dataset_dir, "*.json")))
    files = all_files[dctx.rank::max(dctx.world_size, 1)][:n]
    batches = []
    for fp in files:
        d = json.load(open(fp))
        vel = np.linalg.norm(d.get("velocity", [0, 0]))
        prompt = (f"You are driving. Ego speed {vel:.1f} m/s. "
                  f"Command: {d.get('instruction','keep forward')}. Predict the ego trajectory.")
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = av.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = av.processor(text=[text], images=None, videos=None, padding=True, return_tensors="pt")
        gt = torch.tensor(d["gt_trajectory"], dtype=torch.float32).unsqueeze(0)  # (1,10,3)
        b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in enc.items()}
        b["gt_trajectory"] = gt.to(device)
        batches.append(b)
    dctx.log(f"[data] rank{dctx.rank}: built {len(batches)} text-only batches "
             f"(scenes {[os.path.basename(f)[:8] for f in files]})")
    return batches


# =============================================================================
# Train
# =============================================================================
def save_ckpt(decoder, cfg, step, run_dir, tag):
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, f"fmhead_{tag}.pt")
    torch.save({"fm_decoder": decoder.state_dict(), "config": cfg, "step": step,
                "fm_config": vars(decoder.config)}, path)
    print(f"[ckpt] saved {path}", flush=True)
    return path


def param_signature(module, device):
    """Deterministic scalar signature of the trainable params (for cross-rank sync check)."""
    with torch.no_grad():
        sig = torch.zeros((), dtype=torch.float64, device=device)
        for p in module.parameters():
            if p.requires_grad:
                sig = sig + p.detach().double().sum()
        return sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/fmhead_ddv2.yaml")
    ap.add_argument("--smoke", action="store_true", help="few steps + freeze/loss/ckpt/DDP-sync asserts")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--no_sft_base", action="store_true", help="skip heavy PDMS ckpt (smoke)")
    args = ap.parse_args()

    dctx = DistCtx().init()
    device = dctx.device
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(FMHEAD_DIR, args.config)
    cfg = load_yaml(cfg_path)
    torch.manual_seed(cfg["train"]["seed"])
    text_only = os.environ.get("FMHEAD_TEXT_ONLY", "0") == "1"
    load_sft_base = cfg.get("load_sft_base", True) and not args.no_sft_base
    max_steps = args.max_steps or cfg["train"]["max_steps"]
    # CFG null-style / style-dropout can leave params with zero (but present) grad;
    # default find_unused=False (tested OK). Override via FMHEAD_DDP_FIND_UNUSED=1.
    find_unused = os.environ.get("FMHEAD_DDP_FIND_UNUSED", "0") == "1"

    smoke_scenes = 2
    if args.smoke:
        cfg["train"]["warmup_steps"] = min(cfg["train"]["warmup_steps"], 5)
        cfg["train"]["accumulate_grad_batches"] = 1
        cfg["train"]["ckpt_every"] = 10 ** 9
        if args.max_steps is None:
            max_steps = 250

    dctx.log(f"=== train_fmhead | world_size={dctx.world_size} ddp={dctx.enabled} "
             f"device={device} smoke={args.smoke} text_only={text_only} "
             f"load_sft_base={load_sft_base} max_steps={max_steps} find_unused={find_unused} ===")

    # Optional base-VLM override (e.g. the full-FT extracted base) so the frozen head's
    # content distribution matches the base used at style-training time. None -> config default.
    _base_override = cfg.get("base_ckpt")
    if _base_override and str(_base_override).strip().lower() in ("none", "null", ""):
        _base_override = None
    av, av_config, lora_on = build_frozen_vlm(cfg, device, load_sft_base=load_sft_base,
                                              log=dctx.log, base_ckpt_override=_base_override)
    # Optional dataset override (e.g. point the reused AutoVLA config at the human-GT
    # navtrain set) without editing the AutoVLA repo config.
    if cfg.get("dataset_path"):
        dp = cfg["dataset_path"]
        av_config["data"]["train"]["json_dataset_path"] = [dp]
        if isinstance(av_config["data"].get("val"), dict):
            av_config["data"]["val"]["json_dataset_path"] = dp
        dctx.log(f"[data] dataset override -> {dp}")
    # Style axis OFF when: no LoRA attached, OR style_mode:none, OR style_alpha==0.
    style_off = (not lora_on) or (str(cfg.get("style_mode", "")).lower() == "none") \
        or float(cfg.get("style_alpha", 1.0)) == 0.0
    dctx.log(f"[style] axis {'OFF (s=0, single VLM forward)' if style_off else 'ON (s=alpha*(styled-base))'}")
    decoder = build_decoder(cfg, device)

    # Wrap ONLY the FMHead decoder in DDP; the frozen VLM stays outside DDP.
    if dctx.enabled:
        ddp_device_ids = [dctx.dev_index] if torch.cuda.is_available() else None
        train_module = torch.nn.parallel.DistributedDataParallel(
            decoder, device_ids=ddp_device_ids,
            output_device=(dctx.dev_index if torch.cuda.is_available() else None),
            broadcast_buffers=False, find_unused_parameters=find_unused)
    else:
        train_module = decoder

    n_train = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    n_vlm_grad = sum(p.numel() for p in av.parameters() if p.requires_grad)
    dctx.log(f"[params] FMHead trainable={n_train:,} | VLM requires_grad={n_vlm_grad:,} (must be 0)")
    assert n_vlm_grad == 0, "VLM must be fully frozen"

    opt = torch.optim.AdamW([p for p in decoder.parameters() if p.requires_grad],
                            lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    warmup = cfg["train"]["warmup_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: min(1.0, (s + 1) / max(1, warmup)))

    # rank-0 chooses the run dir and broadcasts it so all ranks agree
    run_dir = os.path.join(cfg["ckpt_dir"], datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                           + ("_smoke" if args.smoke else "") + (f"_x{dctx.world_size}" if dctx.enabled else ""))
    if dctx.enabled:
        obj = [run_dir]
        dist.broadcast_object_list(obj, src=0)
        run_dir = obj[0]

    # data source
    sampler = None
    if text_only:
        smoke_batches = make_textonly_batches(av, n=(smoke_scenes if args.smoke else 4),
                                              device=device, dctx=dctx,
                                              dataset_dir=cfg.get("dataset_path"))

        def data_iter():
            for i in range(max_steps):
                yield smoke_batches[i % len(smoke_batches)]
    else:
        loader, sampler = make_real_dataloader(av, av_config, dctx)

        def data_iter():
            ep, produced = 0, 0
            while produced < max_steps:
                if sampler is not None:
                    sampler.set_epoch(ep)
                for batch in loader:
                    yield batch
                    produced += 1
                    if produced >= max_steps:
                        break
                ep += 1

    decoder.train()
    losses, t0 = [], time.time()
    accum = cfg["train"]["accumulate_grad_batches"]
    vlm_grad_after = None
    for step, batch in enumerate(data_iter()):
        model_inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                        for k, v in batch.items()}
        ctx, ctx_mask, s = extract_c_s(av, model_inputs, alpha=cfg["style_alpha"],
                                       reduce=cfg["c_reduce"], style_off=style_off,
                                       ctx_max_len=decoder.config.ctx_max_len,
                                       content_source=cfg.get("content_source"))
        # STYLE-phase target: with style ON, the flow target should be the STYLE-MATCHED
        # teacher trajectory (e.g. DDv2) so s controls the output; with style OFF the target
        # is human GT (feasibility). cfg["style_target"]="teacher" uses batch["teacher_trajectory"]
        # when present (falls back to gt_trajectory + a warning if the teacher field is absent).
        target = model_inputs["gt_trajectory"]
        if (not style_off) and str(cfg.get("style_target", "gt")).lower() == "teacher":
            if "teacher_trajectory" in model_inputs:
                target = model_inputs["teacher_trajectory"]
            elif step == 0:
                dctx.log("[style] WARNING: style_target=teacher but batch has no "
                         "'teacher_trajectory'; falling back to gt_trajectory (no style signal!)")
        is_accum_boundary = (step + 1) % accum == 0
        # skip DDP all-reduce on non-boundary micro-steps (grad accumulation)
        sync_ctx = (train_module.no_sync() if (dctx.enabled and not is_accum_boundary)
                    else contextlib.nullcontext())
        with sync_ctx:
            loss = train_module(target, ctx, s, ctx_mask=ctx_mask)   # fwd==loss
            (loss / accum).backward()
        if is_accum_boundary:
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), cfg["train"]["grad_clip"])
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        if vlm_grad_after is None:
            vlm_grad_after = sum(int(p.grad is not None) for p in av.parameters())
        if step % cfg["train"]["log_every"] == 0:
            dctx.log(f"[step {step:6d}] loss={loss.item():.4f} lr={sched.get_last_lr()[0]:.2e} "
                     f"({(time.time()-t0)/(step+1)*1000:.0f} ms/step)")
        if step > 0 and step % cfg["train"]["ckpt_every"] == 0:
            dctx.barrier()
            if dctx.is_main:
                save_ckpt(decoder, cfg, step, run_dir, f"step{step}")
            dctx.barrier()

    # final ckpt (rank 0 only, barriers around it)
    dctx.barrier()
    final_path = None
    if dctx.is_main:
        final_path = save_ckpt(decoder, cfg, max_steps, run_dir, "final")
    dctx.barrier()

    # ---- cross-rank DDP param-sync check (all ranks participate) ----
    sync_ok = True
    if dctx.enabled:
        # nccl all_gather needs CUDA tensors; gloo needs CPU tensors.
        comm_dev = device if dctx.backend == "nccl" else torch.device("cpu")
        sig = param_signature(decoder, device).to(comm_dev)
        gathered = [torch.zeros_like(sig) for _ in range(dctx.world_size)]
        dist.all_gather(gathered, sig)
        vals = [g.item() for g in gathered]
        sync_ok = all(abs(v - vals[0]) < 1e-6 for v in vals)
        dctx.log(f"[ddp] param signatures across ranks: {vals} -> "
                 f"{'IN SYNC' if sync_ok else 'DIVERGED'}")

    # ---- smoke assertions (rank 0) ----
    if args.smoke and dctx.is_main:
        init_l, fin_l = float(np.mean(losses[:10])), float(np.mean(losses[-10:]))
        dec2 = build_decoder(cfg, device)
        ck = torch.load(final_path, map_location=device)
        dec2.load_state_dict(ck["fm_decoder"])
        reload_ok = torch.allclose(decoder.traj_mean, dec2.traj_mean)
        loss_ok = fin_l < 0.85 * init_l
        print("\n=== SMOKE SUMMARY ===", flush=True)
        print(f"world_size / ddp         : {dctx.world_size} / {dctx.enabled}")
        print(f"VLM grads after backward : {vlm_grad_after} (want 0) -> "
              f"{'PASS' if vlm_grad_after == 0 else 'FAIL'}")
        print(f"FMHead trainable params  : {n_train:,}")
        print(f"loss (mean first10->last10): {init_l:.4f} -> {fin_l:.4f} (ratio={fin_l/init_l:.2f}) -> "
              f"{'PASS' if loss_ok else 'FAIL'}")
        print(f"DDP param sync across rank: {'PASS' if sync_ok else 'FAIL'}")
        print(f"ckpt save+reload         : {'PASS' if reload_ok else 'FAIL'} ({final_path})")
        ok = (vlm_grad_after == 0 and loss_ok and reload_ok and sync_ok)
        res = {"world_size": dctx.world_size, "ddp": dctx.enabled,
               "vlm_grads_after_backward": vlm_grad_after, "fmhead_trainable": n_train,
               "loss_first10": init_l, "loss_last10": fin_l, "ckpt": final_path,
               "reload_ok": bool(reload_ok), "ddp_param_sync": bool(sync_ok),
               "ran_text_only": text_only, "load_sft_base": load_sft_base,
               "find_unused": find_unused, "all_pass": bool(ok)}
        json.dump(res, open(os.path.join(FMHEAD_DIR, "train_smoke_results.json"), "w"), indent=2)
        print(f"ALL                      : {'PASS' if ok else 'FAIL'}", flush=True)

    dctx.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
