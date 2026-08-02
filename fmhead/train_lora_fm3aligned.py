"""train_lora_fm3aligned.py -- NEW RECIPE: "fm3-aligned persona LoRA".

Distill a teacher trajectory into a persona LoRA THROUGH the FROZEN fm3-kin decoder,
so the LoRA's activation delta is aligned to the fm3-kin CONTROL space (not the old
codebook / discrete-action regime).

This is EXACTLY train_fmhead.py's control-space rectified flow-matching loss, but with
the FLIP of what is trainable:

    train_fmhead.py :  VLM(+LoRA) FROZEN (no_grad)          | FMHead decoder TRAINABLE
    THIS recipe     :  base VLM + fm3-kin head FROZEN       | persona LoRA TRAINABLE

Forward (per step):
  * VLM base (fm3-kin ep6 base) is FROZEN; a FRESH persona LoRA (r=64/alpha=128, attn+MLP,
    matching youdrive-<teacher>-persona-lora-big-l2) is attached and is the ONLY trainable
    thing. A SINGLE LoRA-ON forward (grad retained) produces the styled content sequence
    c = context_sequence_hidden(hidden_last) (the multi-token cross-attn context the
    fm3-kin head expects).
  * AdaLN style s = 0 (no z-encoder, no style-adapter, no CFG) -- identical conditioning
    to how the fm3-kin ep6 head was trained (style_off / s=0).
  * The FROZEN fm3-kin kinematic head predicts the control-space velocity field conditioned
    on c; its params get NO gradient. Gradient flows through the frozen head back into c and
    hence into the LoRA ONLY.

Loss = fm3-kin CONTROL-space flow-matching MSE (reuse FMHeadDecoder.training_loss in
kinematic mode): target = the TEACHER trajectory (batch gt_trajectory of the teacher12k
set) inverted to unicycle controls (a_long, yaw_rate) with v0=first_gt_step and z-scored
with traj_norm_stats_ctrl.json -- exactly the fm3-kin target pipeline, but the teacher
trajectory is the target. grad -> LoRA only.

ADDITIVE / config-gated: this is a NEW entrypoint + NEW configs. train_fmhead.py, fm_head.py,
autovla_fmhead.py, GOLD/fm3-kin base+head, and every existing training mode are UNCHANGED.

Usage:
  # single node, 8 GPU (cluster):
  torchrun --standalone --nproc_per_node=8 train_lora_fm3aligned.py \
      --config config/fmhead_lora_fm3aligned_ddv2.yaml
  # CPU / text-only wiring smoke (no vision, few steps):
  FMHEAD_TEXT_ONLY=1 FMHEAD_DDP_BACKEND=gloo python train_lora_fm3aligned.py \
      --config config/fmhead_lora_fm3aligned_ddv2.yaml --smoke --no_sft_base
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

AUTOVLA_ROOT = "/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA"
FMHEAD_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (AUTOVLA_ROOT, os.path.join(AUTOVLA_ROOT, "navsim"), FMHEAD_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# Reuse the EXISTING (unchanged) building blocks. Importing train_fmhead does not run its
# main() (guarded by __main__) -- we only borrow its helpers so the head/data/DDP plumbing
# stays byte-identical to the head-training path.
from train_fmhead import (  # noqa: E402
    DistCtx, load_yaml, build_decoder, make_real_dataloader, make_textonly_batches,
    param_signature,
)
from autovla_fmhead import context_sequence_hidden  # noqa: E402

torch.set_float32_matmul_precision("high")

# VLM-forward input keys to keep (everything else -- targets / metadata -- is dropped).
_DROP_KEYS = ("gt_trajectory", "gt_action", "has_cot", "input_features", "token",
              "data_path", "text", "image_inputs", "video_inputs", "teacher_trajectory")


# =============================================================================
# Frozen base VLM + FRESH trainable persona LoRA
# =============================================================================
def build_lora_trainable_vlm(cfg, device, load_sft_base=True, log=print, base_ckpt_override=None):
    """AutoVLA with the fm3-kin base loaded + a FRESH (not warm-started) persona LoRA that is
    the ONLY trainable module. Returns (av, av_config, peft_model).

    LoRA config is read from the reused AutoVLA training config's ``model.lora`` block, so it
    matches the ``youdrive-<teacher>-persona-lora-big-l2`` lineage EXACTLY (r=64, alpha=128,
    attn+MLP, dropout=0.05). The base VLM (incl. vision tower) is frozen; only ``lora_*`` train.
    """
    os.chdir(AUTOVLA_ROOT)  # codebook_cache_path etc. are repo-relative
    av_config = load_yaml(os.path.join(AUTOVLA_ROOT, "config", cfg["autovla_config"] + ".yaml"))

    from models.autovla import AutoVLA
    t0 = time.time()
    av = AutoVLA(av_config, device=device)
    log(f"[vlm] AutoVLA(Qwen2.5-VL-3B) built in {time.time()-t0:.1f}s on {device}")

    base_ckpt = base_ckpt_override or av_config["model"].get("sft_model_path")
    if load_sft_base and base_ckpt:
        log(f"[vlm] loading FROZEN base: {base_ckpt}"
            + ("  (OVERRIDE: fm3-kin ep6 base)" if base_ckpt_override else "  (config sft_model_path)"))
        sd = torch.load(base_ckpt, map_location="cpu")["state_dict"]
        sd = {k.replace("autovla.", "").replace("drivevla.", ""): v for k, v in sd.items()}
        msg = av.load_state_dict(sd, strict=False)
        log(f"[vlm] base loaded (missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)})")
    elif not load_sft_base:
        log("[vlm] SKIP base ckpt load (--no_sft_base / smoke): fresh Qwen weights as base")

    # FRESH persona LoRA from the reused AutoVLA config's lora block (NOT warm-started).
    from peft import get_peft_model, LoraConfig, TaskType
    lc = dict(av_config["model"].get("lora", {}))
    assert lc.get("use", False), "autovla_config must define a model.lora block (r/alpha/targets)"
    lora_config = LoraConfig(
        task_type=TaskType[lc.get("task_type", "CAUSAL_LM")],
        target_modules=lc.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
        r=lc.get("r", 16), lora_alpha=lc.get("alpha", 32),
        lora_dropout=lc.get("dropout", 0.05), bias=lc.get("bias", "none"),
    )
    av.vlm = get_peft_model(av.vlm, lora_config)
    log(f"[lora] FRESH persona LoRA attached: r={lora_config.r} alpha={lora_config.lora_alpha} "
        f"dropout={lora_config.lora_dropout} targets={lora_config.target_modules}")

    # Freeze EVERYTHING except lora_* (peft already does this; enforce + count explicitly).
    n_lora = n_base = 0
    for n, p in av.vlm.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True); n_lora += p.numel()
        else:
            p.requires_grad_(False); n_base += p.numel()
    for n, p in av.named_parameters():        # any non-vlm AutoVLA params -> frozen
        if not n.startswith("vlm."):
            p.requires_grad_(False)
    av.to(device)
    av.train()  # LoRA dropout active (matches SFT-LoRA); frozen base LayerNorms are unaffected
    log(f"[lora] TRAINABLE lora params={n_lora/1e6:.2f}M | FROZEN base (VLM+vision)={n_base/1e9:.2f}B")
    return av, av_config, av.vlm


# =============================================================================
# Differentiable content extraction (LoRA-ON forward, grad retained)
# =============================================================================
def extract_ctx_styled(vlm_module, model_inputs, action_start_id, ctx_max_len):
    """SINGLE LoRA-ON forward (NO no_grad, NO disable_adapter) -> (ctx, ctx_mask, s).

    ctx (B,T_ctx,2048) = the styled content sequence the fm3-kin head cross-attends to.
    s = zeros (AdaLN style OFF, exactly the fm3-kin ep6 s=0 conditioning). Gradient is
    retained through the VLM forward so it flows into the persona LoRA (base is frozen).
    ``vlm_module`` is the (possibly DDP-wrapped) peft model so DDP registers the LoRA grads.
    """
    keep = {k: v for k, v in model_inputs.items() if k not in _DROP_KEYS}
    out = vlm_module(**keep, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[-1]                                  # (B, S, 2048), grad-enabled
    ctx, ctx_mask = context_sequence_hidden(
        h, keep["input_ids"], attention_mask=keep.get("attention_mask"),
        action_start_id=action_start_id, max_len=ctx_max_len)
    s = torch.zeros(h.shape[0], h.shape[-1], device=h.device, dtype=h.dtype)  # AdaLN s=0
    return ctx, ctx_mask, s


# =============================================================================
# Frozen fm3-kin head
# =============================================================================
def build_frozen_head(cfg, device, log=print):
    """Build the FMHead decoder, load the CONTROL normalizer + the fm3-kin ep6 head, then
    FREEZE ALL of it. Only the LoRA (outside this module) trains; the head just maps the
    styled content -> control-space velocity field and passes gradient back to the LoRA."""
    dec = build_decoder(cfg, device)                 # loads normalizer + init_fmhead_ckpt (ep6 head)
    n = 0
    for p in dec.parameters():
        p.requires_grad_(False); n += p.numel()
    dec.eval()
    log(f"[head] FROZEN fm3-kin head: {n/1e6:.2f}M params (control-space, s=0), requires_grad=False")
    return dec


# =============================================================================
# Checkpoint (LoRA adapter)
# =============================================================================
def save_lora(peft_model, cfg, step, run_dir, tag):
    os.makedirs(run_dir, exist_ok=True)
    out = os.path.join(run_dir, f"lora_{tag}")
    peft_model.save_pretrained(out)
    print(f"[ckpt] saved LoRA adapter -> {out}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="few steps + grad-flow/frozen/finite/ckpt/DDP-sync asserts")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--no_sft_base", action="store_true", help="skip the heavy 8GB base ckpt (smoke)")
    args = ap.parse_args()

    dctx = DistCtx().init()
    device = dctx.device
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(FMHEAD_DIR, args.config)
    cfg = load_yaml(cfg_path)
    assert bool(cfg.get("train_lora_only", False)), \
        "this entrypoint requires train_lora_only: true in the config (fm3-aligned LoRA recipe)"
    torch.manual_seed(cfg["train"]["seed"])
    text_only = os.environ.get("FMHEAD_TEXT_ONLY", "0") == "1"
    load_sft_base = cfg.get("load_sft_base", True) and not args.no_sft_base
    max_steps = args.max_steps or cfg["train"]["max_steps"]
    find_unused = os.environ.get("FMHEAD_DDP_FIND_UNUSED", "0") == "1"

    smoke_scenes = 2
    if args.smoke:
        cfg["train"]["warmup_steps"] = min(cfg["train"]["warmup_steps"], 2)
        cfg["train"]["accumulate_grad_batches"] = 1
        cfg["train"]["ckpt_every"] = 10 ** 9
        if args.max_steps is None:
            max_steps = 3

    dctx.log(f"=== train_lora_fm3aligned | world_size={dctx.world_size} ddp={dctx.enabled} "
             f"device={device} smoke={args.smoke} text_only={text_only} "
             f"load_sft_base={load_sft_base} max_steps={max_steps} ===")

    _base_override = cfg.get("base_ckpt")
    if _base_override and str(_base_override).strip().lower() in ("none", "null", ""):
        _base_override = None
    av, av_config, peft_model = build_lora_trainable_vlm(
        cfg, device, load_sft_base=load_sft_base, log=dctx.log, base_ckpt_override=_base_override)

    if cfg.get("dataset_path"):
        dp = cfg["dataset_path"]
        av_config["data"]["train"]["json_dataset_path"] = [dp]
        if isinstance(av_config["data"].get("val"), dict):
            av_config["data"]["val"]["json_dataset_path"] = dp
        dctx.log(f"[data] dataset override -> {dp}")

    decoder = build_frozen_head(cfg, device, log=dctx.log)

    # DDP wraps ONLY the trainable LoRA carrier (the peft model). The frozen head stays outside.
    if dctx.enabled:
        ddp_device_ids = [dctx.dev_index] if torch.cuda.is_available() else None
        train_vlm = torch.nn.parallel.DistributedDataParallel(
            peft_model, device_ids=ddp_device_ids,
            output_device=(dctx.dev_index if torch.cuda.is_available() else None),
            broadcast_buffers=False, find_unused_parameters=find_unused)
    else:
        train_vlm = peft_model

    lora_params = [p for p in peft_model.parameters() if p.requires_grad]
    n_lora = sum(p.numel() for p in lora_params)
    n_base_grad = sum(p.numel() for n, p in peft_model.named_parameters()
                      if p.requires_grad and "lora_" not in n)
    n_head_grad = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    dctx.log(f"[params] LoRA trainable={n_lora:,} | base req_grad={n_base_grad:,} (must be 0) | "
             f"head req_grad={n_head_grad:,} (must be 0)")
    assert n_base_grad == 0, "base VLM must be frozen"
    assert n_head_grad == 0, "fm3-kin head must be frozen"

    opt = torch.optim.AdamW(lora_params, lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    warmup = cfg["train"]["warmup_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: min(1.0, (s + 1) / max(1, warmup)))

    run_dir = os.path.join(cfg["ckpt_dir"], datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                           + ("_smoke" if args.smoke else "") + (f"_x{dctx.world_size}" if dctx.enabled else ""))
    if dctx.enabled:
        obj = [run_dir]
        dist.broadcast_object_list(obj, src=0)
        run_dir = obj[0]

    asid = av.action_start_id
    ctx_max_len = decoder.config.ctx_max_len

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

    losses, t0 = [], time.time()
    accum = cfg["train"]["accumulate_grad_batches"]
    lora_grad_elems = base_grad_after = head_grad_after = None
    for step, batch in enumerate(data_iter()):
        model_inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                        for k, v in batch.items()}
        # TARGET = the TEACHER trajectory. In the teacher12k sets, gt_trajectory IS the teacher.
        target = model_inputs["gt_trajectory"]
        is_accum_boundary = (step + 1) % accum == 0
        sync_ctx = (train_vlm.no_sync() if (dctx.enabled and not is_accum_boundary)
                    else contextlib.nullcontext())
        with sync_ctx:
            # LoRA-ON forward (grad) -> styled content; s=0 (AdaLN style off).
            ctx, ctx_mask, s = extract_ctx_styled(train_vlm, model_inputs, asid, ctx_max_len)
            # control-space rectified flow-matching MSE through the FROZEN fm3-kin head.
            loss = decoder.training_loss(target, ctx, s, ctx_mask=ctx_mask, style_dropout_prob=0.0)
            (loss / accum).backward()
        # grad-flow snapshot RIGHT AFTER the first backward (before any opt.zero_grad clears it):
        # LoRA must receive grad; the frozen base VLM + fm3-kin head must receive NONE.
        if lora_grad_elems is None:
            lora_grad_elems = sum(int(p.grad is not None and p.grad.abs().sum() > 0)
                                  for p in lora_params)
            base_grad_after = sum(int(p.grad is not None) for n, p in peft_model.named_parameters()
                                  if "lora_" not in n)
            head_grad_after = sum(int(p.grad is not None) for p in decoder.parameters())
        if is_accum_boundary:
            torch.nn.utils.clip_grad_norm_(lora_params, cfg["train"]["grad_clip"])
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        if step % cfg["train"]["log_every"] == 0:
            dctx.log(f"[step {step:6d}] loss={loss.item():.4f} lr={sched.get_last_lr()[0]:.2e} "
                     f"({(time.time()-t0)/(step+1)*1000:.0f} ms/step)")
        if step > 0 and step % cfg["train"]["ckpt_every"] == 0:
            dctx.barrier()
            if dctx.is_main:
                save_lora(peft_model, cfg, step, run_dir, f"step{step}")
            dctx.barrier()

    dctx.barrier()
    final_path = None
    if dctx.is_main:
        final_path = save_lora(peft_model, cfg, max_steps, run_dir, "final")
    dctx.barrier()

    # cross-rank DDP LoRA-param-sync check
    sync_ok = True
    if dctx.enabled:
        comm_dev = device if dctx.backend == "nccl" else torch.device("cpu")
        sig = param_signature(peft_model, device).to(comm_dev)
        gathered = [torch.zeros_like(sig) for _ in range(dctx.world_size)]
        dist.all_gather(gathered, sig)
        vals = [g.item() for g in gathered]
        sync_ok = all(abs(v - vals[0]) < 1e-6 for v in vals)
        dctx.log(f"[ddp] LoRA param signatures across ranks: {vals} -> "
                 f"{'IN SYNC' if sync_ok else 'DIVERGED'}")

    if args.smoke and dctx.is_main:
        init_l = float(np.mean(losses[:max(1, min(3, len(losses)))]))
        fin_l = float(np.mean(losses[-max(1, min(3, len(losses))):]))
        # ckpt reload check: adapter tensors on disk == in-memory LoRA weights
        reload_ok = False
        try:
            from safetensors.torch import load_file
            saved = load_file(os.path.join(final_path, "adapter_model.safetensors"))
            mem = {n.replace(".weight", "").replace("base_model.model.", ""): p
                   for n, p in peft_model.named_parameters() if "lora_" in n}
            k = next(iter(saved))
            reload_ok = torch.isfinite(saved[k]).all().item() and len(saved) > 0
        except Exception as e:
            print(f"[smoke] reload check error: {e}", flush=True)
        all_finite = all(np.isfinite(losses))
        print("\n=== SMOKE SUMMARY (fm3-aligned persona LoRA) ===", flush=True)
        print(f"world_size / ddp          : {dctx.world_size} / {dctx.enabled}")
        print(f"LoRA trainable params     : {n_lora:,}")
        print(f"LoRA grad elems (want >0) : {lora_grad_elems} -> "
              f"{'PASS' if lora_grad_elems and lora_grad_elems > 0 else 'FAIL'}")
        print(f"base VLM grads (want 0)   : {base_grad_after} -> "
              f"{'PASS' if base_grad_after == 0 else 'FAIL'}")
        print(f"fm3-kin head grads(want 0): {head_grad_after} -> "
              f"{'PASS' if head_grad_after == 0 else 'FAIL'}")
        print(f"loss finite (all steps)   : {all_finite} -> {'PASS' if all_finite else 'FAIL'} "
              f"(first={init_l:.4f} last={fin_l:.4f})")
        print(f"DDP LoRA param sync       : {'PASS' if sync_ok else 'FAIL'}")
        print(f"LoRA save+reload          : {'PASS' if reload_ok else 'FAIL'} ({final_path})")
        ok = (lora_grad_elems and lora_grad_elems > 0 and base_grad_after == 0
              and head_grad_after == 0 and all_finite and reload_ok and sync_ok)
        res = {"world_size": dctx.world_size, "ddp": dctx.enabled,
               "lora_trainable": n_lora, "lora_grad_elems": lora_grad_elems,
               "base_grads_after": base_grad_after, "head_grads_after": head_grad_after,
               "loss_first": init_l, "loss_last": fin_l, "all_finite": bool(all_finite),
               "ddp_param_sync": bool(sync_ok), "reload_ok": bool(reload_ok),
               "ran_text_only": text_only, "load_sft_base": load_sft_base,
               "ckpt": final_path, "all_pass": bool(ok)}
        json.dump(res, open(os.path.join(FMHEAD_DIR, "train_lora_fm3aligned_smoke_results.json"), "w"),
                  indent=2)
        print(f"ALL                       : {'PASS' if ok else 'FAIL'}", flush=True)

    dctx.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
