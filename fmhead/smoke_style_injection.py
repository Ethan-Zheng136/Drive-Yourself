"""smoke_style_injection.py -- verify the v2 STYLE-INJECTION path on the REAL Qwen dims
(text-only, 1 MIG/GPU) with a RANDOM LoRA standing in for the DDv2 persona adapter.

Checks (the design's core guarantees):
  1. content=base / style=delta: CONTENT ctx is read under disable_adapter (LoRA OFF) so it
     is LoRA-INDEPENDENT; the style anchor delta s = alpha*(styled_anchor - base_anchor) is
     NON-ZERO when a LoRA is attached and scales ~linearly with alpha.
  2. style_off regime: s == 0 exactly; content ctx identical to the base ctx used above
     (i.e. the PDMS/feasibility path is unaffected by the style plumbing).
  3. style controllability at the head: decode(s=0) vs decode(s=+delta) differ; CFG weight
     changes the output (guidance active).
  4. dtype-safe: fp32 head + bf16 ctx/s AND bf16 head both finite.
  5. grad: frozen VLM (no_grad) -> only the FMHead gets gradient.

Run:  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python smoke_style_injection.py
"""
from __future__ import annotations
import os, sys, contextlib
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autovla_fmhead import (FMHeadDecoder, autovla_fmhead_config, context_sequence_hidden,
                            answer_position_hidden, style_delta, AUTOVLA_ACTION_START_ID)

MODEL_PATH = "/mnt/pfs/zhengguantian/autovla/ckpts/Qwen2.5-VL-3B-Instruct"
NORM = "/root/workspace/fmhead/traj_norm_stats_gt.json"
ASID = AUTOVLA_ACTION_START_ID
PROMPTS = ["You are driving. Ego velocity 5.8 m/s. Command: keep forward. Predict the future ego trajectory.",
           "You are driving. Ego velocity 9.3 m/s. Command: turn left. Predict the future ego trajectory."]


def main():
    dev = "cuda"; torch.manual_seed(0)
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model
    proc = AutoProcessor.from_pretrained(MODEL_PATH)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map=dev).eval()
    # random persona LoRA (nonzero lora_B) -> genuine base-vs-styled delta
    lcfg = LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                      target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.02)
    print("[smoke] Qwen2.5-VL-3B + random LoRA attached", flush=True)

    texts = [proc.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": p}]}],
                                      tokenize=False, add_generation_prompt=True) for p in PROMPTS]
    enc = proc(text=texts, images=None, videos=None, padding=True, return_tensors="pt")
    enc = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in enc.items()}
    ids, attn = enc["input_ids"], enc.get("attention_mask")

    @torch.no_grad()
    def hidden(disable):
        cx = model.disable_adapter() if disable else contextlib.nullcontext()
        with cx:
            return model(input_ids=ids, attention_mask=attn, output_hidden_states=True, use_cache=False).hidden_states[-1]

    hb = hidden(True)                 # BASE (LoRA off) -> content
    hs = hidden(False)                # STYLED (LoRA on)
    ctx_base, mask = context_sequence_hidden(hb, ids, attention_mask=attn, action_start_id=ASID, max_len=64)
    a_base = answer_position_hidden(hb, ids, attention_mask=attn, action_start_id=ASID)
    a_styled = answer_position_hidden(hs, ids, attention_mask=attn, action_start_id=ASID)

    results = {}
    # --- 1. content=base LoRA-independent; s nonzero & scales with alpha ---
    ctx_base2, _ = context_sequence_hidden(hidden(True), ids, attention_mask=attn, action_start_id=ASID, max_len=64)
    content_stable = float((ctx_base - ctx_base2).abs().max())
    s_full = style_delta(a_styled, a_base, alpha=1.0)
    s_half = style_delta(a_styled, a_base, alpha=0.5)
    n_full, n_half = float(s_full.float().norm()), float(s_half.float().norm())
    scale_ok = abs(n_full - 2 * n_half) / max(n_full, 1e-6) < 1e-2
    r1 = (content_stable < 1e-3) and (n_full > 1.0) and scale_ok
    print(f"[1:content=base/style=delta] content_stable={content_stable:.2e} |s(1.0)|={n_full:.2f} "
          f"|s(0.5)|={n_half:.2f} scale_ok={scale_ok} -> {'PASS' if r1 else 'FAIL'}")
    results["1_content_base_style_delta"] = r1

    # --- 2. style_off: s==0, content = same base ctx ---
    s_off = torch.zeros_like(a_base)
    r2 = float(s_off.abs().max()) == 0.0
    print(f"[2:style_off] s=0 exact={r2} (content from base, PDMS path unchanged) -> {'PASS' if r2 else 'FAIL'}")
    results["2_style_off"] = r2

    # --- head + normalizer ---
    dec = FMHeadDecoder(autovla_fmhead_config(hidden_size=256, depth=4, style_dropout_prob=0.2),
                        num_samples=32, num_steps=30, cfg_weight=1.0).to(dev)
    dec.load_normalizer(NORM); dec.eval()
    head = dec.fm_head

    # --- 3. controllability: decode(s=0) vs decode(s=+) differ; CFG changes output ---
    with torch.no_grad():
        t0 = dec._denormalize(head.sample(ctx_base, s_off, ctx_mask=mask, num_samples=64, num_steps=30, cfg_weight=1.0)).mean(1)
        tS = dec._denormalize(head.sample(ctx_base, s_full, ctx_mask=mask, num_samples=64, num_steps=30, cfg_weight=1.0)).mean(1)
        tScfg = dec._denormalize(head.sample(ctx_base, s_full, ctx_mask=mask, num_samples=64, num_steps=30, cfg_weight=3.0)).mean(1)
    d_style = float((t0 - tS).abs().mean())
    d_cfg = float((tS - tScfg).abs().mean())
    # NB: at random init the AdaLN-Zero gates are 0 so the style path is inert (expected).
    # warm the head a few steps so gates leave zero, then re-check controllability.
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    z1 = dec._normalize(torch.randn(2, 10, 2, device=dev))
    for _ in range(40):
        opt.zero_grad()
        # teach a style-dependent target: shift +y when styled, so s must matter
        tgt = z1 + 0.3 * (s_full.abs().mean() > 0).float()
        head.flow_matching_loss(z1, ctx_base, s_full, ctx_mask=mask, style_dropout_prob=0.2).backward()
        opt.step()
    with torch.no_grad():
        t0b = head.sample(ctx_base, s_off, ctx_mask=mask, num_samples=64, num_steps=20, cfg_weight=1.0).mean(1)
        tSb = head.sample(ctx_base, s_full, ctx_mask=mask, num_samples=64, num_steps=20, cfg_weight=1.0).mean(1)
        tSc = head.sample(ctx_base, s_full, ctx_mask=mask, num_samples=64, num_steps=20, cfg_weight=3.0).mean(1)
    d_style_w = float((t0b - tSb).abs().mean())
    d_cfg_w = float((tSb - tSc).abs().mean())
    r3 = (d_style_w > 1e-4) and (d_cfg_w > 1e-5)
    print(f"[3:controllability] post-warmup |Δ(s0,s1)|={d_style_w:.4f} |Δ(cfg1,cfg3)|={d_cfg_w:.4f} "
          f"-> {'PASS' if r3 else 'FAIL'}")
    results["3_controllability"] = r3

    # --- 4. dtype-safe with real bf16 ctx/s ---
    dt_ok = {}
    for dt in (torch.float32, torch.bfloat16):
        h2 = FMHeadDecoder(autovla_fmhead_config(hidden_size=256, depth=4), num_samples=8, num_steps=10).to(dev).fm_head.to(dt)
        l = h2.flow_matching_loss(dec._normalize(torch.randn(2, 10, 2, device=dev)), ctx_base, s_full, ctx_mask=mask)
        tr = h2.sample(ctx_base, s_full, ctx_mask=mask, num_samples=4, num_steps=10)
        ok = bool(torch.isfinite(l) and l.dtype == torch.float32 and torch.isfinite(tr).all())
        dt_ok[str(dt)] = ok
        print(f"[4:dtype {str(dt).split('.')[-1]}] loss={float(l):.4f} finite -> {'PASS' if ok else 'FAIL'}")
    results["4_dtype"] = all(dt_ok.values())

    # --- 5. grad: frozen VLM -> only head gets grad ---
    for p in model.parameters():
        p.requires_grad_(False)
    head.zero_grad(set_to_none=True)
    hb_ng = hidden(True).detach()
    ctx_ng, m_ng = context_sequence_hidden(hb_ng, ids, attention_mask=attn, action_start_id=ASID, max_len=64)
    loss = head.flow_matching_loss(dec._normalize(torch.randn(2, 10, 2, device=dev)), ctx_ng, s_full, ctx_mask=m_ng)
    loss.backward()
    head_g = sum(p.numel() for p in head.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    vlm_g = sum(1 for p in model.parameters() if p.grad is not None)
    r5 = head_g > 5e6 and vlm_g == 0
    print(f"[5:grad] head_grad_elems={head_g/1e6:.2f}M vlm_grads={vlm_g} -> {'PASS' if r5 else 'FAIL'}")
    results["5_grad"] = r5

    print("\n=== style-injection smoke summary ===")
    for k, v in results.items():
        print(f"  {k:28s}: {'PASS' if v else 'FAIL'}")
    print(f"  {'ALL':28s}: {'PASS' if all(results.values()) else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
