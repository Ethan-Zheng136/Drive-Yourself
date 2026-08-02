"""v2_scale_check.py -- rigorous MAGNITUDE / DIM / GRAD audit of FM head v2 on REAL Qwen dims.

Uses the REAL Qwen2.5-VL-3B-Instruct via the TEXT-ONLY path (this box: 1 MIG slice,
NVML/vision restricted -> text-only real Qwen forward works). Reports actual magnitudes
(mean/std/min/max/norm) for every tensor entering and inside the v2 head, checks the flow
quantities (normalization metres->O(1)), asserts every dim at hidden width 2048, and
verifies gradient flow (frozen: head-only; full-FT: grad reaches the LM). Saves a magnitude
table + PASS/FAIL verdicts to v2_scale_check.json.

Run:  OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python v2_scale_check.py
"""
from __future__ import annotations
import glob, json, os, time, contextlib
import numpy as np
import torch

from autovla_fmhead import (
    AUTOVLA_HIDDEN_SIZE, AUTOVLA_NUM_POSES, AUTOVLA_VLM_DTYPE, AUTOVLA_ACTION_START_ID,
    FMHeadDecoder, autovla_fmhead_config, context_sequence_hidden, answer_position_hidden,
)

MODEL_PATH = "/mnt/pfs/zhengguantian/autovla/ckpts/Qwen2.5-VL-3B-Instruct"
DATA_DIR = "/mnt/pfs/zhengguantian/autovla/persona/nocot_ddv2_teacher12k"
NORM_JSON = "/root/workspace/fmhead/traj_norm_stats_gt.json"
OUT_JSON = "/root/workspace/fmhead/v2_scale_check.json"
ASID = AUTOVLA_ACTION_START_ID
PROMPTS = [
    "You are driving. Ego velocity 5.8 m/s. Command: keep forward. Predict the future ego trajectory.",
    "You are driving. Ego velocity 0.0 m/s, stopped at a red light. Command: stop. Predict the future ego trajectory.",
    "You are driving. Ego velocity 9.3 m/s. Command: turn left at the intersection. Predict the future ego trajectory.",
]


def stat(name, t):
    """mean/std/min/max + overall L2 norm (computed in fp32)."""
    f = t.detach().float()
    return {
        "name": name, "shape": list(t.shape), "dtype": str(t.dtype),
        "mean": float(f.mean()), "std": float(f.std()),
        "min": float(f.min()), "max": float(f.max()), "norm": float(f.norm()),
    }


def fmt_row(s):
    return (f"  {s['name']:24s} {str(s['shape']):18s} {s['dtype']:15s} "
            f"mean={s['mean']:+.4f} std={s['std']:.4f} "
            f"min={s['min']:+.3f} max={s['max']:+.3f} norm={s['norm']:.2f}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "need GPU/MIG for the real Qwen forward"
    torch.manual_seed(0)
    rows, verdicts = [], {}

    def add(name, t):
        s = stat(name, t); rows.append(s); print(fmt_row(s)); return s

    # ---------- load real Qwen (text-only) ----------
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    print(f"[load] Qwen2.5-VL-3B loaded in {time.time()-t0:.1f}s dtype={next(model.parameters()).dtype}")

    # build a REAL text-only batch (no images -> no vision tower)
    texts = []
    for p in PROMPTS:
        messages = [{"role": "user", "content": [{"type": "text", "text": p}]}]
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    enc = processor(text=texts, images=None, videos=None, padding=True, return_tensors="pt")
    enc = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in enc.items()}
    input_ids, attn = enc["input_ids"], enc.get("attention_mask")
    B, S = input_ids.shape
    print(f"[inputs] B={B} S={S} attention_mask_sum={None if attn is None else attn.sum(-1).tolist()}")

    # ---------- 1. context sequence from the REAL VLM ----------
    print("\n=== [1] CONTEXT SEQUENCE (real VLM) ===")
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
    hidden_last = out.hidden_states[-1]                          # (B, S, 2048) bf16
    add("vlm.hidden_last", hidden_last)
    ctx, ctx_mask = context_sequence_hidden(hidden_last, input_ids, attention_mask=attn,
                                            action_start_id=ASID, max_len=64)
    s_ctx = add("ctx", ctx)
    add("ctx_mask", ctx_mask)
    # per-token L2 norm over VALID tokens only
    tok_norm = ctx.float().norm(dim=-1)                          # (B, T_ctx)
    valid = ctx_mask.bool()
    vnorms = tok_norm[valid]
    T_ctx = ctx.shape[1]
    per_tok = {"valid_tokens_per_row": ctx_mask.sum(-1).long().tolist(),
               "tok_norm_mean": float(vnorms.mean()), "tok_norm_min": float(vnorms.min()),
               "tok_norm_max": float(vnorms.max())}
    print(f"  T_ctx={T_ctx}  valid/row={per_tok['valid_tokens_per_row']}  "
          f"per-token L2 norm: mean={per_tok['tok_norm_mean']:.1f} "
          f"min={per_tok['tok_norm_min']:.1f} max={per_tok['tok_norm_max']:.1f}")
    anchor = answer_position_hidden(hidden_last, input_ids, attention_mask=attn, action_start_id=ASID)
    add("anchor(c v1)", anchor)
    # verdicts: bf16, sane per-token norm O(10..1e3), mask sane
    v1_dtype = (ctx.dtype == torch.bfloat16)
    v1_norm = 5.0 < per_tok["tok_norm_mean"] < 5000.0
    v1_mask = bool((ctx_mask.sum(-1) > 0).all()) and set(torch.unique(ctx_mask.float()).tolist()) <= {0.0, 1.0}
    v1_dim = (ctx.shape[-1] == AUTOVLA_HIDDEN_SIZE == 2048)
    verdicts["1_ctx"] = {"pass": bool(v1_dtype and v1_norm and v1_mask and v1_dim),
                         "dtype_bf16": v1_dtype, "pertoken_norm_sane": bool(v1_norm),
                         "mask_sane": bool(v1_mask), "dim2048": bool(v1_dim), **per_tok}

    # ---------- 2. style s in the s=0 regime ----------
    print("\n=== [2] STYLE s (s=0 regime) ===")
    s = torch.zeros_like(anchor)
    ss = add("s (style)", s)
    v2 = (ss["min"] == 0.0 and ss["max"] == 0.0 and not np.isnan(ss["norm"]))
    verdicts["2_style_zero"] = {"pass": bool(v2), "exactly_zero": bool(v2)}

    # ---------- build the head with the REAL normalizer ----------
    dec = FMHeadDecoder(autovla_fmhead_config(hidden_size=256, depth=4),
                        num_samples=16, num_steps=30, cfg_weight=1.0).to(device)
    st = dec.load_normalizer(NORM_JSON)
    head = dec.fm_head
    print(f"\n[norm] loaded {st['n_samples']} samples; endpoint mean/std "
          f"x={st['per_waypoint_mean'][-1][0]:.1f}/{st['per_waypoint_std'][-1][0]:.1f} "
          f"y={st['per_waypoint_mean'][-1][1]:.2f}/{st['per_waypoint_std'][-1][1]:.2f}")
    n_head = sum(p.numel() for p in head.parameters())
    print(f"[head] params={n_head/1e6:.2f}M dtype={head._param_dtype()}")

    # ---------- 3. head input projections + AdaLN params ----------
    print("\n=== [3] HEAD INPUT PROJECTIONS + AdaLN (init) ===")
    pdt = head._param_dtype()
    ctx_kv = head._project_ctx(ctx)                              # (B, T_ctx, 256)
    add("ctx_proj(ctx)", ctx_kv)
    x_t = torch.randn(B, head.config.horizon, head.config.traj_dim, device=device, dtype=pdt)
    add("x_embedder(x_t)", head.x_embedder(x_t))
    add("temporal_pos", head.temporal_pos)
    t = torch.rand(B, device=device, dtype=pdt)
    t_emb = head.t_embedder(t)
    add("t_embedder(t)", t_emb)
    cvec = head._cond_vector(t, s)                               # t + style
    add("cvec (adaLN in)", cvec)
    # AdaLN modulation params of block 0 (9-way): should be ~0 at AdaLN-Zero init
    mod0 = head.blocks[0].adaLN_modulation(cvec)
    parts = mod0.chunk(9, dim=1)
    labels = ["shift_sa","scale_sa","gate_sa","shift_ca","scale_ca","gate_ca","shift_mlp","scale_mlp","gate_mlp"]
    adaln_absmax = float(mod0.abs().max())
    for lb, pt in zip(labels, parts):
        add(f"adaLN.{lb}", pt)
    print(f"  AdaLN abs-max over all 9 params = {adaln_absmax:.2e} (AdaLN-Zero => ~0)")
    v3_proj = 0.01 < stat("p", ctx_kv)["std"] < 100.0
    v3_adaln = adaln_absmax < 1e-3
    verdicts["3_projections"] = {"pass": bool(v3_proj and v3_adaln),
                                 "ctx_proj_std_sane": bool(v3_proj),
                                 "adaLN_zero_at_init": bool(v3_adaln), "adaLN_absmax": adaln_absmax}

    # ---------- 4. FLOW quantities (normalization metres->O(1)) ----------
    print("\n=== [4] FLOW QUANTITIES ===")
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))[:B]
    gts = torch.stack([torch.tensor(json.load(open(f))["gt_trajectory"], dtype=torch.float32)
                       for f in files]).to(device)              # (B,10,3) metres
    gt_xy = gts[:, :AUTOVLA_NUM_POSES, :2]
    add("gt_xy (metres)", gt_xy)
    z1 = dec._normalize(gt_xy.to(pdt))                          # z-score normalized target
    add("z1 (normalized)", z1)
    z0 = torch.randn_like(z1)
    add("z0 (noise N(0,1))", z0)
    u = z1 - z0
    add("u = z1 - z0 (target vel)", u)
    tt = torch.rand(B, device=device, dtype=pdt)
    x_mix = (1 - tt.view(B,1,1)) * z0 + tt.view(B,1,1) * z1
    with torch.no_grad():
        v = head(x_mix, tt, ctx, s, ctx_mask=ctx_mask)
    add("v (predicted vel)", v)
    loss = head.flow_matching_loss(z1, ctx, s, ctx_mask=ctx_mask)
    loss_val = float(loss)
    print(f"  flow_matching_loss = {loss_val:.4f} (dtype={loss.dtype})")
    z1_absmax = float(z1.abs().max())
    gt_absmax = float(gt_xy.abs().max())
    v4_norm = z1_absmax < 6.0            # normalized target must be O(1), not O(10) metres
    v4_metres_big = gt_absmax > 6.0      # raw metres ARE O(10)
    v4_loss = 0.05 < loss_val < 20.0     # O(1), not ~82 (unnormalized bug) nor ~0
    verdicts["4_flow"] = {"pass": bool(v4_norm and v4_metres_big and v4_loss),
                          "gt_absmax_m": gt_absmax, "z1_absmax": z1_absmax,
                          "z1_is_O1": bool(v4_norm), "metres_were_O10": bool(v4_metres_big),
                          "loss": loss_val, "loss_is_O1": bool(v4_loss)}

    # ---------- 5. DIMS end-to-end ----------
    print("\n=== [5] DIMS ===")
    dims_ok = (ctx.shape[-1] == 2048 and ctx_kv.shape[-1] == 256 and
               tuple(x_t.shape[1:]) == (10, 2) and tuple(v.shape) == (B, 10, 2) and
               anchor.shape[-1] == 2048 and s.shape[-1] == 2048)
    print(f"  ctx=2048 proj=256 traj=10x2 vel={tuple(v.shape)} anchor=2048 s=2048 -> {'OK' if dims_ok else 'BAD'}")
    verdicts["5_dims"] = {"pass": bool(dims_ok), "ctx_dim": ctx.shape[-1], "proj_dim": ctx_kv.shape[-1],
                          "vel_shape": list(v.shape)}

    # ---------- 6. GRAD FLOW (real dims) ----------
    print("\n=== [6] GRAD FLOW ===")
    # warm the AdaLN gates off zero so cross-attn gradient can reach ctx (representative)
    ctx_det = ctx.detach()
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    for _ in range(20):
        opt.zero_grad(); head.flow_matching_loss(z1.detach(), ctx_det, s, ctx_mask=ctx_mask).backward(); opt.step()

    # frozen path: VLM under no_grad -> only head grads
    head.zero_grad(set_to_none=True)
    with torch.no_grad():
        h_frozen = model(input_ids=input_ids, attention_mask=attn,
                         output_hidden_states=True, use_cache=False).hidden_states[-1]
    ctx_f, cmask_f = context_sequence_hidden(h_frozen, input_ids, attention_mask=attn,
                                             action_start_id=ASID, max_len=64)
    loss_f = head.flow_matching_loss(z1.detach(), ctx_f, s, ctx_mask=cmask_f)
    loss_f.backward()
    head_grad_params = [p for p in head.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    head_grad_n = sum(p.numel() for p in head_grad_params)
    head_grad_finite = all(torch.isfinite(p.grad).all() for p in head_grad_params)
    vlm_grad_frozen = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"  frozen: head grad params={len(head_grad_params)} ({head_grad_n/1e6:.2f}M elems) "
          f"finite={head_grad_finite}  VLM grads set={vlm_grad_frozen}")

    # full-FT path: unfreeze the LAST LM decoder layer, grad must reach it through ctx
    max_layer = -1
    for name, _ in model.named_parameters():
        for part in name.split("."):
            if part.isdigit():
                max_layer = max(max_layer, int(part))
    for p in model.parameters():
        p.requires_grad_(False)
    unfrozen = []
    tag = f".{max_layer}."
    for name, p in model.named_parameters():
        if tag in name and ("mlp" in name or "self_attn" in name):
            p.requires_grad_(True); unfrozen.append((name, p))
    model.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)
    h_full = model(input_ids=input_ids, attention_mask=attn,
                   output_hidden_states=True, use_cache=False).hidden_states[-1]
    ctx_full, cmask_full = context_sequence_hidden(h_full, input_ids, attention_mask=attn,
                                                   action_start_id=ASID, max_len=64)
    loss_full = head.flow_matching_loss(z1.detach(), ctx_full, s, ctx_mask=cmask_full)
    loss_full.backward()
    lm_grad = [(n, p) for n, p in unfrozen if p.grad is not None and p.grad.abs().sum() > 0]
    lm_grad_finite = all(torch.isfinite(p.grad).all() for _, p in lm_grad)
    print(f"  full-FT: unfroze layer {max_layer} ({len(unfrozen)} tensors); "
          f"got grad on {len(lm_grad)} finite={lm_grad_finite}")
    v6 = (head_grad_n > 5e6 and head_grad_finite and vlm_grad_frozen == 0
          and len(lm_grad) > 0 and lm_grad_finite)
    verdicts["6_grad"] = {"pass": bool(v6),
                          "frozen_head_grad_elems_M": round(head_grad_n/1e6, 2),
                          "frozen_head_grad_finite": bool(head_grad_finite),
                          "frozen_vlm_grads": int(vlm_grad_frozen),
                          "fullft_lm_grad_tensors": len(lm_grad),
                          "fullft_lm_grad_finite": bool(lm_grad_finite),
                          "unfrozen_layer": max_layer}

    # ---------- 7. dtype safety (real bf16 ctx) ----------
    print("\n=== [7] DTYPE SAFETY (real bf16 ctx) ===")
    dt_ok = {}
    for dt in (torch.float32, torch.bfloat16):
        h2 = FMHeadDecoder(autovla_fmhead_config(hidden_size=256, depth=4),
                           num_samples=8, num_steps=10).to(device).fm_head.to(dt)
        l = h2.flow_matching_loss(z1.detach(), ctx_det, torch.zeros_like(s), ctx_mask=ctx_mask)
        tr = h2.sample(ctx_det, torch.zeros_like(s), ctx_mask=ctx_mask, num_samples=4, num_steps=10)
        ok = bool(torch.isfinite(l) and l.dtype == torch.float32 and torch.isfinite(tr).all())
        dt_ok[str(dt)] = {"loss": float(l), "loss_dtype": str(l.dtype), "trajs_finite": bool(torch.isfinite(tr).all()), "pass": ok}
        print(f"  head={str(dt).split('.')[-1]:8s} + bf16 ctx -> loss={float(l):.4f}({l.dtype}) "
              f"trajs_finite={bool(torch.isfinite(tr).all())} -> {'PASS' if ok else 'FAIL'}")
    verdicts["7_dtype"] = {"pass": all(v["pass"] for v in dt_ok.values()), **dt_ok}

    # ---------- summary ----------
    print("\n=== VERDICTS ===")
    allpass = True
    for k, v in verdicts.items():
        allpass &= bool(v["pass"])
        print(f"  {k:16s}: {'PASS' if v['pass'] else 'FAIL'}")
    print(f"  {'ALL':16s}: {'PASS' if allpass else 'FAIL'}")

    result = {"device": device, "B": B, "S": S, "T_ctx": int(T_ctx),
              "head_params_M": round(n_head/1e6, 3), "loss": loss_val,
              "magnitude_table": rows, "verdicts": verdicts, "all_pass": bool(allpass)}
    json.dump(result, open(OUT_JSON, "w"), indent=2)
    print(f"\n[saved] {OUT_JSON}")
    return 0 if allpass else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
