"""real_gpu_smoke.py -- end-to-end FMHead<->AutoVLA smoke on the REAL Qwen2.5-VL-3B.

Loads the actual Qwen2.5-VL-3B-Instruct (bf16, GPU), builds a realistic AutoVLA
prompt from a real navtrain sample, and:
  * extracts a GENUINE action/answer-region hidden state c (dim 2048, bf16),
  * attaches a PEFT LoRA (as AutoVLA does) and derives the base-vs-LoRA style
    delta s = alpha*(h_styled - h_base) via `disable_adapter()`,
  * loads FMHead's normalizer from REAL gt_trajectory stats (traj_norm_stats.json),
  * runs FMHeadDecoder.decode (-> navsim poses (1,10,3)) and training_loss,
  * tiny overfit to the real gt_trajectory (loss must drop),
  * style-flip check using the real c as content + +/- style (controllability).

Run: OMP_NUM_THREADS=8 python real_gpu_smoke.py
"""

from __future__ import annotations

import glob
import json
import os
import time

import numpy as np
import torch

from autovla_fmhead import (
    AUTOVLA_HIDDEN_SIZE, AUTOVLA_NUM_POSES, AUTOVLA_VLM_DTYPE,
    FMHeadDecoder, autovla_fmhead_config, style_delta,
)

MODEL_PATH = "/mnt/pfs/zhengguantian/autovla/ckpts/Qwen2.5-VL-3B-Instruct"
DATA_DIR = "/mnt/pfs/zhengguantian/autovla/persona/nocot_ddv2_teacher12k"
NORM_JSON = "/root/workspace/fmhead/traj_norm_stats.json"
PROMPT = ("You are driving. Front/left/right cameras observe the scene. "
          "Ego velocity 5.8 m/s. Command: keep forward. "
          "Predict the future ego trajectory.")


def load_real_vlm(device):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    print(f"[load] Qwen2.5-VL-3B loaded in {time.time()-t0:.1f}s "
          f"(dtype={next(model.parameters()).dtype})")
    return model, processor


def build_inputs(processor, sample_json, device):
    """Build real VLM inputs; use real camera images if present, else text-only.

    Set FMHEAD_TEXT_ONLY=1 to skip the vision tower entirely (needed on this box:
    NVML is permission-restricted, and the large vision activation trips the CUDA
    caching allocator's NVML free-memory query). The LM hidden state is still a
    genuine Qwen2.5-VL-3B forward of the correct dim (2048)."""
    text_only = os.environ.get("FMHEAD_TEXT_ONLY", "0") == "1"
    imgs = []
    if not text_only:
        for cam in ("front_camera_paths", "front_left_camera_paths", "front_right_camera_paths"):
            p = sample_json.get(cam, [None])[0]
            if p and os.path.exists(p):
                imgs.append(p)
    content = [{"type": "image", "image": p} for p in imgs] + [{"type": "text", "text": PROMPT}]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs = None
    if imgs:
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, _ = process_vision_info(messages)
        except Exception as e:  # noqa: BLE001 - report and fall back to text-only
            print(f"[inputs] image processing failed ({type(e).__name__}: {e}); text-only")
            image_inputs = None
    inputs = processor(text=[text], images=image_inputs, videos=None,
                       padding=True, return_tensors="pt")
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    print(f"[inputs] used {len(imgs)} images; input_ids={tuple(inputs['input_ids'].shape)}")
    return inputs


def attach_random_lora(model):
    """Attach a PEFT LoRA (AutoVLA target modules) with a NON-zero delta.

    PEFT zero-inits lora_B => identity at init => base==styled. We fill lora_B with
    small random weights so the adapter genuinely perturbs activations, exercising
    the real base-vs-LoRA style-delta extraction path (a trained persona LoRA would
    replace these weights).
    """
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(
        task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, cfg)
    n = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.data.normal_(0, 0.02)
                n += 1
    print(f"[lora] attached PEFT LoRA (r=16), randomized {n} lora_B tensors")
    return model


@torch.no_grad()
def extract_hidden(model, inputs, disable_adapter):
    import contextlib
    ctx = model.disable_adapter() if (disable_adapter and hasattr(model, "disable_adapter")) \
        else contextlib.nullcontext()
    with ctx:
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden_last = out.hidden_states[-1]                # (B, S, 2048)
    mask = inputs.get("attention_mask")
    if mask is not None:
        idx = mask[0].nonzero()[-1, 0]
        c = hidden_last[:, idx, :]
    else:
        c = hidden_last[:, -1, :]
    return c                                            # (B, 2048)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "this smoke wants GPU"
    torch.manual_seed(0)
    results = {"device": device, "model_path": MODEL_PATH}

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
    sample = json.load(open(files[0]))
    gt = torch.tensor(sample["gt_trajectory"], dtype=torch.float32, device=device)  # (10,3)
    print(f"[data] sample={os.path.basename(files[0])} gt_trajectory={tuple(gt.shape)} "
          f"end_xy={gt[-1,:2].tolist()}")

    ran_real = False
    try:
        model, processor = load_real_vlm(device)
        model = attach_random_lora(model)
        inputs = build_inputs(processor, sample, device)
        c = extract_hidden(model, inputs, disable_adapter=False)      # styled
        h_base = extract_hidden(model, inputs, disable_adapter=True)  # base (LoRA off)
        s = style_delta(c, h_base, alpha=1.0)
        ran_real = True
        print(f"[real] c={tuple(c.shape)}/{c.dtype}  s={tuple(s.shape)}/{s.dtype}  "
              f"|c|={c.float().norm().item():.2f}  |s|_mean={s.float().abs().mean().item():.4f}")
        del model  # free the VLM; keep c, s
        torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001 - explicit fallback with reason
        import traceback; traceback.print_exc()
        print(f"[real] FELL BACK to correct-dim tensors: {type(e).__name__}: {e}")
        c = torch.randn(1, AUTOVLA_HIDDEN_SIZE, device=device, dtype=AUTOVLA_VLM_DTYPE)
        s = 0.1 * torch.randn(1, AUTOVLA_HIDDEN_SIZE, device=device, dtype=AUTOVLA_VLM_DTYPE)
    results["ran_real_vlm"] = ran_real

    assert tuple(c.shape) == (1, AUTOVLA_HIDDEN_SIZE) and c.dtype == AUTOVLA_VLM_DTYPE
    results["c_shape"], results["c_dtype"] = list(c.shape), str(c.dtype)
    results["style_abs_mean"] = float(s.float().abs().mean())

    # --- FMHead decoder with REAL normalizer ---
    dec = FMHeadDecoder(autovla_fmhead_config(hidden_size=256, depth=4),
                        num_samples=64, num_steps=10, cfg_weight=2.0).to(device)
    st = dec.load_normalizer(NORM_JSON)
    print(f"[norm] loaded {st['n_samples']} samples; endpoint mean/std "
          f"x={st['per_waypoint_mean'][-1][0]:.1f}/{st['per_waypoint_std'][-1][0]:.1f} "
          f"y={st['per_waypoint_mean'][-1][1]:.2f}/{st['per_waypoint_std'][-1][1]:.2f}")

    # --- decode + training_loss shape/grad checks ---
    poses = dec.decode(c, s)                                  # (1,10,3)
    gt_b = gt.unsqueeze(0)                                    # (1,10,3)
    dec.zero_grad(set_to_none=True)
    loss = dec.training_loss(gt_b, c.detach(), s.detach())
    loss.backward()
    n_grad = sum(int(p.grad is not None and p.grad.abs().sum() > 0) for p in dec.fm_head.parameters())
    n_par = sum(1 for _ in dec.fm_head.parameters())
    print(f"[decode] poses={tuple(poses.shape)} finite={torch.isfinite(poses).all().item()}  "
          f"init_loss={loss.item():.4f}  grads={n_grad}/{n_par}")
    results["poses_shape"] = list(poses.shape)

    # --- tiny overfit to the REAL gt trajectory (fixed real c, s) ---
    opt = torch.optim.Adam(dec.fm_head.parameters(), lr=1e-3)
    losses = []
    for _ in range(400):
        loss = dec.training_loss(gt_b, c.detach(), s.detach(), style_dropout_prob=0.2)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        losses.append(loss.item())
    init_l, fin_l = float(np.mean(losses[:20])), float(np.mean(losses[-20:]))
    dec.eval()
    poses2 = dec.decode(c, s)                                 # (1,10,3)
    end_err = float(torch.linalg.norm(poses2[0, -1, :2] - gt[-1, :2]).item())
    print(f"[overfit] loss {init_l:.4f}->{fin_l:.4f}  sampled_end={poses2[0,-1,:2].tolist()}  "
          f"gt_end={gt[-1,:2].tolist()}  end_L2={end_err:.3f} m")
    results["overfit"] = {"init_loss": init_l, "final_loss": fin_l, "end_L2_m": end_err}

    # --- style controllability with the REAL c as content ---
    # style axis = the real delta direction; target mode gated by its sign.
    dec2 = FMHeadDecoder(autovla_fmhead_config(hidden_size=256, depth=4),
                         num_samples=128, num_steps=10, cfg_weight=2.0).to(device)
    dec2.load_normalizer(NORM_JSON)
    T = AUTOVLA_NUM_POSES
    tt = torch.arange(1, T + 1, dtype=torch.float32, device=device)
    left = torch.stack([2.3 * tt, +0.7 * tt], dim=-1)         # left-drifting (y>0)
    right = torch.stack([2.3 * tt, -0.7 * tt], dim=-1)        # right-drifting (y<0)
    s_dir = (s / s.float().norm().clamp_min(1e-6)).to(AUTOVLA_VLM_DTYPE)  # unit style dir
    c_fix = c.detach()
    opt2 = torch.optim.Adam(dec2.fm_head.parameters(), lr=1e-3)
    for _ in range(500):
        B = 128
        sign = torch.where(torch.rand(B, device=device) < 0.5, 1.0, -1.0)
        tgt = torch.where((sign > 0).view(B, 1, 1), left.unsqueeze(0), right.unsqueeze(0))
        tgt = tgt + 0.1 * torch.randn_like(tgt)
        gt3 = torch.cat([tgt, torch.zeros(B, T, 1, device=device)], dim=-1)
        c_b = c_fix.expand(B, AUTOVLA_HIDDEN_SIZE)
        s_b = (sign.view(B, 1).to(AUTOVLA_VLM_DTYPE) * s_dir) * 30.0  # scale to a meaningful magnitude
        l = dec2.training_loss(gt3, c_b, s_b)
        opt2.zero_grad(set_to_none=True); l.backward(); opt2.step()
    dec2.eval()

    def frac_left(sign_val, w):
        s_in = (torch.tensor([[sign_val]], device=device, dtype=AUTOVLA_VLM_DTYPE) * s_dir) * 30.0
        tr = dec2.fm_head.sample(c_fix, s_in, num_samples=128, num_steps=10, cfg_weight=w)
        tr = dec2._denormalize(tr)
        ends = tr[0, :, -1, :2].float().cpu().numpy()
        return float((ends[:, 1] > 0).mean())  # y>0 == left
    pl, mr = frac_left(+1.0, 2.0), 1.0 - frac_left(-1.0, 2.0)
    print(f"[style] real-c content + real-style-dir: s+ frac_left={pl:.3f}  s- frac_right={mr:.3f}")
    results["style"] = {"s_plus_frac_left": round(pl, 3), "s_minus_frac_right": round(mr, 3)}

    ok = (tuple(poses.shape) == (1, 10, 3) and fin_l < 0.6 * init_l and pl > 0.8 and mr > 0.8)
    results["all_pass"] = bool(ok)
    out = "/root/workspace/fmhead/real_gpu_results.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\n=== REAL GPU SMOKE: {'PASS' if ok else 'FAIL'} "
          f"(ran_real_vlm={ran_real}) -> {out} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
