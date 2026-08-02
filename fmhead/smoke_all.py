"""smoke_all.py -- rigorous local pre-cluster smoke of the FMHead pipeline (items 1,3,5).

Loads ONE real (text-only) Qwen2.5-VL-3B via FMHeadSFTAutoVLA and reuses it for:
  1. c-extraction consistency (answer_position_hidden: train vs infer cos, OLD vs NEW)
  3. full-FT single-process: training_step/val forward+backward, BOTH dtype paths,
     grad flow (LM + FMHead), vision frozen
  5. PDMS-agent fm_predict logic: inference prompt (no action tokens) -> aligned c -> decode

Run: OMP_NUM_THREADS=8 FMHEAD_TEXT_ONLY=1 PYTORCH_NVML_BASED_CUDA_CHECK=0 python smoke_all.py
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np, torch, torch.nn.functional as F

FM = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, FM)
from fmhead_sft import AUTOVLA_ROOT, FMHeadSFTAutoVLA, load_config
from autovla_fmhead import action_region_hidden, answer_position_hidden
os.chdir(AUTOVLA_ROOT)

RES = {}
dev = "cuda:0"
cfg = load_config(os.path.join(FM, "config/fmhead-sft-full.yaml"))  # SFT-schema config
cfg["model"]["sft_model_path"] = None            # skip 7GB base for the smoke
torch.manual_seed(0)
model = FMHeadSFTAutoVLA(cfg).to(dev); model.train()
av = model.autovla; proc = av.processor; vlm = av.vlm; asid = av.action_start_id
scene = json.load(open(sorted(glob.glob("/mnt/pfs/zhengguantian/autovla/persona/nocot_navtrain12k/*.json"))[0]))

# ---- build an inference prompt (add_generation_prompt=True, NO action tokens) ----
prompt = f"You are driving. Ego speed 5.8 m/s. Command: {scene.get('instruction','keep forward')}. Predict the ego trajectory."
text = proc.apply_chat_template([{"role":"user","content":[{"type":"text","text":prompt}]}],
                                tokenize=False, add_generation_prompt=True)
enc = proc(text=[text], images=None, videos=None, return_tensors="pt")
pids, pmask = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
# teacher-forced sequence = prompt + GT-ish action tokens (values >= asid)
act = torch.tensor([[asid+3, asid+7, asid+1, asid+9, asid+2]], device=dev)
tids = torch.cat([pids, act], dim=1); tmask = torch.ones_like(tids)

# =========================== ITEM 1: c consistency ===========================
with torch.no_grad():
    hp = vlm(input_ids=pids, attention_mask=pmask, output_hidden_states=True, use_cache=False).hidden_states[-1]
    ht = vlm(input_ids=tids, attention_mask=tmask, output_hidden_states=True, use_cache=False).hidden_states[-1]
labels = tids.clone(); labels[:, :pids.shape[1]] = -100
c_old_inf = action_region_hidden(hp, labels=None, attention_mask=pmask, action_start_id=asid)
c_old_trn = action_region_hidden(ht, labels=labels, attention_mask=tmask, action_start_id=asid)
c_new_inf = answer_position_hidden(hp, pids, pmask, asid)
c_new_trn = answer_position_hidden(ht, tids, tmask, asid)
cos = lambda a,b: F.cosine_similarity(a.float(), b.float(), dim=-1).item()
old_cos, new_cos = cos(c_old_inf, c_old_trn), cos(c_new_inf, c_new_trn)
item1 = (new_cos > 0.9999) and (old_cos < 0.99)
print(f"[1 c-consistency] OLD cos(infer,train)={old_cos:.4f}  NEW cos={new_cos:.4f}  "
      f"|Δ|/|c|_new={((c_new_inf-c_new_trn).float().norm()/c_new_trn.float().norm()):.2e} -> {'PASS' if item1 else 'FAIL'}")
RES["1_c_consistency"] = {"pass": bool(item1), "old_cos": round(old_cos,4), "new_cos": round(new_cos,4)}

# =========================== ITEM 5: agent fm_predict logic ===========================
# exactly what FMHeadAutoVLAAgent.fm_predict does: inference prompt -> aligned c -> decode
with torch.no_grad():
    c_inf = answer_position_hidden(hp, pids, pmask, asid)     # last prompt token (no action tokens)
    s0 = torch.zeros_like(c_inf)
    poses = model.fm_decoder.decode(c_inf, s0)               # (1,10,3) metres
p = poses[0].float().cpu().numpy()
finite = np.isfinite(p).all()
heads = p[:,2]; head_sane = bool(np.all(np.abs(heads) < np.pi))   # no ±pi garbage
xy_prog = float(p[-1,0])                                     # forward progress (x)
shape_ok = tuple(poses.shape) == (1,10,3)
item5 = bool(finite and shape_ok and head_sane)
print(f"[5 agent-decode]  poses={tuple(poses.shape)} finite={finite} heading|max|={np.abs(heads).max():.3f} "
      f"end_x={xy_prog:.2f} end_y={p[-1,1]:.2f} -> {'PASS' if item5 else 'FAIL'}")
RES["5_agent_decode"] = {"pass": item5, "poses_shape": list(poses.shape), "heading_absmax": float(np.abs(heads).max()),
                          "end_xy": [round(float(p[-1,0]),2), round(float(p[-1,1]),2)]}

# =========================== ITEM 3: full-FT single-process ===========================
T = model.fm_decoder.config.horizon
gt = torch.tensor(scene["gt_trajectory"], dtype=torch.float32, device=dev)[:T].unsqueeze(0)  # (1,10,3)
batch = {"input_ids": tids, "attention_mask": tmask, "labels": labels, "gt_trajectory": gt}

def grad_counts():
    lm = sum(int(p.grad is not None and p.grad.abs().sum()>0) for p in av.vlm.model.parameters())
    fm = sum(int(p.grad is not None and p.grad.abs().sum()>0) for p in model.fm_decoder.parameters())
    vis = sum(float(p.grad.abs().sum()) if p.grad is not None else 0.0 for p in av.vlm.visual.parameters())
    vis_tr = sum(int(p.requires_grad) for p in av.vlm.visual.parameters())
    return lm, fm, vis, vis_tr

def run_fullft(bf16):
    tag = "bf16" if bf16 else "fp32"
    if bf16:
        model.fm_decoder.to(torch.bfloat16)
    else:
        model.fm_decoder.to(torch.float32)
    model.zero_grad(set_to_none=True)
    # warm a couple steps so AdaLN-Zero lets grad reach the LM (grad to c is 0 at init)
    opt = torch.optim.AdamW([p for p in model.fm_decoder.parameters() if p.requires_grad], lr=1e-3)
    losses = []
    for _ in range(4):
        model.zero_grad(set_to_none=True)
        loss = model.training_step(batch, 0)      # uses answer_position_hidden internally
        loss.backward(); losses.append(float(loss.item())); opt.step()
    lm, fm, vis, vis_tr = grad_counts()
    with torch.no_grad():
        vloss = model.validation_step(batch, 0)
    ok = np.isfinite(losses).all() and lm > 0 and fm > 0 and vis == 0.0 and vis_tr == 0 and torch.isfinite(vloss)
    print(f"[3 full-FT {tag}] train_loss={losses[0]:.3f}->{losses[-1]:.3f} val={float(vloss):.3f} "
          f"LM_grad={lm} FM_grad={fm} vision_grad_sum={vis:.1e} vision_trainable={vis_tr} -> {'PASS' if ok else 'FAIL'}")
    return bool(ok), {"tag": tag, "lm_grad": lm, "fm_grad": fm, "vision_grad_sum": vis,
                      "vision_trainable": vis_tr, "loss0": losses[0], "lossN": losses[-1], "val": float(vloss)}

ok32, r32 = run_fullft(bf16=False)
ok16, r16 = run_fullft(bf16=True)
item3 = ok32 and ok16
RES["3_fullft_fp32"] = {"pass": ok32, **r32}
RES["3_fullft_bf16"] = {"pass": ok16, **r16}
print(f"[3 full-FT] both dtype paths -> {'PASS' if item3 else 'FAIL'}")

RES["all_pass_local"] = bool(item1 and item5 and item3)
json.dump(RES, open(os.path.join(FM, "smoke_all_results.json"), "w"), indent=2)
print(f"\n[smoke_all] items 1,3,5 -> {'ALL PASS' if RES['all_pass_local'] else 'FAIL'}  (see smoke_all_results.json)")
sys.exit(0 if RES["all_pass_local"] else 1)
