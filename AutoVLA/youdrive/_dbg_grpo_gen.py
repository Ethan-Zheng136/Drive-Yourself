"""Diagnose the GRPO rollout nan/inf: load persona_init policy, run one forward, inspect logits."""
import os, sys, yaml, torch
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "navsim"))
from models.autovla import GRPOAutoVLA
from dataset_utils.rft_dataset import RFTDataset

cfg = yaml.safe_load(open("./config/training/youdrive-ddv2-grpo.yaml"))
print("building model (inference=True, skips reference)...", flush=True)
model = GRPOAutoVLA(cfg, inference=True)
sd = torch.load(cfg['model']['sft_model_path'], map_location="cpu")["state_dict"]
msg = model.load_state_dict(sd, strict=False)
print(f"loaded persona_init. missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}", flush=True)
model.autovla = model.autovla.to("cuda").to(torch.bfloat16)
model.autovla.eval()

ds = RFTDataset(cfg['data']['train'], cfg['model'])
data = ds[0]
inputs = model.autovla.get_prompt(data['input_features'])
mi = {k: (v.to("cuda") if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

with torch.no_grad():
    out = model.autovla.vlm(**{k: v for k, v in mi.items() if isinstance(v, torch.Tensor)})
    logits = out.logits
print("=== raw forward logits ===", flush=True)
print("shape", tuple(logits.shape), "dtype", logits.dtype)
lf = logits.float()
print(f"nan={torch.isnan(lf).sum().item()}  inf={torch.isinf(lf).sum().item()}  "
      f"min={lf.min().item():.3f} max={lf.max().item():.3f}")
last = lf[0, -1]
print(f"last-token logits: nan={torch.isnan(last).sum().item()} inf={torch.isinf(last).sum().item()} "
      f"min={last.min().item():.3f} max={last.max().item():.3f}")
# try sampling probs like generate does (temp 0.9)
probs = torch.softmax(last / 0.9, dim=-1)
print(f"softmax probs: nan={torch.isnan(probs).sum().item()} sum={probs.sum().item():.4f} max={probs.max().item():.4f}")

print("=== full generate_sample (no FSDP) ===", flush=True)
try:
    s = model.generate_sample(data, model=model.autovla, device="cuda:0")
    import numpy as np
    print("generate OK. trajectory poses shape:", np.asarray(s['trajectory'].poses).shape)
    print("token:", s['token'])
except Exception as e:
    import traceback; traceback.print_exc()
    print("generate FAILED:", repr(e)[:120])
print("DONE_DIAG", flush=True)
