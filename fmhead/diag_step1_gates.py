"""Step 1 diagnostic: inspect StyleAdapter gate magnitudes in the trained head ckpt.

Loads ONLY the head checkpoint on CPU and prints the learned style-adapter gate
magnitudes + trunk/head weight norms. No modification of any weights.
"""
import sys
import torch

CKPT = "/mnt/pfs/zhengguantian/autovla/persona/fmhead_ckpts_style_ddv2/2026-07-16_10-56-07_x4/fmhead_final.pt"


def main():
    print(f"[step1] loading head ckpt (CPU): {CKPT}")
    obj = torch.load(CKPT, map_location="cpu", weights_only=False)

    # locate the state_dict
    if isinstance(obj, dict):
        print(f"[step1] top-level ckpt keys: {list(obj.keys())[:20]}")
        if "step" in obj:
            print(f"[step1] step = {obj['step']}")
        if "fm_config" in obj:
            print(f"[step1] fm_config = {obj['fm_config']}")
        for key in ("fm_decoder", "state_dict", "model", "head", "fmhead", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                print(f"[step1] using nested state_dict under obj['{key}']")
                sd = obj[key]
                break
        else:
            sd = obj
    else:
        sd = obj.state_dict() if hasattr(obj, "state_dict") else obj

    # normalize: strip common prefixes so we can find style_adapter.*
    def find(substr):
        return {k: v for k, v in sd.items() if substr in k}

    print("\n===== ALL style_adapter.* keys =====")
    sa_keys = sorted([k for k in sd.keys() if "style_adapter" in k])
    for k in sa_keys:
        v = sd[k]
        print(f"  {k:60s} shape={tuple(v.shape)} dtype={v.dtype}")
    if not sa_keys:
        print("  (NONE FOUND) -- printing all top-level keys for context:")
        for k in list(sd.keys())[:60]:
            print("   ", k, tuple(sd[k].shape) if hasattr(sd[k], "shape") else type(sd[k]))
        return

    # ---- block_gates ----
    bg = find("style_adapter.block_gates")
    fg = find("style_adapter.final_gate")
    print("\n===== GATES =====")
    for k, v in bg.items():
        vf = v.float()
        print(f"[block_gates] key={k}")
        print(f"   raw tensor        = {vf.tolist()}")
        print(f"   abs-mean          = {vf.abs().mean().item():.6e}")
        print(f"   abs-max           = {vf.abs().max().item():.6e}")
        print(f"   per-depth abs     = {[round(x,6) for x in vf.abs().tolist()]}")
    for k, v in fg.items():
        vf = v.float()
        print(f"[final_gate]  key={k}")
        print(f"   raw tensor        = {vf.tolist()}")
        print(f"   abs               = {vf.abs().item():.6e}")

    # ---- weight norms for context ----
    print("\n===== WEIGHT L2 NORMS (context) =====")
    def norm_group(substr):
        g = find(substr)
        for k, v in sorted(g.items()):
            print(f"   {k:60s} L2={v.float().norm().item():.6e}")
        return g
    norm_group("style_adapter.trunk")
    norm_group("style_adapter.block_heads")
    norm_group("style_adapter.final_head")

    # ---- verdict ----
    print("\n===== VERDICT =====")
    all_gate_abs = []
    for v in bg.values():
        all_gate_abs += v.float().abs().tolist()
    for v in fg.values():
        all_gate_abs += v.float().abs().tolist()
    gmax = max(all_gate_abs) if all_gate_abs else 0.0
    print(f"   overall gate abs-max = {gmax:.6e}")
    if gmax < 1e-2:
        print("   >>> DEAD GATES: abs-max < 1e-2 -> adapter never turned on; style barely injected.")
    else:
        print("   >>> GATES ALIVE: abs-max >= 1e-2 -> style IS injected; weak expression is downstream.")


if __name__ == "__main__":
    main()
