"""Verify extracted fm3-kin ep6/ep10 artifacts: strict head load + base ckpt well-formed.
CPU-only, no GPU needed. Mirrors FMHeadAutoVLAAgent.initialize strict-load path.
"""
import sys, torch
from autovla_fmhead import FMHeadDecoder, autovla_fmhead_config

RUN = "/mnt/pfs/zhengguantian/autovla/persona/fmhead_fm3_kin/2026-07-17_12-47-42"
NORM = "/root/workspace/fmhead/traj_norm_stats_ctrl.json"
CONSTRAINT = {"mode": "kinematic", "accel_max": 5.0, "yawrate_max": 1.0, "v_max": 25.0, "dt": 0.5}
EPOCHS = ["ep6", "ep10"]

ok = True
for ep in EPOCHS:
    head_path = f"{RUN}/fullft_fm3_kin_{ep}_head.pt"
    base_path = f"{RUN}/fullft_fm3_kin_{ep}_base.ckpt"
    print(f"\n===== verify {ep} =====")

    # ---- head: strict load into FMHeadDecoder (same code path as the agent) ----
    ck = torch.load(head_path, map_location="cpu")
    assert "fm_decoder" in ck, f"{head_path}: missing 'fm_decoder' key"
    has_adapter = any(k.startswith("fm_head.style_adapter") for k in ck["fm_decoder"])
    cfg = autovla_fmhead_config(style_dropout_prob=0.0, use_style_adapter=has_adapter,
                                constraint=CONSTRAINT)
    dec = FMHeadDecoder(cfg)
    dec.load_normalizer(NORM)
    dec.load_state_dict(ck["fm_decoder"], strict=True)  # HARD error on any mismatch
    dec.eval()
    print(f"  head OK: {len(ck['fm_decoder'])} keys strict-loaded "
          f"(has_adapter={has_adapter}, constraint_mode={dec.config.constraint_mode})")

    # ---- base: well-formed, all autovla.* keys under state_dict ----
    bk = torch.load(base_path, map_location="cpu")
    assert "state_dict" in bk, f"{base_path}: missing 'state_dict' key"
    sd = bk["state_dict"]
    non_autovla = [k for k in sd if not k.startswith("autovla.")]
    assert not non_autovla, f"{base_path}: {len(non_autovla)} non-autovla keys, e.g. {non_autovla[:3]}"
    print(f"  base OK: {len(sd)} keys, all 'autovla.*' "
          f"(sample={list(sd.keys())[0]})")

print("\nVERIFY_ALL_OK")
