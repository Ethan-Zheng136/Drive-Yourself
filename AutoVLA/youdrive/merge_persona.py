"""Merge AutoVLA base + alpha * LoRA(persona) -> a full ckpt the standard eval can load (LORA=false).
Env: BASE_CKPT, ADAPTER, ALPHA, OUT_CKPT, CONFIG."""
import os, yaml, torch
from peft import PeftModel
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.autovla_agent import AutoVLAAgent

REPO = "/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA"
CONFIG = os.environ.get("CONFIG", os.path.join(REPO, "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
BASE_CKPT = os.environ.get("BASE_CKPT", "/mnt/pfs/zhengguantian/autovla/ckpts/AutoVLA/AutoVLA_PDMS_89.ckpt")
ADAPTER = os.environ["ADAPTER"]
ALPHA = float(os.environ["ALPHA"])
OUT_CKPT = os.environ["OUT_CKPT"]

def main():
    os.chdir(REPO)
    cfg = yaml.safe_load(open(CONFIG))
    ts = TrajectorySampling(time_horizon=cfg["model"]["trajectory"]["time_horizon"],
                            interval_length=cfg["model"]["trajectory"]["interval_length"])
    agent = AutoVLAAgent(trajectory_sampling=ts, checkpoint_path=BASE_CKPT, sensor_data_path=".",
                         config_path=CONFIG, lora_conf={"use_lora": False}, device="cpu")
    sd = torch.load(BASE_CKPT, map_location="cpu")["state_dict"]
    agent.autovla.load_state_dict({k.replace("autovla.", ""): v for k, v in sd.items()}, strict=False)
    # attach adapter, scale by alpha, then merge into base weights
    agent.autovla.vlm = PeftModel.from_pretrained(agent.autovla.vlm, ADAPTER)
    for _, mod in agent.autovla.vlm.named_modules():
        if hasattr(mod, "scaling") and isinstance(getattr(mod, "scaling"), dict):
            for adp in list(mod.scaling.keys()):
                mod.scaling[adp] = mod.scaling[adp] * ALPHA
    agent.autovla.vlm = agent.autovla.vlm.merge_and_unload()  # bakes alpha*LoRA into weights
    # save in the format the eval agent expects: {"state_dict": {"autovla.<k>": v}}
    merged = {"autovla." + k: v for k, v in agent.autovla.state_dict().items()}
    os.makedirs(os.path.dirname(OUT_CKPT), exist_ok=True)
    torch.save({"state_dict": merged}, OUT_CKPT)
    print(f"[merge] alpha={ALPHA} -> {OUT_CKPT}  ({len(merged)} tensors)", flush=True)

if __name__ == "__main__":
    main()
