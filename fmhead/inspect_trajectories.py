"""inspect_trajectories.py -- decisive H1-vs-H2 diagnostic for the FMHead PDMS collapse.

For ~N navtest scenes, dump and compare the THREE trajectories exactly as they'd reach PDM:
  * FMHead  : the poses the FMHeadAutoVLAAgent feeds to PDM (SAME code path:
              agent.compute_trajectory(agent_input, scene) -> Trajectory.poses),
  * GT      : the human future trajectory (scene.get_future_trajectory().poses),
  * Codebook: the stock AutoVLA codebook decode from the SAME base weights
              (AutoVLA.predict, i.e. the class method, bypassing the instance monkeypatch).

Per-scene + aggregate metrics:
  forward extent (endpoint x, max x), endpoint (x,y), per-waypoint L2 to GT (ADE/FDE),
  heading |max| / monotonicity, and a geometric-plausibility flag.

KEY QUESTION -> printed verdict:
  H2 (geometry/format/scale/frame bug; retrain won't help): FMHead is systematically
     SHORT / wrong-scale / wrong-frame vs GT (e.g. endpoint-x ratio << 1 or ADE huge)
     -> explains ego_progress ~0.30.
  H1 (trajectories geometrically fine, just not safest; retrain/selection helps):
     FMHead endpoint-x ~ GT, ADE small, headings sane -> the problem is selection/safety.

VISION NODE ONLY (needs real camera forward + navsim scene loading). Command at bottom.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, torch

FM = os.path.dirname(os.path.abspath(__file__))
AV = "/root/workspace/closed_loop/navsim_candidates_survey/repos/AutoVLA"
for p in (FM, AV, AV + "/navsim"):
    if p not in sys.path:
        sys.path.insert(0, p)


def ade_fde(pred_xy, gt_xy):
    L = min(len(pred_xy), len(gt_xy))
    d = np.linalg.norm(np.asarray(pred_xy)[:L, :2] - np.asarray(gt_xy)[:L, :2], axis=-1)
    return float(d.mean()), float(d[-1])


def heading_stats(poses):
    h = np.asarray(poses)[:, 2]
    return float(np.abs(h).max()), bool(np.all(np.diff(h) >= -0.05))  # near-monotone for a smooth turn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmhead_ckpt", required=True)
    ap.add_argument("--base_ckpt", default="/mnt/pfs/zhengguantian/autovla/ckpts/AutoVLA/AutoVLA_PDMS_89.ckpt")
    ap.add_argument("--normalizer", default=os.path.join(FM, "traj_norm_stats_gt.json"))
    ap.add_argument("--config_path", default=os.path.join(AV, "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
    ap.add_argument("--metric_cache", default="/mnt/pfs/zhengguantian/autovla/exp/metric_cache_navtest_v1")
    ap.add_argument("--json_data", default=os.path.join(AV, "dataset/nuplan/navtest_nocot"))
    ap.add_argument("--sensor_blobs", default=os.environ.get("OPENSCENE_SENSOR_BLOBS", ""))
    ap.add_argument("--navsim_log", default=os.environ.get("OPENSCENE_LOG_PATH", ""))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--select", default="medoid", help="medoid|pdm|oracle (what the agent feeds PDM)")
    ap.add_argument("--out", default=os.path.join(FM, "inspect_trajectories_out"))
    args = ap.parse_args()
    os.chdir(AV)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from pathlib import Path
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.common.dataloader import SceneLoader, MetricCacheLoader
    from navsim.common.dataclasses import SceneFilter
    from models.autovla import AutoVLA
    from fmhead_navsim_agent import FMHeadAutoVLAAgent

    ts = TrajectorySampling(num_poses=10, interval_length=0.5, time_horizon=5.0)
    agent = FMHeadAutoVLAAgent(
        trajectory_sampling=ts, checkpoint_path=args.base_ckpt, sensor_data_path=".",
        config_path=args.config_path, lora_conf={"use_lora": False}, device=device,
        fmhead_ckpt_path=args.fmhead_ckpt, fmhead_normalizer_path=args.normalizer,
        fmhead_style_off=True, fmhead_select_mode=args.select,
        fmhead_metric_cache_path=args.metric_cache,
    )
    agent.initialize()

    mcl = MetricCacheLoader(Path(args.metric_cache))
    sf_yaml = os.path.join(AV, "navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml")
    scene_filter: SceneFilter = instantiate(OmegaConf.load(sf_yaml))
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(args.sensor_blobs) if args.sensor_blobs else None,
        data_path=Path(args.navsim_log) if args.navsim_log else None,
        scene_filter=scene_filter, sensor_config=agent.get_sensor_config(),
    )
    tokens = [t for t in scene_loader.tokens if t in mcl.tokens][:args.n]
    print(f"[inspect] {len(tokens)} scenes | select={args.select} base={os.path.basename(args.base_ckpt)}", flush=True)

    rows = []
    for k, token in enumerate(tokens):
        try:
            agent_input = json.load(open(os.path.join(args.json_data, f"{token}.json")))
            scene = scene_loader.get_scene_from_token(token)
            # 1) FMHead (exact agent-to-PDM poses)
            fm_traj, _ = agent.compute_trajectory(agent_input, scene)
            fm = np.asarray(fm_traj.poses)                                  # (10,3)
            # 2) GT
            gt = np.asarray(scene.get_future_trajectory(ts.num_poses).poses)  # (10,3)
            # 3) Codebook (class method -> not the instance monkeypatch)
            features = agent._build_features(agent_input)
            cb_poses, _ = AutoVLA.predict(agent.autovla, features)
            cb = np.asarray(cb_poses)[: ts.num_poses]                        # (10,3)

            fm_ade, fm_fde = ade_fde(fm, gt); cb_ade, cb_fde = ade_fde(cb, gt)
            hmax, mono = heading_stats(fm)
            rows.append(dict(token=token, fm_endx=float(fm[-1, 0]), fm_endy=float(fm[-1, 1]),
                             fm_maxx=float(fm[:, 0].max()), gt_endx=float(gt[-1, 0]), gt_endy=float(gt[-1, 1]),
                             cb_endx=float(cb[-1, 0]), fm_ade=fm_ade, fm_fde=fm_fde, cb_ade=cb_ade,
                             fm_head_absmax=hmax, fm_head_mono=mono))
            print(f"  [{k:02d}] {token[:8]}  FM end=({fm[-1,0]:5.1f},{fm[-1,1]:5.1f}) maxx={fm[:,0].max():5.1f} "
                  f"| GT end=({gt[-1,0]:5.1f},{gt[-1,1]:5.1f}) | CB end=({cb[-1,0]:5.1f}) "
                  f"| FM_ADE={fm_ade:.2f} CB_ADE={cb_ade:.2f} head|max|={hmax:.2f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [{k:02d}] {token[:8]} FAILED: {e!r}", flush=True)

    if not rows:
        print("[inspect] no scenes succeeded"); return 1
    a = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0] if k != "token"}
    # aggregate
    fm_prog = a["fm_endx"].mean(); gt_prog = a["gt_endx"].mean(); cb_prog = a["cb_endx"].mean()
    ratio = fm_prog / gt_prog if gt_prog else float("nan")
    print("\n===== AGGREGATE (n=%d, select=%s) =====" % (len(rows), args.select))
    print(f"  endpoint-x  mean:  FMHead={fm_prog:6.2f}  GT={gt_prog:6.2f}  Codebook={cb_prog:6.2f}   FM/GT ratio={ratio:.2f}")
    print(f"  ADE-to-GT   mean:  FMHead={a['fm_ade'].mean():.3f}  Codebook={a['cb_ade'].mean():.3f}")
    print(f"  FDE-to-GT   mean:  FMHead={a['fm_fde'].mean():.3f}")
    print(f"  FMHead heading |max| mean={a['fm_head_absmax'].mean():.3f}  monotone_frac={a['fm_head_mono'].mean():.2f}")

    # H1/H2 verdict logic
    verdict = []
    if ratio < 0.6 or a["fm_ade"].mean() > 3.0:
        verdict.append("H2 LIKELY: FMHead is systematically short/wrong-scale/frame "
                       f"(FM/GT endpoint-x ratio={ratio:.2f}, ADE={a['fm_ade'].mean():.2f}) -> format/geometry bug; retrain won't fix by itself.")
    if ratio >= 0.8 and a["fm_ade"].mean() < 1.5:
        verdict.append("H1 LIKELY: FMHead trajectories are geometrically fine "
                       f"(FM/GT ratio={ratio:.2f}, ADE={a['fm_ade'].mean():.2f}) -> the problem is safety/selection, not geometry; better selection (pdm) / retrain helps.")
    if a["fm_head_absmax"].mean() > 2.5:
        verdict.append("WARN: FMHead headings near +-pi on average -> heading/format still suspect.")
    if not verdict:
        verdict.append(f"MIXED/INCONCLUSIVE: FM/GT ratio={ratio:.2f}, ADE={a['fm_ade'].mean():.2f} -- inspect per-scene rows.")
    print("\n  VERDICT:"); [print("   - " + v) for v in verdict]

    json.dump({"rows": rows, "aggregate": {k: float(v.mean()) for k, v in a.items()},
               "endpoint_x_ratio_FM_over_GT": float(ratio), "verdict": verdict},
              open(os.path.join(args.out, "inspect_results.json"), "w"), indent=2)

    # optional plot (xy overlay of first up-to-6 scenes)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        m = min(6, len(rows)); fig, axes = plt.subplots(1, m, figsize=(3 * m, 3), squeeze=False)
        for j in range(m):
            tok = rows[j]["token"]
            agent_input = json.load(open(os.path.join(args.json_data, f"{tok}.json")))
            scene = scene_loader.get_scene_from_token(tok)
            fm = np.asarray(agent.compute_trajectory(agent_input, scene)[0].poses)
            gt = np.asarray(scene.get_future_trajectory(ts.num_poses).poses)
            cb = np.asarray(AutoVLA.predict(agent.autovla, agent._build_features(agent_input))[0])[:ts.num_poses]
            ax = axes[0][j]
            ax.plot(gt[:, 0], gt[:, 1], "g-o", ms=3, label="GT")
            ax.plot(fm[:, 0], fm[:, 1], "r-o", ms=3, label="FMHead")
            ax.plot(cb[:, 0], cb[:, 1], "b--o", ms=2, label="Codebook")
            ax.set_title(tok[:8]); ax.axis("equal"); ax.grid(alpha=0.3)
            if j == 0: ax.legend(fontsize=7)
        png = os.path.join(args.out, "inspect_trajectories.png"); fig.tight_layout(); fig.savefig(png, dpi=110)
        print(f"[inspect] plot -> {png}")
    except Exception as e:  # noqa: BLE001
        print(f"[inspect] plot skipped: {e!r}")
    print(f"[inspect] json -> {os.path.join(args.out,'inspect_results.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
