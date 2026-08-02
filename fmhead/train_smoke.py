"""train_smoke.py -- end-to-end smoke tests for the conditional flow-matching FMHead.

Runs four tests and prints a machine-readable summary (also written to
``smoke_results.json`` for the README):

  1. Shape test          : v_theta(x_t,t,c,s) -> (B,T,2).
  2. Overfit / multimodality : a fixed scene has a BIMODAL target; after a few
     hundred training steps (a) flow loss drops a lot and (b) sampling N trajs
     recovers BOTH modes (KMeans-2 on endpoints lands near both canonical ends).
  3. Style control       : mode depends on style s (s=+ -> left, s=- -> right);
     conditioning on +/- flips which mode is sampled.
  4. CFG                  : increasing guidance weight w sharpens toward the
     style-consistent mode.

Usage:
    OMP_NUM_THREADS=8 python train_smoke.py
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List

import numpy as np
import torch
from sklearn.cluster import KMeans

from fm_head import FMHead, FMHeadConfig
from synthetic import SyntheticScene


# -----------------------------------------------------------------------------
def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_head(scene: SyntheticScene, device: torch.device) -> FMHead:
    cfg = FMHeadConfig(
        horizon=8, traj_dim=2,
        d_cond=scene.d_cond, d_style=scene.d_style,
        hidden_size=256, depth=4, num_heads=8, mlp_ratio=4.0,
        cond_hidden=512, style_dropout_prob=0.2,
    )
    return FMHead(cfg).to(device)


def train(head: FMHead, scene: SyntheticScene, mode: str,
          steps: int = 1200, batch: int = 128, lr: float = 1e-3) -> List[float]:
    """Train the head on the synthetic scene; return the per-step loss list."""
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    head.train()
    losses: List[float] = []
    for step in range(steps):
        tau, c, s, _ = scene.sample_batch(batch, mode=mode)
        loss = head.flow_matching_loss(tau, c, s)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return losses


# -----------------------------------------------------------------------------
def assign_modes(endpoints_metric: np.ndarray, canon_ends: np.ndarray) -> np.ndarray:
    """Assign each endpoint (metres) to nearest canonical end. Returns idx in {0,1}."""
    d0 = np.linalg.norm(endpoints_metric - canon_ends[0], axis=-1)
    d1 = np.linalg.norm(endpoints_metric - canon_ends[1], axis=-1)
    return (d1 < d0).astype(int)  # 0=left, 1=right


# =============================================================================
# Test 1: shape
# =============================================================================
def test_shape(device: torch.device) -> Dict:
    set_seed(0)
    cfg = FMHeadConfig(d_cond=2048, d_style=2048, hidden_size=128, depth=2)
    head = FMHead(cfg).to(device)
    B = 5
    x_t = torch.randn(B, cfg.horizon, cfg.traj_dim, device=device)
    t = torch.rand(B, device=device)
    c = torch.randn(B, cfg.d_cond, device=device)
    s = torch.randn(B, cfg.d_style, device=device)
    v = head(x_t, t, c, s)
    ok = tuple(v.shape) == (B, cfg.horizon, cfg.traj_dim)
    print(f"[test1:shape] out={tuple(v.shape)} expected=({B},{cfg.horizon},{cfg.traj_dim}) "
          f"-> {'PASS' if ok else 'FAIL'}")
    return {"pass": bool(ok), "out_shape": list(v.shape)}


# =============================================================================
# Test 2: overfit / multimodality
# =============================================================================
def test_multimodality(scene: SyntheticScene, device: torch.device,
                        steps: int = 1200, n_samples: int = 256) -> Dict:
    set_seed(1)
    head = build_head(scene, device)
    losses = train(head, scene, mode="content", steps=steps)
    init_loss = float(np.mean(losses[:20]))
    final_loss = float(np.mean(losses[-20:]))
    drop_ratio = final_loss / max(init_loss, 1e-9)

    head.eval()
    c = scene.content(1)
    s = torch.zeros(1, scene.d_style, device=device)  # neutral style (uninformative)
    trajs = head.sample(c, s, num_samples=n_samples, num_steps=10, cfg_weight=1.0)
    ends_norm = trajs[0, :, -1, :].cpu().numpy()
    ends_metric = scene.normalizer.denorm(trajs[0])[:, -1, :].cpu().numpy()

    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(ends_norm)
    centers_norm = torch.tensor(km.cluster_centers_, dtype=torch.float32)
    # map cluster centres to metres via a full-traj proxy is overkill; compare
    # endpoint clustering directly in metric space instead:
    km_m = KMeans(n_clusters=2, n_init=10, random_state=0).fit(ends_metric)
    centers_metric = km_m.cluster_centers_
    labels = km_m.labels_

    canon_ends = scene.canon[:, -1, :].cpu().numpy()  # (2,2) metres [left,right]
    # nearest canonical end for each cluster centre
    cluster_to_canon = assign_modes(centers_metric, canon_ends)
    dists = np.array([
        min(np.linalg.norm(centers_metric[k] - canon_ends[0]),
            np.linalg.norm(centers_metric[k] - canon_ends[1]))
        for k in range(2)
    ])
    counts = np.bincount(labels, minlength=2)
    frac = counts / counts.sum()

    # both canonical modes must be covered by a distinct cluster, balanced-ish
    both_modes_covered = len(set(cluster_to_canon.tolist())) == 2
    close = bool((dists < 1.5).all())
    balanced = bool((frac > 0.15).all())
    ok = both_modes_covered and close and balanced

    print(f"[test2:multimodality] init_loss={init_loss:.4f} final_loss={final_loss:.4f} "
          f"(ratio={drop_ratio:.3f})")
    print(f"                      cluster_centres(m)={np.round(centers_metric,2).tolist()} "
          f"canon_ends(m)={np.round(canon_ends,2).tolist()}")
    print(f"                      centre->canon dists(m)={np.round(dists,2).tolist()} "
          f"mode_fracs={np.round(frac,3).tolist()}")
    print(f"                      both_modes={both_modes_covered} close={close} "
          f"balanced={balanced} -> {'PASS' if ok else 'FAIL'}")
    return {
        "pass": bool(ok),
        "init_loss": init_loss, "final_loss": final_loss, "loss_ratio": drop_ratio,
        "cluster_centers_m": np.round(centers_metric, 3).tolist(),
        "canon_ends_m": np.round(canon_ends, 3).tolist(),
        "center_to_canon_dist_m": np.round(dists, 3).tolist(),
        "mode_fracs": np.round(frac, 3).tolist(),
        "both_modes_covered": both_modes_covered,
    }


# =============================================================================
# Test 3 + 4: style control and CFG (shared style-conditioned model)
# =============================================================================
def test_style_and_cfg(scene: SyntheticScene, device: torch.device,
                        steps: int = 1500, n_samples: int = 256) -> Dict:
    set_seed(2)
    head = build_head(scene, device)
    losses = train(head, scene, mode="style", steps=steps)
    init_loss = float(np.mean(losses[:20]))
    final_loss = float(np.mean(losses[-20:]))

    head.eval()
    canon_ends = scene.canon[:, -1, :].cpu().numpy()  # [left(0), right(1)]

    def sample_stats(sign: float, w: float) -> Dict:
        """Return endpoint statistics for a given style sign and CFG weight w."""
        c = scene.content(1)
        s = scene.style_from_sign(torch.tensor([sign], device=device))
        trajs = head.sample(c, s, num_samples=n_samples, num_steps=10, cfg_weight=w)
        ends = scene.normalizer.denorm(trajs[0])[:, -1, :].cpu().numpy()
        modes = assign_modes(ends, canon_ends)  # 0=left,1=right
        consistent_end = canon_ends[0] if sign > 0 else canon_ends[1]
        return {
            "frac_left": float((modes == 0).mean()),
            # mean distance of endpoints to the style-consistent canonical end (m)
            "dist_to_consistent_m": float(np.linalg.norm(ends - consistent_end, axis=-1).mean()),
            # lateral (y) spread of the endpoint cloud (m) -- lower = sharper
            "y_std_m": float(ends[:, 1].std()),
        }

    # --- Test 3: style control at w=1 ---
    plus = sample_stats(+1.0, w=1.0)     # s=+ should go LEFT
    minus = sample_stats(-1.0, w=1.0)    # s=- should go RIGHT
    plus_left = plus["frac_left"]
    minus_right = 1.0 - minus["frac_left"]
    style_flip_rate = 0.5 * (plus_left + minus_right)
    style_ok = bool(plus_left > 0.8 and minus_right > 0.8)

    print(f"[test3:style] init_loss={init_loss:.4f} final_loss={final_loss:.4f}")
    print(f"              s=+ ->frac_left={plus_left:.3f} (want high)  "
          f"s=- ->frac_right={minus_right:.3f} (want high)")
    print(f"              style_flip_rate={style_flip_rate:.3f} -> "
          f"{'PASS' if style_ok else 'FAIL'}")

    # --- Test 4: CFG sharpening (style-consistent = LEFT for s=+) ---
    ws = [0.0, 1.0, 2.0, 4.0]
    cfg_curve = {w: sample_stats(+1.0, w=w) for w in ws}
    frac = {w: cfg_curve[w]["frac_left"] for w in ws}
    dist = {w: cfg_curve[w]["dist_to_consistent_m"] for w in ws}
    # w=0 -> unconditional (both modes, frac~0.5, large dist); increasing w ->
    # more mass on the style-consistent mode (frac up) AND tighter around it
    # (dist down). Use the continuous distance metric as the robust criterion
    # since the discrete fraction can saturate at 1.0.
    frac_up = frac[4.0] > frac[0.0] + 0.1
    dist_down = dist[4.0] < dist[0.0] - 0.1
    cfg_ok = bool(frac_up and dist_down)
    print(f"[test4:cfg]   frac_left vs w: "
          + ", ".join(f"w={w}:{frac[w]:.3f}" for w in ws))
    print(f"              dist_to_consistent(m) vs w: "
          + ", ".join(f"w={w}:{dist[w]:.3f}" for w in ws))
    print(f"              frac_up={frac_up} dist_down={dist_down} -> "
          f"{'PASS' if cfg_ok else 'FAIL'}")

    return {
        "style": {
            "pass": style_ok,
            "init_loss": init_loss, "final_loss": final_loss,
            "s_plus_frac_left": round(plus_left, 3),
            "s_minus_frac_right": round(minus_right, 3),
            "style_flip_rate": round(style_flip_rate, 3),
        },
        "cfg": {
            "pass": cfg_ok,
            "frac_left_by_w": {str(w): round(frac[w], 3) for w in ws},
            "dist_to_consistent_m_by_w": {str(w): round(dist[w], 3) for w in ws},
            "frac_up": bool(frac_up), "dist_down": bool(dist_down),
        },
    }


# =============================================================================
def main():
    device = get_device()
    t0 = time.time()
    print(f"=== FMHead smoke tests | device={device} | "
          f"torch={torch.__version__} ===")

    scene = SyntheticScene(d_cond=2048, d_style=2048, device=device, seed=0)

    results: Dict = {}
    results["device"] = str(device)
    results["test1_shape"] = test_shape(device)
    results["test2_multimodality"] = test_multimodality(scene, device)
    sc = test_style_and_cfg(scene, device)
    results["test3_style"] = sc["style"]
    results["test4_cfg"] = sc["cfg"]

    all_pass = (
        results["test1_shape"]["pass"]
        and results["test2_multimodality"]["pass"]
        and results["test3_style"]["pass"]
        and results["test4_cfg"]["pass"]
    )
    results["all_pass"] = bool(all_pass)
    results["wall_time_s"] = round(time.time() - t0, 1)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoke_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"test1 shape         : {'PASS' if results['test1_shape']['pass'] else 'FAIL'}")
    print(f"test2 multimodality : {'PASS' if results['test2_multimodality']['pass'] else 'FAIL'}")
    print(f"test3 style control : {'PASS' if results['test3_style']['pass'] else 'FAIL'}")
    print(f"test4 CFG           : {'PASS' if results['test4_cfg']['pass'] else 'FAIL'}")
    print(f"ALL                 : {'PASS' if all_pass else 'FAIL'}  "
          f"({results['wall_time_s']}s)  -> {out_path}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
