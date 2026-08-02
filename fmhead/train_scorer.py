"""train_scorer.py -- train the deployable FMHeadScorer by distilling PDM/EPDMS labels.

Consumes the dump_scorer output ({token}.npz: cands, ctx, ctx_mask) + the offline PDM
labels (label_candidates_pdm.py: {token}.npz per-metric (N,) in [0,1]) and trains
fmhead_scorer.FMHeadScorer with per-metric BCE + optional imitation CE (Hydra-MDP KD).
Pool candidates from base + all teacher LoRA dumps to get ONE shared, style-agnostic
scorer. Does NOT touch fm3decoder / any generator.

Eval diagnostics (no GT/simulator at inference): top-1 SELECTION AGREEMENT with the
simulator's argmax, and mean-PDMS-of-selected vs the oracle (max-over-candidates) ceiling.

  # real:
  python train_scorer.py --dump_dir DUMP --labels_dir LABELS --out_dir OUT --epochs 20
  # self-contained smoke (synthetic data; asserts loss-down + agreement-up + ckpt reload):
  python train_scorer.py --smoke
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

FMHEAD_DIR = os.path.dirname(os.path.abspath(__file__))
if FMHEAD_DIR not in sys.path:
    sys.path.insert(0, FMHEAD_DIR)

from fmhead_scorer import FMHeadScorer, V1_METRICS, V1_WEIGHTS  # noqa: E402


class CandidateDataset(Dataset):
    """(ctx, ctx_mask, cands, per-metric labels) per token, intersected over dump & labels."""

    def __init__(self, dump_dir: str, labels_dir: str, metrics=V1_METRICS, limit=None):
        self.metrics = list(metrics)
        dump = {os.path.splitext(os.path.basename(f))[0]: f
                for f in glob.glob(os.path.join(dump_dir, "*.npz"))}
        lab = {os.path.splitext(os.path.basename(f))[0]: f
               for f in glob.glob(os.path.join(labels_dir, "*.npz"))}
        self.tokens = sorted(set(dump) & set(lab))
        if limit:
            self.tokens = self.tokens[:limit]
        self.dump, self.lab = dump, lab
        assert self.tokens, f"no overlapping tokens between {dump_dir} and {labels_dir}"

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        tok = self.tokens[i]
        d = np.load(self.dump[tok])
        lz = np.load(self.lab[tok])
        ctx = torch.from_numpy(d["ctx"].astype(np.float32))          # (T_ctx, 2048)
        ctx_mask = torch.from_numpy(d["ctx_mask"].astype(bool))      # (T_ctx,)
        cands = torch.from_numpy(d["cands"].astype(np.float32))      # (N, T, 3)
        targets = {m: torch.from_numpy(lz[m].astype(np.float32)) for m in self.metrics if m in lz}
        score = torch.from_numpy(lz["score"].astype(np.float32)) if "score" in lz else None
        return ctx, ctx_mask, cands, targets, score


def collate(batch):
    ctx = torch.stack([b[0] for b in batch])
    ctx_mask = torch.stack([b[1] for b in batch])
    cands = torch.stack([b[2] for b in batch])
    metrics = batch[0][3].keys()
    targets = {m: torch.stack([b[3][m] for b in batch]) for m in metrics}
    score = (torch.stack([b[4] for b in batch]) if batch[0][4] is not None else None)
    return ctx, ctx_mask, cands, targets, score


@torch.no_grad()
def evaluate(scorer, loader, device):
    """Top-1 agreement with simulator argmax + selected/oracle mean PDMS."""
    scorer.eval()
    agree = tot = 0
    sel_pdms, oracle_pdms = [], []
    for ctx, ctx_mask, cands, _, score in loader:
        ctx, ctx_mask, cands = ctx.to(device), ctx_mask.to(device), cands.to(device)
        out = scorer(ctx, cands, env_mask=ctx_mask, weights=V1_WEIGHTS)
        sel = out["selected_index"].cpu()
        if score is not None:
            true_best = score.argmax(dim=1)
            agree += int((sel == true_best).sum()); tot += sel.numel()
            for b in range(sel.shape[0]):
                sel_pdms.append(float(score[b, sel[b]]))
                oracle_pdms.append(float(score[b].max()))
    scorer.train()
    res = {}
    if tot:
        res["top1_agreement"] = agree / tot
        res["sel_mean_pdms"] = float(np.mean(sel_pdms))
        res["oracle_mean_pdms"] = float(np.mean(oracle_pdms))
    return res


def train(dump_dir, labels_dir, out_dir, epochs=20, lr=2e-4, batch_size=32,
          d_model=256, n_decoder_layers=3, num_workers=4, limit=None, device=None, log_every=50):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = CandidateDataset(dump_dir, labels_dir, limit=limit)
    # infer T (num_poses) + env_in_dim from the first sample
    ctx0, _, cands0, _, _ = ds[0]
    T, env_in = cands0.shape[1], ctx0.shape[-1]
    # DETERMINISTIC held-out split by token hash (~10% val) so the reported agreement / mean
    # PDMS is measured on scenes the scorer never trained on (leakage-free offline number).
    def _is_val(tok):
        return int(hashlib.md5(tok.encode()).hexdigest(), 16) % 10 == 0
    val_idx = [i for i, t in enumerate(ds.tokens) if _is_val(t)]
    tr_idx = [i for i, t in enumerate(ds.tokens) if not _is_val(t)]
    if not val_idx:                       # tiny (smoke) -> fall back to in-sample eval
        val_idx = tr_idx
    print(f"[train_scorer] tokens={len(ds)} (train={len(tr_idx)} val={len(val_idx)}) "
          f"T={T} env_in={env_in} device={device}", flush=True)
    loader = DataLoader(Subset(ds, tr_idx), batch_size=batch_size, shuffle=True,
                        collate_fn=collate, num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=batch_size, shuffle=False,
                            collate_fn=collate, num_workers=num_workers, drop_last=False)

    scorer = FMHeadScorer(env_in_dim=env_in, d_model=d_model, n_decoder_layers=n_decoder_layers,
                          num_poses=T, metrics=V1_METRICS).to(device)
    opt = torch.optim.AdamW(scorer.parameters(), lr=lr, weight_decay=0.01)
    os.makedirs(out_dir, exist_ok=True)

    losses = []
    step = 0
    t0 = time.time()
    for ep in range(epochs):
        for ctx, ctx_mask, cands, targets, _ in loader:
            ctx, ctx_mask, cands = ctx.to(device), ctx_mask.to(device), cands.to(device)
            targets = {m: t.to(device) for m, t in targets.items()}
            out = scorer(ctx, cands, env_mask=ctx_mask, weights=V1_WEIGHTS)
            loss = scorer.distill_loss(out, targets)["total"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            if step % log_every == 0:
                print(f"[train_scorer] ep{ep} step{step} loss={loss.item():.4f} "
                      f"({(time.time()-t0)/(step+1)*1000:.0f} ms/step)", flush=True)
            step += 1
    ev = evaluate(scorer, val_loader, device)
    ckpt = os.path.join(out_dir, "scorer_final.pt")
    torch.save({"scorer": scorer.state_dict(),
                "config": {"env_in_dim": env_in, "d_model": d_model,
                           "n_decoder_layers": n_decoder_layers, "num_poses": T,
                           "metrics": V1_METRICS},
                "eval": ev}, ckpt)
    print(f"[train_scorer] DONE loss {np.mean(losses[:20]):.4f}->{np.mean(losses[-20:]):.4f} "
          f"eval={ev} -> {ckpt}", flush=True)
    return ckpt, losses, ev


# ---------------------------------------------------------------------------
def _make_synthetic(root, n_tokens=64, N=16, T=10, T_ctx=32, env_in=2048, seed=0):
    """Synthetic dump + labels with a LEARNABLE signal (labels are a deterministic
    function of the candidate), so the scorer's loss must drop and agreement rise."""
    rng = np.random.default_rng(seed)
    dump_dir = os.path.join(root, "dump"); lab_dir = os.path.join(root, "labels")
    os.makedirs(dump_dir, exist_ok=True); os.makedirs(lab_dir, exist_ok=True)
    for i in range(n_tokens):
        tok = f"tok{i:04d}"
        cands = rng.standard_normal((N, T, 3)).astype(np.float32)
        ctx = rng.standard_normal((T_ctx, env_in)).astype(np.float16)
        mask = np.ones(T_ctx, dtype=bool)
        np.savez(os.path.join(dump_dir, f"{tok}.npz"), cands=cands, ctx=ctx, ctx_mask=mask)
        # Realistic learnable targets, all deterministic functions of the candidate:
        #   NC/DAC/DDC -> BINARY {0,1} (per-scene median split so ~half pass -> non-degenerate),
        #   EP/TTC/comfort -> CONTINUOUS in [0,1] (as in real PDM weighted metrics).
        ep_dist = np.linalg.norm(cands[:, -1, :2], axis=-1)          # (N,)
        rough = np.abs(np.diff(cands[:, :, :2], axis=1)).sum(axis=(1, 2))
        head = np.abs(cands[:, :, 2]).sum(axis=1)                     # heading magnitude
        def med_bin(x):                                              # per-scene median split
            return (x > np.median(x)).astype(np.float32)
        def sig(x):
            return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)
        labels = {
            "no_at_fault_collisions": med_bin(-ep_dist),             # closer endpoint -> safer
            "drivable_area_compliance": med_bin(-rough),             # smoother -> in-lane
            "driving_direction_compliance": med_bin(-head),          # less heading swing
            "ego_progress": sig(ep_dist - ep_dist.mean()),           # farther -> more progress
            "time_to_collision_within_bound": sig(ep_dist.mean() - ep_dist),
            "comfort": sig(rough.mean() - rough),
        }
        score = (labels["no_at_fault_collisions"] * labels["drivable_area_compliance"]
                 * labels["driving_direction_compliance"]
                 * (5 * labels["ego_progress"] + 5 * labels["time_to_collision_within_bound"]
                    + 2 * labels["comfort"]) / 12.0).astype(np.float32)
        labels["score"] = score
        np.savez(os.path.join(lab_dir, f"{tok}.npz"), **labels)
    return dump_dir, lab_dir


def _smoke():
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as root:
        # SMALL synthetic (fast CPU smoke): tiny env dim / few ctx tokens / few tokens / tiny net.
        dump_dir, lab_dir = _make_synthetic(root, n_tokens=16, N=8, T=8, T_ctx=8, env_in=64)
        out_dir = os.path.join(root, "out")
        ckpt, losses, ev = train(dump_dir, lab_dir, out_dir, epochs=20, lr=5e-4,
                                 batch_size=8, d_model=64, n_decoder_layers=1,
                                 num_workers=0, device="cpu", log_every=20)
        init, fin = float(np.mean(losses[:20])), float(np.mean(losses[-20:]))
        # reload check (rebuild from saved config so architecture matches exactly)
        sd = torch.load(ckpt, map_location="cpu")
        c = sd["config"]
        sc2 = FMHeadScorer(env_in_dim=c["env_in_dim"], d_model=c["d_model"],
                           n_decoder_layers=c["n_decoder_layers"], num_poses=c["num_poses"],
                           metrics=c["metrics"])
        sc2.load_state_dict(sd["scorer"])
        loss_ok = fin < 0.9 * init
        agree_ok = ev.get("top1_agreement", 0) > 0.25   # chance for N=8 is ~0.125
        print("\n=== SMOKE SUMMARY (train_scorer) ===")
        print(f"loss first20->last20 : {init:.4f} -> {fin:.4f} (ratio={fin/init:.2f}) -> "
              f"{'PASS' if loss_ok else 'FAIL'}")
        print(f"top1 agreement       : {ev.get('top1_agreement'):.3f} (chance~0.125) -> "
              f"{'PASS' if agree_ok else 'FAIL'}")
        print(f"sel/oracle mean PDMS : {ev.get('sel_mean_pdms'):.4f} / {ev.get('oracle_mean_pdms'):.4f}")
        print(f"ckpt save+reload     : PASS ({os.path.basename(ckpt)})")
        ok = loss_ok and agree_ok
        print(f"ALL                  : {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir")
    ap.add_argument("--labels_dir")
    ap.add_argument("--out_dir")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        return _smoke()
    assert args.dump_dir and args.labels_dir and args.out_dir, \
        "need --dump_dir --labels_dir --out_dir (or --smoke)"
    train(args.dump_dir, args.labels_dir, args.out_dir, epochs=args.epochs, lr=args.lr,
          batch_size=args.batch_size, d_model=args.d_model, num_workers=args.num_workers,
          limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
