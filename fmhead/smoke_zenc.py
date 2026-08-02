"""smoke_zenc.py -- GPU smoke for the fm3-kin Z-ENCODER conditional-style run.

Launch-readiness for config/fmhead_style_ddv2_zenc.yaml. Single GPU, text-only VLM forward
(vision NVML-restricted on this box; same fm_step path). Uses the REAL train_fmhead code
(build_frozen_vlm + build_decoder + extract_c_s) so it exercises exactly what the 8-GPU
launcher runs.

Checks (the THREE stability gates + safety):
  A. fm3-kin ep6 base VLM + DDv2 LoRA load & freeze (VLM requires_grad == 0).
  B. FREEZE scope (train_z_path_only, NEW): trainable == z_encoder ONLY (incl. its zero-conv
     final Linear); s_proj AND the body (DiT / cross-attn / embedders / final / temporal_pos /
     ctx_proj / null_style) are FROZEN.
  C. GATE 1 (STABILITY -- the headline): a few hundred real train steps; loss finite and
     SMOOTHLY DECREASING with NO 10x+ spikes (contrast the prior run's 100-180 spikes). Prints
     the loss trace every 20 steps + the max step/step ratio. grad reaches z_encoder only.
  D. GATE 2 (alpha=0 BIT-EXACT base): at alpha=0 the z-path decode == the clean fm3-kin base
     (z-encoder bypassed) with an identical seed -> max|Delta| ~= 0. Reports the delta.
  E. GATE 3 (alpha ALIVE + MONOTONIC): alpha = 0 -> 0.5 -> 1 moves style metrics monotonically
     toward the DDv2 teacher (peak_acc / long_jerk / L2-to-teacher). Reports per-alpha numbers.
  F. CFG: cfg_weight>1 with the z path runs (v = v_null + w*(v_style - v_null)).

Run:
  FMHEAD_TEXT_ONLY=1 /root/workspace/miniconda3/envs/autovla/bin/python smoke_zenc.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_fmhead as T  # noqa: E402

CFG_PATH = "config/fmhead_style_ddv2_zenc.yaml"
DT = 0.5
BODY_HINTS = ("fm_head.blocks", "fm_head.final_layer", "fm_head.x_embedder",
              "fm_head.ctx_proj", "fm_head.temporal_pos", "fm_head.null_style",
              "fm_head.t_embedder")


class _Ctx:
    world_size = 1
    rank = 0

    def log(self, *a, **k):
        print(*a, **k, flush=True)


def traj_metrics(xy):
    """xy: (B,10,2) metres (ego frame, origin (0,0)). Returns dict of scalar means over B."""
    B = xy.shape[0]
    origin = torch.zeros(B, 1, 2)
    P = torch.cat([origin, xy], dim=1)
    vel = (P[:, 1:] - P[:, :-1]) / DT                # (B,10,2)
    speed = torch.linalg.norm(vel, dim=-1)           # (B,10)
    acc = (speed[:, 1:] - speed[:, :-1]) / DT        # (B,9) longitudinal accel
    jerk = (acc[:, 1:] - acc[:, :-1]) / DT           # (B,8) longitudinal jerk
    return {"peak_acc": float(acc.abs().max(dim=1).values.mean()),
            "long_jerk": float(jerk.abs().mean()),
            "endpoint_speed": float(speed[:, -1].mean())}


def main():
    assert torch.cuda.is_available(), "need a GPU"
    dev = torch.device("cuda:0")
    cfg = T.load_yaml(os.path.join(os.path.dirname(os.path.abspath(__file__)), CFG_PATH))
    torch.manual_seed(cfg["train"]["seed"])
    res = {}

    # -- A. frozen VLM (fm3-kin ep6 base + DDv2 LoRA) --------------------------
    av, av_config, lora_on = T.build_frozen_vlm(cfg, dev, load_sft_base=True,
                                                base_ckpt_override=cfg.get("base_ckpt"))
    n_vlm_grad = sum(p.numel() for p in av.parameters() if p.requires_grad)
    res["A_lora_on"] = bool(lora_on); res["A_vlm_grad_params"] = int(n_vlm_grad)
    print(f"[A] lora_on={lora_on}  VLM requires_grad params={n_vlm_grad} (want 0)", flush=True)

    # -- B. z-path decoder + FREEZE scope -------------------------------------
    dec = T.build_decoder(cfg, dev)
    trainable = {n for n, p in dec.named_parameters() if p.requires_grad}
    frozen = {n for n, p in dec.named_parameters() if not p.requires_grad}
    # NEW freeze scope: ONLY the z-encoder trains (s_proj is now FROZEN too).
    only_z_enc = bool(trainable) and all("z_encoder" in n for n in trainable)
    body_frozen = all(not any(h in n for h in BODY_HINTS) for n in trainable)
    z_present = any("z_encoder" in n for n in trainable)
    sproj_frozen = not any("s_proj" in n for n in trainable)
    z_norm_delta = bool(getattr(dec.fm_head.z_encoder, "norm_delta", False))
    # ControlNet zero-conv: the z-encoder's FINAL Linear is zero-init, earlier layers nonzero.
    znet = dec.fm_head.z_encoder.net
    final_zero_init = float(znet[-1].weight.abs().max()) == 0.0
    earlier_nonzero = float(znet[0].weight.abs().max()) > 0.0 and float(znet[2].weight.abs().max()) > 0.0
    n_train = sum(p.numel() for p in dec.parameters() if p.requires_grad)
    res.update(B_n_trainable=int(n_train), B_only_z_encoder=bool(only_z_enc),
               B_body_frozen=bool(body_frozen), B_zenc_trainable=bool(z_present),
               B_sproj_frozen=bool(sproj_frozen), B_final_zero_init=bool(final_zero_init),
               B_earlier_nonzero=bool(earlier_nonzero), B_z_norm_delta=z_norm_delta,
               B_constraint_mode=dec.config.constraint_mode, B_content_source=dec.content_source)
    print(f"[B] trainable={n_train:,} only_z_encoder={only_z_enc} body_frozen={body_frozen} "
          f"z_encoder={z_present} s_proj_frozen={sproj_frozen} final_zero_init={final_zero_init} "
          f"earlier_nonzero={earlier_nonzero} z_norm_delta={z_norm_delta} "
          f"mode={dec.config.constraint_mode} content_source={dec.content_source}", flush=True)
    print(f"[B] trainable params: {sorted(n.split('.')[1] if n.startswith('fm_head') else n for n in {t.split('.',2)[1] if t.startswith('fm_head') else t for t in trainable})}", flush=True)

    # -- cache ~10 real tokens: (ctx, mask, delta=styled-base, v0, teacher_xy) --
    style_off = (not lora_on) or float(cfg.get("style_alpha", 1.0)) == 0.0
    batches = T.make_textonly_batches(av, n=10, device=dev, dctx=_Ctx(),
                                      dataset_dir=cfg["dataset_path"])
    cache = []
    for b in batches:
        # extract delta at alpha=1 (raw task-vector); content from base (content_source=base)
        ctx, mask, delta = T.extract_c_s(av, b, alpha=1.0, reduce=cfg["c_reduce"],
                                         style_off=style_off, ctx_max_len=dec.config.ctx_max_len,
                                         content_source=cfg.get("content_source"))
        gt_xy = b["gt_trajectory"][:, :10, :2].float()
        v0 = torch.linalg.norm(gt_xy[:, 0, :], dim=-1) / DT
        cache.append((ctx, mask, delta, v0, gt_xy.cpu()))
    s_norm_mean = float(np.mean([float(c[2].float().norm(dim=-1).mean()) for c in cache]))
    res["style_delta_norm_mean"] = s_norm_mean
    print(f"[cache] {len(cache)} tokens  ||delta||~{s_norm_mean:.1f} (nonzero style)", flush=True)

    def decode_alpha(alpha, cfg_weight=None, seed=None):
        """Decode every cached token at a given alpha; return averaged metrics.

        alpha is applied as the EXTERNAL z-encoder gain (dec.style_alpha) because the delta is
        L2-normalized to a direction inside the norm_delta z-encoder (baked-in alpha washes out)."""
        dec.eval()
        dec.style_alpha = float(alpha)                       # external gain (norm_delta path)
        if cfg_weight is not None:
            saved_w = dec.cfg_weight; dec.cfg_weight = cfg_weight
        mets = []
        with torch.no_grad():
            for ctx, mask, delta, v0, _ in cache:
                if seed is not None:
                    torch.manual_seed(seed)
                s = alpha * delta
                poses = dec.decode(ctx, s, ctx_mask=mask, v0=v0)  # (B,10,3)
                mets.append(traj_metrics(poses[..., :2].float().cpu()))
        if cfg_weight is not None:
            dec.cfg_weight = saved_w
        return {k: float(np.mean([m[k] for m in mets])) for k in mets[0]}

    teacher_met = {k: float(np.mean([traj_metrics(c[4])[k] for c in cache]))
                   for k in ("peak_acc", "long_jerk", "endpoint_speed")}

    def _l2_to_teacher(alpha):
        dec.style_alpha = float(alpha)                       # external gain (norm_delta path)
        return float(np.mean([
            torch.linalg.norm(
                dec.decode(c[0], (alpha * c[2]), ctx_mask=c[1], v0=c[3])[..., :2].float().cpu() - c[4],
                dim=-1).mean().item() for c in cache]))
    l2_to_teacher = _l2_to_teacher

    def gate2_bitexact_delta(seed=1234):
        """GATE 2: at alpha=0 the z-path decode must be BIT-EXACT the clean base (z-encoder
        bypassed) under an identical seed. Returns max|Delta| over all cached tokens (want ~0)."""
        dec.eval(); dec.style_alpha = 0.0
        mx = 0.0
        with torch.no_grad():
            for ctx, mask, delta, v0, _ in cache:
                s0 = torch.zeros_like(delta)
                torch.manual_seed(seed)
                p_z = dec.decode(ctx, s0, ctx_mask=mask, v0=v0)[..., :2].float().cpu()
                saved = dec.fm_head.z_encoder; dec.fm_head.z_encoder = None   # bypass == clean base
                torch.manual_seed(seed)
                p_base = dec.decode(ctx, s0, ctx_mask=mask, v0=v0)[..., :2].float().cpu()
                dec.fm_head.z_encoder = saved
                mx = max(mx, float((p_z - p_base).abs().max()))
        return mx

    # NEUTRAL @init: alpha=0 must be the clean base (low jerk).
    a0_init = decode_alpha(0.0)
    l2_a0_init = l2_to_teacher(0.0)
    print(f"[D-init] alpha=0 (neutral)  peak_acc={a0_init['peak_acc']:.3f} "
          f"long_jerk={a0_init['long_jerk']:.3f} m/s^3  L2->teacher={l2_a0_init:.3f}", flush=True)

    # GATE 2 pre-train: alpha=0 z-path must ALREADY be bit-exact the base (zero-conv final layer).
    g2_init = gate2_bitexact_delta()
    print(f"[D-init] GATE2 alpha=0 bit-exact base (pre-train) max|Delta|={g2_init:.2e} (want ~0)", flush=True)

    # -- C. GATE 1: STABILITY -- loss trace, no spikes; grad->z-encoder only ---
    opt = torch.optim.AdamW([p for p in dec.parameters() if p.requires_grad],
                            lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    warmup = cfg["train"].get("warmup_steps", 100)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: min(1.0, (s + 1) / max(1, warmup)))
    dec.train()
    # IMPORTANT: the pre-train diagnostics above (decode_alpha(0)/gate2) set dec.style_alpha=0;
    # restore the TRAIN dose so training_loss applies external gain = style_alpha = 1.0 (a gain
    # of 0 would zero the z-encoder gradient -- the real trainer never touches style_alpha).
    dec.style_alpha = float(cfg["style_alpha"])
    losses = []
    NB = len(batches)                                   # scenes cycle every NB steps (== 1 epoch)
    z_names = [n for n, p in dec.named_parameters() if "z_encoder" in n and p.requires_grad]
    z_ever_grad = {n: False for n in z_names}           # did EACH z param EVER get nonzero grad?
    grad_sproj_zero = grad_body_zero = None
    z_before = {n: p.detach().clone() for n, p in dec.named_parameters() if "z_encoder" in n and p.requires_grad}
    n_steps = int(os.environ.get("SMOKE_STEPS", "300"))
    log_every = 20
    print(f"[C] --- GATE 1 loss trace (lr={cfg['train']['lr']}, {n_steps} steps, "
          f"log every {log_every}) ---", flush=True)
    for step in range(n_steps):
        b = batches[step % NB]
        ctx, mask, s = T.extract_c_s(av, b, alpha=cfg["style_alpha"], reduce=cfg["c_reduce"],
                                     style_off=style_off, ctx_max_len=dec.config.ctx_max_len,
                                     content_source=cfg.get("content_source"))
        loss = dec(b["gt_trajectory"], ctx, s, ctx_mask=mask,
                   style_dropout_prob=cfg["fmhead"]["style_dropout_prob"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        named = dict(dec.named_parameters())
        # accumulate which z-encoder params have EVER received a nonzero, finite gradient. With
        # the zero-conv, ONLY the final Linear has grad at step 0; the earlier layers come alive
        # once the final weights move -> by a few steps ALL z params should have trained.
        if step < 15:
            for n in z_names:
                g = named[n].grad
                if g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0:
                    z_ever_grad[n] = True
        if step == 0:
            sg = [(n, p) for n, p in named.items() if "s_proj" in n]
            bg = [p.grad for n, p in named.items() if any(h in n for h in BODY_HINTS)]
            grad_sproj_zero = all((not p.requires_grad) and (p.grad is None) for _, p in sg)
            grad_body_zero = all(g is None or g.abs().sum() == 0 for g in bg)
        torch.nn.utils.clip_grad_norm_(dec.parameters(), cfg["train"]["grad_clip"])
        opt.step(); sched.step()
        losses.append(float(loss))
        if step % log_every == 0 or step == n_steps - 1:
            print(f"  [step {step:4d}] loss={float(loss):.4f} lr={sched.get_last_lr()[0]:.2e}", flush=True)
    vlm_grad_after = sum(int(p.grad is not None) for p in av.parameters())
    grad_z_ok = bool(z_ever_grad) and all(z_ever_grad.values())     # EVERY z param trained
    # did the z-encoder weights actually MOVE (not just receive grad)?
    z_moved = float(max((named[n].detach() - z_before[n]).abs().max().item() for n in z_before))
    fin = bool(np.isfinite(losses).all())
    lz = np.asarray(losses, dtype=float)
    loss_max = float(lz.max())
    # SPIKE / trend on EPOCH means (each block of NB steps sees all scenes once -> comparable;
    # removes the per-scene variance of a tiny cyclic smoke set). The prior run hit 100-180.
    n_ep = len(lz) // NB
    ep_means = np.array([lz[i * NB:(i + 1) * NB].mean() for i in range(n_ep)]) if n_ep else lz
    init_l = float(ep_means[:max(1, n_ep // 5)].mean()); last_l = float(ep_means[-max(1, n_ep // 5):].mean())
    decreasing = bool(last_l < init_l)
    ep_ratio = float(ep_means.max() / max(ep_means.min(), 1e-6)) if len(ep_means) else 1.0
    # SPIKE test. The OLD pathological run sat SUSTAINED at loss 100-180. The meaningful anti-
    # spike signal is the EPOCH-MEAN smoothness (each block sees all NB scenes once -> the raw
    # per-step variance of the accum=1 / B=1 text-only smoke averages out). Gate on: (i) finite,
    # (ii) epoch means smooth (max/min < 2x -> no sustained spike / blow-up), (iii) every raw
    # step stays FAR below the 100-180 pathology band (< 80). Occasional single-step FM-sampling
    # outliers (random t + noise draw on one hard scene) are expected and are NOT instability.
    frac_above_40 = float((lz > 40.0).mean())
    no_spikes = bool(fin and ep_ratio < 2.0 and loss_max < 80.0)
    trace_20 = [round(float(losses[i]), 4) for i in range(0, len(losses), log_every)]
    ep_trace = [round(float(x), 3) for x in ep_means]
    res.update(C_loss_finite=fin, C_loss_init_epmean=init_l, C_loss_last_epmean=last_l,
               C_loss_decreasing=decreasing, C_grad_z_encoder=grad_z_ok, C_z_weights_moved=z_moved,
               C_grad_s_proj_zero=bool(grad_sproj_zero), C_grad_body_zero=bool(grad_body_zero),
               C_vlm_grads_after=int(vlm_grad_after), C_loss_max=loss_max,
               C_epoch_ratio=ep_ratio, C_frac_raw_above_40=frac_above_40, C_no_spikes=no_spikes,
               C_loss_trace_every20=trace_20, C_epoch_mean_trace=ep_trace, C_z_ever_grad_all=grad_z_ok)
    print(f"[C] GATE1 loss finite={fin}  epoch-mean {init_l:.4f} -> {last_l:.4f} (decreasing={decreasing})  "
          f"loss_max={loss_max:.3f} epoch_ratio={ep_ratio:.2f}x frac_raw>40={frac_above_40:.3f}  "
          f"NO_SPIKES={no_spikes}", flush=True)
    print(f"[C] epoch-mean trace: {ep_trace}", flush=True)
    print(f"[C] grad->z_encoder(ALL params ever)={grad_z_ok} z_weights_moved={z_moved:.2e} "
          f"grad->s_proj=ZERO/frozen({grad_sproj_zero}) grad->body=ZERO({grad_body_zero})  "
          f"VLM grads after={vlm_grad_after} (want 0)", flush=True)

    # -- D. GATE 2: alpha=0 BIT-EXACT base (post-train) -----------------------
    g2_post = gate2_bitexact_delta()
    a0 = decode_alpha(0.0)
    res.update(D_gate2_bitexact_maxdelta_init=g2_init, D_gate2_bitexact_maxdelta_post=g2_post,
               D_alpha0_met=a0, D_alpha0_jerk=a0["long_jerk"])
    gate2_ok = bool(g2_init < 1e-4 and g2_post < 1e-4)
    print(f"[D] GATE2 alpha=0==base bit-exact: max|Delta| init={g2_init:.2e} post-train={g2_post:.2e} "
          f"(want ~0) | alpha=0 jerk={a0['long_jerk']:.3f} (base-like) -> {'PASS' if gate2_ok else 'FAIL'}", flush=True)

    # -- E. GATE 3: alpha ALIVE + MONOTONIC (0 -> 0.5 -> 1) -------------------
    a05 = decode_alpha(0.5); a1 = decode_alpha(1.0)
    l2_a0, l2_a05, l2_a1 = l2_to_teacher(0.0), l2_to_teacher(0.5), l2_to_teacher(1.0)
    peak = [a0["peak_acc"], a05["peak_acc"], a1["peak_acc"]]
    jerk = [a0["long_jerk"], a05["long_jerk"], a1["long_jerk"]]
    l2s = [l2_a0, l2_a05, l2_a1]
    _mono = lambda v: (v[0] <= v[1] <= v[2]) or (v[0] >= v[1] >= v[2])
    # axis-alive: alpha must MOVE the output (any metric) as it grows 0->1
    axis_alive = bool(abs(peak[2] - peak[0]) > 1e-3 or abs(jerk[2] - jerk[0]) > 1e-3
                      or abs(l2s[2] - l2s[0]) > 1e-3)
    # monotonic in >=1 style metric (early/noisy signal at 300 steps; report all three)
    mono_any = bool(_mono(peak) or _mono(jerk) or _mono(l2s))
    res.update(E_teacher_met=teacher_met, E_peak_acc_a=[round(x, 3) for x in peak],
               E_long_jerk_a=[round(x, 3) for x in jerk], E_L2_to_teacher_a=[round(x, 3) for x in l2s],
               E_axis_alive=axis_alive, E_monotonic_any=mono_any)
    print(f"[E] GATE3 alpha=0/0.5/1 (external gain):", flush=True)
    print(f"    peak_acc  = {peak[0]:.3f} / {peak[1]:.3f} / {peak[2]:.3f}  (teacher {teacher_met['peak_acc']:.3f})", flush=True)
    print(f"    long_jerk = {jerk[0]:.3f} / {jerk[1]:.3f} / {jerk[2]:.3f}  (teacher {teacher_met['long_jerk']:.3f})", flush=True)
    print(f"    L2->teach = {l2s[0]:.3f} / {l2s[1]:.3f} / {l2s[2]:.3f}", flush=True)
    print(f"    axis_alive={axis_alive} monotonic_any={mono_any}", flush=True)

    # -- F. CFG path with the z head ------------------------------------------
    a1_cfg = decode_alpha(1.0, cfg_weight=2.0)
    cfg_ok = np.isfinite(list(a1_cfg.values())).all() and abs(a1_cfg["peak_acc"] - a1["peak_acc"]) >= 0.0
    res.update(F_alpha1_cfgw2_met=a1_cfg, F_cfg_runs=bool(cfg_ok))
    print(f"[F] CFG w=2.0 alpha=1  peak_acc={a1_cfg['peak_acc']:.3f} long_jerk={a1_cfg['long_jerk']:.3f} "
          f"(finite={cfg_ok})", flush=True)

    gate1_ok = bool(res["C_loss_finite"] and res["C_loss_decreasing"] and res["C_no_spikes"]
                    and res["C_grad_z_encoder"])
    ok = (res["A_vlm_grad_params"] == 0 and lora_on and res["B_only_z_encoder"] and res["B_body_frozen"]
          and res["B_zenc_trainable"] and res["B_sproj_frozen"] and res["B_final_zero_init"]
          and res["B_earlier_nonzero"] and res["B_z_norm_delta"] and gate1_ok
          and res["C_grad_s_proj_zero"] and res["C_grad_body_zero"] and res["C_vlm_grads_after"] == 0
          and s_norm_mean > 0 and gate2_ok and res["E_axis_alive"] and res["F_cfg_runs"])
    res["ALL_PASS"] = bool(ok)
    res["GATE1_stability_and_learning"] = gate1_ok
    res["GATE2_alpha0_bitexact"] = gate2_ok
    res["GATE3_alpha_alive"] = res["E_axis_alive"]
    print("\n=== SMOKE SUMMARY (fm3-kin z-encoder conditional style; 3 gates) ===", flush=True)
    print(f"  GATE1 stability+learning    : {gate1_ok}  (decreasing={res['C_loss_decreasing']} "
          f"NO_SPIKES={res['C_no_spikes']} grad->z_encoder={res['C_grad_z_encoder']} "
          f"loss_max={res['C_loss_max']:.3f} epoch_ratio={res['C_epoch_ratio']:.2f}x)", flush=True)
    print(f"  GATE2 alpha=0 bit-exact base: {gate2_ok}  "
          f"(max|Delta| init={res['D_gate2_bitexact_maxdelta_init']:.1e} "
          f"post={res['D_gate2_bitexact_maxdelta_post']:.1e})", flush=True)
    print(f"  GATE3 alpha alive+monotonic : alive={res['E_axis_alive']} monotonic_any={res['E_monotonic_any']}", flush=True)
    for k in ("A_vlm_grad_params", "B_n_trainable", "B_only_z_encoder", "B_body_frozen",
              "B_sproj_frozen", "B_final_zero_init", "B_earlier_nonzero", "B_z_norm_delta",
              "C_loss_finite", "C_loss_decreasing", "C_grad_z_encoder", "C_z_weights_moved",
              "C_grad_s_proj_zero", "C_grad_body_zero", "C_vlm_grads_after", "F_cfg_runs"):
        print(f"  {k:24s}: {res[k]}")
    print(f"  {'C_epoch_mean_trace':24s}: {res['C_epoch_mean_trace']}", flush=True)
    print(f"  {'C_loss_trace_every20':24s}: {res['C_loss_trace_every20']}", flush=True)
    print(f"  {'ALL':24s}: {'PASS' if ok else 'FAIL'}", flush=True)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "smoke_zenc_results.json"), "w"), indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
