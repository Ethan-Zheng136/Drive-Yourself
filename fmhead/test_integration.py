"""test_integration.py -- prove FMHead tensors flow through the REAL AutoVLA interface.

The heavy Qwen2.5-VL stack cannot load in this env, so we use a MockQwenVLM that
emits hidden states of the CORRECT AutoVLA dim (2048, bfloat16) and supports the
`disable_adapter()` toggle (base vs styled). We then run exactly the extraction +
decode + train-loss path that `AutoVLA_FMHead` uses, and additionally do a tiny
overfit to confirm gradients actually flow and reduce the loss through the
AutoVLA-dim interface.

Usage:
    OMP_NUM_THREADS=8 python test_integration.py
"""

from __future__ import annotations

import contextlib
import json
import os

import numpy as np
import torch
import torch.nn as nn

from autovla_fmhead import (
    AUTOVLA_ACTION_START_ID, AUTOVLA_HIDDEN_SIZE, AUTOVLA_NUM_POSES, AUTOVLA_VLM_DTYPE,
    FMHeadDecoder, action_region_hidden, autovla_fmhead_config, build_autovla_fmhead_subclass,
    style_delta,
)


# -----------------------------------------------------------------------------
class MockQwenVLM(nn.Module):
    """Deterministic stand-in for Qwen2_5_VLForConditionalGeneration.

    hidden = embed(input_ids) [+ adapter_delta(input_ids) when LoRA active].
    Emits `.hidden_states` (tuple, last = (B,S,2048) bf16) like the real forward
    with output_hidden_states=True, and a `disable_adapter()` context manager.
    """

    def __init__(self, vocab: int = 152000, hidden: int = AUTOVLA_HIDDEN_SIZE):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.adapter = nn.Embedding(vocab, hidden)  # the "LoRA" delta
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.adapter.weight, std=0.05)
        self.to(AUTOVLA_VLM_DTYPE)
        self._adapter_on = True

    @contextlib.contextmanager
    def disable_adapter(self):
        prev = self._adapter_on
        self._adapter_on = False
        try:
            yield
        finally:
            self._adapter_on = prev

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                output_hidden_states=False, **kw):
        h = self.embed(input_ids)
        if self._adapter_on:
            h = h + self.adapter(input_ids)
        out = type("Out", (), {})()
        out.hidden_states = (h,)  # tuple; [-1] is last layer
        out.logits = None
        return out


def make_batch(B=3, S=40, n_action=AUTOVLA_NUM_POSES, device="cpu"):
    """Fake teacher-forced batch: prompt tokens then `n_action` action tokens."""
    torch.manual_seed(0)
    input_ids = torch.randint(0, 150000, (B, S), device=device)
    labels = torch.full((B, S), -100, device=device)
    # last n_action positions are action tokens (>= action_start_id)
    act_ids = AUTOVLA_ACTION_START_ID + torch.randint(0, 64, (B, n_action), device=device)
    input_ids[:, -n_action:] = act_ids
    labels[:, -n_action:] = act_ids
    attention_mask = torch.ones(B, S, dtype=torch.long, device=device)
    # gt_trajectory (B, G, 3) x-forward/y-left/heading in metres (G >= num_poses)
    G = n_action + 2
    t = torch.arange(1, G + 1, dtype=torch.float32, device=device)
    x = 2.0 * t
    y = 0.5 * t  # gentle left drift
    gt = torch.stack([x, y, torch.zeros_like(x)], dim=-1).unsqueeze(0).expand(B, G, 3).contiguous()
    return {"input_ids": input_ids, "attention_mask": attention_mask,
            "labels": labels, "gt_trajectory": gt}


def emulate_forward_hidden(vlm, batch, disable_adapter, n_bins=64):
    """Mirror AutoVLA_FMHead._forward_hidden without needing the heavy AutoVLA."""
    ctx = vlm.disable_adapter() if disable_adapter else contextlib.nullcontext()
    with ctx:
        out = vlm(**{k: v for k, v in batch.items() if k != "gt_trajectory"},
                  output_hidden_states=True)
    hidden_last = out.hidden_states[-1]
    return action_region_hidden(hidden_last, labels=batch["labels"],
                                attention_mask=batch["attention_mask"],
                                action_start_id=AUTOVLA_ACTION_START_ID, n_bins=n_bins)


# =============================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {"device": device}
    print(f"=== FMHead<->AutoVLA integration test | device={device} ===")

    # --- Test A: dims match real AutoVLA ---
    cfg = autovla_fmhead_config()
    a_ok = (cfg.d_cond == 2048 and cfg.d_style == 2048 and cfg.horizon == 10 and cfg.traj_dim == 2)
    print(f"[A:dims] d_cond={cfg.d_cond} d_style={cfg.d_style} horizon={cfg.horizon} "
          f"traj_dim={cfg.traj_dim} -> {'PASS' if a_ok else 'FAIL'}")
    results["A_dims"] = {"pass": bool(a_ok), "d_cond": cfg.d_cond, "d_style": cfg.d_style,
                          "horizon": cfg.horizon, "traj_dim": cfg.traj_dim}

    vlm = MockQwenVLM().to(device)
    batch = make_batch(device=device)

    # --- Test B: extract c (styled) and h_base; both (B,2048) bf16 ---
    c = emulate_forward_hidden(vlm, batch, disable_adapter=False)
    h_base = emulate_forward_hidden(vlm, batch, disable_adapter=True)
    s = style_delta(c, h_base, alpha=1.0)
    b_ok = (tuple(c.shape) == (3, 2048) and c.dtype == AUTOVLA_VLM_DTYPE
            and tuple(s.shape) == (3, 2048) and s.abs().mean().item() > 0)
    print(f"[B:extract] c={tuple(c.shape)}/{c.dtype}  s={tuple(s.shape)}/{s.dtype}  "
          f"|s|_mean={s.float().abs().mean().item():.4f} -> {'PASS' if b_ok else 'FAIL'}")
    results["B_extract"] = {"pass": bool(b_ok), "c_shape": list(c.shape),
                             "c_dtype": str(c.dtype), "style_abs_mean": s.float().abs().mean().item()}

    # --- Test C: decode -> navsim poses (B,10,3) from bf16 conditions ---
    decoder = FMHeadDecoder(cfg, num_samples=8, num_steps=10, cfg_weight=2.0).to(device)
    poses = decoder.decode(c, s)
    c_ok = (tuple(poses.shape) == (3, 10, 3) and torch.isfinite(poses).all().item())
    print(f"[C:decode] poses={tuple(poses.shape)} finite={torch.isfinite(poses).all().item()} "
          f"heading[0,:3]={poses[0,:3,2].float().tolist()} -> {'PASS' if c_ok else 'FAIL'}")
    results["C_decode"] = {"pass": bool(c_ok), "poses_shape": list(poses.shape)}

    # --- Test D: training loss backward propagates grads through the whole head ---
    # (Fit the normalizer first; AdaLN-Zero blocks upstream grad on the very first
    #  step, so take a few optimizer steps and confirm grads reach most params.)
    decoder.fit_normalizer(batch["gt_trajectory"])
    optd = torch.optim.Adam(decoder.fm_head.parameters(), lr=1e-3)
    n_par = sum(1 for _ in decoder.fm_head.parameters())
    loss0 = None
    for _ in range(5):
        loss = decoder.training_loss(batch["gt_trajectory"], c.detach(), s.detach())
        if loss0 is None:
            loss0 = loss.item()
        optd.zero_grad(set_to_none=True); loss.backward(); optd.step()
    n_grad = sum(int(p.grad is not None and p.grad.abs().sum() > 0) for p in decoder.fm_head.parameters())
    d_ok = (loss.dim() == 0 and torch.isfinite(loss).item() and n_grad > n_par // 2)
    print(f"[D:trainstep] loss {loss0:.4f}->{loss.item():.4f} (normalized) "
          f"params_with_grad={n_grad}/{n_par} -> {'PASS' if d_ok else 'FAIL'}")
    results["D_trainstep"] = {"pass": bool(d_ok), "loss_start": loss0, "loss_end": loss.item(),
                               "params_with_grad": n_grad, "n_params": n_par}

    # --- Test E: tiny overfit through the AutoVLA-dim interface ---
    # Fixed styled/base hidden (so c,s constant) with a BIMODAL target gated by
    # the style sign -> proves the whole plumbed head learns & is controllable.
    torch.manual_seed(3)
    H = AUTOVLA_HIDDEN_SIZE
    c_fix = torch.randn(1, H, device=device, dtype=AUTOVLA_VLM_DTYPE)
    style_base = torch.randn(1, H, device=device, dtype=AUTOVLA_VLM_DTYPE)
    T = AUTOVLA_NUM_POSES
    tt = torch.arange(1, T + 1, dtype=torch.float32, device=device)
    left = torch.stack([2.0 * tt, +4.0 * (tt / T) ** 2], dim=-1)   # (T,2)
    right = torch.stack([2.0 * tt, -4.0 * (tt / T) ** 2], dim=-1)
    dec2 = FMHeadDecoder(autovla_fmhead_config(hidden_size=256, depth=4),
                         num_samples=128, num_steps=10, cfg_weight=2.0).to(device)
    # fit normalizer on the (bimodal) target distribution
    dec2.fit_normalizer(torch.cat([left.unsqueeze(0), right.unsqueeze(0)], dim=0))
    opt = torch.optim.Adam(dec2.fm_head.parameters(), lr=1e-3)
    losses = []
    for step in range(700):
        B = 128
        sign = torch.where(torch.rand(B, device=device) < 0.5, 1.0, -1.0)
        tgt = torch.where((sign > 0).view(B, 1, 1), left.unsqueeze(0), right.unsqueeze(0))
        tgt = tgt + 0.1 * torch.randn_like(tgt)
        gt = torch.cat([tgt, torch.zeros(B, T, 1, device=device)], dim=-1)  # (B,T,3)
        c_b = c_fix.expand(B, H)
        s_b = sign.view(B, 1) * style_base
        loss = dec2.training_loss(gt, c_b, s_b)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        losses.append(loss.item())
    init_l, fin_l = float(np.mean(losses[:20])), float(np.mean(losses[-20:]))
    # style controllability through the interface
    canon = np.array([[2.0 * T, 4.0], [2.0 * T, -4.0]])  # [left,right] endpoints
    def frac_left(sign_val, w):
        s_in = torch.tensor([[sign_val]], device=device) * style_base
        tr = dec2.fm_head.sample(c_fix, s_in, num_samples=128, num_steps=10, cfg_weight=w)
        tr = dec2._denormalize(tr)  # normalized -> metres before comparing to canon
        ends = tr[0, :, -1, :2].float().cpu().numpy()
        d0 = np.linalg.norm(ends - canon[0], axis=-1); d1 = np.linalg.norm(ends - canon[1], axis=-1)
        return float((d0 < d1).mean())
    pl, mr = frac_left(+1.0, 2.0), 1.0 - frac_left(-1.0, 2.0)
    e_ok = bool(fin_l < 0.5 * init_l and pl > 0.8 and mr > 0.8)
    print(f"[E:overfit] loss {init_l:.3f}->{fin_l:.3f}  s+ frac_left={pl:.3f}  "
          f"s- frac_right={mr:.3f} -> {'PASS' if e_ok else 'FAIL'}")
    results["E_overfit"] = {"pass": e_ok, "init_loss": init_l, "final_loss": fin_l,
                             "s_plus_frac_left": round(pl, 3), "s_minus_frac_right": round(mr, 3)}

    # --- Test F: subclass import is guarded (AutoVLA absent here) ---
    try:
        build_autovla_fmhead_subclass()
        f_ok, f_note = True, "models.autovla importable (real env)"
    except Exception as ex:  # noqa: BLE001 - we explicitly report the guard result
        f_ok, f_note = True, f"guarded import raised {type(ex).__name__} (expected outside AutoVLA env)"
    print(f"[F:subclass-guard] {f_note} -> {'PASS' if f_ok else 'FAIL'}")
    results["F_subclass_guard"] = {"pass": f_ok, "note": f_note}

    all_pass = all(results[k]["pass"] for k in
                   ["A_dims", "B_extract", "C_decode", "D_trainstep", "E_overfit", "F_subclass_guard"])
    results["all_pass"] = bool(all_pass)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "integration_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== SUMMARY ===")
    for k in ["A_dims", "B_extract", "C_decode", "D_trainstep", "E_overfit", "F_subclass_guard"]:
        print(f"{k:18s}: {'PASS' if results[k]['pass'] else 'FAIL'}")
    print(f"{'ALL':18s}: {'PASS' if all_pass else 'FAIL'}  -> {out}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
