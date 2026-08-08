# DDv2-Persona on AutoVLA — Research Log / Memory

Goal: inject DiffusionDriveV2 (DDv2) driving **style** into AutoVLA via LoRA **without
losing safety** (navtest PDMS, base = 0.8946).

## Arc so far (chronological conclusions)

1. **Pure imitation CE (SFT)** on DDv2 teacher trajectories → **style IS expressible via LoRA**,
   but PDMS collapsed **0.89 → 0.74** (α=1.0). Unsafe. (Style learned, safety lost.)

2. **KL anchor** (constrain distribution shift toward base) to fix safety:
   - β=20 and β=5 → PDMS recovers to ~0.88, but **style flattened to base** (social metrics, DAI/PAI/SAI, ADE-to-DDv2 all ≈ base).
   - even **β=1 → same**: PDMS ~0.88, style still gone.
   - Root cause: **style IS distribution shift; KL penalizes exactly that → KL is the wrong axis, saturates at β=1.** → **KL fully abandoned.**

3. **GRPO (PDMS reward)** = the SELECTIVE/outcome-space safety constraint. But **standalone GRPO is meaningless**:
   pure PDMS reward chases safety → drifts to base → loses style.

## Current design (the conclusion we are executing)

**The imitation CE (the original SFT train_loss) is the MANDATORY backbone for style.**
**GRPO/PDMS is only a safety constraint layered on top. KL is dropped entirely.**

```
L = ce_weight   * CE(model, DDv2 target action tokens)     # style backbone (the SFT train_loss)
  + pdms_weight * GRPO_policy_gradient(PDMS reward)          # safety constraint only
   (NO KL)
```
- ce_weight == pdms_weight for now (tune later).
- May add MORE style-strengthening losses later (CE is necessary but maybe not sufficient).

## Validation that matters
Not just PDMS. After training MUST measure BOTH:
- safety: navtest PDMS (+ sub-metrics), target ≥ ~0.87 (safe zone).
- style: social metrics (thw/ttc/lead_gap/SAI) + ADE-to-DDv2, must move toward DDv2 (not base).
Pipeline: eval_persona.sh (PDMS) + dump_persona_sweep_8gpu.sh + env_interact.py + persona_compare_full.py.

## Infra notes
- GRPO trainer: tools/run_rft.py + GRPOAutoVLA (models/autovla.py). Multi-node via torchrun (WORLD_SIZE=#nodes, NPROC_PER_NODE, RANK, MASTER_ADDR/PORT; c10d rdzv). DDP not FSDP (FSDP+generate → nan). bf16 dtype fix in action_tokenizer.
- GroupSampler: all ranks share one prompt/step → more GPUs = bigger group (advantage quality), NOT faster; epoch length fixed (~11988 prompts), ~3s/step. Use MAX_STEPS to cap.
- navtrain reward cache: /mnt/pfs/zhengguantian/autovla/exp/metric_cache_navtrain12k (11988 tokens).
- DDv2 teacher trajectories: /mnt/pfs/zhengguantian/autovla/persona/ddv2_navtrain12k.json ; SFT-tokenized DDv2 targets: nocot_ddv2_teacher12k.
