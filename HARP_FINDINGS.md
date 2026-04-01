# HaRP: Hallucination-aware Representation Probing
## Complete Project Findings — Honest Evaluation (Exp 01–24)

---

## Overview

HaRP is a hallucination detection study for Qwen 2.5 3B on TruthfulQA
(700 factual QA samples). It spans 28 experiments. Exp 24 corrects a data leakage bug
in Exp 23 and replaces inflated numbers with true out-of-sample results. Experiments
25–28 form a validation suite characterising signal scope, deployability, and specificity.

**Model:** Qwen 2.5 3B (36-layer transformer, 2048 hidden dim)
**Dataset:** TruthfulQA — 700 samples (393 correct, 307 hallucinated)
**Task:** Binary classification: correct (1) vs hallucinated (0)

---

## Correction Log (Exp 24)

A data leakage bug was identified and corrected before finalising results.

**The bug:** In Exp 09, `probe_conf` was computed by running a probe (fitted on all 700
samples) back over those same 700 training samples. In-sample, this probe has AUROC = 1.0000
— it recovers the label perfectly because it was fit on it. When this feature entered the
failure estimator's 5-fold CV, the outer CV did not correct for the inner leakage.

**Measured inflation:** +0.2427 AUROC on probe_conf alone. Failure estimator inflation: +0.19.

**Fix (Exp 09b):** Out-of-fold (OOF) computation — for each fold, the probe and class centroids
are fitted on the training fold only, then applied to the held-out test fold. This gives
true out-of-sample probe_conf with AUROC 0.7573 (matching Exp 06's cross-validated 0.7583).

| Metric | Exp 23 (leaked) | Exp 24 (honest) | Delta |
|--------|----------------|----------------|-------|
| AUROC A+B+C (5-fold CV) | 0.9620 | **0.7714** | −0.1906 |
| AUROC SOTA (5-fold CV) | 0.9624 | **0.7753** | −0.1871 |
| Test AUROC | 0.9521 | **0.7871** | −0.1650 |
| Hall rate (ACCEPT) | 0.0000 | **0.1667** | +0.1667 |
| Accept rate | 31.7% | **4.3%** | −27.4pp |

The zero-hallucination claim was an artifact of leakage. All results below use the honest numbers.

---

## Phase 1 — Data Collection and Baselines (Exp 01–05)

- 700 TruthfulQA prompts; model responses labeled via ROUGE-L + semantic matching
- Label distribution: 393 correct (56.1%), 307 hallucinated (43.9%)
- Raw LLM hallucination rate: **39.53%**
- Full hidden states extracted: 700 × 37 layers × 2048 dim = `hidden_states.npz` (94 MB)

---

## Phase 2 — Mechanistic Analysis (Exp 06–11)

### Exp 06: Layer-wise Probing
- **Best layer: L32** — cross-validated AUROC = **0.7583 ± 0.0096**
- Plateau from L28 onward; L32 is the plateau onset

### Exp 07: Geometry of Truth
- PC1 correlation with label at L32: **0.4241** — strong linear truth direction
- Separability score: **0.684**

### Exp 08: Logit Lens
- Hallucinated answers commit to wrong tokens at mean layer **L4.07**
- 27.2% of samples show no clear commitment

### Exp 09b: Out-of-Fold Group C (Leak Correction)
- In-sample probe_conf AUROC: **1.0000** (leaked — memorised training labels)
- OOF probe_conf AUROC: **0.7573** (honest — matches Exp 06 CV result)
- cos_correct/cos_hallucinated: minor centroid leakage (1/N contribution), corrected via OOF centroids
- h_norm: label-free, no correction needed

### Exp 11: Universal Activation Steering
- Steering `h' = h + α·v` at L32, α ∈ {0.0, 0.5, 1.0, 1.5, 2.0}, n=100 hallucinated
- **No improvement at any α.** Hall rate degrades from 0.67 (α=0) to 0.74 (α=2.0)
- Universal steering is ineffective; failure mode identity must determine intervention

---

## Phase 3 — Feature Engineering and Policy (Exp 12–21)

### Feature Groups

| Group | Features | Solo AUROC (honest) |
|-------|----------|---------------------|
| A | entropy, max_entropy, logit_var, conf_gap, top1_prob, first_entropy | ~0.55 |
| B | consistency_score, consistency_variance | ~0.59 (A+B) |
| C (OOF) | cos_correct, cos_hallucinated, h_norm, probe_conf | ~0.77 (A+B+C) |
| C2 | attn_entropy_mean, attn_entropy_std | marginal |
| D | traj_total_variation, traj_stabilization_layer, traj_max_change_layer, traj_early_commitment, icr_score | +0.005 |
| E | simplex_volume, embedding_spread, centroid_drift | −0.001 |

### Exp 15: Calibration
- ECE (Raw LLM): 0.072
- ECE (HaRP estimator): **0.039** (well-calibrated)

---

## Phase 4 — Final Evaluation (Exp 22–24)

### Exp 22–22c: Quadrant-Adaptive Steering

4-quadrant taxonomy (data-driven thresholds: entropy=0.3704, consistency=0.8575):

| Quadrant | n | Description |
|----------|---|-------------|
| Epistemic | 182 | High entropy, low consistency — knowledge gap |
| Aleatoric | 168 | High entropy, high consistency — inherent uncertainty |
| Confident Hallucination | 248 | Low entropy, low consistency — confidently wrong |
| Confident Correct | 102 | Low entropy, high consistency |

**Key finding (Exp 22c):** Confident hallucinations are immune to activation steering at
ALL tested alpha values including negative alpha (δ = −0.007, within noise bounds at n=57,
temperature=0.9). This is a rigorous null result: the negative-alpha hypothesis is falsified.
Epistemic samples show ~5pp improvement at α=1.5, but this is not statistically distinguishable
from generation noise (±14pp at n≈80, temperature=0.9).

**Interpretation:** Activation steering with a mean-difference vector at L32 does not
reliably correct hallucinations in Qwen 2.5 3B on TruthfulQA, for any failure mode.
The ABSTAIN routing decision for confident hallucinations is the correct governance action,
but this conclusion follows from the failure estimator signal alone — the steering experiments
are expensive null results that confirm rather than motivate it.

### Exp 24: Honest Evaluation (DEFINITIVE RESULTS)

**Failure estimator ablation (5-fold CV, n=700 HS samples, OOF Group C):**

| System | AUROC | ±std | F1 |
|--------|-------|------|----|
| Group A only (token signals) | 0.5734 | 0.0558 | 0.5692 |
| A+B (+consistency) | 0.5936 | 0.0730 | 0.6039 |
| **A+B+C (+repr. geometry, OOF)** | **0.7714** | 0.0474 | 0.7708 |
| Full v3 (+attn entropy) | 0.7714 | 0.0474 | 0.7708 |
| v4 (+trajectory dynamics) | 0.7765 | 0.0419 | 0.7540 |
| SOTA (all groups) | 0.7753 | 0.0416 | 0.7557 |
| MLP SOTA | 0.7332 | 0.0309 | 0.7353 |

**AUROC improvements (honest):**
- Group A → A+B+C: +0.1979 (representation geometry contributes meaningfully)
- A+B+C → SOTA (D+E): +0.0039 (Groups D and E add nothing beyond noise)

**Policy evaluation (test set, n=140, OOF features):**

| Metric | Value |
|--------|-------|
| Test AUROC | 0.7871 |
| Bootstrap 95% CI | [0.7057, 0.8659] |
| AUARC | 0.7495 |
| Optimal α* | 0.15 |
| Optimal β* | 0.90 |
| ACCEPT rate | 4.3% (6/140) |
| REGENERATE rate | 90.0% (126/140) |
| ABSTAIN rate | 5.7% (8/140) |
| Hall rate (ACCEPT) | 0.1667 (1/6) |
| Clopper-Pearson 95% CI | [0.004, 0.641] |
| Hall rate (served, incl. regen) | 0.4091 |

**Important:** With AUROC 0.7753, the estimator does not have enough discrimination to
tightly separate samples into a confident-accept bucket. The optimal policy sends 90% of
queries to REGENERATE. With only 6 accepted samples in the test set, the hall_rate CI
is [0.004, 0.641] — too wide to draw strong conclusions about acceptance quality.

---

## Honest Key Findings

1. **Representation geometry is discriminative** — adding OOF Group C (cosine distances,
   probe confidence at L32) lifts AUROC from 0.57 to 0.77. A genuine and honest contribution.

2. **Token entropy alone is near-random (0.57)** — clean negative result.

3. **Groups D and E (trajectory + geometric volume) add nothing** — +0.004 AUROC over
   A+B+C, well within noise (std ~0.05). These are null contributions.

4. **The probe's true generalisation AUROC is ~0.76** — consistent across Exp 06 CV
   and Exp 09b OOF. The representation signal at L32 is real but moderate, not dominant.

5. **Universal and quadrant-adaptive activation steering both fail** — five experiments,
   consistently null results. This is a rigorous negative finding worth publishing.

6. **The governance policy requires better signal to be practically useful** — at AUROC 0.78,
   the estimator is risk-averse: it regenerates 90% of queries and can only confidently
   accept a small fraction. Useful as a conservative filter, not as a confident gatekeeper.

7. **The 4-quadrant uncertainty taxonomy is operationally valid** but the "formal Bayesian
   decomposition" framing overstates what is implemented (entropy+consistency thresholding).

8. **The signal is scale-emergent** — GPT-2 117M: null, GPT-2 Medium 345M: 0.579 (weak),
   Qwen 2.5 3B: 0.775 (strong). The geometry strengthens monotonically with model capacity.
   (Exp 25)

9. **The signal is generation-time specific** — HaRP probe zero-shot transfer to HaluEval
   yields AUROC 0.094 (anti-correlated). The probe captures internal uncertainty during
   generation, not textual semantic correctness. An in-domain probe on HaluEval achieves
   0.997, confirming the geometry is present but domain-specific. (Exp 26)

10. **Routing overhead is 23.9ms** — negligible relative to generation latency (7,691ms
    forward pass paid regardless). The system is deployable on current hardware. (Exp 27)

11. **The L32 signal is hallucination-specific** — sycophancy probe AUROC 0.404, below
    chance. The probe does not fire for social compliance failures or general "wrong answer"
    geometry. The specificity is a property of the signal. (Exp 28)

---

## What Is Publishable

### Strong claims (supported by honest numbers):
- Layer 32 hidden states encode a linear hallucination signal (AUROC 0.7583, Exp 06)
- Token entropy alone is insufficient for hallucination detection (AUROC 0.57)
- Adding hidden-state geometry at the best layer improves AUROC to 0.77 (+0.20)
- A linear probe at the best layer is the single most informative signal
- Universal activation steering with mean-difference vector does not reduce hallucination rate
- Negative activation steering for confident hallucinations is also ineffective (Exp 22c)
- Cross-layer trajectory and geometric volume signals (Groups D, E) do not add discriminative power beyond the best-layer geometry

### Weaker claims (require qualification):
- Governance policy can reduce hallucination rate below raw LLM (true, but at high abstention cost)
- 4-quadrant uncertainty taxonomy maps failure modes (valid conceptually, thresholds are heuristic)

### Claims to drop:
- "SOTA" system — no empirical comparison to prior work on shared benchmarks
- Zero hallucination rate — artifact of leakage; honest number is 1/6 accepted (wide CI)
- "Formal Bayesian uncertainty decomposition" — replace with "entropy-consistency proxy"

---

## Experiment Completion Status

| Exp | Description | Status | Key Honest Output |
|-----|-------------|--------|-------------------|
| 01–05 | Data, labeling, hidden states | Complete | 700 samples, hall_rate=0.3953 |
| 06 | Layer probing | Complete | best_layer=32, AUROC=0.7583 |
| 07 | Geometry of truth | Complete | PC1 corr=0.4241 |
| 08 | Logit lens | Complete | commit_hall=L4.07 |
| 09 | Feature extraction (in-sample) | Superseded by 09b | — |
| **09b** | **OOF Group C (leak fix)** | **Complete** | **probe_conf OOF AUROC=0.7573** |
| 10 | Steering vector | Complete | steering_vector.npy |
| 11 | Universal steering | Complete | No improvement at any α |
| 12–21 | Policy, calibration, ablations | Complete | ECE=0.039 |
| 22, 22c | Quadrant steering | Complete | Null result; conf_hall immune |
| 23 | Final eval (leaked) | Superseded by 24 | — |
| **24** | **Honest evaluation** | **Complete** | **AUROC=0.7753, honest CI** |
| **25** | **Cross-model validation (GPT-2 Medium)** | **Complete** | **Best AUROC=0.579 at L18 — scale-emergent** |
| **26** | **OOD generalization (HaluEval)** | **Complete** | **Zero-shot AUROC=0.094 — generation-time specific** |
| **27** | **Routing latency profiling** | **Complete** | **23.9ms ± 30.2ms overhead (p95: 66.6ms)** |
| **28** | **Sycophancy geometry probe** | **Complete** | **Probe AUROC=0.404 — signal is hallucination-specific** |

---

## Output Files

```
results/
  logs/
    final_results.json              ← Single source of truth (honest numbers)
    24_honest_results.json          ← Exp 24 detailed output
    09b_oof_group_c.json            ← Leakage quantification
    23_sota_results.json            ← Archived (inflated, superseded)
  plots/
    24_honest_ablation.png          ← Honest ablation vs leaked baseline
  hidden_states/
    hidden_states.npz               ← 700×37×2048
    steering_vector.npy
    best_layer_probe.pkl            ← Used for deployment (not evaluation)
data/
  processed/
    features_combined_oof.csv       ← Leak-free features (700 HS samples)
    features_combined.csv           ← Original (contains leakage, archived)
```

---

---

## Output Files (Exp 25–28)

```
results/
  logs/
    25_cross_model_validation.json    ← Exp 25: Cross-model validation results
    26_halueval_ood.json              ← Exp 26: HaluEval OOD results
    27_latency_profiling.json         ← Exp 27: Latency profiling results
    28_sycophancy_results.json        ← Exp 28: Sycophancy probe results
    28_sycophancy_biased_responses.csv ← Exp 28: 500-sample behavioral labels
  hidden_states/
    25_gpt2_medium_hidden_states.npz  ← Exp 25: GPT-2 Medium L18–L22 states
experiments/
  25_cross_model_validation.py       ← Exp 25 script
  26_halueval_ood.py                 ← Exp 26 script (local version)
  27_latency_profiling.py            ← Exp 27 script
  28_sycophancy_geometry_probe.py    ← Exp 28 script
```

---

*HaRP project complete. 28 experiments. March 2026.*
