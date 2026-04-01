# HaRP — Technical Deep Dive

> This document covers the methodology, mathematical foundations, and design decisions
> behind every component. For numerical results, see [RESULTS.md](RESULTS.md).
> For the high-level narrative, see [README.md](README.md).

---

## Table of Contents

1. [Research Evolution — How We Got Here](#1-research-evolution)
2. [Problem Framing](#2-problem-framing)
3. [Dataset and Labeling](#3-dataset-and-labeling)
4. [Signal Architecture — Seven Feature Groups](#4-signal-architecture)
5. [Layer Selection and Probing](#5-layer-selection-and-probing)
6. [Geometry of Truth](#6-geometry-of-truth)
7. [Data Leakage — The Bug, The Fix, The Lesson](#7-data-leakage)
8. [Failure Estimator — Architecture and Choices](#8-failure-estimator)
9. [Routing Policy — Decision Rule and Threshold Optimisation](#9-routing-policy)
10. [4-Quadrant Uncertainty Taxonomy](#10-4-quadrant-uncertainty-taxonomy)
11. [Activation Steering — Design and Null Results](#11-activation-steering)
12. [Evaluation Methodology](#12-evaluation-methodology)
13. [What Holds Up and What Doesn't](#13-what-holds-up-and-what-doesnt)

---

## 1. Research Evolution

The project went through four distinct versions. Understanding why each version was
abandoned explains why the final design looks the way it does.

**Version 0 — Can token entropy detect hallucinations?**
Hypothesis: the model is uncertain when it hallucinates, and that uncertainty shows up
in the output token distribution. Six entropy/logit features computed from Ollama
inference. AUROC = 0.573. Hypothesis rejected. Token entropy on TruthfulQA is near-random.

Implication: the model *sounds* confident when it hallucinates. Surface signals are
insufficient. We need to look deeper.

**Version 1 — Does multi-sample consistency help?**
Hypothesis: if outputs are inconsistent across multiple generations, the model is
uncertain. Five responses per prompt, embedded with sentence-transformers, cosine
similarity averaged. AUROC = 0.594 (+0.021). Small improvement, insufficient discrimination.
Consistency requires 5× inference overhead. The ROI does not justify deployment.

Implication: consistency captures something token entropy misses — but not enough.
The signal we need is not in the outputs. It is in the hidden states.

**Version 2 — What does L32 see that L1 doesn't?**
This was the turning point. Extracting full hidden states from Qwen 2.5 3B (700 × 37 × 2048)
and probing each layer with a linear classifier. L32 AUROC = 0.758. L1 AUROC ≈ 0.55.
The signal grows monotonically through the network and plateaus at L28–L32.

This confirmed the core hypothesis: the model's internal representation at the right layer
*does* encode factual correctness — even when the output token distribution looks confident.

**Version 3 — Can we engineer better features from L32?**
Four features from L32 geometry: cosine distance to μ_correct (cos_correct), cosine
distance to μ_hallucinated (cos_hallucinated), L2 norm (h_norm), and probe confidence
(probe_conf). These are Group C. Combined with Groups A and B: AUROC = 0.771.

This version contained the data leakage bug. probe_conf was computed in-sample, inflating
the system AUROC to 0.962. The bug was caught during final evaluation review.

**Version 4 — What are the honest numbers?**
Out-of-fold correction (Exp 09b) recomputes probe_conf and centroids per fold. Honest
AUROC = 0.775. This is the definitive result. Additional signal groups (D, E) add noise.
The system is complete. The honest numbers replace the inflated ones throughout.

---

## 2. Problem Framing

**Task:** Given a factual question Q and a generated response R from Qwen 2.5 3B,
predict — using signals available at inference time — whether R is factually incorrect
(hallucinated) before it is delivered to the user.

**Why internal signals:** Post-hoc fact-checking requires retrieval against an external
knowledge base. Internal signal detection is:
- Faster (one forward pass, no retrieval)
- Cheaper (no external API calls)
- Model-intrinsic (works without reference corpus)
- Complementary to retrieval (can be combined)

**Selective prediction framing:** Rather than a binary pass/block filter, HaRP implements
a three-action routing policy following Geifman & El-Yaniv (2017):

```
p_fail < α      →  ACCEPT      — high confidence, serve the response
α ≤ p_fail < β  →  REGENERATE  — uncertain, retry once with the same prompt
p_fail ≥ β      →  ABSTAIN     — high failure risk, return "I don't know"
```

This framing has advantages over binary filtering: REGENERATE gives the model a second
chance on genuinely uncertain cases, while ABSTAIN reserves refusal for high-risk cases.
The governance loss `L = hall_rate(ACCEPT) + 0.5 × abstain_rate` balances safety against
utility by penalising both false acceptance and excessive refusal.

---

## 3. Dataset and Labeling

**TruthfulQA (Lin et al. 2022):** 817 questions across 38 categories, specifically
designed to expose misconceptions that language models have absorbed from training data.
The questions are adversarial — correct answers often contradict common myths.

We use 700 samples (the subset with full hidden-state extraction coverage).

**Two-tier labeling:**
1. ROUGE-L recall ≥ 0.4 against any reference answer → **correct**
2. Semantic keyword overlap ≥ 0.5 → **correct**
3. Otherwise → **hallucinated**

**Why ROUGE-L + semantic overlap:** Pure ROUGE-L misses paraphrased correct answers
(it penalises valid reformulations). Semantic keyword overlap catches cases where the
model answers correctly using different surface forms. The two-tier design reduces
false-negative labeling errors.

**Known limitations:**
- ROUGE-L penalises verbose but correct answers
- Keyword overlap can miss subtle semantic errors
- ~5% of labels are borderline (manual spot-check confirmed acceptable quality)

**Distribution:** 393 correct (56.1%), 307 hallucinated (43.9%). Approximately balanced,
consistent with TruthfulQA's design to produce ~50% misconception responses from capable models.

---

## 4. Signal Architecture

HaRP uses two computation modes producing seven signal groups.

### Mode 1 — Ollama (local, no GPU, no hidden states)

**Group A — Token-level uncertainty**
Six features from the output token probability distribution at the first response token:

| Feature | Formula | Intuition |
|---------|---------|-----------|
| entropy | H = −Σ p_i log p_i | Overall output uncertainty |
| max_entropy | max(H_t) across response | Worst-case uncertainty position |
| logit_var | Var(logit_i) | Spread of the pre-softmax scores |
| conf_gap | p_1 − p_2 | Margin between top-2 tokens |
| top1_prob | max(p_i) | Peak output confidence |
| first_entropy | H at first token | Uncertainty at generation onset |

Solo AUROC: 0.573. These features fail because TruthfulQA elicits *confident* hallucinations.

**Group B — Multi-sample consistency**
Generate k=5 responses per prompt (temperature=0.9). Embed each with
sentence-transformers (all-MiniLM-L6-v2, 384-dim). Compute mean pairwise cosine
similarity (consistency_score) and variance across the 5 embeddings.

The hypothesis: inconsistent outputs across samples indicate model uncertainty.
Marginal gain: +0.021 AUROC over Group A.

**Group B2 — Semantic entropy (Kuhn et al. 2023)**
NLI-based clustering of the k=5 responses: group semantically equivalent responses
(bidirectional NLI entailment), compute entropy over cluster-level probabilities.
Contribution: marginal, not reported in main ablation.

### Mode 2 — HuggingFace (full model, hidden states required)

**Group C — Representation geometry at L32**

Critical: these four features must be computed out-of-fold (see §7).

| Feature | Formula | Intuition |
|---------|---------|-----------|
| cos_correct | 1 − cosine(h_32, μ_correct) | Distance to correct class centroid |
| cos_hallucinated | 1 − cosine(h_32, μ_hallucinated) | Distance to hallucinated class centroid |
| h_norm | ‖h_32‖₂ | L2 magnitude of the hidden state |
| probe_conf | LR.predict_proba(h_32)[1] | Probe's P(correct) at L32 |

These four features together add +0.177 AUROC to the A+B baseline. This is the entire
discriminative contribution of the HaRP system.

**Group C2 — Attention entropy**
Mean and standard deviation of per-head attention entropy at L32. Zero marginal gain.

**Group D — Cross-layer trajectory (ICR-inspired, Zhang et al. 2025)**
Five features tracking how the token prediction changes across layers:
total_variation, stabilization_layer, max_change_layer, early_commitment_score, icr_score.
Combined contribution: +0.006 AUROC — within noise.

**Group E — Geometric volume (Granese et al. 2025)**
Three features measuring the geometric spread of the hidden-state trajectory:
simplex_volume, embedding_spread, centroid_drift.
Combined contribution: −0.001 AUROC — null.

### Honest Contribution Assessment

| Group | Marginal AUROC | Conclusion |
|-------|---------------|------------|
| A | 0.573 | Baseline, near random |
| B | +0.021 | Real but insufficient |
| C (OOF) | +0.177 | The signal — all the gain |
| C2 | +0.000 | Null |
| D | +0.006 | Noise (within std 0.042) |
| E | −0.001 | Null |

---

## 5. Layer Selection and Probing

**Experimental setup (Exp 06):**
For each of 37 hidden-state positions (position 0 = embedding, positions 1–36 =
transformer layers L1–L36), extract the last-token hidden state h ∈ ℝ²⁰⁴⁸.
Train a logistic regression probe via 5-fold stratified cross-validation.
Report mean AUROC ± std across folds.

**Why last-token hidden state:**
In a causal (decoder-only) transformer, attention is restricted to left context via
the causal mask. The last token's hidden state at any layer has attended to all
preceding tokens — it accumulates the full left-context via the attention mechanism.
This makes it the natural "decision state" from which the next token is predicted.

**Why linear probe:**
- Interpretable: coefficients identify which of the 2048 dimensions carry the signal
- Appropriate for N=700: non-linear probes (MLP, SVM RBF) are prone to overfitting
  at this dataset scale
- Consistent with the geometry-of-truth literature: if the signal is linearly encoded,
  a linear probe will find it; non-linearity would suggest a more complex structure

**Layer selection:**
The probing curve shows monotonically increasing AUROC from the embedding through L28,
then a plateau from L28 to L36 with AUROC ≈ 0.758.
L32 is selected as the plateau onset — the earliest layer with maximum discrimination.
Using the plateau onset (rather than the peak) avoids over-fitting the layer selection
to noise at the tail.

**Comparison to MECH-INT (GPT-2, 12 layers):**
MECH-INT found peak discrimination at L8–L9 ≈ 67–75% depth.
HaRP finds peak at L32 = 89% depth (of 36 layers).
This depth scaling is consistent with the hypothesis that larger models encode
higher-level semantic abstractions deeper in the network.

---

## 6. Geometry of Truth

**PCA at L32 (Exp 07):**
Apply PCA to the 700 × 2048 hidden-state matrix at L32.
Project to 2D. Measure Pearson correlation of PC1 scores with binary labels.
Compute linear separability: fraction of samples correctly classified by PC1 alone.

**Mathematical structure:**
Let H ∈ ℝ^{700 × 2048} be the hidden-state matrix at L32.
Let y ∈ {0, 1}^{700} be the labels (1 = correct, 0 = hallucinated).
Let u₁ ∈ ℝ²⁰⁴⁸ be the first principal component (argmax_u Var(Hu)).
Then corr(Hu₁, y) = 0.424.

This is not trivial: PCA maximises variance, not label correlation.
The fact that the direction of maximum variance is also the direction of maximum
label correlation means the correct/hallucinated distinction is the dominant source
of variance in the L32 representation space.

**Class means and the steering direction:**
μ_correct = (1/n_c) Σ_{i: y_i=1} h_i  ∈ ℝ²⁰⁴⁸
μ_hallucinated = (1/n_h) Σ_{i: y_i=0} h_i  ∈ ℝ²⁰⁴⁸
v = μ_correct − μ_hallucinated  ∈ ℝ²⁰⁴⁸  (norm = 2.68)

This vector v defines:
1. The steering direction (Exp 10–11): add α·v to h[32] at inference time
2. The cosine distance features (Group C): cos_correct = 1 − cosine(h, μ_correct)
3. The linear truth direction for PCA interpretation

**Separability (68.4%):** If we threshold PC1 scores at the class boundary, 68.4% of
samples are correctly classified. This is not clean separation — but it is meaningful
structure above chance (50%) and consistent with the probe AUROC (a linear classifier
on 2048 features achieves 0.758, which is better than a 1D projection).

---

## 7. Data Leakage — The Bug, The Fix, The Lesson

This section documents the most important methodological failure and correction in the project.
Understanding this in detail is more valuable than any single AUROC number.

### What Went Wrong (Exp 09)

The pipeline in Exp 09 computed `probe_conf` as follows:

```python
# Step 1: Fit probe on ALL 700 samples (in Exp 06, saved as best_layer_probe.pkl)
probe = LogisticRegression()
probe.fit(H_all, y_all)          # H_all: [700 × 2048], y_all: [700]

# Step 2: Compute probe_conf for ALL 700 samples
probe_conf = probe.predict_proba(H_all)[:, 1]   # ← WRONG: evaluates on training data
```

Then, when building the failure estimator in the outer 5-fold CV:

```python
for train_idx, test_idx in StratifiedKFold(5).split(X_features, y):
    clf.fit(X_features[train_idx], y[train_idx])
    preds[test_idx] = clf.predict_proba(X_features[test_idx])[:, 1]
    # X_features includes probe_conf — which was computed in-sample on ALL 700 samples
    # The probe already saw y[test_idx]. The outer CV cannot fix this.
```

The outer CV correctly evaluates the failure estimator (clf) out-of-fold.
But `probe_conf` — a *feature* of clf — was computed using a sub-model (the probe)
that already saw all 700 labels, including the test fold labels. This is
**nested model leakage**: the inner model contaminates the outer evaluation.

**Consequence:** An in-sample probe achieves AUROC = 1.000 (it recovers the labels
it was trained on). When this perfect feature is included in the outer clf's feature
set, clf learns that `probe_conf ≈ label`. The reported AUROC inflates to 0.962.

### The Fix (Exp 09b)

Out-of-fold (OOF) computation via a nested CV:

```python
probe_conf_oof = np.zeros(700)
cos_correct_oof = np.zeros(700)
cos_hall_oof = np.zeros(700)

for train_idx, test_idx in StratifiedKFold(5).split(H, y):
    H_train, H_test = H[train_idx], H[test_idx]
    y_train = y[train_idx]

    # Fit everything on TRAINING FOLD ONLY
    probe = LogisticRegression().fit(H_train, y_train)
    mu_correct = H_train[y_train == 1].mean(axis=0)
    mu_hall    = H_train[y_train == 0].mean(axis=0)

    # Compute features on HELD-OUT TEST FOLD ONLY
    probe_conf_oof[test_idx] = probe.predict_proba(H_test)[:, 1]
    cos_correct_oof[test_idx] = 1 - cosine_similarity(H_test, mu_correct)
    cos_hall_oof[test_idx]    = 1 - cosine_similarity(H_test, mu_hall)
```

**Validation of the fix:** OOF probe_conf AUROC = 0.757 — exactly matching Exp 06's
cross-validated probe AUROC (0.758). These are the same signal, now measured correctly.
If the OOF value had differed significantly from Exp 06, that would have indicated a
different bug.

**h_norm requires no correction:** The L2 norm of the hidden state is label-free.
No sub-model is involved. h_norm is computed directly from the forward pass.

### The Lesson

This is a classical **nested model leakage** pattern. It occurs whenever:
1. A sub-model is trained on the full dataset
2. That sub-model's predictions are used as features
3. The outer evaluation uses cross-validation

The outer CV cannot retroactively correct leakage in the features. The fix is always
the same: compute sub-model predictions out-of-fold, matching the outer CV splits.

**Checklist for future projects:**
- [ ] Any feature derived from a trained model must be computed OOF
- [ ] Any centroid/mean computed from labeled data must be computed OOF
- [ ] Validate OOF features against an independent cross-validated baseline
- [ ] If OOF AUROC ≫ independent baseline, there is likely still leakage

---

## 8. Failure Estimator — Architecture and Choices

### Architecture

```
sklearn.pipeline.Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('clf',     LogisticRegression(
                    class_weight='balanced',
                    max_iter=1000,
                    C=1.0
                ))
])
```

**Why this architecture:**

**SimpleImputer (median):** 230 of 930 total samples lack hidden states (Ollama-only
collection in Phase 1). In Exp 24, we restrict to the 700 HS samples to avoid imputation
artifacts. The imputer handles the few remaining NaN values from failed Ollama calls.

**StandardScaler:** Features have very different scales (entropy ∈ [0, 5], cosine distances
∈ [0, 2], h_norm ∈ [O(10²)]). Standardisation is essential for logistic regression.

**LogisticRegression with class_weight='balanced':** The 393/307 class imbalance
(56/44) is mild but non-trivial. Balanced weighting upweights the minority class
(hallucinated) during training, preventing the classifier from optimising only for
the majority class.

**Why not MLP:** At N=700, D=22, a multi-layer perceptron achieves AUROC 0.733 vs
logistic regression's 0.775. This is consistent with the signal being genuinely linear —
Group C features are cosine distances and probe probabilities that already encode
the geometric relationship in a form accessible to a linear classifier. Adding non-linearity
introduces overfitting at this dataset scale.

**Why not SVM or GBM:** The goal is interpretability alongside discrimination. Logistic
regression coefficients directly reveal feature importance. The AUROC gap between LR
and more complex models is negative or zero at N=700.

### Training Protocol

**Cross-validation evaluation:** 5-fold stratified CV on the 700 HS samples.
Stratification preserves the 56/44 class ratio in each fold.
Reports: mean AUROC ± std across folds.

**Policy threshold optimisation:** Final production estimator fitted on the
full training split (60% of 700 = 420 samples). Thresholds α*, β* optimised
on the validation split (20% = 140 samples). Test evaluation on the held-out
test split (20% = 140 samples).

**Important:** The cross-validated AUROC (0.775, n=700) and test AUROC (0.787, n=140)
are from different evaluation protocols. The test AUROC uses a final model trained on
the training split. The CV AUROC pools predictions across all 5 folds.

---

## 9. Routing Policy — Decision Rule and Threshold Optimisation

### Decision Rule

```
p_fail = 1 − estimator.predict_proba(features)[:, 1]

if p_fail < α:       route → ACCEPT     (serve the response)
elif p_fail < β:     route → REGENERATE (retry with same prompt)
else:                route → ABSTAIN    (return "I don't know")
```

**p_fail** is the complement of the estimator's P(correct) score. It is the
model's assessment of failure probability for this specific response.

### Threshold Optimisation

Grid search over α ∈ {0.10, 0.15, …, 0.55} and β ∈ {0.50, 0.55, …, 0.95}:

```
L(α, β) = hall_rate(ACCEPT bucket) + λ × abstain_rate
```

λ = 0.5 balances safety (minimise hallucinations in accepted outputs) against
utility (avoid excessive abstention). The optimal thresholds α* = 0.15, β* = 0.90
reflect the estimator's limited discrimination: a tight acceptance window
(only samples with p_fail < 0.15 are accepted) and a wide regeneration band.

**Honest assessment of optimal thresholds:**
α* = 0.15 means the estimator is only trusted when it is very confident of correctness.
This leads to a 4.3% acceptance rate — only 6 test samples qualify.
The CI on the hallucination rate of those 6 samples is [0.004, 0.641] — too wide
to support strong governance conclusions.

**What better thresholds require:** Higher AUROC (≥ 0.85) would allow a wider
acceptance window while maintaining safety. This requires either more data,
a stronger signal, or a different model/benchmark.

### AUARC — Area Under Accuracy-Rejection Curve

Sort all 140 test samples by p_correct (descending — most confident first).
At each rejection threshold t ∈ [0, 1], compute accuracy on the top (1−t) fraction.
AUARC = integral of this accuracy-retention curve.

AUARC = 1.0 → perfect: the top-ranked samples are all correct.
AUARC = base_rate → the confidence ranking adds nothing over random ordering.

HaRP AUARC = 0.749. Base rate ≈ 0.561. The estimator's ranking is meaningfully
better than random — it successfully prioritises correct samples.

---

## 10. 4-Quadrant Uncertainty Taxonomy

### Design

Two signals define two binary axes:

```
Axis 1 — Token entropy:
    entropy > entropy_threshold  →  HIGH uncertainty
    entropy ≤ entropy_threshold  →  LOW uncertainty

Axis 2 — Multi-sample consistency:
    consistency_score < consistency_threshold  →  INCONSISTENT (low agreement)
    consistency_score ≥ consistency_threshold  →  CONSISTENT (high agreement)
```

This produces four quadrants:

| | Consistent | Inconsistent |
|---|---|---|
| **High entropy** | Aleatoric (high H, agrees across samples → inherent ambiguity) | Epistemic (high H, disagrees → knowledge gap) |
| **Low entropy** | Confident Correct (low H, agrees → correctly confident) | Confident Hallucination (low H, disagrees → confidently wrong) |

### Critical: Data-Driven Thresholds

**All 700 HS samples have consistency_score > 0.503.**
A hardcoded threshold of 0.5 (the natural midpoint) assigns all 700 samples to
the "consistent" half, making the epistemic and confident-hallucination quadrants
empty. This was the bug in Exp 22a and 22b.

**Fix:** Use median thresholds computed from the 700 HS samples:
- entropy_threshold = 0.3704 (median entropy)
- consistency_threshold = 0.8575 (median consistency_score)

Medians guarantee a balanced 4-quadrant split by construction. Each quadrant
receives approximately 175 samples (actual: 182/168/248/102).

### Conceptual Framing

The epistemic/aleatoric decomposition in Bayesian machine learning (Kendall & Gal 2017,
Malinin & Gales 2018) refers to:
- **Epistemic uncertainty**: reducible with more data, comes from model ignorance
- **Aleatoric uncertainty**: irreducible, comes from inherent noise in the task

The entropy × consistency proxy approximates this decomposition:
- High entropy + inconsistent ≈ the model doesn't know (epistemic)
- High entropy + consistent ≈ the question is genuinely hard (aleatoric)

**Important caveat:** This is an *operational proxy*, not a formal Bayesian decomposition.
A formal decomposition would require ensemble inference or explicit epistemic variance
estimation. Describing it as "entropy-consistency proxy decomposition" is accurate.
Describing it as "formal Bayesian epistemic-aleatoric decomposition" overstates what is implemented.

### Governance Implications

The quadrant analysis reveals that the most dangerous failure mode — Confident Hallucination
(35% of samples, low entropy, wrong) — is **immune to activation steering** and
**invisible to entropy-based filters**. For this quadrant:
- Token entropy cannot detect it (definitionally — low entropy)
- Activation steering cannot fix it (Exp 22c, delta = 0.007 at α=−1.0)
- Only representation geometry at L32 (Group C features) can flag it

The correct governance action for confident hallucinations is ABSTAIN.

---

## 11. Activation Steering — Design and Null Results

### Design

**Steering vector (Exp 10):**
v = μ_correct − μ_hallucinated  ∈ ℝ²⁰⁴⁸  (norm = 2.68)

Computed from all 700 labeled HS samples. Note: this is *not* OOF — the steering
vector is intended for use at inference time on new samples, not evaluated on training
samples. Using the full 700 samples for v maximises the signal-to-noise ratio of the
population direction.

**Intervention (Exp 11):**
At inference time, intercept the forward pass at layer 32:
```python
def hook_fn(module, input, output):
    output[0][:, -1, :] += alpha * steering_vector
    return output

model.model.layers[32].register_forward_hook(hook_fn)
```
Continue the forward pass from layer 33. Generate the response.

**Tested values:** α ∈ {0.0, 0.5, 1.0, 1.5, 2.0}, n=100 hallucinated samples per run.

### Three-Level Null Result

**Level 1 — Universal steering (Exp 11):**
Single vector v applied to all samples. No improvement at any α.
Performance degrades monotonically above α=0.5.
The vector is not a universal correction — it moves all samples in the same direction
regardless of where they are in the representation space.

**Level 2 — Quadrant-adaptive steering (Exp 22):**
Four separate vectors, one per quadrant. Same v-direction computation within each quadrant:
v_epistemic = μ_correct_epistemic − μ_hall_epistemic
Best result: hall_rate = 0.68 — equal to the unsteered baseline.
Quadrant-specific vectors add no information over the global vector.

**Level 3 — Per-quadrant alpha + negative alpha (Exp 22c):**
Full grid search over α ∈ {−1.0, −0.5, 0.0, 0.5, 1.0, 1.5, 2.0} per quadrant.
Confident hallucination at α=−1.0: delta = −0.007 (negative alpha = worse).
With n≈57 and T=0.9, generation stochasticity introduces ±14pp noise.
Delta = 0.007 is firmly within noise. The negative-alpha hypothesis is falsified.

### Why Steering Fails — A Mechanistic Hypothesis

The mean-difference vector v is a population-level aggregate. It represents the
average direction from hallucinated to correct class centroids. But:

1. **Individual samples do not align with v.** The within-class variance of the
   2048-dim hidden states is large. An individual hallucinated sample may be far
   from μ_hallucinated — adding v does not necessarily move it toward the correct
   manifold.

2. **Confident hallucinations may be orthogonal to v.** The confident hallucination
   quadrant's geometry may not be opposed to v — if these samples lie near μ_hallucinated
   already, then both +v and −v interventions fail.

3. **Single-layer, single-position intervention is coarse.** Modifying only h[32]
   at the last-token position ignores the full attention context. The error may be
   distributed across many positions and layers.

4. **Generation stochasticity (T=0.9) obscures small effects.** Temperature sampling
   introduces noise that overwhelms effects smaller than ~10pp at n<100.

**What might work:**
- Per-sample steering directions (not population means)
- Token-level intervention (modify each generated token's hidden state)
- Temperature=0 evaluation to remove generation noise
- Contrastive decoding rather than hidden-state intervention

These remain open questions. The null result is a contribution — it rules out the
simplest possible intervention and motivates more targeted approaches.

---

## 12. Evaluation Methodology

### AUROC — Primary Metric

Area Under the Receiver Operating Characteristic Curve.
Interpretation: probability that the estimator ranks a randomly chosen hallucinated
sample higher (more likely to fail) than a randomly chosen correct sample.
AUROC = 0.5 → random. AUROC = 1.0 → perfect discrimination.

**Why AUROC:** Threshold-free, handles class imbalance, directly measures the quality
of the probability ranking independent of decision threshold.

**Why not accuracy:** Accuracy depends on the threshold and is misleading with
imbalanced classes. A trivial classifier that always predicts "correct" achieves 56%
accuracy on this dataset.

### AUARC — Secondary Metric

Area Under the Accuracy-Rejection Curve (from SelectLLM, ACL 2025).

Sort test samples by P(correct) descending. At each retention level r ∈ [0, 1],
compute accuracy on the top-r fraction. AUARC = integral of accuracy(r) dr.

AUARC = 1.0 → perfect: always retain correct, always reject hallucinated.
AUARC = base_rate → confidence ranking adds nothing over random ordering.

HaRP AUARC = 0.749. This confirms the ranking is useful for selective prediction
(you can improve average accuracy by selectively accepting only high-confidence responses).

### Bootstrap AUROC Confidence Interval

Resample test set with replacement 1000 times (seed=42). Compute AUROC on each
resample. Report (2.5th percentile, 97.5th percentile) as the 95% CI.

HaRP test AUROC CI: [0.706, 0.866]. The lower bound (0.706) confirms the signal
is real — the CI does not include 0.5. The width (0.16) reflects the 140-sample test size.

### Clopper-Pearson CI on Hall Rate

For small-n binary rate estimation, the Clopper-Pearson exact binomial interval is correct.
The normal approximation (p ± z√(p(1−p)/n)) is invalid when n < 30 or p ∈ {0, 1}.

```python
from scipy.stats import beta
lo = beta.ppf(0.025, n_hall,     n_accept - n_hall + 1)
hi = beta.ppf(0.975, n_hall + 1, n_accept - n_hall)
```

For n_accept=6, n_hall=1: CI = [0.004, 0.641].
The width of this interval is why we cannot make strong governance claims.

### ECE — Expected Calibration Error

Binned calibration error:
ECE = Σ_b (|b|/n) × |acc(b) − conf(b)|

where b ranges over confidence bins, acc(b) is the fraction of correct samples
in bin b, and conf(b) is the mean predicted probability in bin b.

ECE = 0 → perfect calibration. ECE = 0.039 (HaRP) means the estimator's
P(correct) scores are accurate probability estimates, not just rankings.

---

## 13. What Holds Up and What Doesn't

### Solid Findings — Supported by Honest Numbers

| Finding | Evidence | Why Trustworthy |
|---------|----------|----------------|
| L32 hidden states encode hallucination signal | Probe AUROC 0.758 ± 0.010 | Consistent across 5 folds; matches OOF estimate |
| Token entropy alone is insufficient | AUROC 0.573 ± 0.056 | Stable, near-random; reproducible from any LLM API |
| Group C (OOF) adds +0.177 AUROC | 0.594 → 0.771 | OOF computation; matches Exp 06 probe AUROC |
| Signal is linear at N=700 | LR 0.775 vs MLP 0.733 | Expected; Group C features encode geometry linearly |
| Universal steering fails | 5 α values, all flat/degraded | GPU experiment; three replication attempts |
| Negative alpha fails | delta=0.007, within ±14pp noise | Correct statistical framing of null result |
| Groups D, E add nothing | +0.004 — within std 0.042 | Multiple feature groups tested; consistent null |
| Estimator is well-calibrated | ECE = 0.039 | Meaningful probability scores, not just rankings |
| Early wrong commitment (L4.07) | Logit lens, GPU | Mechanistic explanation for L32 signal |

### Findings That Need Qualification

| Claim | Issue | Corrected Framing |
|-------|-------|-------------------|
| "AUROC 0.962" | In-sample leakage | Honest OOF AUROC is 0.775 |
| "Zero hallucination rate" | Leaked estimator + n=59 | Honest: 1/6, CI [0.004, 0.641] |
| "Formal Bayesian decomposition" | Operationally a proxy | "Entropy-consistency proxy decomposition" |
| "SOTA system" | No comparison to prior work | "Competitive on TruthfulQA" pending benchmarking |
| "Governance framework" | 90% regeneration rate, 4.3% accept | "Conservative filter" with useful but limited signal |

### What Would Strengthen the Work

| Direction | Requirement | Impact |
|-----------|------------|--------|
| Second benchmark | HaluBench / SelfAware evaluation | Generalization evidence |
| Larger test set | 5,000+ samples | Meaningful policy rate estimation |
| Direct comparison to prior work | ICR Probe, HaluNet on same data | Positioning relative to SOTA |
| Temperature=0 steering | Rerun Exp 22c without stochasticity | Separate signal from generation noise |
| Per-sample steering | Sample-specific direction computation | Address population-level failure |
| Epistemic steering validation | n≥300 epistemic samples | Confirm/reject the ~5pp signal |

### On Research Transparency

The most important methodological contribution of HaRP is not the AUROC number —
it is the process of discovering a data leakage bug, correcting it, and reporting the
honest numbers while explicitly documenting what changed and why.

Research transparency of this kind is uncommon but essential for:
- Trust in the final numbers (they have been stress-tested)
- Reproducibility (the exact bug and fix are documented)
- Methodological learning (the Exp 09 vs 09b comparison teaches the OOF pattern)

The honest number (0.775) is a more interesting finding than the leaked number (0.962)
would have been — it shows that representation geometry is real but moderate, that
AUROC alone is insufficient for governance, and that the steering null result is
not a bug in the detection system but an intrinsic limitation of population-level vectors.
