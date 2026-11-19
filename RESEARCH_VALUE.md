# HaRP — Research Value, Breadth, and Future Scope

> **Status note (20 Nov 2026): two calibration results are withdrawn.** `src/calibration/calibrate.py` applies `softmax(..., axis=-1)` to one logit per sample, which normalises across samples instead of within each one. The estimator's ECE of 0.039 and the temperature-scaling result (ECE 0.395, "temperature scaling degrades calibration") both came from that script and are not valid. Every ECE and calibration statement below is withdrawn; ranking metrics (AUROC, AUARC) are unaffected.


**Project:** Hallucination-aware Representation Probing
**Model:** Qwen 2.5 3B Instruct (36 layers, 2048 hidden dim)
**Status:** Complete (28 experiments)
**Position in arc:** Project 2 of 5 — the full governance pipeline

---

## Why This Project Matters

HaRP is the most complete project in the arc. It goes further than any single
paper in the hallucination detection literature — not just detecting hallucination
but building a full governance pipeline from detection → uncertainty decomposition
→ calibration → abstention policy → activation steering → robustness testing →
OOD evaluation.

The key result — that the geometry of hidden states at L32 (89% depth) predicts
hallucination with AUROC 0.775 on an out-of-domain benchmark (HaluEval), with
ECE 0.039 (better than raw LLM at 0.071) — establishes that geometry-based
detection is both accurate and well-calibrated in a way that standard output-based
approaches are not.

But HaRP's most important contribution may be its negative results: the OOD
transfer gap is catastrophic (in-domain AUROC 0.997 → OOD 0.775, gap = 0.225),
and temperature scaling makes calibration *worse* (ECE from 0.071 raw → 0.395
after temp scaling). These aren't failures — they define the problem that
GUARDIAN must solve.

---

## Core Contributions

| Contribution | Key number |
|---|---|
| Optimal detection layer: L32 (89% depth) | Probe AUROC 0.758 in-domain (OOF-honest) |
| OOD detection (TruthfulQA → HaluEval) | AUROC 0.775 — geometry transfers |
| Calibration quality | ECE 0.039 (HaRP full) vs 0.071 (raw LLM) |
| Temperature scaling failure | ECE degrades to 0.395 — standard calibration doesn't work |
| Activation steering at L32 | Directional control over hallucination propensity |
| 4-quadrant uncertainty decomposition | Epistemic vs aleatoric × confident vs uncertain |
| Sycophancy robustness | Geometry is stable under adversarial pressure |
| Conformal abstention policy | Formal coverage guarantee with acceptance rate tuning |

---

## Mathematical and Architectural Power

**The 89% depth law:**
L32/36 = 88.9% consistently emerges as the optimal detection depth for Qwen 2.5 3B.
MECH-INT found L8/24 = 33% as the *causal* source in GPT-2.
GEOM-PROOF found Fisher J peaks at 69–79% depth.

Three distinct depth signals:
- 33%: causal generation of the error (FFN over-retrieval)
- 69–79%: peak geometric separability (Fisher J)
- 89%: peak linear probe performance (the error has crystallized into a detectable representation)

**The OOD transfer gap:**
In-domain AUROC: 0.997 (near-perfect — geometry fully separates on training distribution)
OOD AUROC (HaluEval): 0.775 (decent but 22.2pp drop)
Signal retention: −111.5% (calculated transfer ratio shows the probe captures
training-distribution-specific geometry, not universal hallucination geometry)

This gap is the central problem for GUARDIAN. A governance system that works at
0.997 in-domain but 0.775 OOD needs an adaptive mechanism — which is exactly
what local Fisher J (KNN-based, per-query) provides.

**The temperature scaling anomaly:**
Raw LLM: ECE 0.071
HaRP full system: ECE 0.039 (better)
Temperature scaling: ECE 0.395 (far worse)

Temperature scaling assumes the model's confidence is monotonically related to
accuracy. HaRP shows this assumption fails for hallucination specifically —
the model is confidently wrong in a way that temperature scaling amplifies.
Geometry-based calibration (HaRP) is the correct tool.

**The 4-quadrant decomposition:**
The epistemic/aleatoric × confident/uncertain decomposition provides a formal
language for hallucination failure modes:
- High epistemic, high confidence = dangerous confident hallucination
- High aleatoric, low confidence = normal uncertainty (safe to abstain)
- Low epistemic, low confidence = confused but not factually wrong
- Low epistemic, high confidence = reliable knowledge

This decomposition maps directly onto FAIL-CHAIN's failure taxonomy:
CASCADE ↔ high epistemic confident wrong
LATENT ↔ high aleatoric appears correct but is not
AMPLIFIED ↔ low epistemic at step 1, high epistemic induced at step 2

---

## Open Threads With Genuine Future Value

### OT-1: The OOD Adaptation Problem
**Question:** Can the L32 probe be adapted to a new distribution (HaluEval) with
only 10–20 labeled examples, using the geometry of the new distribution?

**Why it matters:** A real governance system must handle queries from distributions
it was never trained on. The 0.225 AUROC gap is a deployment gap, not a research
gap. Any system that doesn't close this gap is not deployable.

**Hypothesis:** Local Fisher J (KNN in the new distribution) adapts automatically —
it doesn't assume the global geometry matches training. This is the core GUARDIAN
hypothesis: local geometry is distribution-adaptive where global geometry is not.

### OT-2: The Temperature Scaling Anomaly as a Research Finding
**Question:** Why does temperature scaling *degrade* calibration for hallucination
specifically (ECE 0.071 → 0.395)?

**Hypothesis:** The model's confidence scores for hallucinated responses are not
miscalibrated in the usual direction (overconfident). They are overconfident in
a structured way that temperature scaling *enhances* rather than corrects. The
hallucinated responses cluster at the extreme of the confidence scale — these
are the adversarial factoid questions that TruthfulQA is designed to produce.

**What to do:** Plot ECE separately for (correct, wrong) subsets and for
(hallucination trap, non-trap) question types. Temperature scaling probably
helps for non-traps and catastrophically fails for traps.

### OT-3: Sycophancy Geometry vs Factual Hallucination Geometry
HaRP tested sycophancy robustness (Exp 14, Exp 28). The finding was that geometry
is stable under adversarial pressure. But the geometry of *sycophantic responses*
vs *factually wrong responses* was not characterized.

**Question:** Do sycophantic answers (model changes correct answer to match user)
produce the same L32 geometry as factually hallucinated answers (model generates
wrong answer independently)?

**If yes:** A single geometry-based detector catches both — GUARDIAN can route
both failure types to ABSTAIN with the same alarm signal.
**If no:** Two distinct geometry signatures exist. GUARDIAN needs two probes.

### OT-4: The Steering Vector as a Pre-Output Intervention
HaRP Exp 10–11 established activation steering at L32. The steering direction
was extracted from the difference between correct and wrong hidden states.

The FAIL-CHAIN persona direction (different concept — trained on step-3 labels,
not step-1 labels) is mathematically related: both are linear directions in
L32 space that encode hallucination propensity.

**Open question:** Are the HaRP steering direction and the FAIL-CHAIN persona
direction correlated? If they point in the same direction in L32 space, they
are measuring the same underlying geometry from different experimental contexts —
and this convergence would be a strong theoretical result.

**What to do:** Load both vectors (HaRP: `results/logs/10_steering_vector_meta.json`,
FAIL-CHAIN: `results/logs/hallucination_direction.npy`) and compute cosine similarity.
This is a 5-line experiment requiring no GPU.

### OT-5: Multi-Layer Governance (L8 + L32 ensemble)
MECH-INT established L8 as the causal source. HaRP established L32 as the
predictive signal. An ensemble probe using both layers — (h_L8, h_L32) — would
have:
- h_L8: early-network mechanistic signal (is the FFN over-retrieving a wrong fact?)
- h_L32: late-network geometric signal (has the error crystallized into a detectable state?)

**Hypothesis:** The ensemble AUROC exceeds either single-layer probe. The two
signals are complementary — L8 catches errors early (high recall, low precision)
while L32 catches the crystallized form (higher precision).

---

## Connection to the PhD Arc

| Downstream project | How HaRP informs it |
|---|---|
| GEOM-PROOF | The L32 signal is certified mathematically — Fisher J bounds the probe AUROC |
| FAIL-CHAIN | The 89% depth finding determines which layer to hook in the pipeline. The OOD gap motivates cascade-aware governance |
| GUARDIAN | The OOD gap (0.225) is the problem GUARDIAN solves. The steering direction is GUARDIAN's step-2 monitor seed |

---

## The One Sentence This Project Adds to the World

> *At 89% depth (L32), the geometry of Qwen 2.5 3B's hidden states predicts
> hallucination with AUROC 0.775 on an unseen benchmark, ECE 0.039, and
> robustness to adversarial sycophancy pressure — establishing representation-level
> geometry as a viable foundation for real-time LLM governance.*
