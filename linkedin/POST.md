# HaRP — LinkedIn Post

---

**HaRP:**
Probing a language model's internal geometry to detect hallucinations the output alone cannot reveal — and building a governance policy that acts on that signal.

I wanted to know whether a model's internal state "knows" an answer is wrong before it says it.

Not just detecting hallucinations from the output. From inside.

I built a 24-experiment pipeline on Qwen 2.5 3B + TruthfulQA (700 samples, 39.5% raw hallucination rate) to test this. A data-leakage bug was found mid-project, corrected, and published with full transparency. These are the honest numbers.

**Core Finding: Layer 32 Encodes a Detectable Hallucination Signal**

The model's internal representation at Layer 32 (the second-to-last layer) looks measurably different for a wrong answer versus a correct one — even when the output text sounds identical.

This is not detectable from the output. It is only visible inside.

**Key Results (honest, leak-corrected)**

◆ Token entropy alone → AUROC 0.573
→ Surface-level signals are near chance. Hallucinations sound confident.

◆ + Internal geometry at L32 → AUROC 0.771  (+0.198 lift)
→ Cosine distance to correct/hallucinated centroids + linear probe confidence

◆ Held-out test AUROC → 0.787  [95% CI: 0.706 – 0.866]
→ True out-of-sample performance, OOF-corrected

◆ Calibration error (ECE) → 0.039
→ When the system says 80% confident, it is right ~80% of the time

◆ Governance policy: ACCEPT / REGENERATE / ABSTAIN
→ 4.3% accepted · 90% regenerated · 5.7% abstained

◆ Activation steering → 0 improvement across all conditions tested
→ 5 experiments. Pushing the internal state did not fix the output.

**What makes this distinct**

- Applies hidden-state geometry (INSIDE, Chen et al. 2024) to a 3B model + governance policy
- The geometry signal is absent at GPT-2 scale (companion project: MECH-INT) — confirms this is a scale-dependent phenomenon
- A data-leakage bug (+0.19 AUROC inflation) was caught, corrected, and documented openly
- Activation steering fails in every configuration tested — a rigorous negative finding worth knowing
- All code, data, and results are fully reproducible and open-source

**Honest Caveat**
AUROC 0.77 is real and consistent, but not enough for production-grade deployment. The governance policy sends 90% of queries to REGENERATE — it works as a conservative filter, not a confident gatekeeper. Improving signal quality (from 0.77 → 0.90+) is the clear next step.

**Acknowledgements & Idea Lineage**
- Li et al. (NeurIPS 2023, ITI) — Inference-Time Intervention via representation probing
- Chen et al. (ACL 2024, INSIDE) — Cosine geometry for hallucination discrimination
- Kuhn et al. (ICLR 2023, Semantic Uncertainty) — NLI-based semantic entropy
- Zhang et al. (ACL 2025, ICR Probe) — Cross-layer trajectory consistency
- Granese et al. (arXiv 2024) — Geometric volume as entropy upper bound

Their work directly shaped the signal architecture and methodology of this project.

**Interactive dashboard (all 24 experiments, live results):**
https://github.com/Lakshmi-Chakradhar-Vijayarao/harp

**Full code & pipeline:**
https://github.com/Lakshmi-Chakradhar-Vijayarao/harp

If you've worked on hallucination detection at scale — does the geometry signal strengthen past 7B? I'd be curious.

Open to feedback and discussion.

#AIResearch #LLM #HallucinationDetection #NLP #DeepLearning #TrustworthyAI #MachineLearning #AIAlignment #InterpretableAI #RepresentationLearning
