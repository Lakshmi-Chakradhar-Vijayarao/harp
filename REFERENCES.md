# References — HaRP

*Papers cited or directly compared against in this project.*

---

## Representation Probing

**[1] Burns et al. — Discovering Latent Knowledge in Language Models Without Supervision (CCS)**
Collin Burns, Haotian Ye, Dan Klein, Jacob Steinhardt.
*NeurIPS*, 2022 / *ACL*, 2023.
> Establishes that representation probing works: linear probes find concentrated truthfulness directions. Core methodological justification for our L32 probe.

**[2] Slobodkin et al. — The Land of Lost Context: A Tale of LLMs and Hallucination**
Avi Slobodkin, Omer Goldman, Avi Caciularu, Ido Dagan, Shauli Ravfogel.
*arXiv*, 2023.
> Representation probing for hallucination in context-dependent settings.

**[3] Li et al. — Inference-Time Intervention: Eliciting Truthful Answers from a Language Model (ITI)**
Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, Martin Wattenberg.
*NeurIPS*, 2023.
> Shows later layers are better for truth signals. Directly motivates our 89% depth finding at L32.

**[4] Marks & Tegmark — The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets**
Samuel Marks, Max Tegmark.
*COLM*, 2024.
> Geometric analysis of truth representations. Later layers better for truth — consistent with our L32 finding.

---

## Uncertainty Quantification

**[5] Kendall & Gal — What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?**
Alex Kendall, Yarin Gal.
*NeurIPS*, 2017.
> Epistemic vs. aleatoric uncertainty decomposition. Theoretical foundation for our 4-quadrant uncertainty analysis.

**[6] Malinin & Gales — Predictive Uncertainty Estimation via Prior Networks**
Andrey Malinin, Mark Gales.
*NeurIPS*, 2018.
> Epistemic-aleatoric decomposition in neural networks.

**[7] Kuhn et al. — Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation**
Lorenz Kuhn, Yarin Gal, Sebastian Farquhar.
*ICLR*, 2023.
> Semantic entropy baseline. Our token entropy feature (AUROC 0.573) confirms entropy alone is insufficient.

**[8] Kadavath et al. — Language Models (Mostly) Know What They Know**
Saurav Kadavath, Tom Conerly, Amanda Askell, et al. (Anthropic)
*arXiv*, 2022.
> Token entropy / self-knowledge. Confirms entropy insufficient as standalone signal.

**[9] Xiong et al. — Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs**
Miao Xiong, Zhiyuan Hu, Xinyang Lu, Yifei Li, et al.
*ICLR*, 2024.
> Token entropy for uncertainty — further confirms limitations addressed by our probe approach.

---

## Selective Prediction & Abstention

**[10] Geifman & El-Yaniv — Selective Prediction in Deep Neural Networks**
Yonatan Geifman, Ran El-Yaniv.
*NeurIPS*, 2017.
> Foundational selective prediction framework. Theoretical basis for our ACCEPT/REGENERATE/ABSTAIN policy.

**[11] Kamath et al. — Selective Question Answering under Domain Shift**
Amita Kamath, Robin Jia, Percy Liang.
*ACL*, 2020.
> Selective prediction in QA settings — directly motivates our deployment framing.

---

## Consistency & Output-Level Methods

**[12] Manakul et al. — SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models**
Potsawee Manakul, Amos Azaria, Mark Gales.
*EMNLP*, 2023.
> Multi-sample consistency signal. Our probe adds +0.021 AUROC over consistency baseline.

---

## Early Commitment / Logit Lens

**[13] Nostalgebraist — Interpreting GPT: The Logit Lens**
*Blog post*, 2020.
https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
> Logit lens for tracking prediction formation. Our data shows mean early-commitment layer at L4.07.

**[14] Belrose et al. — Eliciting Latent Predictions from Transformers with the Tuned Lens**
Nora Belrose, Zac Hatfield-Dodds, et al.
*arXiv*, 2023.
> Tuned lens extension of the logit lens — early commitment analysis.

---

## Activation Steering (Negative Result)

**[15] Turner et al. — Activation Addition: Steering Language Models Without Optimization**
Alexander Matt Turner, Lisa Thiergart, David Udell, et al.
*arXiv*, 2023.
> Activation addition / steering. Our result: steering fails (null AUROC at all α) — steering is context-dependent, consistent with this paper's caveats.

---

## Cross-Layer & Geometric Features

**[16] Zhang et al. — ICR: A Benchmark for In-Context Learning of Concepts with Real-World Priors (ICR-inspired probe features)**
*arXiv*, 2025.
> Cross-layer trajectory signals used as Group D features in our 22-feature set.

**[17] Granese et al. — Geometric Volume Features for Hallucination Detection**
*arXiv*, 2025.
> Geometric volume features (Group E in our feature set).

---

## Datasets

**[18] Lin et al. — TruthfulQA: Measuring How Models Mimic Human Falsehoods**
Stephanie Lin, Jacob Hilton, Owain Evans.
*ACL*, 2022.
> Primary training/evaluation dataset. 817 questions, 700 used after filtering.

**[19] Li et al. — HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models**
Junyi Li, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun Nie, Ji-Rong Wen.
*EMNLP*, 2023.
> OOD evaluation dataset. 500-sample subset used for transfer evaluation (TQA → HaluEval OOD AUROC 0.775).

---

*Total: 19 references*
