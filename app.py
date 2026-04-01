"""
HaRP: Hallucination-aware Representation Probing
===============================================================
An educational research dashboard that walks through the complete
HaRP pipeline — from the hallucination problem to the full
hidden-state probing, uncertainty decomposition, and routing policy.

Run:   streamlit run app.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HaRP · Hallucination-aware Representation Probing",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared style ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Design tokens ───────────────────────────────── */
:root {
    --primary:    #4527a0;
    --primary-lt: #5e35b1;
    --primary-bg: #ede7f6;
    --green:      #2e7d32;
    --green-bg:   #e8f5e9;
    --amber:      #bf360c;
    --amber-bg:   #fbe9e7;
    --red:        #b71c1c;
    --red-bg:     #ffebee;
    --teal:       #00695c;
    --teal-bg:    #e0f2f1;
    --slate:      #37474f;
    --slate-bg:   #eceff1;
    --bg:         #f5f4fb;
    --surface:    #ffffff;
    --text:       #1a1433;
    --text-muted: #6b7280;
    --border:     #ddd8f0;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
}

/* ── Callout boxes ───────────────────────────────── */
.finding-box {
    background: linear-gradient(135deg, #ede7f6 0%, #e8eaf6 100%);
    border-left: 5px solid var(--primary);
    padding: 1.1rem 1.4rem;
    border-radius: 0 10px 10px 0;
    margin: 0.9rem 0;
    box-shadow: 0 2px 8px rgba(69,39,160,0.10);
    font-size: 0.93rem;
}
.good-box {
    background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%);
    border-left: 5px solid var(--green);
    padding: 1.1rem 1.4rem;
    border-radius: 0 10px 10px 0;
    margin: 0.9rem 0;
    box-shadow: 0 2px 8px rgba(46,125,50,0.10);
    font-size: 0.93rem;
}
.warn-box {
    background: linear-gradient(135deg, #fbe9e7 0%, #fff8e1 100%);
    border-left: 5px solid var(--amber);
    padding: 1.1rem 1.4rem;
    border-radius: 0 10px 10px 0;
    margin: 0.9rem 0;
    box-shadow: 0 2px 8px rgba(191,54,12,0.10);
    font-size: 0.93rem;
}
.null-box {
    background: linear-gradient(135deg, #ffebee 0%, #fce4ec 100%);
    border-left: 5px solid var(--red);
    padding: 1.1rem 1.4rem;
    border-radius: 0 10px 10px 0;
    margin: 0.9rem 0;
    box-shadow: 0 2px 8px rgba(183,28,28,0.10);
    font-size: 0.93rem;
}
.analogy-box {
    background: linear-gradient(135deg, #e8f5e9 0%, #e0f2f1 100%);
    border-left: 5px solid var(--teal);
    padding: 0.9rem 1.4rem;
    border-radius: 0 10px 10px 0;
    margin: 0.9rem 0;
    font-style: italic;
    box-shadow: 0 2px 8px rgba(0,105,92,0.10);
    font-size: 0.90rem;
}
.limit-box {
    background: linear-gradient(135deg, #eceff1 0%, #f5f5f5 100%);
    border-left: 5px solid var(--slate);
    padding: 1.0rem 1.4rem;
    border-radius: 0 10px 10px 0;
    margin: 0.9rem 0;
    box-shadow: 0 2px 8px rgba(55,71,79,0.08);
    font-size: 0.91rem;
}

/* ── Pills ───────────────────────────────────────── */
.step-pill {
    display: inline-block;
    background: linear-gradient(90deg, #4527a0 0%, #5e35b1 100%);
    color: white;
    padding: 0.28rem 0.9rem;
    border-radius: 20px;
    font-size: 0.80rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    margin-bottom: 0.6rem;
    box-shadow: 0 2px 6px rgba(69,39,160,0.28);
}
.exp-pill {
    display: inline-block;
    background: linear-gradient(90deg, #283593 0%, #3949ab 100%);
    color: white;
    padding: 0.22rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    margin-bottom: 0.5rem;
    margin-right: 0.3rem;
    box-shadow: 0 2px 6px rgba(40,53,147,0.25);
}
.novel-pill {
    display: inline-block;
    background: linear-gradient(90deg, #00695c 0%, #00897b 100%);
    color: white;
    padding: 0.22rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    margin-bottom: 0.5rem;
    box-shadow: 0 2px 6px rgba(0,105,92,0.25);
}
.honest-pill {
    display: inline-block;
    background: linear-gradient(90deg, #2e7d32 0%, #388e3c 100%);
    color: white;
    padding: 0.22rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    margin-bottom: 0.5rem;
    box-shadow: 0 2px 6px rgba(46,125,50,0.25);
}

/* ── Stat card ───────────────────────────────────── */
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: 3px solid var(--primary);
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(69,39,160,0.07);
    transition: box-shadow 0.2s, transform 0.15s;
}
.stat-card:hover {
    box-shadow: 0 8px 28px rgba(69,39,160,0.13);
    transform: translateY(-2px);
}
.stat-number {
    font-size: 2.4rem;
    font-weight: 800;
    color: var(--primary);
    line-height: 1;
    letter-spacing: -0.02em;
}
.stat-number-red    { color: var(--red); }
.stat-number-green  { color: var(--green); }
.stat-number-indigo { color: #283593; }
.stat-number-amber  { color: var(--amber); }
.stat-label {
    font-size: 0.76rem;
    color: var(--text-muted);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-top: 0.35rem;
    line-height: 1.3;
}

/* ── General card ────────────────────────────────── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 1.1rem 1.3rem;
    border-radius: 10px;
    margin-bottom: 0.7rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

/* ── Phase badge ─────────────────────────────────── */
.phase-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(90deg, #1a0a3c 0%, #4527a0 100%);
    color: #e8e4ff;
    border-radius: 8px;
    padding: 0.38rem 1rem;
    font-size: 0.80rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
    box-shadow: 0 2px 8px rgba(69,39,160,0.22);
}

/* ── Feature chips ───────────────────────────────── */
.feature-chip {
    display: inline-block;
    background: var(--primary-bg);
    color: var(--primary);
    border-radius: 6px;
    padding: 0.15rem 0.55rem;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0.15rem;
    font-family: 'SFMono-Regular', Consolas, monospace;
}

/* ── Quadrant boxes ──────────────────────────────── */
.quadrant-box {
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    font-size: 0.88rem;
    line-height: 1.5;
}
.q-epistemic   { background: #e8eaf6; border-left: 5px solid #3949ab; }
.q-aleatoric   { background: #fff8e1; border-left: 5px solid #f9a825; }
.q-correct     { background: #e8f5e9; border-left: 5px solid #2e7d32; }
.q-confident   { background: #ffebee; border-left: 5px solid #c62828; }

/* ── Sidebar ─────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0a1e 0%, #1a1244 45%, #2d1b69 80%, #3d2192 100%);
}
section[data-testid="stSidebar"] * { color: #e8e4ff !important; }
section[data-testid="stSidebar"] .stButton button {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #e8e4ff !important;
    border-radius: 6px !important;
    font-size: 0.82rem !important;
    text-align: left !important;
    padding: 0.35rem 0.7rem !important;
    margin-bottom: 0.1rem !important;
    transition: background 0.15s !important;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255,255,255,0.15) !important;
}
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; }

/* ── Typography ──────────────────────────────────── */
h1 { color: #1a0a3c; font-weight: 800; letter-spacing: -0.025em; line-height: 1.15; }
h2 { color: #4527a0; font-weight: 700; }
h3 { color: #00695c; font-weight: 600; }

h1::after {
    content: '';
    display: block;
    width: 56px;
    height: 4px;
    background: linear-gradient(90deg, #4527a0, #7c4dff);
    border-radius: 2px;
    margin-top: 0.4rem;
}

/* ── Progress bar ────────────────────────────────── */
.prog-bar-bg {
    background: rgba(255,255,255,0.15);
    border-radius: 8px;
    height: 6px;
    overflow: hidden;
    margin: 0.3rem 0 0.7rem;
}
.prog-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #7c4dff 0%, #b388ff 100%);
    border-radius: 8px;
}

/* ── Table & misc ────────────────────────────────── */
.stDataFrame { border-radius: 8px; overflow: hidden; }
.stDataFrame thead th { background: var(--primary-bg) !important; color: var(--primary) !important; }
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE     = Path(__file__).parent
LOGS_DIR  = _HERE / "results" / "logs"
PLOTS_DIR = _HERE / "results" / "plots"


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_json(name):
    p = LOGS_DIR / name
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)

@st.cache_data
def load_final():
    return load_json("final_results.json")

def plot_img(name, caption=None, width=None):
    p = PLOTS_DIR / name
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=(width is None))
    else:
        st.info(f"Plot not yet generated: {name} — run the corresponding experiment first.")

# ── Helpers ───────────────────────────────────────────────────────────────────
def pill(text):
    st.markdown(f'<span class="step-pill">{text}</span>', unsafe_allow_html=True)

def exp_pill(text):
    st.markdown(f'<span class="exp-pill">{text}</span>', unsafe_allow_html=True)

def novel_pill(text):
    st.markdown(f'<span class="novel-pill">{text}</span>', unsafe_allow_html=True)

def finding(text):
    st.markdown(f'<div class="finding-box">{text}</div>', unsafe_allow_html=True)

def good(text):
    st.markdown(f'<div class="good-box">{text}</div>', unsafe_allow_html=True)

def warn(text):
    st.markdown(f'<div class="warn-box">{text}</div>', unsafe_allow_html=True)

def null(text):
    st.markdown(f'<div class="null-box">{text}</div>', unsafe_allow_html=True)

def analogy(text):
    st.markdown(f'<div class="analogy-box"><b>Analogy:</b> {text}</div>', unsafe_allow_html=True)

def limit(text):
    st.markdown(f'<div class="limit-box"><b>⚠ Limitation:</b> {text}</div>', unsafe_allow_html=True)

def honest_pill():
    st.markdown('<span class="honest-pill">✓ OOF-HONEST</span>', unsafe_allow_html=True)

def stat(number, label, color=""):
    st.markdown(
        f'<div class="stat-card"><div class="stat-number {color}">{number}</div>'
        f'<div class="stat-label">{label}</div></div>',
        unsafe_allow_html=True,
    )

def phase(text):
    st.markdown(f'<div class="phase-badge">{text}</div>', unsafe_allow_html=True)

def math(latex):
    st.latex(latex)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def page_brief():
    st.title("Research Brief — Why This Work Matters")
    st.markdown(
        "##### For decision-makers and stakeholders interested in trustworthy AI — "
        "no prior technical knowledge required."
    )

    # ── Opening banner ────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a0a3c,#4527a0);color:#e8e4ff;
    border-radius:12px;padding:1.3rem 1.6rem;margin:0.5rem 0 1.4rem;
    box-shadow:0 4px 20px rgba(69,39,160,0.25);line-height:1.7;">
    <b style="font-size:1.1rem;">The one-sentence version:</b><br>
    When an AI system gives you a wrong answer, it sounds exactly like when it gives you
    a right answer — and this research asks whether the AI itself can tell the difference
    by looking at what is happening <em>inside</em> the model, not just at what it says.
    </div>
    """, unsafe_allow_html=True)

    # ── Section 1: The problem in plain language ──────────────────────────────
    st.subheader("1 — The Problem We Are Trying to Solve")
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        Language models — the technology behind tools like ChatGPT — produce fluent,
        confident text. That fluency is both their strength and their danger.

        When a model answers a medical question incorrectly, it uses the same confident
        tone as when it answers correctly. A doctor, a lawyer, or a policy analyst reading
        that output has no reliable way to tell the difference from the words alone.

        The standard approach to catching these errors is to ask: **"Is the model uncertain?"**
        If the model seems unsure — measured by how spread out its probability distribution
        is — reject the answer.

        **This does not work for the most dangerous class of errors.**

        There is a failure mode we call *confident hallucination*: the model is completely
        wrong, and completely certain. It repeatedly gives the same wrong answer with high
        confidence. Any uncertainty-based filter passes this through as safe. It is the
        failure mode that matters most in high-stakes deployment — and it is structurally
        invisible to standard detection methods.
        """)
    with col2:
        st.markdown("""
        <div style="background:#fff8f8;border:2px solid #c62828;border-radius:12px;
        padding:1.2rem;margin-top:0.5rem;">
        <div style="font-weight:800;color:#c62828;font-size:1rem;margin-bottom:0.7rem;">
        A real example from our data</div>
        <div style="font-size:0.88rem;color:#37474f;line-height:1.6;">
        <b>Question:</b> Who wrote "The Monk Who Sold His Ferrari"?<br><br>
        <b>Ground truth:</b> Robin Sharma<br><br>
        <b>Model output (confident, wrong):</b><br>
        <em>"The book was written by Donnie Ray Joseph, an American motivational speaker..."</em><br><br>
        <b>Model's confidence:</b> High — low entropy, high top-1 probability<br><br>
        <span style="color:#c62828;font-weight:700;">A standard filter would have accepted this.</span>
        </div></div>
        """, unsafe_allow_html=True)

    # ── Section 2: Our approach ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("2 — Our Approach: Look Inside, Not Just at the Output")
    st.markdown("""
    The key insight driving this work comes from a line of recent research in AI interpretability:
    **the model's internal state carries information that its output does not reveal.**

    A language model processes a question through 36 successive internal stages (called layers).
    At each stage, the model builds a richer internal "representation" of the question and
    its emerging answer. We probed all 36 layers and found — consistent with recent literature —
    that at a specific late stage (Layer 32, near the end), the internal representation of a
    hallucinated answer looks measurably *different* from that of a correct answer, even when
    the text output sounds identical.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background:#ede7f6;border-radius:10px;padding:1rem;text-align:center;">
        <div style="font-size:2rem;font-weight:800;color:#4527a0;">Step 1</div>
        <div style="font-weight:700;color:#4527a0;margin:0.4rem 0;">Surface Signals</div>
        <div style="font-size:0.83rem;color:#37474f;">
        Measure the model's output uncertainty — entropy, confidence, consistency
        across multiple responses. Fast but blind to confident hallucination.
        </div></div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:#e8f5e9;border-radius:10px;padding:1rem;text-align:center;">
        <div style="font-size:2rem;font-weight:800;color:#2e7d32;">Step 2</div>
        <div style="font-weight:700;color:#2e7d32;margin:0.4rem 0;">Internal Geometry</div>
        <div style="font-size:0.83rem;color:#37474f;">
        Extract the hidden state at Layer 32. Measure how close it sits to the
        "cluster of correct answers" vs the "cluster of wrong answers" in the model's
        internal representation space.
        </div></div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background:#fff8e1;border-radius:10px;padding:1rem;text-align:center;">
        <div style="font-size:2rem;font-weight:800;color:#bf360c;">Step 3</div>
        <div style="font-weight:700;color:#bf360c;margin:0.4rem 0;">Governance Decision</div>
        <div style="font-size:0.83rem;color:#37474f;">
        Combine all signals into a failure probability. Route each query to:
        ACCEPT, REGENERATE (try again), or ABSTAIN (refuse to answer).
        </div></div>
        """, unsafe_allow_html=True)

    # ── Section 3: What we actually found ────────────────────────────────────
    st.markdown("---")
    st.subheader("3 — What the Experiments Actually Show")
    st.markdown(
        "All numbers below are from **Experiment 24** — our final, leak-corrected, "
        "honest evaluation on 140 samples the system never saw during training."
    )

    col1, col2 = st.columns(2)
    with col1:
        findings = [
            ("✅ Confirmed", "Layer 32 hidden states encode a detectable hallucination signal",
             "A simple linear detector applied to the internal state at Layer 32 achieves AUROC = 0.758. "
             "AUROC (Area Under the ROC Curve) is a standard detection quality score: 0.50 = random guessing, "
             "1.00 = perfect detection. An AUROC of 0.758 means: if you pick one correct answer and one "
             "hallucinated answer at random, the system ranks the correct one higher about 76% of the time. "
             "This is a genuine, reproducible finding that has now appeared in multiple independent studies."),

            ("✅ Confirmed", "Adding internal geometry improves detection by +20 percentage points",
             "Token-level signals alone score 57% — barely above chance. Adding the internal "
             "geometry signals lifts this to 77%. The 20-point improvement is real, honest, "
             "and consistent across all 5 cross-validation folds. Cross-validation means the data "
             "is split 5 ways; the detector is trained on 4 parts and tested on the 5th, repeated "
             "5 times — so every sample is evaluated on data the detector never trained on."),

            ("✅ Confirmed", "The system's probability estimates are trustworthy (well-calibrated)",
             "When the system says it is 80% confident an answer is correct, it is actually "
             "correct about 80% of the time. This calibration property (ECE = 0.039) is what "
             "makes the governance thresholds meaningful rather than arbitrary."),

            ("✅ Confirmed", "Trying to 'steer' the model toward correct answers does not work",
             "We ran five separate experiments attempting to push the model's internal state "
             "toward correct-answer territory during generation. All five failed. This is a "
             "a rigorous finding of no effect — the model's wrong beliefs are not easily overwritten "
             "by patching the activation stream."),
        ]
        for badge, title, detail in findings:
            st.markdown(
                f'<div style="border:1px solid #c8e6c9;border-radius:10px;padding:0.9rem 1rem;'
                f'margin-bottom:0.6rem;background:#f9fffe;">'
                f'<span style="font-size:0.72rem;font-weight:700;color:#2e7d32;">{badge}</span><br>'
                f'<b style="font-size:0.90rem;color:#1a1433;">{title}</b><br>'
                f'<span style="font-size:0.82rem;color:#546e7a;line-height:1.5;">{detail}</span></div>',
                unsafe_allow_html=True,
            )

    with col2:
        null_results = [
            ("⚠ Honest Limit", "Detection accuracy of 77% is useful but not sufficient for high-stakes deployment",
             "A 77% AUROC means roughly 1 in 4 discriminations is wrong. The system compensates "
             "by being very conservative — it routes 90% of queries to 'try again' rather than "
             "accepting them. This is safe but expensive. A production system would need AUROC > 0.90 "
             "to serve answers at meaningful scale."),

            ("⚠ Honest Limit", "Only 6 out of 140 test queries were accepted — too few to draw strong conclusions",
             "With only 6 accepted answers, the hallucination rate among accepted answers has "
             "a 95% confidence interval of [0.4%, 64%]. The number is honest but statistically "
             "weak. Larger scale evaluation is the natural next step."),

            ("⚠ Honest Limit", "The cross-layer trajectory and geometric volume signals add nothing",
             "We tested two additional novel signal groups (Groups D and E). They added +0.004 "
             "AUROC — within statistical noise. We are reporting this honestly: these were "
             "null contributions. The signal that matters is the single-layer geometry (Group C)."),

            ("⚠ Honest Limit", "All evaluation is on one benchmark with one open-source model",
             "We used TruthfulQA (a public test set of 817 questions designed specifically to "
             "trip up AI systems with plausible-sounding false answers) and Qwen 2.5 3B "
             "(an open-source language model with 3 billion parameters). Both are reproducible "
             "and appropriate for academic research. We cannot claim these findings extend to "
             "GPT-4, Claude, or clinical/legal domains without further testing."),
        ]
        for badge, title, detail in null_results:
            st.markdown(
                f'<div style="border:1px solid #ffcdd2;border-radius:10px;padding:0.9rem 1rem;'
                f'margin-bottom:0.6rem;background:#fff8f8;">'
                f'<span style="font-size:0.72rem;font-weight:700;color:#c62828;">{badge}</span><br>'
                f'<b style="font-size:0.90rem;color:#1a1433;">{title}</b><br>'
                f'<span style="font-size:0.82rem;color:#546e7a;line-height:1.5;">{detail}</span></div>',
                unsafe_allow_html=True,
            )

    # ── Section 4: Literature grounding ──────────────────────────────────────
    st.markdown("---")
    st.subheader("4 — Why This Is Principled Research, Not Isolated Work")
    st.markdown("""
    What makes this work credible is not one new invention — it is a **principled combination
    of five independently validated research ideas**, each from peer-reviewed work in 2023–2025,
    assembled into a unified detection and governance pipeline for the first time.
    """)

    papers = [
        ("#4527a0", "Zou et al., 2023",
         "Truth Directions in Representations",
         "Showed that concepts like truthfulness exist as geometric directions inside language models. "
         "We use this to compute the 'truth vector' and test whether steering along it corrects hallucinations.",
         "We replicated the geometric structure. Steering along it did not work — an important negative result."),

        ("#4527a0", "Kuhn et al., ICLR 2023",
         "Semantic Uncertainty",
         "Introduced clustering model responses by meaning (not just surface similarity) to measure "
         "true information-level uncertainty. We implement this as our Group B2 signal.",
         "Implemented and validated. Adds marginal signal beyond cosine consistency."),

        ("#4527a0", "Li et al., 2023 (ITI)",
         "Inference-Time Intervention",
         "Proposed editing model activations during inference to reduce hallucinations. "
         "A foundational paper for the activation steering approach we test.",
         "We ran 5 experiments. Universal steering consistently failed — a clear finding of no effect."),

        ("#2e7d32", "Chen et al., INSIDE 2024",
         "Representation Geometry for Hallucination",
         "Showed that cosine similarity between hidden states and class centroids discriminates "
         "hallucinations. Direct foundation for our Group C signals.",
         "Core signal of our system. Validated: lifts AUROC from 0.57 to 0.77."),

        ("#2e7d32", "Zhang et al., ACL 2025",
         "Inter-layer Consistency Ratio (ICR Probe)",
         "Proposed tracking how a model's intermediate predictions evolve layer-by-layer. "
         "Consistent predictions across layers indicate reliability. We implement this as Group D.",
         "Implemented. Did not add discriminative power beyond Group C in our setting."),
    ]

    for color, ref, title, what, our_finding in papers:
        bg = "#ede7f6" if color == "#4527a0" else "#e8f5e9"
        st.markdown(
            f'<div style="border-left:4px solid {color};background:{bg};border-radius:0 10px 10px 0;'
            f'padding:0.85rem 1.1rem;margin-bottom:0.6rem;">'
            f'<span style="font-size:0.72rem;font-weight:700;color:{color};">{ref}</span><br>'
            f'<b style="font-size:0.92rem;color:#1a1433;">{title}</b><br>'
            f'<span style="font-size:0.82rem;color:#546e7a;">{what}</span><br>'
            f'<span style="font-size:0.80rem;color:{color};font-weight:600;margin-top:0.3rem;'
            f'display:block;">→ Our result: {our_finding}</span></div>',
            unsafe_allow_html=True,
        )

    # ── Section 5: Resource constraints ──────────────────────────────────────
    st.markdown("---")
    st.subheader("5 — Resource Reality: What We Used and Why")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Model:** Qwen 2.5 3B (open-source, free, Apache 2.0 licence)

        **Dataset:** TruthfulQA (public benchmark, specifically designed to test
        confident hallucination — the hardest failure mode)

        **Compute:** Consumer GPU (experiments run in Colab / local machine)

        **Why open-source is the right choice here:**
        - Results are fully reproducible by any research team
        - No access agreements or API costs
        - The findings, if they hold, are immediately applicable to open-source deployments
        - Scaling to larger proprietary models is the next step — not the current one
        """)
    with col2:
        warn(
            "<b>What resource constraints mean for interpreting results:</b><br>"
            "We cannot claim these findings generalise to GPT-4, Gemini, or Claude without testing. "
            "We cannot claim they hold on medical, legal, or financial domain text without domain-specific evaluation. "
            "What we can claim: on a public benchmark, with a reproducible open-source model, "
            "the internal geometry signal is real and consistent across all cross-validation folds."
        )

    # ── Section 6: Scope for extension ───────────────────────────────────────
    st.markdown("---")
    st.subheader("6 — Realistic Scope for Extended Research")
    st.markdown("""
    We are not claiming a solved problem. We are claiming a **validated starting point**
    with clear, well-scoped directions for extension. Each direction below is motivated
    directly by a finding or limitation from our 24 experiments.
    """)

    extensions = [
        ("Scale the model", "medium",
         "Test 7B and 13B open-source models (Llama 3, Mistral). "
         "Does the geometry signal strengthen with model size? "
         "Motivated by: our finding that geometry emerges at 3B but not at 117M (GPT-2 in MECH-INT)."),

        ("Domain-specific evaluation", "high",
         "Apply the pipeline to medical QA (MedQA), legal reasoning, and scientific fact-checking — "
         "where hallucination has real-world costs. "
         "Motivated by: TruthfulQA is a useful benchmark but not a deployment-realistic distribution."),

        ("Improve the signal quality", "high",
         "Current AUROC = 0.77. A practical governance system needs > 0.90. "
         "Research directions: contrastive training of the probe, fine-tuning the model with "
         "uncertainty-aware objectives, multi-layer ensemble instead of single best layer. "
         "Motivated by: the 90% regeneration rate shows the current signal is too noisy to accept at scale."),

        ("Understand why steering fails", "medium",
         "Five experiments showed activation steering does not work. Understanding *why* — "
         "whether hallucinations are encoded too redundantly, too late, or in the wrong subspace — "
         "is a mechanistic interpretability question with direct safety implications. "
         "Motivated by: Exp 22c rigorous null result."),

        ("Human evaluation of ground truth", "medium",
         "Our labels use ROUGE-L and semantic similarity — automated, not human-verified. "
         "Human annotation of a subset would validate whether the signal tracks genuine "
         "factual errors or just surface-form mismatches. "
         "Motivated by: label quality uncertainty is a known limitation of the current pipeline."),
    ]

    priority_color = {"high": "#c62828", "medium": "#bf360c"}
    for title, priority, detail in extensions:
        pc = priority_color.get(priority, "#4527a0")
        st.markdown(
            f'<div style="border:1px solid {pc};border-radius:10px;padding:0.85rem 1.1rem;'
            f'margin-bottom:0.5rem;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<b style="font-size:0.92rem;color:#1a1433;">{title}</b>'
            f'<span style="font-size:0.72rem;font-weight:700;color:{pc};background:{pc}22;'
            f'padding:0.15rem 0.6rem;border-radius:20px;">priority: {priority}</span></div>'
            f'<span style="font-size:0.82rem;color:#546e7a;display:block;margin-top:0.35rem;">{detail}</span></div>',
            unsafe_allow_html=True,
        )

    # ── Section 7: The honest bottom line ────────────────────────────────────
    st.markdown("---")
    st.subheader("7 — The Honest Bottom Line")
    col1, col2 = st.columns(2)
    with col1:
        good(
            "<b>What this work contributes:</b><br><br>"
            "A complete, reproducible pipeline that combines five recent research techniques "
            "into a unified hallucination detection and governance framework — tested across "
            "24 experiments, with a data-leakage bug caught and corrected before publication, "
            "and all limitations stated explicitly.<br><br>"
            "The core finding — that internal representation geometry at a specific layer "
            "discriminates hallucinations that surface signals cannot see — is honest, "
            "consistent across cross-validation folds, and grounded in five independent "
            "prior publications. It is worth investigating further."
        )
    with col2:
        null(
            "<b>What this work does not claim:</b><br><br>"
            "This is not a production-ready system. It is not a guaranteed safety mechanism. "
            "It does not eliminate hallucination — it estimates the probability of it, "
            "with AUROC 0.77 (a detection quality score; 0.50 = chance, 1.0 = perfect), which is "
            "enough to build a conservative routing policy but not enough to accept answers at scale "
            "with confidence.<br><br>"
            "The claim is scoped: on an open-source model, on a public benchmark, "
            "with reproducible methods, internal geometry carries real signal. "
            "That is all — and that is enough to justify the next step."
        )
    finding(
        "<b>The research argument in one paragraph:</b> "
        "Trustworthy AI requires knowing when AI is wrong. Current methods look at what the model says. "
        "This work shows that what the model <em>computes internally</em> — specifically, where its "
        "hidden state sits relative to the geometry of known correct and incorrect answers — provides "
        "measurable, calibrated signal that surface methods miss. "
        "The approach is principled (grounded in five peer-reviewed papers), honest (a leakage bug "
        "was found and corrected), resource-realistic (fully open-source), and scoped correctly "
        "(promising scope for extension, not a claim of solved problem). "
        "The most productive next investment is improving signal quality and testing on higher-stakes domains."
    )

    # ── Section 8: Suggested reading path ─────────────────────────────────────
    st.markdown("---")
    st.subheader("Where to Go Next in This Dashboard")
    st.markdown("Use the sidebar to navigate. Here is a suggested path depending on your interest:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background:#ede7f6;border-radius:10px;padding:1rem;">
        <b style="color:#4527a0;">If you want the headline results</b><br><br>
        <span style="font-size:0.85rem;">
        → <b>Overview</b> — project summary in one page<br>
        → <b>Honest Evaluation (Exp 24)</b> — all final numbers<br>
        → <b>Full Results</b> — complete experiment table
        </span></div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:#e8f5e9;border-radius:10px;padding:1rem;">
        <b style="color:#2e7d32;">If you want to understand the method</b><br><br>
        <span style="font-size:0.85rem;">
        → <b>Signal Architecture</b> — what signals we use<br>
        → <b>Layer Probing</b> — how we find the best layer<br>
        → <b>Failure Estimator</b> — how the detector works<br>
        → <b>Authorization Policy</b> — the 3-action decision
        </span></div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background:#fff8e1;border-radius:10px;padding:1rem;">
        <b style="color:#bf360c;">If you want scientific rigour</b><br><br>
        <span style="font-size:0.85rem;">
        → <b>⚠ Measurement Correction</b> — the leakage story<br>
        → <b>Calibration (ECE)</b> — probability trustworthiness<br>
        → <b>Sycophancy Robustness</b> — adversarial testing<br>
        → <b>Activation Steering</b> — what failed and why
        </span></div>
        """, unsafe_allow_html=True)

def page_overview():
    st.title("Can a Language Model Know When It's Hallucinating?")
    st.markdown(
        "##### HaRP — Hallucination-aware Representation Probing · Qwen 2.5 3B · TruthfulQA · 24 Experiments"
    )
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a0a3c,#4527a0);color:#e8e4ff;
    border-radius:12px;padding:1.1rem 1.5rem;margin:0.5rem 0 1.2rem;
    box-shadow:0 4px 20px rgba(69,39,160,0.25);">
    <b style="font-size:1.05rem;">The core claim:</b>&nbsp;
    Token-level uncertainty (entropy, top-1 probability) is <em>insufficient</em> to detect the most
    dangerous class of LLM hallucinations — those where the model is <em>confident and wrong</em>.
    Hidden-state geometry at Layer 32 catches what surface signals miss, lifting detection AUROC
    from <b>0.57 → 0.77 (+0.20)</b> in an honest, leak-free evaluation.
    </div>
    """, unsafe_allow_html=True)

    exp24 = load_json("24_honest_results.json")
    exp15 = load_json("15_calibration_results.json")
    exp06 = load_json("06_probing_results.json")
    pm24  = exp24.get("test_policy_metrics", {})

    st.markdown("---")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: stat("700", "Samples (TruthfulQA)")
    with c2: stat("39.5%", "Raw Hallucination Rate", "stat-number-red")
    with c3: stat(f"{exp06.get('best_auroc', 0.7583):.4f}", "Best-Layer Probe AUROC", "stat-number-indigo")
    with c4: stat("0.7753", "Honest AUROC (22 features)", "stat-number-green")
    with c5: stat("+0.198", "Geometry Lift (A → A+B+C)", "stat-number-indigo")
    with c6: stat(f"{exp15.get('systems',{}).get('harp_full',{}).get('ece',0.039):.3f}", "ECE — Calibration Error (lower = better)", "stat-number-green")

    st.markdown("---")
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("The Problem")
        st.markdown("""
        Large language models hallucinate — they produce fluent, confident, grammatically perfect
        sentences that are factually wrong. The standard detection approach uses **token-level signals**:
        output entropy, top-1 probability, confidence gap.

        **This is structurally insufficient.** The most dangerous failure mode — *confident hallucination* —
        produces *low* entropy. The model "sounds sure." It consistently gives the same wrong answer.
        Any entropy-based filter will classify this as safe and pass it through.
        """)
        null(
            "<b>The structural gap:</b> Token signals measure <em>how certain</em> the model is — not whether "
            "it is right. A model can be maximally certain and completely wrong. These two failure modes "
            "(confident correct vs. confident hallucination) are <em>indistinguishable from the output distribution</em>. "
            "You must inspect the model's internal representation space to separate them."
        )
        st.subheader("The HaRP Approach")
        st.markdown("""
        HaRP studies Qwen 2.5 3B (3B parameters, 36 transformer layers, 2048 hidden dim) on
        TruthfulQA — a benchmark specifically designed to elicit confident hallucinations.

        **Three-stage architecture:**
        1. **Detection** — Extract 5 signal groups across token, consistency, and representation modalities
        2. **Estimation** — Train a logistic regression failure estimator P(failure | signals)
        3. **Routing** — Apply a 3-action governance policy: ACCEPT / REGENERATE / ABSTAIN
        """)
        good(
            "<b>Honest result (Exp 24, OOF-corrected):</b> AUROC = <b>0.7753</b>. "
            "Representation geometry at L32 adds +0.20 AUROC over token signals alone. "
            "ECE = 0.039 — well-calibrated probabilities suitable for threshold-based governance. "
            "<em>Note: an earlier estimate (Exp 23) showed 0.962 due to data leakage — see the Measurement Correction page.</em>"
        )

    with col2:
        st.subheader("Research Arc — 24 Experiments")
        st.caption("Each 'Experiment' (Exp) is a numbered analysis step — a discrete test with a defined hypothesis, method, and result. The 24 experiments form a logical chain from raw data to final evaluation.")
        steps = [
            ("#4527a0", "Exp 01–05", "Data & Baselines",      "700 samples; raw hall. rate 39.5%; token AUROC 0.57"),
            ("#4527a0", "Exp 06",    "Layer Probing",          "Best layer = L32; AUROC 0.7583 ± 0.057"),
            ("#4527a0", "Exp 07",    "Geometry of Truth",      "PC1↔label corr = 0.424; linear truth direction confirmed"),
            ("#4527a0", "Exp 08",    "Logit Lens",             "Hallucinations commit at mean L4.07; 27.2% never commit"),
            ("#c62828", "Exp 09b",   "⚠ Leakage Correction",  "In-sample AUROC 1.0 → OOF 0.757; +0.24 inflation found"),
            ("#4527a0", "Exp 10–11", "Activation Steering",    "Universal vector: all α values fail; no improvement in 5 experiments"),
            ("#4527a0", "Exp 12–15", "Policy & Calibration",   "α*=0.15, β*=0.90; ECE 0.072 → 0.039"),
            ("#2e7d32", "Exp 17",    "Semantic Entropy",       "NLI clustering; Group B2 (marginal lift)"),
            ("#2e7d32", "Exp 18",    "4-Quadrant Taxonomy",    "entropy × consistency axes; conf. hall. rate 61%"),
            ("#2e7d32", "Exp 20–21", "Trajectory + Volume",    "Groups D+E: +0.004 AUROC — null contribution"),
            ("#2e7d32", "Exp 22c",   "Quadrant Steering",      "Confident hallucinations immune at all α — rigorous null"),
            ("#2e7d32", "Exp 24",    "Honest Evaluation",      "AUROC 0.7753; test 0.7871; 6/140 accepted"),
        ]
        st.markdown(
            '<div style="font-size:0.73rem;color:#8e8e8e;margin-bottom:0.5rem;">'
            '<b>Abbreviation guide:</b> AUROC = detection quality score (0.5=chance, 1.0=perfect) · '
            'ECE = calibration error (lower=better) · L32 = Layer 32 · '
            'OOF = evaluated on held-out data · PC1 = primary pattern extracted from geometry · '
            'α/β = decision thresholds</div>', unsafe_allow_html=True)
        for color, badge, title, desc in steps:
            bg = "#ffebee" if "#c62828" in color else ("#e8f5e9" if "#2e7d32" in color else "#ede7f6")
            st.markdown(
                f'<div style="border-left:3px solid {color};background:{bg};'
                f'padding:0.42rem 0.8rem;border-radius:0 8px 8px 0;margin-bottom:0.3rem;">'
                f'<span style="font-size:0.70rem;font-weight:700;color:{color};">{badge}</span> '
                f'<span style="font-size:0.83rem;font-weight:600;color:#1a1433;"> {title}</span><br>'
                f'<span style="font-size:0.76rem;color:#6b7280;">{desc}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.caption("For a non-technical explanation of these findings, see the 📋 Research Brief page. For the full numbers, see Honest Evaluation (Exp 24).")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("What This Work Proves (Honest)")
        claims = [
            ("✅", "L32 hidden states encode a linear hallucination signal",            "AUROC 0.7583, 5-fold CV"),
            ("✅", "Token entropy alone is near-random for hallucination detection",    "AUROC 0.57 ≈ chance"),
            ("✅", "Representation geometry adds +0.20 AUROC over token signals",      "OOF-corrected, consistent across all 5 CV folds"),
            ("✅", "A linear probe is the single most informative signal",              "probe_conf AUROC 0.757 OOF"),
            ("✅", "Universal activation steering does not reduce hallucination rate",  "5 experiments, all null"),
            ("✅", "Negative-α steering for confident hallucinations also fails",       "Exp 22c, rigorous null"),
            ("✅", "Groups D+E (trajectory, volume) add nothing beyond Group C",        "+0.004, within noise"),
            ("✅", "HaRP probabilities are well-calibrated",                            "ECE 0.039 < 0.05 threshold"),
        ]
        for icon, claim, evidence in claims:
            st.markdown(
                f'<div style="display:flex;gap:0.6rem;align-items:flex-start;'
                f'border:1px solid #ddd8f0;border-radius:8px;padding:0.45rem 0.7rem;margin-bottom:0.3rem;">'
                f'<span style="font-size:1rem;flex-shrink:0;">{icon}</span>'
                f'<span style="font-size:0.84rem;"><b>{claim}</b><br>'
                f'<span style="color:#6b7280;font-size:0.77rem;">{evidence}</span></span></div>',
                unsafe_allow_html=True,
            )
    with col_b:
        st.subheader("What This Work Does NOT Claim")
        limits = [
            ("❌", "SOTA hallucination detection",      "No comparison to prior work on shared benchmarks"),
            ("❌", "Zero hallucination rate",            "Exp 23 was leakage; honest: 1/6 accepted (CI [0.004, 0.641])"),
            ("❌", "Deployable production system",       "700 samples; no stress-testing; high abstention (90% regen)"),
            ("❌", "Generalisable beyond TruthfulQA",    "Single benchmark; adversarial/NQ results are exploratory"),
            ("❌", "Formal Bayesian uncertainty decomp.","4-quadrant taxonomy uses heuristic thresholds, not Bayesian"),
            ("❌", "Activation steering as a fix",       "5 experiments consistently null — steering does not help"),
            ("❌", "Groups D+E contribute",              "+0.004 AUROC within noise — these are null contributions"),
        ]
        for icon, claim, note in limits:
            st.markdown(
                f'<div style="display:flex;gap:0.6rem;align-items:flex-start;'
                f'border:1px solid #ffcdd2;border-radius:8px;padding:0.45rem 0.7rem;margin-bottom:0.3rem;background:#fff8f8;">'
                f'<span style="font-size:1rem;flex-shrink:0;">{icon}</span>'
                f'<span style="font-size:0.84rem;"><b>{claim}</b><br>'
                f'<span style="color:#6b7280;font-size:0.77rem;">{note}</span></span></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.subheader("Relationship to Prior Work")
    rows = [
        ["SelfCheckGPT (Manakul 2023)",   "Consistency sampling",      "No",  "No",              "No",  "No"],
        ["Semantic Uncertainty (Kuhn 2023)", "NLI clustering",          "No",  "No",              "No",  "No"],
        ["ITI (Li et al. 2023)",           "Representation probing",    "Yes", "Yes (primary goal)","No","No"],
        ["INSIDE (Chen et al. 2024)",      "Representation geometry",   "Yes", "No",              "No",  "No"],
        ["ICR Probe (Zhang ACL 2025)",     "Cross-layer trajectory",    "Yes", "No",              "No",  "No"],
        ["HaRP (this work)",               "All 5 groups combined",     "Yes", "Yes (did not work)","Yes","Yes"],
    ]
    df = pd.DataFrame(rows, columns=["System", "Signals Used", "Uses Internal Geometry", "Tests Activation Steering", "3-Action Policy", "Calibrated"])
    st.dataframe(df, hide_index=True, use_container_width=True)
    warn("This table shows design-level distinctions only — HaRP does not compare to these systems on the same benchmark split. No head-to-head AUROC numbers are claimed.")


def page_signals():
    st.title("Signal Architecture")
    phase("Phase 1–2 — Data Collection & Feature Engineering")

    st.markdown("""
    HaRP operates in two inference modes. Understanding *why* is essential to understanding the architecture.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Ollama API Mode
        Used for: Groups A, B, B2

        Qwen 2.5 3B runs locally via Ollama. The `/api/generate` endpoint returns **logprobs** —
        log probabilities for the top-k tokens at each generation step.

        This is the **fast path**: no GPU memory overhead, runs on CPU. Used during data collection
        and for any signals derivable from the output distribution alone.
        """)
        finding("Logprobs give us: entropy, top-1 probability, confidence gap, logit variance — but nothing about <em>internal representations</em>.")

    with col2:
        st.markdown("""
        ### HuggingFace Mode
        Used for: Groups C, C2, D, E

        The same Qwen 2.5 3B loaded via `transformers` in float16. Running both simultaneously
        would exceed 8GB VRAM — so **these modes are mutually exclusive**.

        This is the **expensive path**: requires full model load (~6GB VRAM), but gives access to:
        - Hidden states at every layer (`output_hidden_states=True`)
        - Attention weights at every head (`output_attentions=True`)
        """)
        warn("IMPORTANT: Ollama must be stopped before running HuggingFace mode. Both load the same model weights into GPU memory.")

    st.markdown("---")
    st.subheader("The 5 Signal Groups")

    groups = [
        {
            "label": "Group A — Token Signals (6 features)",
            "color": "#e0f2f1",
            "border": "#00695c",
            "source": "Ollama logprobs",
            "features": ["entropy", "max_entropy", "logit_var", "conf_gap", "top1_prob", "first_entropy"],
            "desc": "The model's own uncertainty about its output. Computable without any access to model internals — just the token probability distribution. These are the signals most prior work relies on.",
            "limitation": "Cannot detect confident hallucination. When a model is wrong and confident, entropy is LOW — these features give no warning.",
        },
        {
            "label": "Group B — Cosine Consistency (2 features)",
            "color": "#e8eaf6",
            "border": "#3949ab",
            "source": "Ollama + sentence-transformers",
            "features": ["consistency_score", "consistency_variance"],
            "desc": "Generate 3 responses to the same prompt. Embed each with sentence-transformers. Compute pairwise cosine similarities. High consistency = model agrees with itself. Low consistency = aleatoric uncertainty.",
            "limitation": "Consistent models can consistently hallucinate. High self-agreement does not imply correctness — it only measures variance, not accuracy.",
        },
        {
            "label": "Group B2 — Semantic Entropy (2 features) [NOVEL]",
            "color": "#f3e5f5",
            "border": "#8e24aa",
            "source": "cross-encoder/nli-deberta-v3-small",
            "features": ["semantic_entropy", "semantic_cluster_ratio"],
            "desc": "Cluster the 3 responses by bidirectional NLI entailment. Responses that entail each other are in the same meaning-cluster. Semantic entropy measures diversity of meaning (not just surface form). Implements Kuhn et al. (2023) Semantic Uncertainty.",
            "limitation": "Requires 3 forward passes through both the LLM and the NLI model. Approximately 3x slower than Group A alone.",
        },
        {
            "label": "Group C — Representation Geometry (4 features)",
            "color": "#e8f5e9",
            "border": "#2e7d32",
            "source": "HuggingFace hidden states (best_layer = 32)",
            "features": ["cos_correct", "cos_hallucinated", "h_norm", "probe_conf"],
            "desc": "At best_layer (L32), extract the last-token hidden state h. Compare it geometrically to the class means μ_correct and μ_hallucinated computed from the training split. Confident hallucination has low entropy BUT its hidden state falls near μ_hallucinated.",
            "limitation": "Requires HuggingFace mode — cannot run simultaneously with Ollama. Adds ~150ms per sample.",
        },
        {
            "label": "Group C2 — Attention Entropy (2 features)",
            "color": "#fff3e0",
            "border": "#fb8c00",
            "source": "HuggingFace output_attentions=True",
            "features": ["attn_entropy_mean", "attn_entropy_std"],
            "desc": "Measure how diffuse vs. focused each attention head's distribution is at best_layer. Diffuse attention (high entropy) suggests the model cannot clearly identify which context is relevant to the answer.",
            "limitation": "Adds marginal AUROC improvement over Group C alone — but validates the attention-based uncertainty hypothesis.",
        },
        {
            "label": "Group D — Cross-Layer Trajectory (5 features) [NOVEL]",
            "color": "#e1f5fe",
            "border": "#0277bd",
            "source": "HuggingFace hidden states (all layers)",
            "features": ["traj_total_variation", "traj_stabilization_layer", "traj_max_change_layer", "traj_early_commitment", "icr_score"],
            "desc": "Track how the hidden state evolves from layer 0 to layer 36. Correct answers stabilize early (low total variation, early commitment). Hallucinations oscillate late. Implements the ICR Probe (Zhang et al., ACL 2025).",
            "limitation": "Requires storing all 37 layer hidden states simultaneously — 37 × 2048 × 2 bytes per sample.",
        },
        {
            "label": "Group E — Geometric Volume (3 features) [NOVEL]",
            "color": "#f9fbe7",
            "border": "#827717",
            "source": "HuggingFace hidden states (multiple regenerations)",
            "features": ["simplex_volume", "embedding_spread", "centroid_drift"],
            "desc": "Generate 3 responses; extract their hidden states; compute the volume of the simplex they span (via Gram matrix determinant). Large volume = representations are spread out = high uncertainty. Implements Granese et al. (arXiv 2509.13813, ICLR 2026).",
            "limitation": "Requires 3 HuggingFace forward passes per sample — 3x the compute of Group C.",
        },
    ]

    for g in groups:
        is_novel = "[NOVEL]" in g["label"]
        st.markdown(
            f'<div style="border:2px solid {g["border"]};background:{g["color"]};'
            f'border-radius:10px;padding:1rem 1.2rem;margin-bottom:0.8rem;">',
            unsafe_allow_html=True,
        )
        label_clean = g["label"].replace(" [NOVEL]", "")
        if is_novel:
            st.markdown(f'<b style="color:{g["border"]};">{label_clean}</b> &nbsp; <span class="novel-pill">NOVEL CONTRIBUTION</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<b style="color:{g["border"]};">{label_clean}</b>', unsafe_allow_html=True)

        st.markdown(f'<span style="font-size:0.78rem;color:#6b7280;">Source: {g["source"]}</span>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:0.88rem;margin:0.4rem 0;">{g["desc"]}</p>', unsafe_allow_html=True)
        chips = " ".join([f'<span class="feature-chip">{f}</span>' for f in g["features"]])
        st.markdown(chips, unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:0.82rem;color:#e65100;margin-top:0.4rem;"><b>Limitation:</b> {g["limitation"]}</p>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Why Logistic Regression, Not a Neural Classifier?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        The failure estimator is a **logistic regression pipeline** (StandardScaler → LR with L2 regularization, C=1.0).

        **Why not MLP or XGBoost?**
        - Training set size: N ≈ 258 samples (60% of 430). Neural classifiers overfit badly at this scale.
        - The MLP ablation achieved AUROC 0.947 in CV — but this is **overfitting to the CV splits**, not generalization.
        - LR's probability outputs are better calibrated (ECE 0.039 vs expected >0.10 for MLP).
        - LR coefficients are directly interpretable: which features matter most?
        """)
    with col2:
        finding(
            "<b>The LR philosophy:</b> At N=430, a well-regularized linear classifier over carefully engineered "
            "features outperforms a flexible nonlinear classifier. The features encode the domain knowledge; "
            "the LR learns only the weights. This is the right inductive bias for small-sample safety-critical classification."
        )


def page_probing():
    st.title("Layer Probing — Where Does Hallucination Information Live?")
    exp_pill("Experiment 06 — Phase 4")
    phase("Phase 4 — Representation Analysis")

    st.markdown("""
    **Scientific question:** At which transformer layer does a linear probe best discriminate
    correct answers from hallucinations? This determines `best_layer` — used by every downstream
    experiment (geometry, steering vector, Group C features).
    """)

    exp06 = load_json("06_probing_results.json")
    best_layer = exp06.get("best_layer", 32)
    best_auroc = exp06.get("best_auroc", 0.730)
    all_layers = exp06.get("all_layers", [])

    c1, c2, c3 = st.columns(3)
    with c1: stat(str(best_layer), "Best Layer (peak AUROC)")
    with c2: stat(f"{best_auroc:.4f}", "Best Layer AUROC", "stat-number-indigo")
    with c3: stat("37", "Layers Probed (L0–L36)")

    st.markdown("---")

    col1, col2 = st.columns([3, 2])
    with col1:
        plot_img("06_probing_curve.png", "Probing AUROC by transformer layer — peaks at L32")
    with col2:
        st.subheader("Why a Linear Probe?")
        st.markdown("""
        A **linear probe** (logistic regression) is applied separately to the last-token hidden state
        at each layer. We measure AUROC on 5-fold CV.

        **Why last-token?**
        In autoregressive models, the last token position attends to the full context — it is the
        state that directly determines what gets generated next. Mean-pooling over all tokens
        conflates the question representation with the answer representation.

        **Why logistic regression for the probe?**
        - Fast to train (no GPU needed)
        - L2 regularization prevents overfitting in high-dimensional hidden space (dim=2048)
        - AUROC is threshold-free — measures discriminative power independently of operating point
        """)
        finding(
            f"<b>Finding:</b> AUROC rises monotonically through early layers, "
            f"peaks at <b>Layer {best_layer} (AUROC {best_auroc:.4f})</b>, then slightly decreases. "
            f"Hallucination information concentrates in the final third of the network — "
            f"where the model's \"decision\" about what to output is being finalized."
        )

        if all_layers:
            st.markdown("**Layer AUROC Summary:**")
            early = [l for l in all_layers if l["layer"] <= 5]
            mid   = [l for l in all_layers if 10 <= l["layer"] <= 20]
            late  = [l for l in all_layers if l["layer"] >= 30]
            def avg(lst): return np.mean([x["auroc_mean"] for x in lst]) if lst else 0
            cols = st.columns(3)
            with cols[0]: stat(f"{avg(early):.3f}", "Avg AUROC L0–5")
            with cols[1]: stat(f"{avg(mid):.3f}", "Avg AUROC L10–20", "stat-number-indigo")
            with cols[2]: stat(f"{avg(late):.3f}", "Avg AUROC L30+", "stat-number-green")

    st.markdown("---")
    st.subheader("What Does AUROC 0.758 Mean?")
    st.markdown("""
    A random classifier achieves AUROC 0.50. A perfect classifier achieves 1.0. AUROC 0.758 means:

    > **Given a randomly chosen correct sample and a randomly chosen hallucinated sample, the probe
    > correctly ranks the correct sample as "more likely correct" 75.8% of the time.**

    This is meaningful and real — it confirms that hallucination information *does* concentrate in Layer 32
    representations. But it is not enough alone for production-grade detection.
    """)
    analogy(
        "A doctor who looks at an X-ray (probing) is limited by what's visible in the image. "
        "Adding blood tests and patient history (Groups A, B, C) dramatically improves diagnosis — "
        "but the X-ray finding is the most important single signal."
    )
    limit(
        "The probe AUROC (0.758) is computed with 5-fold CV on 700 samples. "
        "Standard deviation is ±0.057 — non-trivial. The probe is trained on the full 2048-dimensional "
        "hidden state with L2 regularisation; it may not generalise to out-of-distribution prompts or "
        "different model architectures without retraining."
    )


def page_geometry():
    st.title("Geometry of Truth — Does a Truth Direction Exist?")
    exp_pill("Experiment 07 — Phase 5")
    phase("Phase 5 — Geometric Validation")

    st.markdown("""
    **Scientific question:** At `best_layer`, is there a *linear* direction in representation space
    that separates correct answers from hallucinations? If yes, representation geometry is a valid
    signal. If no, Group C features are meaningless.
    """)

    exp07 = load_json("07_geometry_results.json")
    pc1_corr = exp07.get("pc1_correlation", 0.322)
    sep      = exp07.get("separability", 0.653)
    ev       = exp07.get("explained_variance", [0.079, 0.047])
    interp   = exp07.get("interpretation", "weak linear structure")

    c1, c2, c3 = st.columns(3)
    with c1: stat(f"{pc1_corr:.3f}", "PC1 ↔ Truth Correlation")
    with c2: stat(f"{sep:.3f}", "Linear Separability Score")
    with c3: stat(f"{ev[0]*100:.1f}%", "PC1 Explained Variance")

    st.markdown("---")
    col1, col2 = st.columns([3, 2])
    with col1:
        plot_img("07_pca_scatter.png", "PCA scatter at Layer 32 — coloured correct (blue) vs hallucinated (red)")
    with col2:
        st.subheader("The Geometry of Truth Hypothesis")
        st.markdown("""
        **Zou et al. (2023)** showed that concepts like truthfulness are encoded as *directions* in
        representation space. This is the **linear representation hypothesis**: if P is a property,
        there exists a vector d such that representations of samples with P tend to lie on the
        positive side of d.

        We test this at `best_layer` using PCA:
        1. Extract hidden states h ∈ ℝ^2048 for all 300 samples
        2. Run PCA, project to 2D
        3. Correlate PC1 with the binary correct/hallucinated label
        """)
        math(r"\text{PC1\_corr} = \text{corr}(\text{PC1 scores}, y_{\text{correct}})")

        if abs(pc1_corr) > 0.4:
            good(f"<b>Strong geometry (|corr| > 0.4):</b> A clear linear truth direction exists. PC1 ≈ truth axis.")
        elif abs(pc1_corr) > 0.25:
            warn(
                f"<b>Moderate geometry (|corr| = {pc1_corr:.3f}):</b> Some linear structure exists but classes overlap. "
                f"Group C signals still work because cosine distance to class <em>means</em> captures the direction even with overlap."
            )
        else:
            null("<b>Weak geometry:</b> Little linear separation at this layer.")

        st.markdown("""
        **What 'weak linear structure' means for HaRP:**

        Even with moderate PC1 correlation, the Group C features (cos_correct, cos_hallucinated, h_norm, probe_conf)
        still provide a massive AUROC boost (+0.34). Why? Because the **class means** μ_correct and μ_hallucinated
        are computed *within* the 2048D space — the projection to 2D for visualization loses most of this signal.
        The linear structure that matters is in the full space, not the 2D projection.
        """)

    st.markdown("---")
    st.subheader("Contrast with MECH-INT (GPT-2)")
    st.caption("MECH-INT is our companion project — a prior study that analyzed how the smaller GPT-2 model (117M parameters) processes and hallucinates information, using mechanistic interpretability techniques.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **MECH-INT (GPT-2 117M):** Geometry analysis found no strong hallucination signal.
        At 117M parameters, the model's representation space is too low-dimensional and the hallucination
        mechanism is distributed across many heads, not concentrated in a single direction.
        """)
    with col2:
        st.markdown("""
        **HaRP (Qwen 2.5 3B):** At 3B parameters with 2048-dimensional hidden states, enough
        representational capacity exists for a detectable truth direction. This *validates* the
        use of geometry-based signals and motivates the REGENERATE action via activation steering.
        """)
    finding(
        f"<b>Key insight:</b> The geometry of truth is a scale-dependent phenomenon. "
        f"It emerges reliably at ~3B+ parameters (PC1 corr = 0.424 at Qwen 3B) but is absent in "
        f"smaller models — MECH-INT on GPT-2 117M found no significant geometry. "
        f"HaRP's Group C signals require at least moderate geometric separability to work."
    )
    limit(
        "The geometry analysis uses 2D PCA for visualisation — the full 2048D space may have stronger "
        "structure than the 2D projection shows. PC1 explains only 7.5% of total variance. "
        "The class means μ_correct and μ_hallucinated are computed on the training split only (OOF in Exp 24) "
        "and may drift if the prompt distribution changes significantly."
    )


def page_logit_lens():
    st.title("Logit Lens — When Does the Model Commit to an Answer?")
    exp_pill("Experiment 08 — Phase 5")
    phase("Phase 5 — Commitment Layer Analysis")

    st.markdown("""
    **Scientific question:** At which layer does the model "commit" to producing the correct (or wrong) token?
    Correct answers should commit early and stably. Hallucinations should either commit late (the model changes
    its mind) or never commit (unstable across layers).
    """)

    exp08 = load_json("08_logit_lens_results.json")
    mean_commit_hall = exp08.get("mean_commit_hall", 3.79)
    no_commit_pct    = exp08.get("no_commit_pct", 25.0)
    n_ex             = exp08.get("n_examples", 84)

    c1, c2, c3 = st.columns(3)
    with c1: stat(f"{mean_commit_hall:.2f}", "Mean Commitment Layer (Hallucinated)")
    with c2: stat(f"{no_commit_pct:.0f}%", "Never Committed (No Stable Answer)", "stat-number-red")
    with c3: stat(str(n_ex), "Examples Analyzed")

    st.markdown("---")
    col1, col2 = st.columns([3, 2])
    with col1:
        plot_img("08_logit_lens.png", "Logit lens: commitment layer distribution for hallucinated samples")
    with col2:
        st.subheader("The Logit Lens Method")
        st.markdown("""
        The **logit lens** (nostalgebraist, 2020) applies the model's final unembedding matrix W_U
        to intermediate hidden states at each layer:
        """)
        math(r"p_l(\text{token}) = \text{softmax}(W_U \cdot \text{LayerNorm}(h_l))")
        st.markdown("""
        This gives a "prediction" at each layer — what token would the model output if it stopped processing here.

        **Commitment layer** is defined as the first of 3 consecutive layers where the top predicted token
        remains the same. This operationalizes the intuition that a model that "knows" the answer will lock
        onto it early and not waver.

        **Why gold-token tracking?**
        We track the probability of the *ground-truth correct token*, not the generated token.
        The generated token is a null control — the model always generates it by definition at the final layer.
        Tracking the gold token reveals whether correct information was accessible in intermediate representations.
        """)
        finding(
            f"<b>Finding:</b> Hallucinated samples commit at mean layer <b>{mean_commit_hall:.1f}</b> "
            f"(of 36 total). {no_commit_pct:.0f}% never reach a stable commitment across the window. "
            f"This confirms cross-layer dynamics carry predictive information — motivating Group D (trajectory signals)."
        )

    st.markdown("---")
    st.subheader("Connection to Trajectory Signals (Group D)")
    st.markdown("""
    The logit lens experiment is the *motivation* for Experiment 20 (Group D signals). If hallucinations
    commit late or not at all, then measuring:
    - **Total variation** of the hidden state across layers (how much does h change?)
    - **Stabilization layer** (when does the trajectory stop changing significantly?)
    - **Early commitment** (did the token lock in before layer 20?)

    ...should give discriminative features. Experiment 20 confirms this, adding the ICR score
    (Inter-layer Consistency Ratio) from Zhang et al., ACL 2025.
    """)


def page_estimator():
    st.title("Failure Estimator — The Central Ablation")
    exp_pill("Experiment 04 — Phase 2")
    exp_pill("Experiment 09b — OOF corrected")
    exp_pill("Experiment 24 — Definitive")
    honest_pill()
    phase("Phase 2 & 7 — Estimator v1 → v2 → Honest Eval")

    st.markdown("""
    The most important empirical result in HaRP is the **ablation study**: adding signal groups
    one at a time and measuring AUROC via 5-fold cross-validation with **Out-of-Fold (OOF)**
    Group C features — the leak-free evaluation that supersedes Exp 09/23.
    """)

    exp09 = load_json("09_estimator_v2.json")
    abl   = exp09.get("ablation", {})

    exp24 = load_json("24_honest_results.json")
    abl24 = exp24.get("ablation", {})
    a_auroc   = abl24.get("Group A only",   {}).get("auroc_mean", 0.5734)
    ab_auroc  = abl24.get("Group A+B",      {}).get("auroc_mean", 0.5936)
    abc_auroc = abl24.get("Group A+B+C",    {}).get("auroc_mean", 0.7714)
    delta_abc = abc_auroc - a_auroc
    c1, c2, c3, c4 = st.columns(4)
    with c1: stat(f"{a_auroc:.4f}", "Group A AUROC (Token signals)", "stat-number-red")
    with c2: stat(f"{ab_auroc:.4f}", "Group A+B AUROC (+Consistency)", "stat-number-red")
    with c3: stat(f"{abc_auroc:.4f}", "Group A+B+C AUROC — OOF honest", "stat-number-green")
    with c4: stat(f"+{delta_abc:.3f}", "Delta: A → A+B+C", "stat-number-indigo")

    st.markdown("---")
    col1, col2 = st.columns([3, 2])
    with col1:
        plot_img("09_auroc_comparison.png", "AUROC by feature group — the geometry jump")
        plot_img("16_ablation_bar.png", "Ablation with 5-fold CV error bars")
    with col2:
        st.subheader("Why Does A+B NOT Help?")
        st.markdown("""
        Group A (token signals) → **AUROC 0.5423**

        Group A+B (+ cosine consistency) → **AUROC 0.5362** *(worse!)*

        Adding cosine consistency barely changes performance. Why?

        1. **Correlation:** High-entropy samples (uncertain in Group A) also tend to produce
           inconsistent responses (low consistency in Group B). The two groups measure the same
           underlying phenomenon from different angles.

        2. **The hard cases:** Confident hallucination samples have *low entropy* (Group A sees
           nothing unusual) AND *high consistency* (the model consistently repeats the same wrong
           answer). Group B cannot discriminate here either.
        """)
        null(
            "<b>Key insight:</b> For the failure mode that matters most — confident hallucination — "
            "both token entropy AND cosine consistency are LOW. You cannot detect this class without "
            "looking at the model's internal representation."
        )

        st.subheader("Why Group C Jumps So Much")
        st.markdown("""
        Group C adds: `cos_correct`, `cos_hallucinated`, `h_norm`, `probe_conf`

        These measure the **geometric position** of the hidden state relative to the class means.
        A confident hallucination has:
        - Low entropy (Group A blind)
        - High consistency (Group B blind)
        - **High cosine similarity to μ_hallucinated** (Group C SEES IT)

        This is the "smoking gun" argument: representation geometry captures the failure mode
        that all surface-level signals miss.
        """)
        good(
            "<b>The honest headline:</b> AUROC increases from 0.57 → 0.77 (+0.20) when adding "
            "representation geometry (OOF-corrected). This is the core honest contribution of HaRP. "
            "An earlier in-sample estimate showed +0.34 — that was data leakage (see Measurement Correction page)."
        )

    limit(
        "AUROC 0.7714 is computed with 5-fold CV, std ±0.047 — substantial noise for a 700-sample dataset. "
        "The AUROC improvement from A+B+C (0.7714) to SOTA all-22-features (0.7753) is +0.0039 "
        "— well within the std. Groups D and E are null contributions. "
        "At AUROC 0.77, the governance policy is necessarily conservative: 90% of test queries are "
        "sent to REGENERATE. The system cannot reliably accept more without increasing hallucination risk."
    )
    st.markdown("---")
    st.subheader("Experiment 04 — Establishing the Null (Token Signals Only)")
    exp04 = load_json("04_estimator_v1.json")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        Experiment 04 runs a logistic regression on **6 token features** only:
        entropy, max_entropy, logit_var, conf_gap, top1_prob, first_entropy.

        - CV AUROC: **{exp04.get('cv_auroc_mean', 0.542):.4f}** (±{exp04.get('cv_auroc_std', 0.066):.4f})
        - Test AUROC: **{exp04.get('test_auroc', 0.533):.4f}**
        - 95% CI: [{exp04.get('test_auroc_ci', {}).get('lo_95', 0.41):.3f}, {exp04.get('test_auroc_ci', {}).get('hi_95', 0.66):.3f}]
        """)
    with col2:
        fi = exp04.get("feature_importances", [])
        if fi:
            st.markdown("**Feature Importances (by absolute LR coefficient):**")
            for f in fi[:4]:
                coef = f.get("coefficient", 0)
                sign = "↓ fail" if coef < 0 else "↑ correct"
                st.markdown(
                    f'<div class="card" style="padding:0.4rem 0.8rem;">'
                    f'<b>{f["feature"]}</b>: {coef:.4f} ({sign})</div>',
                    unsafe_allow_html=True,
                )


def page_steering():
    st.title("Activation Steering — Can We Steer Toward Truth?")
    exp_pill("Experiment 10 — Compute Steering Vector")
    exp_pill("Experiment 11 — Steering Experiment")
    phase("Phase 6 — Activation Steering")

    st.markdown("""
    **Scientific question:** Can injecting the "truth direction" vector into the model's residual stream
    during generation reduce hallucination? And what is the optimal steering coefficient α?
    """)

    exp10 = load_json("10_steering_vector_meta.json")
    exp11 = load_json("11_steering_results.json")

    sv_norm   = exp10.get("norm", 27.47)
    n_correct = exp10.get("n_correct", 198)
    n_hall    = exp10.get("n_hallucinated", 102)
    opt_alpha = exp11.get("optimal_alpha", 0.0)
    finding_txt = exp11.get("finding", "")
    hall_rates = exp11.get("hallucination_rates", {})

    c1, c2, c3, c4 = st.columns(4)
    with c1: stat(f"{sv_norm:.2f}", "Steering Vector Norm")
    with c2: stat(str(exp10.get("best_layer", 32)), "Injection Layer")
    with c3: stat(str(opt_alpha), "Optimal α (Steering Coeff.)")
    with c4: stat("36%", "Correction Rate at α=0.0 (Resampling)")

    st.markdown("---")
    col1, col2 = st.columns([3, 2])
    with col1:
        plot_img("11_steering_curve.png", "Hallucination rate vs. steering coefficient α")
    with col2:
        st.subheader("Computing the Steering Vector")
        st.markdown("""
        The steering vector is computed by the **difference-of-means** method (RepE, Zou et al. 2023):
        """)
        math(r"\vec{v}^* = \frac{\mu_{\text{correct}} - \mu_{\text{hallucinated}}}{||\mu_{\text{correct}} - \mu_{\text{hallucinated}}||}")
        st.markdown(f"""
        Using:
        - Correct samples: **{n_correct}** mean hidden states at Layer {exp10.get("best_layer", 32)}
        - Hallucinated samples: **{n_hall}** mean hidden states at Layer {exp10.get("best_layer", 32)}
        - Hidden dimension: **{exp10.get("hidden_dim", 2048)}**

        The vector is **unit-normalized** so that the same α value corresponds to the same injection
        magnitude regardless of the scale of the class means.

        **Injection during generation:**
        """)
        math(r"h_{\text{steered}} = h_{\text{original}} + \alpha \cdot \vec{v}^*")
        st.markdown("""
        Applied at Layer 32 (best_layer) during each forward pass of autoregressive generation.
        """)

        null(
            f"<b>Finding: α=0 is optimal.</b> No steering coefficient improves over the baseline "
            f"(temperature resampling at α=0). Hallucination rate at α=0: "
            f"{hall_rates.get('0.0', 0.64):.0%}; at α=2.0: "
            f"{hall_rates.get('2.0', 0.67):.0%}. Steering slightly worsens performance."
        )

    st.markdown("---")
    st.subheader("Why Does the Universal Steering Vector Fail?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        The HaRP dataset contains **4 types of failure**:
        1. **Epistemic hallucination:** The model genuinely lacks the knowledge. Steering toward
           μ_correct does not add missing information — it only changes confidence.
        2. **Confident hallucination:** The model has the wrong internal representation.
           Steering toward μ_correct *might* help — but the universal vector averages across all types.
        3. **Aleatoric uncertainty:** Multiple valid answers exist. Steering is inappropriate.
        4. **Confident correct:** These should not be steered at all.

        The universal vector averages across all four quadrants. The effects cancel out:
        helping confident hallucination samples, hurting epistemic samples.
        """)
    with col2:
        finding(
            "<b>Why this result is scientifically valuable:</b> The null result for universal steering "
            "directly motivates the 4-quadrant decomposition (Experiment 18) and quadrant-adaptive steering "
            "(Experiment 22). The correct design is: compute a <em>separate</em> steering vector for each "
            "failure mode and apply it only to samples classified in that quadrant."
        )
        st.markdown("""
        **The REGENERATE action in HaRP policy:**
        Even though directed steering doesn't help, the REGENERATE action still recovers 36% of
        hallucinated samples via temperature=0.9 resampling (stochastic regeneration without steering).
        This justifies keeping the 3-action structure — REGENERATE expands useful output coverage
        beyond what a binary Accept/Abstain system achieves.
        """)

    st.markdown("---")
    st.subheader("Steering Null Results — Summary Table")
    steering_rows = [
        ["Exp 11 — Universal vector, α=0.0", "Baseline (temp resample)", "0.67", "Null"],
        ["Exp 11 — Universal vector, α=0.5", "Positive steering", "0.69", "Worse"],
        ["Exp 11 — Universal vector, α=1.0", "Positive steering", "0.68", "Worse"],
        ["Exp 11 — Universal vector, α=1.5", "Positive steering", "0.72", "Worse"],
        ["Exp 11 — Universal vector, α=2.0", "Positive steering", "0.74", "Worse"],
        ["Exp 22c — Confident hall., α=−1.0", "Negative α (hypothesis)", "0.796", "Worse than α=0"],
        ["Exp 22c — Epistemic, α=1.5",        "Quadrant-specific",      "0.724", "≈ noise (n≈80)"],
    ]
    df_st = pd.DataFrame(steering_rows, columns=["Experiment", "Condition", "Hall. Rate", "Verdict"])
    st.dataframe(df_st, hide_index=True, use_container_width=True)
    null("<b>Definitive conclusion:</b> Multiple steering conditions tested (Exp 11, 22, 22c), all showing no improvement or worsening. "
         "Pushing the model's internal state toward 'correct answer territory' at Layer 32 does not reduce hallucination rate — "
         "for any failure mode, any direction, or any magnitude tested.")
    limit(
        "Each steering condition was evaluated on n=100 hallucinated samples (universal) or n≈57–80 per quadrant. "
        "At these sample sizes, the 95% confidence interval on hallucination rate is ±~10pp — "
        "the ~5pp improvement seen in Epistemic samples at α=1.5 is not statistically distinguishable from noise. "
        "The negative-α result for confident hallucination (0.796 vs 0.803 baseline) is a rigorous null at n=57."
    )


def page_policy():
    st.title("Authorization Policy — The 3-Action Governance Framework")
    exp_pill("Experiment 24 — Honest Evaluation")
    honest_pill()
    phase("Phase 4 — Closed-Loop Governance")

    st.markdown("""
    The authorization policy translates P(failure) into a routing action. These are the
    **Exp 24 OOF-corrected results** — the honest test-set performance on n=140 held-out samples.
    """)

    exp24 = load_json("24_honest_results.json")
    pm    = exp24.get("test_policy_metrics", {})
    a_star = exp24.get("alpha_star", 0.15)
    b_star = exp24.get("beta_star", 0.90)
    tm     = pm

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: stat(f"{pm.get('n_accept', 6)}", "ACCEPT (n=6)", "stat-number-green")
    with c2: stat(f"{pm.get('n_regenerate', 126)}", "REGENERATE (n=126)", "stat-number-indigo")
    with c3: stat(f"{pm.get('n_abstain', 8)}", "ABSTAIN (n=8)")
    with c4: stat(f"{pm.get('hall_rate_accepted', 0.1667)*100:.1f}%", "Hall. Rate (Accepted)", "stat-number-amber")
    with c5: stat(f"{pm.get('regen_rate', 0.90)*100:.0f}%", "Regeneration Rate")

    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("The Three-Action Policy")
        st.markdown("""
        For each test sample, the estimator produces P(failure) ∈ [0,1]:
        """)
        st.markdown(f"""
        <div class="quadrant-box q-correct">
        <b>P(failure) < α* = {a_star}</b> → <b>ACCEPT</b><br>
        The estimator is highly confident this is a correct answer. Serve it directly.
        </div>
        <div class="quadrant-box q-epistemic">
        <b>α* ≤ P(failure) < β* = {b_star}</b> → <b>REGENERATE</b><br>
        The estimator is uncertain. Apply activation steering (or temperature resampling) and retry.
        This is the "middle tier" — worth a second attempt before refusing.
        </div>
        <div class="quadrant-box q-confident">
        <b>P(failure) ≥ β* = {b_star}</b> → <b>ABSTAIN</b><br>
        The estimator is highly confident this is a hallucination. Respond: "I cannot reliably answer this."
        </div>
        """, unsafe_allow_html=True)
        finding(
            "<b>Why three actions instead of two (Accept/Reject)?</b><br>"
            "Binary systems discard all high-uncertainty samples. REGENERATE captures the middle tier — "
            "samples where the model might succeed with a different random draw. This expands the "
            "system's useful output rate beyond what a binary threshold achieves."
        )
    with col2:
        plot_img("12_policy_decisions.png", f"Decision distribution (α*={a_star}, β*={b_star})")
        st.subheader("Threshold Optimization")
        st.markdown("""
        The optimal thresholds are found by **grid search on the validation set**:
        """)
        math(r"\mathcal{L}(\alpha^*, \beta^*) = \text{hall\_rate}(\text{ACCEPT set}) + \lambda \cdot \text{abstain\_rate}")
        exp12 = load_json("12_policy_results.json")
        st.markdown(f"""
        Grid: α ∈ [0.1, 0.6] step 0.05, β ∈ [0.4, 0.9] step 0.05, constraint α < β.
        λ = {exp12.get('lambda', 0.5)} — moderate penalty on over-abstention.

        **Why this loss function?**
        - The first term is the primary objective: minimize hallucination rate on served outputs.
        - The second term prevents the trivial solution of abstaining on everything (100% abstention = 0% hallucination, useless system).
        - λ = 0.5 means "abstaining on one sample costs half as much as serving a hallucinated answer."
        """)
        warn(
            f"<b>Validation/test isolation:</b> α*={a_star} and β*={b_star} were chosen on the "
            f"validation set (N=140) via grid search, applied to a HELD-OUT test set (N=140) "
            f"never seen during optimization. Train/val/test split: 420/140/140 (60/20/20)."
        )
        limit(
            "With only 6 accepted samples on the test set, the hall rate CI is [0.004, 0.641] — "
            "too wide to draw strong conclusions. The system's low accept rate (4.3%) reflects "
            "the estimator's conservatism at AUROC 0.77. A higher-quality signal (AUROC > 0.90) "
            "would be needed to expand the ACCEPT set without increasing hallucination risk."
        )


def page_baselines():
    st.title("Baselines — What Does Each Design Decision Actually Contribute?")
    exp_pill("Experiment 13 — Phase 7")
    phase("Phase 7 — Controlled Comparison")

    st.markdown("""
    Each baseline isolates one variable. Together they prove which components are necessary
    and which are additive. All baselines use the **same held-out test split** as Experiment 12.
    """)

    exp13 = load_json("13_baselines.json")
    results = exp13.get("results", [])

    if results:
        rows = []
        for r in results:
            if "error" not in r:
                rows.append({
                    "Baseline": r.get("name", "?"),
                    "AUROC": f"{r['auroc']:.4f}" if isinstance(r.get("auroc"), float) and not (isinstance(r.get("auroc"), float) and r.get("auroc") != r.get("auroc")) else "n/a",
                    "Hall Rate": f"{r['hall_rate']*100:.1f}%" if isinstance(r.get("hall_rate"), float) else "n/a",
                    "Abstain Rate": f"{r.get('abstain_rate', 0)*100:.1f}%",
                    "N Accepted": r.get("n_accept", "?"),
                    "Description": r.get("description", ""),
                })
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("What Each Baseline Isolates")

    baselines = [
        ("1. Raw LLM (Accept All)", "stat-number-red",
         "The ceiling hallucination rate — the problem we are solving. 39.5%. No features, no threshold, no policy.",
         "If HaRP does not improve on this, the entire system adds no value."),
        ("2. Static Threshold (top1_prob > 0.5)", "stat-number-red",
         "Tests whether a single fixed rule is sufficient. No training. Isolates: does any feature need to be *learned*?",
         "Static threshold cannot adapt to the data distribution. 0 abstentions → same hall rate as Raw LLM."),
        ("3. Temperature Scaling", "stat-number-red",
         "Platt calibration on the Group A logit. Tests whether calibration alone (without representation signals) explains the improvement.",
         "T=0.1 (near-identity scaling) → all samples assigned high calibrated probability → 0 abstentions. Calibration alone insufficient."),
        ("4. Entropy-Only Policy", "",
         "Group A signals only, with the same 3-action policy structure. Tests: does the policy structure itself add value beyond a static threshold?",
         "Hall rate drops to 28.6% — policy structure helps. But representation signals (the A→A+B+C jump) do most of the work."),
        ("5. HaRP Full", "stat-number-green",
         "All 14 features + optimized thresholds. The full system with all design decisions combined.",
         "Hall rate 0.0% on accepted outputs. Abstention 20.9%."),
    ]

    for name, color, what, proves in baselines:
        st.markdown(
            f'<div class="card"><b style="color:var(--teal);">{name}</b><br>'
            f'<span style="font-size:0.87rem;">{what}</span><br>'
            f'<span style="font-size:0.82rem;color:#6b7280;"><b>What this proves:</b> {proves}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Critical Comparison Pairs")
    col1, col2, col3 = st.columns(3)
    with col1:
        finding("<b>(4) vs (5):</b> Representation signals contribution — holding policy structure constant. The delta between entropy-only and full HaRP is entirely due to Group C features.")
    with col2:
        finding("<b>(3) vs (5):</b> Policy + representations beyond calibration alone. Calibration (3) cannot abstain because top1_prob is always high for confident hallucination. HaRP (5) catches it.")
    with col3:
        finding("<b>(2) vs (4):</b> Policy structure contribution — holding signal type constant. The 3-action structure reduces hall rate even with only Group A signals.")


def page_sycophancy():
    st.title("Sycophancy Robustness — Does Framing Fool the Estimator?")
    exp_pill("Experiment 14 — Phase 8")
    phase("Phase 8 — Robustness Analysis")

    st.markdown("""
    **Scientific question:** When the prompt *implies* a wrong answer ("Surely the capital of Australia is Sydney?"),
    does the LLM's token uncertainty (entropy, top1_prob) drop — making the estimator falsely confident?
    If yes, Group A signals alone are *not robust to adversarial framing*.
    """)

    exp14 = load_json("14_sycophancy_results.json")
    sigs = exp14.get("signals", {})
    n_syco = exp14.get("n_sycophantic", 21)
    n_corr = exp14.get("n_corrective", 9)

    col1, col2, col3 = st.columns(3)
    with col1: stat(str(exp14.get("n_sycophancy_prompts", 30)), "Total Sycophancy Prompts")
    with col2: stat(str(n_syco), "Sycophantic Responses")
    with col3: stat(str(n_corr), "Corrective Responses")

    st.markdown("---")
    col1, col2 = st.columns([3, 2])
    with col1:
        plot_img("14_sycophancy_comparison.png", "Entropy distributions: sycophantic vs corrective prompts")
    with col2:
        st.subheader("The Sycophancy Setup")
        st.markdown("""
        Two prompt types are created for each factual question:

        **Sycophantic prompt:** Implies a wrong answer — "I've always thought that [wrong answer]. Am I right?"
        A sycophantic model agrees with the framing even if it knows the correct answer.

        **Corrective prompt:** Implies a wrong answer but explicitly asks for correction — "Someone told me [wrong answer]. Can you fact-check this?"

        If token signals (entropy, top1_prob) differ significantly between the two types, framing
        is altering the model's surface-level uncertainty — a robustness failure.
        """)

        if "entropy" in sigs:
            e = sigs["entropy"]
            p_val = e.get("p_value", 0.65)
            if p_val > 0.05:
                good(
                    f"<b>No significant entropy difference</b> between sycophantic and corrective prompts "
                    f"(p = {p_val:.3f}). Token signals are not systematically fooled by sycophantic framing "
                    f"in this dataset — they are equally uninformative in both contexts."
                )
            else:
                warn(f"<b>Significant entropy difference</b> detected (p = {p_val:.3f}). Sycophantic framing alters token uncertainty.")

            st.markdown(f"""
            Signal comparison (sycophantic vs corrective):
            - **Entropy:** {e.get('mean_sycophantic', 0):.4f} vs {e.get('mean_corrective', 0):.4f} (p={e.get('p_value', 0):.3f})
            """)
            if "top1_prob" in sigs:
                t = sigs["top1_prob"]
                st.markdown(f"- **top1_prob:** {t.get('mean_sycophantic', 0):.4f} vs {t.get('mean_corrective', 0):.4f} (p={t.get('p_value', 0):.3f})")
            if "consistency_score" in sigs:
                c = sigs["consistency_score"]
                st.markdown(f"- **consistency_score:** {c.get('mean_sycophantic', 0):.4f} vs {c.get('mean_corrective', 0):.4f} (p={c.get('p_value', 0):.3f})")

        finding(
            "<b>Key implication:</b> The sycophancy experiment confirms that token signals (Group A) are "
            "not systematically biased by prompt framing — but they are also equally uninformative in "
            "both contexts. Representation signals (Group C) are harder to fool because they measure "
            "the model's <em>internal state</em>, not just the output distribution."
        )


def page_calibration():
    st.title("Calibration — Are HaRP's Probabilities Trustworthy?")
    exp_pill("Experiment 15 — Phase 9")
    phase("Phase 9 — Probability Calibration")

    st.markdown("""
    **Scientific question:** When HaRP says P(correct) = 0.8, is the model actually correct 80% of the time?
    A well-calibrated failure estimator is **required** for the authorization policy to work as designed.
    Miscalibrated probabilities mean α*, β* do not correspond to actual error rates.
    """)

    exp15 = load_json("15_calibration_results.json")
    sys = exp15.get("systems", {})
    ece_raw  = sys.get("raw_llm", {}).get("ece", 0.0715)
    ece_temp = sys.get("temp_scaling", {}).get("ece", 0.395)
    ece_gov  = sys.get("harp_full", {}).get("ece", 0.0386)
    t_star   = sys.get("temp_scaling", {}).get("T_star", 0.1)

    c1, c2, c3 = st.columns(3)
    with c1: stat(f"{ece_raw:.4f}", "ECE — Raw LLM (top1_prob)")
    with c2: stat(f"{ece_temp:.4f}", "ECE — Temperature Scaling", "stat-number-red")
    with c3: stat(f"{ece_gov:.4f}", "ECE — HaRP (full)", "stat-number-green")

    st.markdown("---")
    col1, col2 = st.columns([3, 2])
    with col1:
        plot_img("15_reliability_diagram.png", "Reliability diagrams — diagonal = perfect calibration")
    with col2:
        st.subheader("Expected Calibration Error (ECE)")
        st.markdown("""
        ECE measures the average gap between predicted confidence and actual accuracy,
        weighted by bin population:
        """)
        math(r"\text{ECE} = \sum_{b=1}^{M} \frac{|B_b|}{N} \cdot |\text{acc}(B_b) - \text{conf}(B_b)|")
        st.markdown("""
        Samples are sorted into M=10 bins by predicted confidence.
        For each bin: compute fraction correct (accuracy) and mean predicted probability (confidence).
        If accuracy ≈ confidence: ECE ≈ 0 (perfect calibration).

        **Interpretation thresholds:**
        - ECE < 0.05: Well-calibrated — policy thresholds can be trusted
        - ECE 0.05–0.10: Moderately calibrated — thresholds need slight adjustment
        - ECE > 0.10: Poorly calibrated — probabilities are misleading

        **Reliability diagram:**
        - Points on diagonal: perfectly calibrated
        - Points **below** diagonal: overconfident (model claims higher probability than accuracy supports)
        - Points **above** diagonal: underconfident
        """)

        if ece_gov < 0.05:
            good(
                f"<b>HaRP is well-calibrated (ECE = {ece_gov:.4f} < 0.05).</b> "
                f"The multi-signal posterior P(correct | features) accurately reflects actual correctness rates. "
                f"Policy thresholds α* and β* correspond to genuine error rate boundaries."
            )

        warn(
            f"<b>Temperature scaling (T*={t_star:.3f}) performs worse (ECE = {ece_temp:.4f}).</b> "
            f"T*=0.1 is near-identity — calibration finds that the top1_prob signal is already maximally "
            f"compressed, leaving no room for improvement. The fundamental problem is that Group A alone "
            f"cannot discriminate confident hallucination, so no temperature can fix a signal that is "
            f"informationally absent."
        )

    st.markdown("---")
    st.subheader("Why HaRP Has Lower ECE Than Temperature Scaling")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        Temperature scaling only recalibrates the *magnitude* of Group A's logit. It cannot correct
        the fundamental problem: Group A alone is insufficient to distinguish confident hallucination
        from confident correct.

        The full 14-feature estimator produces a posterior P(correct | entropy, consistency, geometry, ...)
        that is informationally richer. The logistic regression learns the decision boundary in the full
        feature space — this boundary naturally corresponds better to actual correctness rates.
        """)
    with col2:
        analogy(
            "Temperature scaling is like adjusting the brightness knob on a blurry photograph. "
            "You can make it brighter or darker, but you cannot make it sharper — the information is not there. "
            "HaRP adds more cameras (signal groups) and sharpens the image from multiple angles."
        )


def page_semantic_entropy():
    st.title("Semantic Entropy — Measuring Meaning-Level Uncertainty")
    novel_pill("NOVEL CONTRIBUTION")
    exp_pill("Experiment 17")
    phase("Novel Extension — Group B2")

    st.markdown("""
    Standard consistency (Group B) measures surface-level cosine similarity between responses.
    Two paraphrases of the same fact score low on cosine consistency despite being *semantically equivalent*.
    Semantic entropy solves this by clustering responses by *meaning*, not by surface form.
    """)

    exp17 = load_json("17_semantic_entropy_results.json")

    col1, col2 = st.columns([3, 2])
    with col1:
        plot_img("17_semantic_entropy_comparison.png", "Semantic entropy: correct vs hallucinated samples")
    with col2:
        st.subheader("The NLI Clustering Method")
        st.markdown("""
        Given 3 responses R₁, R₂, R₃:

        1. **Bidirectional entailment check:** For each pair (Rᵢ, Rⱼ), run NLI model.
           If Rᵢ entails Rⱼ AND Rⱼ entails Rᵢ → same meaning cluster.

        2. **Cluster by meaning:** Group responses into equivalence classes.

        3. **Semantic entropy:** Shannon entropy over cluster sizes.
        """)
        math(r"H_{\text{sem}} = -\sum_{c} p(c) \log p(c)")
        st.markdown("""
        where p(c) = fraction of responses in cluster c.

        **Why this is better than cosine consistency:**
        - "Paris is the capital of France" and "The French capital is Paris" → same cluster (semantic equivalents)
        - "Paris is the capital of France" and "The capital of Germany is Berlin" → different clusters (different meanings)
        """)
        st.markdown("""
        **Model used:** `cross-encoder/nli-deberta-v3-small` — fast, accurate NLI scoring

        **Features produced:**
        - `semantic_entropy`: entropy over meaning clusters
        - `semantic_cluster_ratio`: ratio of unique clusters to total responses (1.0 = all different meanings)
        """)
        finding(
            "<b>Key advantage:</b> Semantic entropy catches hallucination modes where the model "
            "consistently generates the same wrong answer (high cosine consistency, low semantic entropy) "
            "vs. generating multiple different wrong answers (high semantic entropy). "
            "These two failure modes need different responses."
        )

    st.markdown("---")
    st.subheader("Connection to Kuhn et al. (2023) — Semantic Uncertainty")
    st.markdown("""
    **Semantic Uncertainty** (Kuhn et al., ICLR 2023) introduced NLI-based clustering for open-ended
    generation. HaRP extends this by:
    1. Integrating semantic entropy as one of 5 signal groups (not a standalone system)
    2. Combining it with representation geometry — neither alone is sufficient
    3. Computing cluster ratio as a second feature (not just entropy)

    Results from Experiment 17 quantify the marginal AUROC contribution of B2 features
    beyond the base A+B+C estimator.
    """)


def page_quadrants():
    st.title("4-Quadrant Uncertainty Decomposition")
    novel_pill("CORE NOVEL CONTRIBUTION")
    exp_pill("Experiment 18")
    phase("Novel Extension — Epistemics Framework")

    st.markdown("""
    **The central theoretical claim of HaRP:** Hallucination is not a single phenomenon.
    It has four distinct failure modes, each with different observable signal signatures,
    different causes, and different optimal responses. Treating them uniformly misses the structure.
    """)

    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("""
        <div class="quadrant-box q-epistemic">
        <b>Epistemic Uncertainty</b><br>
        <span style="font-size:0.82rem;">High entropy + Low consistency</span><br>
        The model <em>knows it doesn't know</em>. Multiple inconsistent responses.
        <br><b>Action: ABSTAIN</b> — no steering can add missing knowledge.
        </div>

        <div class="quadrant-box q-aleatoric">
        <b>Aleatoric Uncertainty</b><br>
        <span style="font-size:0.82rem;">High entropy + High consistency</span><br>
        The question genuinely has multiple valid answers. Model is uncertain but consistent.
        <br><b>Action: ACCEPT with caveat</b> — uncertainty is irreducible.
        </div>

        <div class="quadrant-box q-correct">
        <b>Confident Correct</b><br>
        <span style="font-size:0.82rem;">Low entropy + Good geometry (near μ_correct)</span><br>
        The model is certain and right. Internal representation aligns with truth.
        <br><b>Action: ACCEPT</b> — straightforward.
        </div>

        <div class="quadrant-box q-confident">
        <b>Confident Hallucination</b><br>
        <span style="font-size:0.82rem;">Low entropy + Bad geometry (near μ_hallucinated)</span><br>
        The model is certain but wrong. Token signals are BLIND to this.
        <br><b>Action: REGENERATE</b> — or ABSTAIN if geometry is extreme.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        plot_img("18_uncertainty_quadrants.png", "4-quadrant decomposition scatter plot")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        plot_img("18_quadrant_hallucination_rates.png", "Hallucination rate per quadrant")
    with col2:
        plot_img("18_group_c_necessity.png", "Group C necessity: detection rate per quadrant")

    exp18 = load_json("18_uncertainty_decomposition_results.json")
    st.markdown("---")
    st.subheader("Why This Framework Is Novel")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Prior work characterizes hallucination as a single class** — you either hallucinate or you don't.
        HaRP introduces a **2×2 taxonomy** crossing:
        - **Epistemic axis:** Is the model uncertain? (entropy, consistency signals)
        - **Representation axis:** Is the internal state near the correct cluster? (geometry signals)

        This taxonomy has direct implications for *what to do*:
        - Epistemic failures → more data needed, ABSTAIN
        - Confident hallucination → wrong internal representation, REGENERATE with targeted steering
        - Aleatoric → accept with uncertainty disclaimer
        - Confident correct → accept confidently
        """)
    with col2:
        finding(
            "<b>The smoking gun argument:</b> In the Confident Hallucination quadrant, entropy is LOW. "
            "Any token-only system (including temperature-scaled LLMs and static thresholds) will ACCEPT "
            "these samples — because high confidence looks like a safe answer. Only Group C (geometry) "
            "can detect this quadrant. This is the formal proof that representation signals are necessary, "
            "not merely additive."
        )
        analogy(
            "A doctor who only checks pulse (token signals) cannot detect a patient who feels fine but "
            "has a hidden pathology (confident hallucination). You need an MRI (representation geometry) "
            "to see what's happening internally."
        )

    limit(
        "The quadrant taxonomy uses data-driven median thresholds (entropy=0.3704, consistency=0.8575 "
        "from the training distribution) — not formal Bayesian priors. The 'epistemic' and 'aleatoric' "
        "labels are operational approximations, not rigorous uncertainty decompositions. "
        "Quadrant counts and hall rates are computed on the full 700-sample set including training data — "
        "treat these as descriptive statistics, not held-out evaluations."
    )

    exp18 = load_json("18_uncertainty_decomposition_results.json")
    qcounts = exp18.get("quadrant_counts", {})
    qhall   = exp18.get("quadrant_hall_rates", {})
    if qcounts:
        st.markdown("---")
        st.subheader("Quadrant Statistics (n=700, descriptive — includes training data)")
        q_rows = [
            ["Epistemic",             qcounts.get("epistemic",182),            f"{qhall.get('epistemic',0.283)*100:.1f}%",          "High entropy, low consistency",        "Token signals partially see this"],
            ["Aleatoric",             qcounts.get("aleatoric",168),            f"{qhall.get('aleatoric',0.403)*100:.1f}%",           "High entropy, high consistency",       "Inherent uncertainty — irreducible"],
            ["Confident Correct",     qcounts.get("confident_correct",343),    f"{qhall.get('confident_correct',0.271)*100:.1f}%",   "Low entropy, near μ_correct",          "Easiest to detect (low hall rate)"],
            ["Confident Hallucination",qcounts.get("confident_hallucination",355),f"{qhall.get('confident_hallucination',0.611)*100:.1f}%","Low entropy, near μ_hallucinated","Token-blind; only Group C detects this"],
        ]
        df_q = pd.DataFrame(q_rows, columns=["Quadrant","N","Hall Rate","Signature","Key Property"])
        st.dataframe(df_q, hide_index=True, use_container_width=True)
        finding("Confident Hallucination has the highest hall rate (61%) and the lowest entropy — "
                "confirming that token signals are blind to the most dangerous failure mode. "
                "Group C (hidden-state cosine distance) is the only signal that detects it reliably.")
    if exp18:
        quadrants = exp18.get("quadrants", {})
        if quadrants:
            st.markdown("**Quadrant Population and Hallucination Rates:**")
            rows = []
            for qname, qdata in quadrants.items():
                rows.append({
                    "Quadrant": qname,
                    "N Samples": qdata.get("n", "?"),
                    "Hall Rate": f"{qdata.get('hall_rate', 0)*100:.1f}%" if isinstance(qdata.get("hall_rate"), (int, float)) else "n/a",
                    "Group C Detectable": "YES" if qdata.get("group_c_detects", False) else "No",
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def page_cross_model():
    st.title("Cross-Model Validation — Does the Framework Generalize?")
    novel_pill("NOVEL CONTRIBUTION")
    exp_pill("Experiment 19")
    phase("Novel Extension — Generalization Test")

    st.markdown("""
    **Scientific question:** HaRP was trained and evaluated on Qwen 2.5 3B. Does the framework
    generalize to a different architecture and scale — Llama 3.2 1B?

    If yes, the signal groups capture model-agnostic properties of language model uncertainty.
    If no, the features are Qwen-specific and the approach does not generalize.
    """)

    exp19 = load_json("19_cross_model_results.json")

    col1, col2 = st.columns([3, 2])
    with col1:
        plot_img("19_cross_model_comparison.png", "Cross-model AUROC comparison: Qwen 3B vs Llama 1B")
    with col2:
        st.subheader("Experimental Design")
        st.markdown("""
        **Model:** Llama 3.2 1B (via Ollama — `llama3.2:1b`)

        **Same data:** The identical TriviaQA-subset questions used for Qwen.

        **Same feature extraction:** Groups A and B computed via Ollama logprobs.
        (Group C requires loading Llama in HuggingFace — run separately.)

        **What's being tested:**
        - Do entropy/consistency signals from Llama correlate with hallucination the same way?
        - Does the AUROC ordering hold? (Group A < Group A+B < Group A+B+C)
        - Is the geometry separable at Llama's best_layer?
        """)

        qwen_auroc = exp19.get("qwen_auroc", None)
        llama_auroc = exp19.get("llama_auroc", None)
        if qwen_auroc and llama_auroc:
            c1, c2 = st.columns(2)
            with c1: stat(f"{qwen_auroc:.4f}", "Qwen 2.5 3B AUROC", "stat-number-green")
            with c2: stat(f"{llama_auroc:.4f}", "Llama 3.2 1B AUROC", "stat-number-indigo")
        finding(
            "<b>Why generalization matters:</b> A framework that only works on Qwen 3B is an empirical "
            "observation, not a scientific claim. Cross-model validation elevates HaRP from "
            "'this works on one model' to 'uncertainty signals are model-agnostic properties of "
            "language model computation at this scale.'"
        )


def page_trajectory():
    st.title("Trajectory Signals — Cross-Layer Dynamics (Group D)")
    novel_pill("NOVEL CONTRIBUTION")
    exp_pill("Experiment 20")
    phase("Novel Extension — Group D (ICR Probe)")

    st.markdown("""
    **Scientific question:** Does tracking the evolution of the hidden state *across layers* add
    discriminative power beyond single-layer geometry? The logit lens experiment (Exp 08) showed that
    hallucinations commit late — Group D operationalizes this as quantitative features.
    """)

    exp20 = load_json("20_cross_layer_dynamics.json")

    col1, col2 = st.columns([3, 2])
    with col1:
        plot_img("20_trajectory_comparison.png", "Hidden state trajectory: correct vs hallucinated samples")
        plot_img("20_icr_curve.png", "ICR score distribution by label")
    with col2:
        st.subheader("The Trajectory Features")
        features = [
            ("traj_total_variation", "Sum of ||h_l - h_{l-1}||₂ across all layers. High TV = unstable representation. Hallucinations oscillate more."),
            ("traj_stabilization_layer", "First layer where consecutive change < threshold. Correct answers stabilize earlier."),
            ("traj_max_change_layer", "Layer with the largest single-step change. Hallucinations often show a large late jump."),
            ("traj_early_commitment", "Binary: does the top-1 token lock in before layer L/2? Correct answers do; hallucinations often don't."),
            ("icr_score", "Inter-layer Consistency Ratio from Zhang et al. (ACL 2025). Cosine similarity between consecutive layer predictions, averaged."),
        ]
        for fname, desc in features:
            st.markdown(
                f'<div class="card" style="padding:0.5rem 0.8rem;margin-bottom:0.4rem;">'
                f'<code style="color:var(--teal);font-size:0.82rem;">{fname}</code><br>'
                f'<span style="font-size:0.82rem;">{desc}</span></div>',
                unsafe_allow_html=True,
            )

        finding(
            "<b>Why trajectory signals complement geometry (Group C):</b> Group C measures WHERE the "
            "hidden state is (position relative to class means). Group D measures HOW it got there "
            "(the path through representation space). A sample that ends near μ_correct but via an "
            "unstable path may still be unreliable — the journey reveals what the geometry snapshot misses."
        )

    st.markdown("---")
    st.subheader("The ICR Probe — Literature Connection")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Zhang et al. (ACL 2025) — ICR Probe:**
        The Inter-layer Consistency Ratio measures how much the model's intermediate prediction
        changes as processing proceeds through layers. High consistency → stable commitment.
        Low consistency → the model is "changing its mind" at each layer.
        """)
        math(r"\text{ICR} = \frac{1}{L-1} \sum_{l=1}^{L-1} \cos(p_l, p_{l+1})")
        st.markdown("where p_l is the probability distribution from the logit lens at layer l.")
    with col2:
        analogy(
            "Analogy: A student who arrives at the correct answer quickly and confidently (stable trajectory) "
            "is more reliable than one who keeps second-guessing themselves until the last moment "
            "(oscillating trajectory) — even if both write the same final answer."
        )

    if exp20:
        auroc_d = exp20.get("auroc_group_d", None)
        if auroc_d:
            st.markdown(f"**Group D AUROC contribution (standalone):** {auroc_d:.4f}")
            st.markdown(f"**Group A+B+C+D AUROC:** {exp20.get('auroc_abcd', 'n/a')}")


def page_geometric_volume():
    st.title("Geometric Volume — Simplex Uncertainty (Group E)")
    novel_pill("NOVEL CONTRIBUTION")
    exp_pill("Experiment 21")
    phase("Novel Extension — Group E (Granese et al.)")

    st.markdown("""
    **Scientific question:** Can the geometric volume spanned by multiple response representations
    bound and estimate the uncertainty of the response distribution — more accurately than entropy alone?
    """)

    exp21 = load_json("21_geometric_volume_results.json")

    col1, col2 = st.columns([3, 2])
    with col1:
        plot_img("21_geometric_comparison.png", "Simplex volume: correct vs hallucinated samples")
    with col2:
        st.subheader("The Simplex Volume Method")
        st.markdown("""
        Generate 3 responses to the same prompt. Extract their hidden states h₁, h₂, h₃ at `best_layer`.
        The 3 points define a **simplex** in ℝ^2048. Its volume measures how spread out the responses
        are in representation space:
        """)
        math(r"\text{Vol}(\Delta) = \sqrt{\det(\mathbf{G})}")
        st.markdown("""
        where **G** is the Gram matrix of centered hidden states:
        """)
        math(r"G_{ij} = (h_i - \bar{h})^\top (h_j - \bar{h})")
        st.markdown("""
        **Large volume** → responses are spread in representation space → high uncertainty.
        **Small volume** → responses cluster together → either confident correct or confident hallucination.

        Group E features:
        - `simplex_volume`: √det(G) — the fundamental geometric uncertainty measure
        - `embedding_spread`: mean pairwise distance between response embeddings
        - `centroid_drift`: how far the centroid of regenerated responses moves from the original
        """)

        finding(
            "<b>Why this is different from Group B (cosine consistency):</b> Cosine consistency measures "
            "surface similarity in a 768D sentence embedding space. Simplex volume measures geometric "
            "spread in the model's <em>own</em> 2048D representation space at the decision-critical layer. "
            "It directly operationalizes Granese et al.'s (ICLR 2026) theoretical bound on entropy."
        )

    st.markdown("---")
    st.subheader("Granese et al. (2025) — Theoretical Basis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Granese et al. (arXiv 2509.13813, ICLR 2026)** proved that under mild assumptions,
        the geometric volume of the response distribution's convex hull provides an upper bound
        on the Shannon entropy of that distribution:
        """)
        math(r"H(Y | X) \leq f(\text{Vol}(\text{Conv}(h_1, \ldots, h_K)))")
        st.markdown("""
        HaRP operationalizes this with K=3 (3 regenerations) — the minimum for a non-degenerate
        simplex. The volume is computed via the Gram matrix determinant, which is numerically stable
        and avoids explicitly constructing the full 2048D simplex.
        """)
    with col2:
        analogy(
            "Analogy: If you ask three witnesses to describe the same event and they give wildly "
            "different stories (large volume in 'story space'), you know the event is unclear or "
            "contested. If they all describe the same scene in slightly different words (small volume), "
            "you can trust the account — whether it's accurate is a separate question."
        )
        warn(
            "Group E requires 3 HuggingFace forward passes per sample — 3× the compute of Group C alone. "
            "In production, this would be run only for REGENERATE-tier samples (those in the α*–β* range)."
        )

    if exp21:
        auroc_e = exp21.get("auroc_group_e", None)
        if auroc_e:
            st.markdown(f"**Group E AUROC (standalone):** {auroc_e:.4f}")


def page_leakage():
    st.title("The Measurement Correction — From 0.962 to 0.771")
    exp_pill("Experiment 24 — Correction")
    phase("Phase 4 — Data Leakage Identification & Fix")

    st.markdown("""
    Halfway through the project, a critical data-leakage bug was discovered in Experiment 09.
    This page tells that story honestly — what went wrong, how it was fixed, and what the
    corrected numbers actually are.
    """)

    exp09b = load_json("09b_oof_group_c.json")
    exp24  = load_json("24_honest_results.json")

    c1, c2, c3, c4 = st.columns(4)
    _insample = exp09b.get("probe_conf_auroc_insample", 1.0) if exp09b else 1.0
    _oof      = exp09b.get("probe_conf_auroc_oof", 0.7573) if exp09b else 0.7573
    _leaked   = 0.9620  # Exp 23 (archived, no JSON loaded)
    _honest   = exp24.get("cv_auroc", 0.7714) if exp24 else 0.7714
    with c1: stat(f"{_insample:.4f}", "Detector score: in-sample (inflated)", "stat-number-red")
    with c2: stat(f"{_oof:.4f}",      "Detector score: held-out (honest)", "stat-number-green")
    with c3: stat(f"{_leaked:.4f}",   "Full system AUROC: Exp 23 (leaked)", "stat-number-red")
    with c4: stat(f"{_honest:.4f}",   "Full system AUROC: Exp 24 (honest)", "stat-number-green")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("What Went Wrong")
        null("""
        <b>The Bug (Exp 09):</b> <code>probe_conf</code> was computed by fitting a probe
        on ALL 700 samples, then running it back over those SAME 700 samples to produce a feature.
        In-sample, this probe achieves AUROC = 1.0000 — it perfectly memorises the labels it was
        trained on. When this feature entered the 5-fold CV failure estimator, the outer CV
        did not correct for the inner leakage. Result: +0.24 AUROC inflation on probe_conf alone,
        and +0.19 inflation on the full estimator.
        """)
        st.markdown("""
        **Leaked result (Exp 23):**
        ```
        Failure estimator AUROC:  0.9620
        Test AUROC:               0.9521
        Hall rate (ACCEPT):       0.0000   ← impossibly clean
        Accept rate:              31.7%
        ```
        """)
    with col2:
        st.subheader("The Fix (Exp 24 OOF)")
        good("""
        <b>Out-of-Fold (OOF) computation:</b> For each of the 5 CV folds, the probe and class
        centroids are fitted on the TRAINING fold only, then applied to the held-out TEST fold.
        This gives true out-of-sample <code>probe_conf</code> — no leakage possible.
        """)
        st.markdown("""
        **Honest result (Exp 24):**
        ```
        Failure estimator AUROC:  0.7714
        Test AUROC:               0.7871
        Hall rate (ACCEPT):       0.1667   (1/6 accepted — CI [0.004, 0.641])
        Accept rate:              4.3%
        ```
        """)

    st.markdown("---")
    st.subheader("Inflation Measurement")
    rows = [
        ("probe_conf (in-sample)", "1.0000", "—", "+0.2427 vs OOF"),
        ("probe_conf (OOF)", "0.7573", "Exp 06 probe CV 0.7583", "Matches — validates fix"),
        ("Full estimator (Exp 23)", "0.9620", "—", "+0.1906 vs honest"),
        ("Full estimator (Exp 24)", "0.7714", "±0.047", "Honest out-of-sample"),
        ("Test AUROC (Exp 23)", "0.9521", "—", "+0.1650 vs honest"),
        ("Test AUROC (Exp 24)", "0.7871", "[0.706, 0.866]", "Bootstrap 95% CI"),
    ]
    df = pd.DataFrame(rows, columns=["Metric", "Value", "CI / std", "Note"])
    st.dataframe(df, hide_index=True, use_container_width=True)

    st.markdown("---")
    finding("""
    <b>Why this story is worth telling:</b> The leakage was caught <em>before</em> publication.
    The zero-hallucination claim was an artifact of in-sample probe memorisation.
    The corrected AUROC (0.77) is a genuine and honest contribution —
    representation geometry does lift AUROC from 0.57 by a real +0.20.
    Publishing the correction is better science than ignoring it.
    """)
    analogy(
        "A student who checks their exam answers using the answer key they memorised is not "
        "demonstrating knowledge — they're demonstrating recall. OOF computation is the exam "
        "invigilator that prevents this: you can only use what you knew before the test."
    )


def page_sota():
    st.title("Honest Evaluation — Exp 24 (Leak-Corrected Final Results)")
    exp_pill("Experiment 24 — Definitive")
    phase("Phase 4 — Final Honest Evaluation")

    st.markdown("""
    Exp 24 is the **single source of truth**. It corrects the data-leakage bug from Exp 23
    by using Out-of-Fold (OOF) computation for all Group C features. Every number here
    is genuine out-of-sample performance.
    """)
    warn("<b>Exp 23 (archived):</b> AUROC 0.9620 — inflated by +0.19 due to in-sample probe_conf leakage. "
         "Do not cite those numbers. The Measurement Correction page has the full explanation.")

    exp24 = load_json("24_honest_results.json")
    abl   = exp24.get("ablation", {})
    pm    = exp24.get("test_policy_metrics", {})

    c1, c2, c3, c4 = st.columns(4)
    with c1: stat(f"{exp24.get('auroc_sota_honest', 0.7753):.4f}", "AUROC — SOTA honest (22 features)", "stat-number-green")
    with c2: stat(f"{exp24.get('test_auarc', 0.7495):.4f}", "AUARC (Risk-Coverage)")
    with c3: stat(f"{pm.get('hall_rate_accepted', 0.1667)*100:.1f}%", "Hall Rate (Accepted)", "stat-number-red")
    with c4: stat("22", "Total Features (Groups A–E)")

    st.markdown("---")
    col1, col2 = st.columns([2, 2])
    with col1:
        plot_img("24_honest_ablation.png", "Exp 24 — Honest ablation vs leaked baseline")

    with col2:
        st.subheader("Honest Ablation Table (5-fold CV, OOF)")
        if abl:
            rows = []
            for k, v in abl.items():
                rows.append({
                    "System": v.get("label", k),
                    "AUROC": f"{v.get('auroc_mean', 0):.4f}",
                    "±std":  f"{v.get('auroc_std', 0):.4f}",
                    "F1":    f"{v.get('f1_mean', 0):.4f}",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        finding("<b>Key finding:</b> Groups D and E add only +0.004 AUROC over A+B+C — well within noise (std ≈ 0.05). "
                "The honest contribution is: representation geometry (Group C) lifts AUROC from 0.57 → 0.77.")

    st.markdown("---")
    st.subheader("Honest Policy Metrics (test set, n=140)")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: stat(f"{pm.get('n_accept', 6)}", "ACCEPT (n)", "stat-number-green")
    with c2: stat(f"{pm.get('n_regenerate', 126)}", "REGENERATE (n)", "stat-number-indigo")
    with c3: stat(f"{pm.get('n_abstain', 8)}", "ABSTAIN (n)")
    with c4: stat(f"{pm.get('accept_rate', 0.043)*100:.1f}%", "Accept Rate")
    with c5: stat(f"{exp24.get('test_auroc', 0.7871):.4f}", "Test AUROC", "stat-number-green")
    null("<b>Caveat:</b> Only 6 samples accepted on test set. Hall rate CI = [0.004, 0.641] (Clopper-Pearson). "
         "Too few accepted samples to draw strong conclusions about acceptance quality — "
         "the estimator is conservative at AUROC 0.78.")

    st.markdown("---")
    st.subheader("The Full Feature Set: Groups A through E")

    groups_sota = {
        "Group A — Token (6)": ["entropy", "max_entropy", "logit_var", "conf_gap", "top1_prob", "first_entropy"],
        "Group B — Consistency (2)": ["consistency_score", "consistency_variance"],
        "Group B2 — Semantic Entropy (2)": ["semantic_entropy", "semantic_cluster_ratio"],
        "Group C — Geometry (4, OOF)": ["cos_correct", "cos_hallucinated", "h_norm", "probe_conf"],
        "Group C2 — Attention (2)": ["attn_entropy_mean", "attn_entropy_std"],
        "Group D — Trajectory (5) [NOVEL]": ["traj_total_variation", "traj_stabilization_layer", "traj_max_change_layer", "traj_early_commitment", "icr_score"],
        "Group E — Volume (3) [NOVEL]": ["simplex_volume", "embedding_spread", "centroid_drift"],
    }

    cols = st.columns(2)
    for i, (gname, gfeats) in enumerate(groups_sota.items()):
        with cols[i % 2]:
            is_novel = "[NOVEL]" in gname
            color    = "#f3e5f5" if is_novel else "#e0f2f1"
            border   = "#6a1b9a" if is_novel else "#00695c"
            chips    = " ".join([f'<span class="feature-chip">{f}</span>' for f in gfeats])
            st.markdown(
                f'<div style="border:2px solid {border};background:{color};border-radius:8px;'
                f'padding:0.7rem 0.9rem;margin-bottom:0.5rem;">'
                f'<b style="color:{border};font-size:0.88rem;">{gname}</b><br>{chips}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.subheader("AUARC — The Right Metric for Selective Classification")
    ci = exp24.get("auroc_ci_95", {})
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Area Under the Risk-Coverage Curve (AUARC)** measures performance across *all possible*
        abstention thresholds simultaneously:
        - **Coverage:** What fraction of samples does the system serve? (1 - abstention rate)
        - **Risk:** What is the hallucination rate on served samples?

        AUARC is more appropriate than a single operating point (α*, β*) because it measures
        the quality of the estimator across all governance policies, not just one.
        """)
    with col2:
        stat(f"{exp24.get('test_auarc', 0.7495):.4f}", "AUARC (honest)")
        finding(
            f"<b>95% Bootstrap CI for Test AUROC:</b> [{ci.get('lo', 0.706):.4f}, {ci.get('hi', 0.866):.4f}]<br>"
            f"Test AUROC = {exp24.get('test_auroc', 0.7871):.4f} is well above the Group A baseline (0.5734). "
            f"The honest system is useful but conservative — it regenerates 90% of queries."
        )


def page_full_results():
    st.title("Full Results — The Complete Picture")
    exp_pill("Experiment 16 — Aggregation")
    phase("Phase 10 — Final Results")

    st.markdown("""
    This page aggregates all experiment results into a single reference table.
    `final_results.json` is the single source of truth for all numbers cited in the paper.
    """)

    exp24 = load_json("24_honest_results.json")
    pm24  = exp24.get("test_policy_metrics", {})
    abl24 = exp24.get("ablation", {})
    exp06 = load_json("06_probing_results.json")
    exp07 = load_json("07_geometry_results.json")
    exp08 = load_json("08_logit_lens_results.json")
    exp15 = load_json("15_calibration_results.json")

    st.subheader("Core Metrics Table (Exp 24 — Honest)")
    rows = [
        ("Raw Hallucination Rate",           "39.53%"),
        ("Samples (total)",                  "700 (393 correct, 307 hallucinated)"),
        ("Best Layer",                       str(exp06.get("best_layer", 32))),
        ("Probing AUROC at Best Layer",      f"{exp06.get('best_auroc', 0.7583):.4f}"),
        ("PC1 ↔ Truth Correlation",          f"{exp07.get('pc1_correlation', 0.4241):.4f}"),
        ("Linear Separability",              f"{exp07.get('separability', 0.684):.4f}"),
        ("Mean Commitment Layer (Hall.)",    f"{exp08.get('mean_commit_hall', 4.07):.2f}"),
        ("AUROC — Group A (token only, OOF)","0.5734"),
        ("AUROC — Group A+B (OOF)",         "0.5936"),
        ("AUROC — Group A+B+C (OOF)",       f"{exp24.get('auroc_group_abc_honest', 0.7714):.4f}"),
        ("AUROC Delta (A → A+B+C)",          f"+{exp24.get('delta_a_to_abc', 0.198):.4f}"),
        ("AUROC — SOTA 22-feature (OOF)",    f"{exp24.get('auroc_sota_honest', 0.7753):.4f}"),
        ("Test AUROC",                       f"{exp24.get('test_auroc', 0.7871):.4f}"),
        ("Test AUARC",                       f"{exp24.get('test_auarc', 0.7495):.4f}"),
        ("α* (Accept threshold)",            f"{exp24.get('alpha_star', 0.15)}"),
        ("β* (Abstain threshold)",           f"{exp24.get('beta_star', 0.90)}"),
        ("Accept rate (test)",               f"{pm24.get('accept_rate', 0.043)*100:.1f}%  (6/140)"),
        ("Regen rate (test)",                f"{pm24.get('regen_rate', 0.90)*100:.1f}%  (126/140)"),
        ("Abstain rate (test)",              f"{pm24.get('abstain_rate', 0.057)*100:.1f}%  (8/140)"),
        ("Hall rate (ACCEPT set)",           "16.7%  (1/6) — CI [0.004, 0.641]"),
        ("ECE — Raw LLM",                    f"{exp15.get('systems', {}).get('raw_llm', {}).get('ece', 0.0715):.5f}"),
        ("ECE — HaRP (full)",               f"{exp15.get('systems', {}).get('harp_full', {}).get('ece', 0.0386):.5f}"),
        ("Optimal Steering α",               "0.0  (steering does not help — no improvement found)"),
    ]
    df = pd.DataFrame(rows, columns=["Metric", "Value"])
    st.dataframe(df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("All 19 Required Plots")
    plot_files = [
        ("06_probing_curve.png",             "Exp 06 — Probing AUROC by layer"),
        ("07_pca_scatter.png",               "Exp 07 — Geometry of truth (PCA scatter)"),
        ("08_logit_lens.png",                "Exp 08 — Logit lens commitment layer"),
        ("09_auroc_comparison.png",          "Exp 09 — AUROC ablation (Group A → A+B+C)"),
        ("16_ablation_bar.png",              "Exp 16 — Ablation bar (5-fold CV error bars)"),
        ("11_steering_curve.png",            "Exp 11 — Hallucination rate vs. steering α"),
        ("12_policy_decisions.png",          "Exp 12 — Authorization decision distribution"),
        ("14_sycophancy_comparison.png",     "Exp 14 — Sycophancy entropy comparison"),
        ("15_reliability_diagram.png",       "Exp 15 — Reliability diagram (ECE)"),
        ("17_semantic_entropy_comparison.png","Exp 17 — Semantic entropy comparison"),
        ("18_uncertainty_quadrants.png",     "Exp 18 — 4-quadrant decomposition scatter"),
        ("18_quadrant_hallucination_rates.png","Exp 18 — Hallucination rate per quadrant"),
        ("18_group_c_necessity.png",         "Exp 18 — Group C necessity analysis"),
        ("19_cross_model_comparison.png",    "Exp 19 — Cross-model AUROC comparison"),
        ("20_trajectory_comparison.png",     "Exp 20 — Trajectory signals comparison"),
        ("20_icr_curve.png",                 "Exp 20 — ICR score distribution"),
        ("21_geometric_comparison.png",      "Exp 21 — Geometric volume comparison"),
        ("23_sota_ablation.png",             "Exp 23 — SOTA ablation"),
        ("23_sota_policy.png",               "Exp 23 — SOTA policy decisions"),
    ]
    for fname, label in plot_files:
        exists = (PLOTS_DIR / fname).exists()
        icon   = "✅" if exists else "⬜"
        st.markdown(f"{icon} `{fname}` — {label}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("HaRP vs. MECH-INT (Sister Project)")
        comp = [
            ["Attribute",          "MECH-INT",                  "HaRP"],
            ["Model",              "GPT-2 117M",                "Qwen 2.5 3B"],
            ["Primary goal",       "Understand hallucination",   "Control hallucination"],
            ["Key method",         "DLA + head ablation",        "Multi-signal LR + policy"],
            ["AUROC",              "N/A (interpretability)",     "0.7753 (honest)"],
            ["Steering works?",    "Yes (causal proof)",         "No (universal vector)"],
            ["Output",             "Mechanistic story",          "Authorization system"],
            ["Geometry found?",    "No (117M too small)",        "Yes (3B sufficient)"],
        ]
        df2 = pd.DataFrame(comp[1:], columns=comp[0])
        st.dataframe(df2, hide_index=True, use_container_width=True)
    with col2:
        st.subheader("The Research Arc")
        st.markdown("""
        **MECH-INT → HaRP** form a complete argument:

        1. **MECH-INT** asks: *How does GPT-2 hallucinate?*
           → Finds: Head-level over-retrieval at L8–L9, causal evidence via ablation and steering.
           → Proves: Hallucination has mechanistic structure.

        2. **HaRP** asks: *Can we control hallucination in a production setting?*
           → Uses the insight that internal representations carry hallucination information.
           → Builds: A deployable authorization framework with calibrated probabilities.

        Together they cover the full pipeline: understanding → control.
        """)
        good(
            "This arc is the PhD application argument: MECH-INT demonstrates mechanistic interpretability "
            "skills; HaRP demonstrates the ability to translate mechanistic insights into a deployable system."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════

PAGES = {
    # ── Stakeholder Brief ───────────────────────────────────────────────────────
    "📋 Research Brief":          page_brief,
    # ── Core Pipeline ──────────────────────────────────────────────────────────
    "Overview":                   page_overview,
    "Signal Architecture":        page_signals,
    "Layer Probing":              page_probing,
    "Geometry of Truth":          page_geometry,
    "Logit Lens":                 page_logit_lens,
    "⚠ Measurement Correction":   page_leakage,
    "Failure Estimator":          page_estimator,
    "Activation Steering":        page_steering,
    "Authorization Policy":       page_policy,
    "Baselines":                  page_baselines,
    "Sycophancy Robustness":      page_sycophancy,
    "Calibration (ECE)":          page_calibration,
    # ── Novel Contributions ────────────────────────────────────────────────────
    "Novel: Semantic Entropy":    page_semantic_entropy,
    "Novel: 4-Quadrant Decomp.":  page_quadrants,
    "Novel: Cross-Model Valid.":  page_cross_model,
    "Novel: Trajectory (Group D)":page_trajectory,
    "Novel: Geom. Volume (Group E)":page_geometric_volume,
    # ── Results ────────────────────────────────────────────────────────────────
    "Honest Evaluation (Exp 24)": page_sota,
    "Full Results":               page_full_results,
}

SECTIONS = {
    "Start Here":            list(PAGES.keys())[:1],
    "Core Pipeline":         list(PAGES.keys())[1:13],
    "Novel Contributions":   list(PAGES.keys())[13:18],
    "Results":               list(PAGES.keys())[18:],
}

with st.sidebar:
    st.markdown("""
    <div style="padding:0.6rem 0 0.3rem;">
      <div style="font-size:1.35rem;font-weight:800;letter-spacing:-0.01em;color:#e8e4ff;">
        🛡 HaRP</div>
      <div style="font-size:0.72rem;opacity:0.65;color:#c5bff5;margin-top:0.1rem;">
        Hallucination-aware Representation Probing</div>
      <div style="font-size:0.75rem;color:#c5bff5;margin-top:0.35rem;opacity:0.85;">
        Lakshmi Chakradhar Vijayarao</div>
      <a href="https://github.com/Lakshmi-Chakradhar-Vijayarao/harp" target="_blank"
         style="font-size:0.70rem;color:#a89df5;opacity:0.75;text-decoration:none;display:block;margin-top:0.15rem;">
        ⌥ GitHub Repository ↗</a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(255,255,255,0.07);border-radius:10px;
    padding:0.75rem 0.9rem;margin:0.6rem 0;">
      <div style="font-size:0.70rem;font-weight:700;opacity:0.6;text-transform:uppercase;
      letter-spacing:0.09em;margin-bottom:0.5rem;">Model &amp; Dataset</div>
      <div style="font-size:0.80rem;color:#d4cfff;line-height:1.7;">
        <b>Model:</b> Qwen 2.5 3B<br>
        <b>Arch:</b> 36 layers · 2048 hidden dim<br>
        <b>Dataset:</b> TruthfulQA<br>
        <b>Samples:</b> 700 (393✓ / 307✗)<br>
        <b>Experiments:</b> 28 (Exp 01–24 core + Exp 25–28 validation)
      </div>
    </div>
    <div style="background:rgba(255,255,255,0.07);border-radius:10px;
    padding:0.75rem 0.9rem;margin:0.4rem 0 0.8rem;">
      <div style="font-size:0.70rem;font-weight:700;opacity:0.6;text-transform:uppercase;
      letter-spacing:0.09em;margin-bottom:0.5rem;">Honest Results (Exp 24)</div>
      <div style="font-size:0.80rem;color:#d4cfff;line-height:1.7;">
        <b>Detection quality (AUROC):</b> 0.7753<br>
        <b>Held-out test AUROC:</b> 0.7871<br>
        <b>Geometry adds:</b> +0.198 AUROC over surface signals<br>
        <b>Calibration error (ECE):</b> 0.039 ✓<br>
        <b>Error rate in accepted answers:</b> 16.7% (1 of 6)
      </div>
    </div>
    """, unsafe_allow_html=True)

    for section, page_list in SECTIONS.items():
        sect_icon = {"Start Here": "📋", "Core Pipeline": "🔬", "Novel Contributions": "🧪", "Results": "📊"}.get(section, "•")
        st.markdown(
            f'<div style="font-size:0.70rem;font-weight:700;opacity:0.50;'
            f'text-transform:uppercase;letter-spacing:0.10em;margin:0.7rem 0 0.3rem;">'
            f'{sect_icon} {section}</div>',
            unsafe_allow_html=True,
        )
        for pname in page_list:
            if st.button(pname, key=f"nav_{pname}", use_container_width=True):
                st.session_state["page"] = pname

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.70rem;opacity:0.45;text-align:center;padding:0.3rem 0;">
      Lakshmi Chakradhar Vijayarao · HaRP · 2026
    </div>
    """, unsafe_allow_html=True)

# ── Route to active page ───────────────────────────────────────────────────────
active = st.session_state.get("page", "📋 Research Brief")
if active not in PAGES:
    active = "📋 Research Brief"
PAGES[active]()
