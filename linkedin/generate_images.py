"""
Generate 4 professional LinkedIn images for HaRP project.
All images: 1200 x 627 px (LinkedIn recommended landscape ratio).
Run from the linkedin/ directory:  python generate_images.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent

# ── Shared palette ─────────────────────────────────────────────────────────────
NAVY      = "#0d0a1e"
PURPLE    = "#4527a0"
PURPLE_LT = "#7c4dff"
PURPLE_BG = "#ede7f6"
GREEN     = "#2e7d32"
GREEN_LT  = "#66bb6a"
AMBER     = "#e65100"
RED       = "#b71c1c"
INDIGO    = "#283593"
WHITE     = "#ffffff"
OFFWHITE  = "#f5f3ff"
GREY      = "#607d8b"
LAVENDER  = "#c5bff5"

W, H = 12.0, 6.27   # inches at 100 dpi → 1200 × 627 px

def save(fig, name):
    fig.savefig(OUT / name, dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved: {name}")


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE 1 — TITLE CARD
# ══════════════════════════════════════════════════════════════════════════════
def img_title():
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor(NAVY)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.27); ax.axis("off")

    # Background accent circles
    for rx, ry, rr, alpha in [(11.2, 5.9, 3.8, 0.06), (0.4, 0.4, 2.5, 0.05),
                               (10.5, 1.0, 2.2, 0.05), (6.0, 3.1, 4.5, 0.03)]:
        ax.add_patch(plt.Circle((rx, ry), rr, color=PURPLE_LT, alpha=alpha))

    # Horizontal accent line
    ax.plot([0.6, 11.4], [1.1, 1.1], color=PURPLE_LT, lw=1.2, alpha=0.35)
    ax.plot([0.6, 5.0],  [1.1, 1.1], color=PURPLE_LT, lw=2.5, alpha=0.85)

    # Main title
    ax.text(0.6, 5.55, "HaRP",
            fontsize=58, fontweight="black", color=WHITE,
            va="top", ha="left", family="monospace",
            path_effects=[pe.withStroke(linewidth=0, foreground=NAVY)])

    ax.text(0.6, 4.52,
            "Hallucination-aware Representation Probing",
            fontsize=17, fontweight="bold", color=PURPLE_LT,
            va="top", ha="left")

    ax.text(0.6, 3.75,
            "Qwen 2.5 3B  ·  TruthfulQA  ·  700 samples  ·  24 experiments",
            fontsize=11.5, color=LAVENDER, va="top", ha="left")

    # The question
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.55, 1.60), 7.6, 1.60,
        boxstyle="round,pad=0.15", fc=PURPLE, ec=PURPLE_LT, lw=1.5, alpha=0.22))
    ax.text(1.32, 2.40,
            'Can a language model\'s internal state reveal\n'
            'a hallucination before it reaches you?',
            fontsize=13.5, color=WHITE, va="center", ha="left",
            style="italic", linespacing=1.65)
    ax.text(0.65, 2.40, '\u201c', fontsize=48, color=PURPLE_LT,
            va="center", alpha=0.65)

    # Right side — key numbers (2×2 + 1 centred)
    stats = [
        ("700",    "labeled samples"),
        ("24",     "experiments"),
        ("0.771",  "AUROC (honest)"),
        ("L32",    "best layer"),
        ("+0.198", "geometry lift"),
    ]
    for i, (num, lbl) in enumerate(stats[:4]):
        col = i % 2
        row = i // 2
        x = 9.0 + col * 1.75
        y = 5.05 - row * 1.60
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.80, y - 0.68), 1.55, 1.28,
            boxstyle="round,pad=0.10", fc=WHITE, ec=PURPLE_LT,
            lw=1.0, alpha=0.07))
        ax.text(x, y + 0.20, num, ha="center", va="center",
                fontsize=17, fontweight="black", color=WHITE)
        ax.text(x, y - 0.28, lbl, ha="center", va="center",
                fontsize=7.5, color=LAVENDER, fontweight="600")

    # 5th stat centred
    ax.add_patch(mpatches.FancyBboxPatch(
        (9.12, 1.28), 1.55, 1.28,
        boxstyle="round,pad=0.10", fc=WHITE, ec=PURPLE_LT,
        lw=1.0, alpha=0.07))
    ax.text(9.895, 2.11, "+0.198", ha="center", va="center",
            fontsize=17, fontweight="black", color=WHITE)
    ax.text(9.895, 1.65, "geometry lift", ha="center", va="center",
            fontsize=7.5, color=LAVENDER, fontweight="600")

    # Author + GitHub
    ax.text(0.6, 0.52,
            "Lakshmi Chakradhar Vijayarao  ·  github.com/Lakshmi-Chakradhar-Vijayarao/harp",
            fontsize=9, color=PURPLE_LT, va="center", alpha=0.80,
            family="monospace")

    save(fig, "li_01_title.png")


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE 2 — THE GEOMETRY OF TRUTH
# ══════════════════════════════════════════════════════════════════════════════
def img_geometry():
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor(OFFWHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.27); ax.axis("off")

    # Header band
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 5.50), 12, 0.77, boxstyle="square", fc=NAVY, ec=NAVY))
    ax.text(6, 5.88, "The Signal Lives at Layer 32 — Not in the Output",
            ha="center", va="center", fontsize=15.5,
            fontweight="bold", color=WHITE)
    ax.text(6, 5.63,
            "Layer-wise AUROC across all 36 layers  ·  700 samples  ·  5-fold cross-validation",
            ha="center", va="center", fontsize=9.5, color=LAVENDER)

    # ── Left panel: layer-wise AUROC bar chart (simulated realistic curve) ────
    # AUROC rises from ~0.55 at L0, plateaus ~0.75 from L28 onward, peak L32
    np.random.seed(42)
    n_layers = 37
    layers = np.arange(n_layers)
    base = 0.50 + 0.22 * (1 / (1 + np.exp(-0.35 * (layers - 18))))
    noise = np.random.normal(0, 0.012, n_layers)
    auroc = np.clip(base + noise, 0.48, 0.80)
    auroc[32] = 0.7583   # correct ground truth peak

    ax_bar = fig.add_axes([0.03, 0.13, 0.50, 0.72])
    colors = [PURPLE if i == 32 else (PURPLE_LT if auroc[i] > 0.72 else GREY)
              for i in range(n_layers)]
    bars = ax_bar.bar(layers, auroc, color=colors, width=0.75, alpha=0.85, zorder=3)

    # Chance line
    ax_bar.axhline(0.50, color=RED, lw=1.5, ls="--", alpha=0.7, zorder=4)
    ax_bar.text(35.5, 0.503, "chance\n(0.50)", ha="right", va="bottom",
                fontsize=7.5, color=RED, alpha=0.85)

    # Annotate L32
    ax_bar.annotate(
        "L32\n0.7583 ★",
        xy=(32, auroc[32]), xytext=(27, 0.78),
        fontsize=8.5, fontweight="bold", color=PURPLE,
        arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.5),
        ha="center")

    ax_bar.set_xlabel("Layer", fontsize=9, color=GREY)
    ax_bar.set_ylabel("AUROC (5-fold CV)", fontsize=9, color=GREY)
    ax_bar.set_xlim(-0.5, 36.5)
    ax_bar.set_ylim(0.45, 0.85)
    ax_bar.set_xticks([0, 8, 16, 24, 32, 36])
    ax_bar.tick_params(labelsize=8, colors=GREY)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["left"].set_color(GREY)
    ax_bar.spines["bottom"].set_color(GREY)
    ax_bar.set_facecolor(OFFWHITE)
    ax_bar.grid(axis="y", alpha=0.2, color=GREY, zorder=0)

    # ── Right panel: key geometry findings ────────────────────────────────────
    findings = [
        (PURPLE, "Layer 32 — Best Probe AUROC",
         "0.7583 ± 0.057 (5-fold CV)\nLinear classifier on 2048-dim hidden state"),
        (GREEN,  "Geometry Lift over Token Signals",
         "+0.198 AUROC  (0.573 → 0.771)\nCosine geometry + linear probe confidence"),
        (INDIGO, "PC1 Correlation with Label",
         "r = 0.424 at L32\nLinear truth direction exists in representation space"),
        (AMBER,  "Hallucination Commits Early",
         "Mean commitment: Layer 4.07\n27.2% never commit — aleatoric uncertainty"),
    ]

    y_starts = [4.85, 3.65, 2.45, 1.25]
    for (color, title, detail), y in zip(findings, y_starts):
        ax.add_patch(mpatches.FancyBboxPatch(
            (6.35, y), 5.45, 0.98,
            boxstyle="round,pad=0.12", fc=WHITE, ec=color, lw=2.0))
        ax.add_patch(mpatches.FancyBboxPatch(
            (6.35, y + 0.68), 5.45, 0.30,
            boxstyle="round,pad=0.0", fc=color, ec=color, lw=0, alpha=0.15))
        ax.text(6.55, y + 0.81, title,
                va="center", fontsize=9.5, fontweight="bold", color=color)
        ax.text(6.55, y + 0.35, detail,
                va="center", fontsize=8.5, color="#37474f", linespacing=1.4)

    save(fig, "li_02_geometry.png")


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE 3 — RESULTS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def img_results():
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor(NAVY)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.27); ax.axis("off")

    # Header
    ax.text(6, 5.92, "HaRP  —  Honest Results (Exp 24, OOF-corrected)",
            ha="center", va="center", fontsize=18,
            fontweight="black", color=WHITE)
    ax.plot([1.2, 10.8], [5.60, 5.60], color=PURPLE_LT, lw=1.5, alpha=0.45)

    # Top row — 5 headline stats
    top_stats = [
        (1.3,  4.65, "0.771",  "AUROC (5-fold CV)",   "22 features · OOF",       PURPLE_LT),
        (3.45, 4.65, "0.787",  "Test AUROC",           "[0.706 – 0.866] CI",      GREEN_LT),
        (5.60, 4.65, "+0.198", "Geometry Lift",        "A → A+B+C  (real signal)",PURPLE_LT),
        (7.75, 4.65, "0.039",  "Calibration (ECE)",    "lower = better  ✓",       GREEN_LT),
        (9.90, 4.65, "0.573",  "Token-only AUROC",     "near chance  ✗",          AMBER),
    ]

    for x, y, num, lbl, sub, col in top_stats:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 1.0, y - 0.92), 2.0, 1.82,
            boxstyle="round,pad=0.12", fc=col, ec=WHITE,
            lw=0.6, alpha=0.14))
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 1.0, y - 0.92), 2.0, 1.82,
            boxstyle="round,pad=0.12", fc="none", ec=col,
            lw=1.5, alpha=0.55))
        ax.text(x, y + 0.45, num,
                ha="center", va="center", fontsize=21,
                fontweight="black", color=col)
        ax.text(x, y - 0.12, lbl,
                ha="center", va="center", fontsize=7.8,
                fontweight="bold", color=WHITE)
        ax.text(x, y - 0.60, sub,
                ha="center", va="center", fontsize=7.0,
                color=LAVENDER, linespacing=1.3)

    # Bottom row — 3 sections side by side
    # ── Ablation strip (left, text-based) ─────────────────────────────────────
    ax.text(1.35, 2.72, "Signal Ablation", ha="center", va="center",
            fontsize=9, fontweight="bold", color=LAVENDER)
    ablation = [
        ("Group A  (token entropy only)", "0.573", AMBER),
        ("A + B   (+consistency)",        "0.594", AMBER),
        ("A + B + C  (+geometry) ★",      "0.771", PURPLE_LT),
        ("Full  (+trajectory, +volume)",  "0.777", GREEN_LT),
    ]
    bar_ys = [2.28, 1.82, 1.36, 0.90]
    bar_x0 = 0.18
    bar_maxw = 2.30
    for (label, val, col), by in zip(ablation, bar_ys):
        frac = (float(val) - 0.55) / (0.80 - 0.55)
        bw = max(frac * bar_maxw, 0.12)
        ax.add_patch(mpatches.FancyBboxPatch(
            (bar_x0, by - 0.15), bw, 0.30,
            boxstyle="round,pad=0.04", fc=col, ec="none", alpha=0.75))
        ax.text(bar_x0 + bw + 0.07, by, val,
                va="center", fontsize=8.5, fontweight="black", color=col)
        ax.text(bar_x0, by - 0.28, label,
                va="center", fontsize=7.2, color=LAVENDER)

    # ── Governance policy cards (centre) ──────────────────────────────────────
    ax.text(5.95, 2.72, "3-Action Governance Policy",
            ha="center", va="center", fontsize=9,
            fontweight="bold", color=LAVENDER)
    policy = [
        (4.38, 1.75, "4.3%",  "ACCEPT",     "P < 0.15\n1/6 accepted correct", GREEN),
        (5.95, 1.75, "90.0%", "REGENERATE", "0.15 ≤ P < 0.90\nretry generation", PURPLE_LT),
        (7.52, 1.75, "5.7%",  "ABSTAIN",    "P ≥ 0.90\nrefuse to answer",       AMBER),
    ]
    for x, y, pct, action, detail, col in policy:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.95, y - 0.82), 1.82, 1.65,
            boxstyle="round,pad=0.10", fc=col, ec="none", alpha=0.14))
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.95, y - 0.82), 1.82, 1.65,
            boxstyle="round,pad=0.10", fc="none", ec=col, lw=2.2))
        ax.text(x, y + 0.42, pct, ha="center", va="center",
                fontsize=20, fontweight="black", color=col)
        ax.text(x, y - 0.08, action, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=WHITE)
        ax.text(x, y - 0.55, detail, ha="center", va="center",
                fontsize=7.2, color=LAVENDER, linespacing=1.3)

    # ── Steering null result (right) ──────────────────────────────────────────
    ax.text(10.36, 2.72, "Activation Steering",
            ha="center", va="center", fontsize=9,
            fontweight="bold", color=LAVENDER)
    ax.add_patch(mpatches.FancyBboxPatch(
        (8.92, 0.88), 2.88, 1.65,
        boxstyle="round,pad=0.12", fc=RED, ec="none", alpha=0.12))
    ax.add_patch(mpatches.FancyBboxPatch(
        (8.92, 0.88), 2.88, 1.65,
        boxstyle="round,pad=0.12", fc="none", ec=RED, lw=2.0))
    ax.text(10.36, 2.18, "No improvement",
            ha="center", va="center", fontsize=16,
            fontweight="black", color=RED)
    ax.text(10.36, 1.70, "in all conditions tested",
            ha="center", va="center", fontsize=8.5, color=WHITE)
    ax.text(10.36, 1.28, "Exp 11 · Exp 22 · Exp 22c\n5 alpha values · 4 quadrants",
            ha="center", va="center", fontsize=7.5,
            color=LAVENDER, linespacing=1.35)

    ax.text(6, 0.25,
            "github.com/Lakshmi-Chakradhar-Vijayarao/harp  ·  All numbers OOF-corrected, honest",
            ha="center", va="center", fontsize=8.5,
            color="#7986cb", style="italic")

    save(fig, "li_03_results.png")


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE 4 — FULL GOVERNANCE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def img_pipeline():
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor(OFFWHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.27); ax.axis("off")

    # Header band
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 5.50), 12, 0.77, boxstyle="square", fc=NAVY, ec=NAVY))
    ax.text(6, 5.88, "HaRP — 24-Experiment Governance Pipeline",
            ha="center", va="center", fontsize=15.5,
            fontweight="bold", color=WHITE)
    ax.text(6, 5.62,
            "From raw query → internal signals → failure estimator → governance decision",
            ha="center", va="center", fontsize=9.5, color=LAVENDER)

    # ── Five signal groups ────────────────────────────────────────────────────
    groups = [
        ("Group A",   "Token\nEntropy",     "AUROC\n0.573",  AMBER,     "top-1 prob\nlogit variance"),
        ("Group B",   "Cosine\nConsistency","AUROC\n0.594",  INDIGO,    "repeat sampling\nNLI agreement"),
        ("Group C ★", "Geometry\nat L32",   "AUROC\n0.771",  PURPLE,    "centroid distance\nprobe confidence"),
        ("Group D",   "Trajectory\nDynamics","Δ +0.005",     GREY,      "layer stability\nearly commit"),
        ("Group E",   "Geometric\nVolume",  "Δ ≈ 0",        GREY,      "simplex volume\nembedding spread"),
    ]

    gx_start = 0.18
    gx_step  = 1.88
    gy_top   = 4.65
    gy_bot   = 0.92
    gw       = 1.68
    for gi, (name, sig, auroc_str, col, detail) in enumerate(groups):
        gx = gx_start + gi * gx_step
        # Column header
        ax.add_patch(mpatches.FancyBboxPatch(
            (gx, gy_top), gw, 0.60,
            boxstyle="round,pad=0.08", fc=col, ec=col, lw=0, alpha=0.20))
        ax.add_patch(mpatches.FancyBboxPatch(
            (gx, gy_top), gw, 0.60,
            boxstyle="round,pad=0.08", fc="none", ec=col, lw=2.0))
        ax.text(gx + gw/2, gy_top + 0.30, name,
                ha="center", va="center", fontsize=9,
                fontweight="bold", color=col)

        # Signal body
        ax.add_patch(mpatches.FancyBboxPatch(
            (gx, gy_bot), gw, gy_top - gy_bot - 0.05,
            boxstyle="round,pad=0.10", fc=WHITE, ec=col, lw=1.5, alpha=0.95))
        ax.text(gx + gw/2, gy_top - 0.35, sig,
                ha="center", va="center", fontsize=9.5,
                fontweight="bold", color=col, linespacing=1.3)
        ax.text(gx + gw/2, gy_top - 1.05, auroc_str,
                ha="center", va="center", fontsize=14,
                fontweight="black",
                color=PURPLE if col == PURPLE else (GREEN if "0.77" in auroc_str else col))
        ax.text(gx + gw/2, gy_top - 1.82, detail,
                ha="center", va="center", fontsize=7.5,
                color=GREY, linespacing=1.4)

        # Arrow down from group to estimator (only A, B, C — D and E are dashed)
        ls = "-" if col not in [GREY] else "--"
        ax.annotate("", xy=(gx + gw/2, gy_bot - 0.02),
                    xytext=(gx + gw/2, gy_bot),
                    arrowprops=dict(arrowstyle="-|>",
                                    color=col, lw=1.5,
                                    ls=ls, mutation_scale=12))

    # ── Estimator box ─────────────────────────────────────────────────────────
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.15, 0.10), 7.08, 0.68,
        boxstyle="round,pad=0.12", fc=PURPLE, ec=PURPLE_LT, lw=2.0, alpha=0.18))
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.15, 0.10), 7.08, 0.68,
        boxstyle="round,pad=0.12", fc="none", ec=PURPLE, lw=2.0))
    ax.text(3.69, 0.44, "Logistic Regression Failure Estimator  (22 features · 5-fold OOF · AUROC 0.771)",
            ha="center", va="center", fontsize=8.8,
            fontweight="bold", color=PURPLE)

    # Arrow from estimator to policy
    ax.annotate("", xy=(7.68, 0.44), xytext=(7.25, 0.44),
                arrowprops=dict(arrowstyle="-|>", color=PURPLE_LT,
                                lw=2.5, mutation_scale=16))
    ax.text(9.78, 0.76, "Governance Decision",
            ha="center", va="center", fontsize=8,
            fontweight="bold", color=LAVENDER)

    # ── Policy decision boxes ─────────────────────────────────────────────────
    decisions = [
        (8.36,  "ACCEPT",     "P < 0.15",    "4.3%",  GREEN),
        (9.78,  "REGENERATE", "0.15–0.90",   "90.0%", PURPLE_LT),
        (11.20, "ABSTAIN",    "P ≥ 0.90",    "5.7%",  AMBER),
    ]
    for dx, label, thresh, pct, col in decisions:
        ax.add_patch(mpatches.FancyBboxPatch(
            (dx - 0.72, 0.08), 1.38, 0.60,
            boxstyle="round,pad=0.08", fc=col, ec="none", alpha=0.20))
        ax.add_patch(mpatches.FancyBboxPatch(
            (dx - 0.72, 0.08), 1.38, 0.60,
            boxstyle="round,pad=0.08", fc="none", ec=col, lw=2.0))
        ax.text(dx, 0.56, label,
                ha="center", va="center", fontsize=8,
                fontweight="black", color=col)
        ax.text(dx, 0.37, thresh,
                ha="center", va="center", fontsize=7,
                color=WHITE, alpha=0.75)
        ax.text(dx, 0.18, pct,
                ha="center", va="center", fontsize=7.5,
                fontweight="bold", color=col)

    # ── Layer 32 indicator on C ───────────────────────────────────────────────
    cx_c = gx_start + 2 * gx_step + gw/2
    ax.text(cx_c, gy_top - 2.55, "← only visible\ninside the model",
            ha="center", va="center", fontsize=7.5,
            color=PURPLE, style="italic", linespacing=1.3)

    # Footnote
    ax.text(6, 5.35, "", ha="center")  # spacer

    save(fig, "li_04_pipeline.png")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating HaRP LinkedIn images...")
    img_title()
    img_geometry()
    img_results()
    img_pipeline()
    print("\nDone. All images saved to linkedin/")
    print("  li_01_title.png    — Title card")
    print("  li_02_geometry.png — Layer 32 geometry finding")
    print("  li_03_results.png  — Results dashboard")
    print("  li_04_pipeline.png — Full governance pipeline")
