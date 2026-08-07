#!/usr/bin/env python3
"""
Phase 4 - QAPINN circuit-design decision tree.
Reproducible figure. All explanatory text lives INSIDE the boxes,
in a smaller secondary font. Only edge labels sit outside the boxes.
No non-standard dashes; standard hyphen "-" only.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# ---- palette ---------------------------------------------------------------
NAVY   = "#1f3f8f"   # decision flow / primary
RED    = "#b3202c"   # avoid / classical-preferred
GREEN  = "#2e7d32"   # core requirement / cheap regime
AMBER  = "#9a7d00"   # protocol guardrail
DARK   = "#20242c"   # body text
INK    = "#111318"

FILL_IN   = "#dfe7f7"
FILL_CORE = "#e8f3e8"
FILL_DEC  = "#eef1f7"
FILL_BAD  = "#f7e3e5"
FILL_GOOD = "#e8f3e8"
FILL_WARN = "#fdf6da"
FILL_PRED = "#eceff5"

# primary (title) vs secondary (detail) font sizes - deliberately different
FS_HEAD  = 11.5
FS_BODY  = 9.5
FS_DETAIL= 8.0
FS_EDGE  = 9.0

fig, ax = plt.subplots(figsize=(12.0, 14.0))
ax.set_xlim(0, 100); ax.set_ylim(0, 124); ax.axis("off")


def box(x, y, w, h, head, detail=None, fc=FILL_DEC, ec=NAVY, tc=INK,
        head_fs=FS_BODY, head_weight="bold", detail_fs=FS_DETAIL):
    """A rounded box: bold head line(s) + optional smaller detail block."""
    b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.35,rounding_size=1.8",
                       fc=fc, ec=ec, lw=1.6, zorder=3)
    ax.add_patch(b)
    if detail is None:
        ax.text(x, y, head, ha="center", va="center",
                fontsize=head_fs, weight=head_weight, color=tc, zorder=4)
    else:
        ax.text(x, y + h*0.30, head, ha="center", va="center",
                fontsize=head_fs, weight=head_weight, color=tc, zorder=4)
        ax.text(x, y - h*0.16, detail, ha="center", va="center",
                fontsize=detail_fs, color=DARK, zorder=4, linespacing=1.25)


def arrow(x1, y1, x2, y2, color=NAVY, ls="-", lw=1.7):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=16, lw=lw, color=color, ls=ls, zorder=2))


def edge_label(x, y, text, color):
    ax.text(x, y, text, ha="center", va="center", fontsize=FS_EDGE,
            weight="bold", color=color, zorder=6,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.95))


# ---- title -----------------------------------------------------------------
ax.text(50, 121.5, "Problem-Specific QAPINN Circuit Design",
        ha="center", fontsize=17, weight="bold", color=NAVY)
ax.text(50, 118.2,
        "A decision procedure derived from Phase 3 findings "
        "(Burgers $\\nu=0.05$, $\\nu=0.01/\\pi$; Heat $\\alpha=0.10$)",
        ha="center", fontsize=10.5, style="italic", color=DARK)

# ---- INPUT -----------------------------------------------------------------
box(50, 112, 60, 6,
    "INPUT: target PDE configuration",
    "smoothness  |  frequency content  |  nonlinearity  |  dimensionality",
    fc=FILL_IN, ec=NAVY, head_fs=FS_HEAD, detail_fs=8.6)

arrow(50, 109, 50, 105.6)

# ---- STEP 1 : CORE ---------------------------------------------------------
box(50, 100.5, 80, 9.2,
    "STEP 1   Non-negotiable core",
    "Same at every benchmark tested:\n"
    "$\\bullet$  Encoding: data re-uploading + trainable input scaling\n"
    "$\\bullet$  Entanglement: all-to-all      $\\bullet$  Readout: Pauli-$Z$ expectation values\n"
    "$\\bullet$  Body: deep classical MLP after the circuit",
    fc=FILL_CORE, ec=GREEN, head_fs=FS_HEAD, detail_fs=8.4)

arrow(50, 95.9, 50, 92.6)

# ---- STEP 2 : SPECTRAL GATE ------------------------------------------------
box(50, 88, 62, 7.6,
    "STEP 2   Spectral gate",
    "Does a parameter-matched classical PINN\n"
    "already recover the target spectrum $\\varphi_{>k_{\\mathrm{cut}}}$?",
    fc=FILL_DEC, ec=NAVY, head_fs=FS_HEAD, detail_fs=8.6)

# YES -> classical
arrow(27, 87, 14, 80.5, color=RED)
edge_label(18.5, 84.3, "YES (reach ceiling)", RED)
box(13, 74.5, 22, 9.5,
    "Classical suffices",
    "Quantum layer cannot\nextend reach.\n"
    "$\\bullet$  use narrow $n$, or\n$\\bullet$  prefer a classical PINN\n"
    "WS-A: both recover 99% of\nhigh-freq; $n{=}3$ already 97.5%",
    fc=FILL_BAD, ec=RED, head_fs=10, detail_fs=7.4)

# NO -> quantum candidate
arrow(73, 87, 86, 80.5, color=NAVY)
edge_label(81.5, 84.3, "NO", NAVY)
box(87, 75.5, 22, 7.5,
    "Quantum candidate",
    "A quantum layer is\nwarranted. Size it in\nStep 3.",
    fc=FILL_IN, ec=NAVY, head_fs=10, detail_fs=7.8)

# feed into step 3
arrow(87, 71.7, 60, 65.2, color=NAVY)
arrow(13, 69.7, 40, 65.2, color=RED, ls=(0, (5, 3)))
edge_label(27, 68.6, "quantum used anyway", RED)

# ---- STEP 3 : QUBIT SIZING -------------------------------------------------
box(50, 61, 74, 8.4,
    "STEP 3   Size qubit count $n$ to the spectrum, not to a maximum",
    "Each qubit adds accessible frequencies (Schuld), but beyond the target support:\n"
    "$\\bullet$  no accuracy gain      $\\bullet$  gradient variance decays 14-29$\\times$\n"
    "$\\bullet$  wall-clock cost grows $\\sim 2^{n}$ (28$\\times$ over $n{=}3$ to $8$)",
    fc=FILL_DEC, ec=NAVY, head_fs=FS_HEAD, detail_fs=8.2)

arrow(50, 56.8, 50, 53.4)

# three sizing regimes
box(20, 48, 27, 8.2,
    "Smooth / low-freq",
    "e.g. Heat\n$\\Rightarrow n = 3\\!-\\!4$\naccuracy saturates by $q4$",
    fc=FILL_GOOD, ec=GREEN, head_fs=10, detail_fs=8.0)
box(50, 48, 27, 8.2,
    "Shock / moderate-freq",
    "e.g. Burgers\n$\\Rightarrow n = 4\\!-\\!5$\nimproves through $q5$, then stops",
    fc=FILL_GOOD, ec=GREEN, head_fs=10, detail_fs=8.0)
box(80, 48, 27, 8.2,
    "Wide circuits  $n \\geq 6$",
    "avoid:\nno accuracy gain,\ngradient decay, severe cost",
    fc=FILL_BAD, ec=RED, head_fs=10, detail_fs=8.0)

arrow(50, 53.4, 20, 52.2, color=GREEN)
arrow(50, 53.4, 50, 52.2, color=GREEN)
arrow(50, 53.4, 80, 52.2, color=RED)

# ---- STEP 4 : PROTOCOL -----------------------------------------------------
arrow(20, 43.9, 20, 40.4, color=AMBER)
arrow(50, 43.9, 50, 40.4, color=AMBER)
arrow(80, 43.9, 80, 40.4, color=AMBER)

box(50, 37, 84, 4.8,
    "STEP 4   Training and reporting protocol (every branch)",
    fc=FILL_WARN, ec=AMBER, head_fs=FS_HEAD)

arrow(50, 34.6, 50, 31.6, color=AMBER)

box(20, 26.5, 30, 8.6,
    "Seeds as a distribution",
    "run $\\geq 3$ seeds, report spread;\nseed variance dominates\narchitecture in WS-A, B, C",
    fc=FILL_WARN, ec=AMBER, head_fs=9.6, detail_fs=7.8)
box(50, 26.5, 30, 8.6,
    "Spend effort on the residual",
    "loss is residual-dominated\n($\\mathrm{MSE}_f/\\mathrm{MSE}_u = 20\\!-\\!186$);\n"
    "collocation and weighting,\nnot more qubits",
    fc=FILL_WARN, ec=AMBER, head_fs=9.6, detail_fs=7.8)
box(80, 26.5, 30, 8.6,
    "Match training budget",
    "fix steps across widths;\ntapered budgets confounded\nthe easy sweep (hard: 2016)",
    fc=FILL_WARN, ec=AMBER, head_fs=9.6, detail_fs=7.8)

arrow(50, 31.6, 20, 30.9, color=AMBER)
arrow(50, 31.6, 50, 30.9, color=AMBER)
arrow(50, 31.6, 80, 30.9, color=AMBER)

# ---- FALSIFIABLE PREDICTIONS ----------------------------------------------
arrow(50, 22.1, 50, 19.0, color=DARK)

box(50, 16.2, 86, 4.4,
    "FALSIFIABLE PREDICTIONS   (the procedure must be able to fail)",
    fc=FILL_PRED, ec=INK, head_fs=FS_HEAD, tc=INK)

arrow(50, 14.0, 50, 11.5, color=DARK)

box(28, 6.6, 44, 8.4,
    "P1 / P2",
    "P1  no-entanglement circuit fails\n(relative $L^2 \\approx 1$)\n"
    "P2  qubits beyond support give no\naccuracy gain and lower gradient var.",
    fc="#f4f6fb", ec=INK, head_fs=9.6, detail_fs=8.0)
box(73, 6.6, 44, 8.4,
    "P3 / P4",
    "P3  wide QAPINN not more accurate\nthan its twin, at far higher cost\n"
    "P4  advantage shrinks toward the\ntwin on a smooth target",
    fc="#f4f6fb", ec=INK, head_fs=9.6, detail_fs=8.0)

arrow(50, 14.0, 28, 11.0, color=DARK)
arrow(50, 14.0, 73, 11.0, color=DARK)

# ---- legend ----------------------------------------------------------------
leg = [Line2D([0], [0], color=GREEN, lw=3.4, label="core requirement / cheap regime"),
       Line2D([0], [0], color=RED,   lw=3.4, label="avoid / classical-preferred"),
       Line2D([0], [0], color=AMBER, lw=3.4, label="protocol guardrail"),
       Line2D([0], [0], color=NAVY,  lw=3.4, label="decision flow")]
ax.legend(handles=leg, loc="lower center", ncol=4, fontsize=8.5,
          frameon=False, bbox_to_anchor=(0.5, -0.015))

plt.tight_layout()
plt.savefig("figures/decision_tree.png", dpi=200, bbox_inches="tight", facecolor="white")
print("wrote figures/decision_tree.png")
