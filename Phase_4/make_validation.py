#!/usr/bin/env python3
"""
Phase 4 - validation of the four falsifiable predictions against Phase 3 data.
No in-figure commentary, no overlaps, standard hyphens only.
All numbers are quoted directly from the Phase 3 workstream reports.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAVY  = "#1f3f8f"
RED   = "#b3202c"
GREEN = "#2e7d32"
AMBER = "#9a7d00"
GREY  = "#8a8f98"

plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})

fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.8))
fig.subplots_adjust(hspace=0.42, wspace=0.30, top=0.90, bottom=0.09,
                    left=0.09, right=0.93)

# ---- P1 : entanglement ablation (Roland, easy Burgers) --------------------
ax = axes[0, 0]
labels = ["none", "linear", "all-to-all\n($q=5$)"]
l2 = [1.045, 0.0182, 0.0049]
cols = [RED, AMBER, GREEN]
bars = ax.bar(labels, l2, color=cols, alpha=0.92, width=0.62)
ax.set_yscale("log")
ax.set_ylim(2e-3, 3.0)
ax.set_ylabel(r"relative $L^2$ error")
ax.set_title("P1   Entanglement is required")
ax.axhline(1.0, ls=":", color=GREY, lw=1.1)
# numbers INSIDE / on the bars, positioned to avoid the L2=1 guide line
for b, v in zip(bars, l2):
    ax.text(b.get_x() + b.get_width()/2, v*0.55, f"{v:.3f}",
            ha="center", va="top", fontsize=9, weight="bold",
            color="white" if v > 0.01 else "black")

# ---- P2 : gradient variance decays with width, QAPINN only (hard) ---------
ax = axes[0, 1]
n = np.array([3, 4, 5, 6, 7, 8])
q  = np.array([0.490, 0.209, 0.112, 0.129, 0.138, 0.035])
tw = np.array([2.909, 7.411, 1.198, 1.691, 2.758, 2.838])
ax.plot(n, q,  "o-",  color=NAVY, lw=2, label="QAPINN (quantum first layer)")
ax.plot(n, tw, "s--", color=RED,  lw=2, label="classical twin")
ax.set_yscale("log")
ax.set_xlabel("qubit count / width $n$")
ax.set_ylabel("first-layer gradient variance")
ax.set_title("P2   Added qubits cost gradient signal")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8.5, loc="lower left")

# ---- P3 : wide QAPINN not more accurate than twin, at rising cost ---------
ax = axes[1, 0]
n2 = np.array([3, 4, 5, 6, 7])
qL2  = np.array([11.8, 9.1, 2.6, 1.7, 5.3])
twL2 = np.array([1.3, 4.5, 19.8, 0.6, 7.0])
ax.plot(n2, qL2,  "o-",  color=NAVY, lw=2, label="QAPINN")
ax.plot(n2, twL2, "s--", color=RED,  lw=2, label="classical twin")
ax.set_xlabel("width $n$")
ax.set_ylabel(r"relative $L^2$ error (%)")
ax.set_title("P3   Wide QAPINN is not more accurate")
ax.set_ylim(0, 22)
ax.grid(alpha=0.3)
ax.legend(fontsize=8.5, loc="upper left", title="accuracy", title_fontsize=8)

ax2 = ax.twinx()
cost = np.array([17, 65, 111, 239, 471])
ax2.plot(n2, cost, "^:", color=GREEN, lw=1.7, alpha=0.9,
         label="cost ratio")
ax2.set_ylabel(r"QAPINN / twin cost per step ($\times$)", color=GREEN)
ax2.tick_params(axis="y", labelcolor=GREEN)
ax2.set_ylim(0, 520)
ax2.legend(fontsize=8.5, loc="upper right", title="cost", title_fontsize=8)

# ---- P4 : info-retention advantage shrinks on a smooth target -------------
ax = axes[1, 1]
cats = ["Burgers\n(shock, hard)", "Burgers\n(shock, easy)", "Heat\n(smooth)"]
ratio = [1.40, 1.34, 1.06]
cols2 = [GREEN, GREEN, RED]
bars = ax.bar(cats, ratio, color=cols2, alpha=0.92, width=0.6)
ax.axhline(1.0, ls=":", color=GREY, lw=1.2)
ax.set_ylabel("CKA ratio   QAPINN / twin")
ax.set_title("P4   Advantage shrinks on a smooth PDE")
ax.set_ylim(0.9, 1.55)
for b, v in zip(bars, ratio):
    ax.text(b.get_x() + b.get_width()/2, v - 0.05, f"{v:.2f}",
            ha="center", va="top", fontsize=9.5, weight="bold", color="white")

fig.suptitle("Methodology validation: four falsifiable predictions against Phase 3 data",
             fontsize=13, weight="bold", y=0.965)

plt.savefig("figures/validation.png", dpi=200, bbox_inches="tight", facecolor="white")
print("wrote figures/validation.png")
