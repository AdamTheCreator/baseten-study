import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.patches as mpatches
import numpy as np

fig = plt.figure(figsize=(14, 11))
fig.suptitle("Why bigger batch = cheaper tokens BUT slower per user\n(the amortized weight read)",
             fontsize=16, fontweight="bold")
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.35, wspace=0.22)

c_w = "#4C72B0"   # weight read
c_tok = "#55A868" # tokens
c_idle = "#CCCCCC"
c_red = "#B22222"

# ============================================================
# PANEL 1: batch = 1  (wasteful)
# ============================================================
ax = fig.add_subplot(gs[0, 0])
ax.set_title("batch = 1  →  read ALL weights, get 1 token", fontsize=12, fontweight="bold", loc="left")
# weight read bar
ax.add_patch(FancyBboxPatch((0.1, 2.2), 6.0, 0.7, boxstyle="round,pad=0.02", fc=c_w, ec="white", lw=1.5))
ax.text(3.1, 2.55, "READ ~140 GB of model weights from VRAM", ha="center", va="center",
        color="white", fontsize=9.5, fontweight="bold")
# 1 token out
ax.add_patch(Rectangle((0.1, 1.1), 0.55, 0.5, fc=c_tok, ec="white"))
ax.text(0.85, 1.35, "= 1 token", fontsize=10, va="center", fontweight="bold", color=c_tok)
# wasted bandwidth
ax.text(0.1, 0.4, "Expensive weight read produces only ONE token.\nGPU mostly idle (decode is memory-bound).  cost/token = HIGH",
        fontsize=9.5, color=c_red, va="center")
ax.set_xlim(0, 6.5); ax.set_ylim(0, 3.2); ax.axis("off")

# ============================================================
# PANEL 2: batch = 8  (amortized)
# ============================================================
ax = fig.add_subplot(gs[0, 1])
ax.set_title("batch = 8  →  read the SAME weights ONCE, get 8 tokens", fontsize=12, fontweight="bold", loc="left")
ax.add_patch(FancyBboxPatch((0.1, 2.2), 6.0, 0.7, boxstyle="round,pad=0.02", fc=c_w, ec="white", lw=1.5))
ax.text(3.1, 2.55, "READ ~140 GB of model weights from VRAM (once!)", ha="center", va="center",
        color="white", fontsize=9.5, fontweight="bold")
for i in range(8):
    ax.add_patch(Rectangle((0.1 + i*0.62, 1.1), 0.55, 0.5, fc=c_tok, ec="white"))
ax.text(0.1, 0.65, "= 8 tokens (one per request)", fontsize=10, fontweight="bold", color=c_tok)
ax.text(0.1, 0.2, "Weight read amortized across 8 users.\nThroughput x8, cost/token / 8.  But the step does 8x compute.",
        fontsize=9.5, color="#222", va="center")
ax.set_xlim(0, 6.5); ax.set_ylim(0, 3.2); ax.axis("off")

# ============================================================
# PANEL 3: the tradeoff curves
# ============================================================
ax = fig.add_subplot(gs[1, 0])
ax.set_title("The tradeoff as batch size grows", fontsize=12, fontweight="bold", loc="left")
b = np.arange(1, 65)
# system throughput: rises, saturates (memory-bound -> compute-bound)
thru = 100 * b / (1 + b/24)
thru = thru / thru.max() * 100
# per-user TPOT: roughly flat then rises (compute-bound regime)
tpot = 20 + 0.9*b + 0.01*b**2
tpot = tpot / tpot.max() * 100

ax.plot(b, thru, color=c_tok, lw=2.6, label="System throughput (tok/s)  ↑ good")
ax.plot(b, tpot, color=c_red, lw=2.6, label="Per-user TPOT (latency)  ↑ bad")
ax.axvspan(8, 20, color="#FFE9A8", alpha=0.5)
ax.text(14, 50, "sweet\nspot", ha="center", fontsize=10, fontweight="bold", color="#8a6d00")
ax.set_xlabel("batch size  →"); ax.set_ylabel("relative")
ax.legend(fontsize=9, loc="center right"); ax.grid(alpha=0.25)

# ============================================================
# PANEL 4: the decision
# ============================================================
ax = fig.add_subplot(gs[1, 1])
ax.set_title("Which way do you push it?  →  depends on the WORKLOAD", fontsize=12, fontweight="bold", loc="left")
ax.axis("off")
txt = (
    "ONLINE  (chat, code, voice)\n"
    "  latency-sensitive\n"
    "  -> SMALL batch, low TPOT\n"
    "  -> accept higher cost/token\n\n"
    "OFFLINE (transcription, embeddings,\n"
    "         moderation, batch jobs)\n"
    "  throughput-sensitive\n"
    "  -> BIG batch, max throughput\n"
    "  -> accept higher per-request latency\n\n"
    "RULE OF THUMB:\n"
    "  Raise batch until you hit their\n"
    "  TPOT SLA (e.g. ~50 ms/token for\n"
    "  smooth reading), then STOP.\n"
    "  = max throughput within latency budget"
)
ax.text(0.0, 0.98, txt, fontsize=10.3, va="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="#F4F4F4", ec="#3B3B3B"))

handles = [mpatches.Patch(color=c_w, label="Model weight read (the expensive part)"),
           mpatches.Patch(color=c_tok, label="Output token")]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10, frameon=False)
plt.tight_layout(rect=[0, 0.03, 1, 0.93])
plt.savefig("notes/quiz/batching_tradeoff.png", dpi=150, bbox_inches="tight")
print("saved")
