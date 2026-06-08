import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import matplotlib.patches as mpatches

fig = plt.figure(figsize=(14, 10.5))
fig.suptitle("Speculative Decoding: spend idle decode-compute on MORE TOKENS per weight-read",
             fontsize=15.5, fontweight="bold")
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.32)

c_w = "#4C72B0"; c_draft="#DD8452"; c_ok="#2CA02C"; c_bad="#C44E52"; c_box="#333"

# ============================================================
# TOP: the mechanism
# ============================================================
ax = fig.add_subplot(gs[0])
ax.set_title("1.  One forward pass (one weight-read) now yields N+1 tokens", fontsize=12.5, fontweight="bold", loc="left")

# normal decode
ax.text(0.05, 3.1, "NORMAL decode:", fontsize=10.5, fontweight="bold")
ax.add_patch(FancyBboxPatch((2.2, 2.9), 3.0, 0.5, boxstyle="round,pad=0.02", fc=c_w, ec="white"))
ax.text(3.7, 3.15, "read all weights", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
ax.add_patch(Rectangle((5.5, 2.9), 0.5, 0.5, fc=c_ok, ec="white"))
ax.text(6.2, 3.15, "= 1 token", va="center", fontsize=9.5, fontweight="bold")

# speculative
ax.text(0.05, 1.7, "SPECULATIVE:", fontsize=10.5, fontweight="bold")
# draft step
ax.text(2.2, 2.25, "(1) fast speculator drafts 4 guesses:", fontsize=8.7, color=c_draft)
for i,lab in enumerate(["the","cat","sat","on"]):
    ax.add_patch(Rectangle((2.2+i*0.7, 1.7), 0.6, 0.45, fc=c_draft, ec="white"))
    ax.text(2.5+i*0.7, 1.92, lab, ha="center", va="center", color="white", fontsize=8)
# validate step
ax.add_patch(FancyBboxPatch((2.2, 0.9), 3.0, 0.5, boxstyle="round,pad=0.02", fc=c_w, ec="white"))
ax.text(3.7, 1.15, "read all weights ONCE -> validate all 4", ha="center", va="center", color="white", fontsize=8.3, fontweight="bold")
# result
for i,(lab,col) in enumerate([("the",c_ok),("cat",c_ok),("sat",c_ok),("on",c_bad)]):
    ax.add_patch(Rectangle((5.6+i*0.55, 0.9), 0.5, 0.5, fc=col, ec="white"))
    ax.text(5.85+i*0.55, 1.15, lab, ha="center", va="center", color="white", fontsize=7.5)
ax.add_patch(Rectangle((5.6+4*0.55, 0.9), 0.5, 0.5, fc=c_ok, ec="white", hatch="//"))
ax.text(5.6+4*0.55+0.25, 0.6, "target's\nown +1", ha="center", fontsize=7)
ax.text(5.6, 0.35, "3 drafts accepted (green) + 1 rejected (red) + 1 target token = 4 tokens / 1 weight-read",
        fontsize=8.5, color=c_box)
ax.text(2.2, 0.35, "rejected token →\nrest of drafts discarded", fontsize=7.5, color=c_bad)

ax.set_xlim(0, 9.5); ax.set_ylim(0, 3.5); ax.axis("off")

# ============================================================
# BOTTOM: the unifying insight + competition
# ============================================================
ax = fig.add_subplot(gs[1])
ax.set_title("2.  Two ways to spend the same idle decode-compute — and they COMPETE",
             fontsize=12.5, fontweight="bold", loc="left")
ax.axis("off")

ax.text(0.02, 0.95,
        "BATCHING  →  width\n"
        "  more USERS per weight-read\n"
        "  (throughput, lower cost/token)\n"
        "  wants HIGH batch",
        fontsize=10.3, va="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="#EAF2FB", ec=c_w))

ax.text(0.36, 0.95,
        "SPECULATION  →  depth\n"
        "  more TOKENS per user per weight-read\n"
        "  (latency, faster TPS for one user)\n"
        "  wants LOW batch (needs spare compute)",
        fontsize=10.3, va="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="#FBEEE6", ec=c_draft))

ax.text(0.02, 0.45,
        "THE CONFLICT:\n"
        "  High batch fills compute with users  →  no spare cycles  →  speculation gets DISABLED.\n"
        "  So speculation is a LOW-batch, latency-first tool. Adopting it REDUCES batch size,\n"
        "  which LOWERS throughput and RAISES cost. You trade money for speed.",
        fontsize=10.3, va="top", color=c_bad,
        bbox=dict(boxstyle="round,pad=0.4", fc="#FFF4F4", ec=c_bad))

ax.set_xlim(0,1); ax.set_ylim(0,1)

handles=[mpatches.Patch(color=c_w,label="weight read"),mpatches.Patch(color=c_draft,label="draft token"),
         mpatches.Patch(color=c_ok,label="accepted / target token"),mpatches.Patch(color=c_bad,label="rejected draft")]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9.5, frameon=False)
plt.tight_layout(rect=[0,0.03,1,0.95])
plt.savefig("notes/quiz/speculative_decoding.png", dpi=150, bbox_inches="tight")
print("saved")
