import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.patches as mpatches

fig = plt.figure(figsize=(14, 11))
fig.suptitle("Anatomy of the KV Cache  —  why concurrency is a memory problem",
             fontsize=17, fontweight="bold")

c_layer = "#4C72B0"
c_k = "#DD8452"
c_v = "#55A868"
c_box = "#3B3B3B"
gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.0, 0.9], hspace=0.42, wspace=0.22)

# ============================================================
# PANEL 1 (top, full width): the transformer stack + one token's KV
# ============================================================
ax = fig.add_subplot(gs[0, :])
ax.set_title("1.  A transformer is a STACK of layers. Each layer stores K and V for every token.",
             fontsize=12.5, fontweight="bold", loc="left")

# draw stack of layers
n_show = 5
for i in range(n_show):
    y = i * 0.7
    ax.add_patch(FancyBboxPatch((0.2, y), 3.2, 0.5, boxstyle="round,pad=0.02",
                                fc=c_layer, ec="white", lw=1.5, alpha=0.9))
    ax.text(1.8, y+0.25, f"Layer {i+1}  (attention sublayer)", ha="center", va="center",
            color="white", fontsize=9.5, fontweight="bold")
ax.text(1.8, n_show*0.7 + 0.05, "...80 layers total (Llama-3 70B)...",
        ha="center", fontsize=9, style="italic", color=c_box)

# zoom into one layer's per-token KV
ax.annotate("", xy=(4.3, 1.6), xytext=(3.5, 1.6),
            arrowprops=dict(arrowstyle="->", lw=2, color=c_box))
ax.text(4.4, 2.55, "Inside ONE layer, for ONE token:", fontsize=10, fontweight="bold")

# K heads
ax.text(4.4, 2.15, "K (keys):", fontsize=9.5, fontweight="bold", color=c_k)
for h in range(8):
    ax.add_patch(Rectangle((5.4 + h*0.42, 1.95), 0.36, 0.32, fc=c_k, ec="white", lw=1))
ax.text(5.4 + 8*0.42 + 0.2, 2.11, "8 KV heads", fontsize=8.5, va="center", color=c_k)

# V heads
ax.text(4.4, 1.55, "V (values):", fontsize=9.5, fontweight="bold", color=c_v)
for h in range(8):
    ax.add_patch(Rectangle((5.4 + h*0.42, 1.35), 0.36, 0.32, fc=c_v, ec="white", lw=1))
ax.text(5.4 + 8*0.42 + 0.2, 1.51, "8 KV heads", fontsize=8.5, va="center", color=c_v)

# head_dim annotation
ax.annotate("each head = a vector of head_dim = 128 numbers",
            xy=(5.58, 1.95), xytext=(5.0, 0.7),
            fontsize=8.7, color=c_box,
            arrowprops=dict(arrowstyle="->", lw=1.2, color=c_box))

ax.set_xlim(0, 11)
ax.set_ylim(0.3, 3.0)
ax.axis("off")

# ============================================================
# PANEL 2 (middle left): the formula
# ============================================================
ax = fig.add_subplot(gs[1, 0])
ax.set_title("2.  The formula (per token)", fontsize=12.5, fontweight="bold", loc="left")
ax.axis("off")
formula = (
    "KV bytes/token =\n"
    "   2          (K and V)\n"
    " × layers     (80)\n"
    " × kv_heads   (8)\n"
    " × head_dim   (128)\n"
    " × bytes      (2 = fp16)\n"
    "──────────────────────\n"
    "= 327,680 B  ≈  0.31 MB / token"
)
ax.text(0.02, 0.92, formula, fontsize=11.5, family="monospace", va="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="#F4F4F4", ec=c_box))
ax.text(0.02, 0.05, "Then ×  sequence_length  ×  concurrent_requests", fontsize=10,
        fontweight="bold", color="#B22222")

# ============================================================
# PANEL 3 (middle right): GQA vs MHA
# ============================================================
ax = fig.add_subplot(gs[1, 1])
ax.set_title("3.  GQA shrinks KV ~8x", fontsize=12.5, fontweight="bold", loc="left")
ax.axis("off")
ax.text(0.0, 0.9,
        "MHA (old): 64 query heads → 64 KV heads\n"
        "   2×80×64×128×2 ≈ 2.6 MB / token\n\n"
        "GQA (Llama-3): 64 query heads SHARE\n"
        "just 8 KV heads\n"
        "   2×80×8×128×2 ≈ 0.31 MB / token\n\n"
        "Same quality, 1/8 the KV cache.\n"
        "This is WHY 70B is servable at all.",
        fontsize=10.3, va="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="#FFF6E9", ec=c_k))

# ============================================================
# PANEL 4 (bottom, full): concurrency = budget / per-request
# ============================================================
ax = fig.add_subplot(gs[2, :])
ax.set_title("4.  Concurrency is just division (FP8 70B on H100: ~10 GB KV budget)",
             fontsize=12.5, fontweight="bold", loc="left")
ax.axis("off")

rows = [
    ("context = 2K tokens", 0.31*2048/1024, ),
    ("context = 8K tokens", 0.31*8192/1024, ),
    ("context = 32K tokens", 0.31*32768/1024, ),
]
y = 0.75
for label, per_req_gb in rows:
    conc = 10 / per_req_gb
    ax.text(0.01, y, f"{label}:", fontsize=10.5, fontweight="bold")
    ax.text(0.28, y, f"~{per_req_gb:.2f} GB / request", fontsize=10.5, family="monospace")
    ax.text(0.62, y, f"→  10 GB ÷ {per_req_gb:.2f}  ≈  {conc:.0f} concurrent",
            fontsize=10.5, family="monospace", color="#B22222", fontweight="bold")
    y -= 0.28
ax.text(0.01, -0.12, "Longer context  →  bigger per-request KV  →  FEWER concurrent users on the SAME GPU.",
        fontsize=10.5, style="italic", color=c_box)
ax.set_ylim(-0.25, 1.0)
ax.set_xlim(0, 1)

handles = [mpatches.Patch(color=c_layer, label="Transformer layer"),
           mpatches.Patch(color=c_k, label="K (keys) per head"),
           mpatches.Patch(color=c_v, label="V (values) per head")]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, frameon=False)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("notes/quiz/kv_cache_math.png", dpi=150, bbox_inches="tight")
print("saved notes/quiz/kv_cache_math.png")
