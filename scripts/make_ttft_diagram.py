import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10))
fig.suptitle("Perceived TTFT = Queue Wait + Prefill   (decode comes after)",
             fontsize=16, fontweight="bold")

# ---------- Colors ----------
c_queue   = "#E8743B"   # infra (orange)
c_prefill = "#1F77B4"   # compute (blue)
c_decode  = "#2CA02C"   # memory bandwidth (green)
c_text    = "#222222"

# ============================================================
# PANEL 1: Anatomy of a single request on a timeline
# ============================================================
ax1.set_title("1.  Anatomy of one request's timeline", fontsize=13, fontweight="bold", loc="left")
y = 0.5
h = 0.32

# segments: (label, start, width, color)
segs = [
    ("QUEUE WAIT\n(infrastructure)", 0.0, 2.5, c_queue),
    ("PREFILL\n(compute-bound)",     2.5, 2.0, c_prefill),
]
for label, x0, w, color in segs:
    ax1.add_patch(FancyBboxPatch((x0, y), w, h, boxstyle="round,pad=0.01",
                                 fc=color, ec="white", lw=2, alpha=0.92))
    ax1.text(x0 + w/2, y + h/2, label, ha="center", va="center",
             color="white", fontsize=10, fontweight="bold")

# decode tokens (small ticks after first token)
dec_start = 4.5
tok_w = 0.45
gap = 0.12
x = dec_start
for i in range(7):
    ax1.add_patch(FancyBboxPatch((x, y), tok_w, h, boxstyle="round,pad=0.005",
                                 fc=c_decode, ec="white", lw=1.5, alpha=0.9))
    x += tok_w + gap
ax1.text((dec_start + x)/2, y + h/2, "DECODE  (memory-bandwidth-bound)",
         ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")

# first token marker
ft_x = 4.5
ax1.annotate("FIRST TOKEN\nstreamed", xy=(ft_x, y + h), xytext=(ft_x, y + h + 0.45),
             ha="center", fontsize=9.5, fontweight="bold", color=c_text,
             arrowprops=dict(arrowstyle="->", lw=1.6, color=c_text))

# TTFT bracket (0 -> first token)
ax1.annotate("", xy=(0, y - 0.18), xytext=(ft_x, y - 0.18),
             arrowprops=dict(arrowstyle="<->", lw=2.2, color="#B22222"))
ax1.text(ft_x/2, y - 0.40, "TTFT  (what the user feels before anything appears)",
         ha="center", fontsize=10.5, fontweight="bold", color="#B22222")

# TPOT / ITL bracket
ax1.annotate("", xy=(dec_start, y - 0.78), xytext=(dec_start + tok_w + gap, y - 0.78),
             arrowprops=dict(arrowstyle="<->", lw=1.8, color=c_decode))
ax1.text(dec_start + 1.6, y - 0.95, "TPOT / ITL  (time per output token)",
         ha="center", fontsize=9.5, fontweight="bold", color=c_decode)

# request arrives marker
ax1.annotate("request\narrives", xy=(0, y + h/2), xytext=(-0.9, y + h/2),
             ha="center", va="center", fontsize=9, color=c_text,
             arrowprops=dict(arrowstyle="->", lw=1.4, color=c_text))

ax1.set_xlim(-1.4, x + 0.4)
ax1.set_ylim(-1.2, 1.6)
ax1.axis("off")

# ============================================================
# PANEL 2: Same bad TTFT, two different root causes
# ============================================================
ax2.set_title("2.  Same bad TTFT (~5s) — two completely different root causes & fixes",
              fontsize=13, fontweight="bold", loc="left")

def draw_scenario(ax, y, q_w, p_w, title, diagnosis, fix, diag_color):
    h = 0.30
    ax.text(-1.6, y + h/2, title, ha="left", va="center", fontsize=10.5, fontweight="bold")
    x0 = 1.0
    ax.add_patch(FancyBboxPatch((x0, y), q_w, h, boxstyle="round,pad=0.008",
                                fc=c_queue, ec="white", lw=2, alpha=0.92))
    if q_w > 0.6:
        ax.text(x0 + q_w/2, y + h/2, "queue", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")
    ax.add_patch(FancyBboxPatch((x0 + q_w, y), p_w, h, boxstyle="round,pad=0.008",
                                fc=c_prefill, ec="white", lw=2, alpha=0.92))
    if p_w > 0.6:
        ax.text(x0 + q_w + p_w/2, y + h/2, "prefill", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")
    # first token + ttft bracket
    ft = x0 + q_w + p_w
    ax.annotate("", xy=(x0, y - 0.12), xytext=(ft, y - 0.12),
                arrowprops=dict(arrowstyle="<->", lw=1.8, color="#B22222"))
    ax.text((x0 + ft)/2, y - 0.30, "TTFT ≈ 5s", ha="center", fontsize=9, color="#B22222", fontweight="bold")
    ax.text(ft + 0.25, y + h/2, diagnosis, ha="left", va="center",
            fontsize=9.5, color=diag_color, fontweight="bold")
    ax.text(ft + 0.25, y - 0.22, fix, ha="left", va="center", fontsize=8.7, color=c_text, style="italic")

# Scenario A: model/compute bound -> short queue, long prefill
draw_scenario(ax2, 1.3, q_w=0.5, p_w=4.0,
              title="A) prefill is SLOW",
              diagnosis="Root cause: MODEL / compute",
              fix="Fix: prompt caching, smaller model,\nfaster GPU, shorter input",
              diag_color=c_prefill)

# Scenario B: infra bound -> long queue, short prefill
draw_scenario(ax2, 0.1, q_w=4.0, p_w=0.5,
              title="B) prefill is FAST",
              diagnosis="Root cause: INFRASTRUCTURE",
              fix="Fix: raise min replicas / max replicas,\nconcurrency target, faster scale-up",
              diag_color=c_queue)

ax2.text(1.0, 2.15,
         '"When inference time is fast but end-to-end time is slow,\n turn your attention to infrastructure rather than model performance."  — Inference Engineering, p.39',
         fontsize=9.5, style="italic", color="#555555")

ax2.set_xlim(-1.7, 9.5)
ax2.set_ylim(-0.6, 2.6)
ax2.axis("off")

# legend
handles = [mpatches.Patch(color=c_queue, label="Queue wait (infrastructure / autoscaling)"),
           mpatches.Patch(color=c_prefill, label="Prefill (compute-bound) -> sets TTFT"),
           mpatches.Patch(color=c_decode, label="Decode (bandwidth-bound) -> sets TPS/ITL")]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9.5, frameon=False)

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.savefig("notes/quiz/ttft_anatomy.png", dpi=150, bbox_inches="tight")
print("saved notes/quiz/ttft_anatomy.png")
