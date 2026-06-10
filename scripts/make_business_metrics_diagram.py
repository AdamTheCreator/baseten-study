import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_title("The Business of Inference — every metric ladders to EXPERIENCE or ECONOMICS",
             fontsize=15.5, fontweight="bold")

c_exp = "#3C7DC4"   # experience
c_eco = "#C77A2E"   # economics
c_box = "#333"

def row(ax, x, y, w, metric, feels, business, color):
    ax.add_patch(FancyBboxPatch((x, y), 2.0, 0.85, boxstyle="round,pad=0.03", fc=color, ec="white", lw=1.5, alpha=0.92))
    ax.text(x+1.0, y+0.42, metric, ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")
    ax.annotate("", xy=(x+3.0, y+0.42), xytext=(x+2.05, y+0.42), arrowprops=dict(arrowstyle="->", lw=1.4, color=c_box))
    ax.text(x+3.1, y+0.42, feels, ha="left", va="center", fontsize=8.7, color=c_box)
    ax.annotate("", xy=(x+8.4, y+0.42), xytext=(x+7.5, y+0.42), arrowprops=dict(arrowstyle="->", lw=1.4, color=color))
    ax.text(x+8.5, y+0.42, business, ha="left", va="center", fontsize=8.7, color=color, fontweight="bold")

# headers
ax.text(0.1, 9.5, "METRIC", fontsize=9, fontweight="bold", color=c_box)
ax.text(3.2, 9.5, "WHAT THE USER / CUSTOMER FEELS", fontsize=9, fontweight="bold", color=c_box)
ax.text(8.6, 9.5, "BUSINESS CONSEQUENCE", fontsize=9, fontweight="bold", color=c_box)

# EXPERIENCE band
ax.text(0.1, 9.0, "① EXPERIENCE  — \"will users love it?\"", fontsize=12, fontweight="bold", color=c_exp)
row(ax, 0.1, 8.0, 2, "TTFT", "blank-screen wait before\nANYTHING appears", "bounce / abandonment;\nvoice feels broken", c_exp)
row(ax, 0.1, 7.0, 2, "TPOT / ITL\n(perceived TPS)", "how fast text streams\nas they read", "feels snappy vs sluggish;\nengagement", c_exp)
row(ax, 0.1, 6.0, 2, "p95 / p99\nlatency", "the UNLUCKY requests —\nworst experiences", "SLA breaches; churn of\nyour heaviest users; trust", c_exp)

# ECONOMICS band
ax.text(0.1, 5.2, "② ECONOMICS  — \"can we afford to scale it profitably?\"", fontsize=12, fontweight="bold", color=c_eco)
row(ax, 0.1, 4.2, 2, "Total TPS\n(throughput)", "tokens the whole system\npumps per GPU-second", "directly sets cost/token", c_eco)
row(ax, 0.1, 3.2, 2, "Cost / token", "what each request\ncosts to serve", "gross margin; can you\nafford to grow", c_eco)
row(ax, 0.1, 2.2, 2, "Utilization /\nconcurrency", "how FULL the GPU is\n(idle = wasted rent)", "wasted spend vs\nefficiency", c_eco)
row(ax, 0.1, 1.2, 2, "Availability /\nuptime", "is it up when\nthey need it", "B2B SLAs; revenue\nprotection; trust", c_eco)

# tension arrow
ax.text(0.1, 0.45, "THE TENSION:  ① wants LOW batch (fast per user)   ⟷   ② wants HIGH batch (cheap per token).",
        fontsize=11, fontweight="bold", color="#B22222")
ax.text(0.1, 0.05, "THE SA's JOB:  find the balance point on that seesaw that fits THIS customer's business.",
        fontsize=11, fontweight="bold", color=c_box)

ax.set_xlim(0, 14); ax.set_ylim(0, 9.9); ax.axis("off")
plt.tight_layout()
plt.savefig("notes/quiz/business_of_inference.png", dpi=150, bbox_inches="tight")
print("saved")
