"""
Whale vs Star Bot - Comprehensive Strategy Analysis Charts
Compares 238K whale trades against 101 paper trades.
Generates a single figure with 6 subplots saved to whale_analysis_charts.png.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Dark theme setup ──────────────────────────────────────────────────────────
plt.style.use("dark_background")
COLORS = {
    "whale": "#00bfff",       # deep sky blue
    "ours": "#ff6347",        # tomato red
    "accent": "#ffd700",      # gold
    "green": "#00e676",
    "yellow": "#ffeb3b",
    "red": "#ff5252",
    "bg": "#0d1117",
    "card": "#161b22",
    "text": "#c9d1d9",
    "muted": "#8b949e",
}

fig, axes = plt.subplots(3, 2, figsize=(18, 14), facecolor=COLORS["bg"])
fig.suptitle(
    "Whale vs Star Bot - Strategy Analysis  (238K whale trades vs 101 paper trades)",
    fontsize=16,
    fontweight="bold",
    color="white",
    y=0.98,
)

for ax in axes.flat:
    ax.set_facecolor(COLORS["card"])
    ax.tick_params(colors=COLORS["text"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(COLORS["muted"])

# ══════════════════════════════════════════════════════════════════════════════
# Chart 1 - Price Bucket WR Comparison (grouped bar)
# ══════════════════════════════════════════════════════════════════════════════
ax1 = axes[0, 0]
buckets = ["<$0.15", "$0.15-0.25", "$0.25-0.35", "$0.35-0.45", "$0.45-0.55"]
whale_wr = [83.1, 64.7, 54.0, 41.8, 71.9]
our_wr = [16.7, 68.0, 69.7, 72.7, 50.0]

x = np.arange(len(buckets))
w = 0.35
bars1 = ax1.bar(x - w / 2, whale_wr, w, label="Whale WR%", color=COLORS["whale"], edgecolor="none", alpha=0.9)
bars2 = ax1.bar(x + w / 2, our_wr, w, label="Our WR%", color=COLORS["ours"], edgecolor="none", alpha=0.9)

for bar, val in zip(bars1, whale_wr):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2, f"{val:.0f}%",
             ha="center", va="bottom", fontsize=8, color=COLORS["whale"], fontweight="bold")
for bar, val in zip(bars2, our_wr):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2, f"{val:.0f}%",
             ha="center", va="bottom", fontsize=8, color=COLORS["ours"], fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels(buckets, fontsize=9)
ax1.set_ylabel("Win Rate %", color=COLORS["text"])
ax1.set_title("Price Bucket WR Comparison", fontsize=12, fontweight="bold", color="white")
ax1.legend(fontsize=9, loc="upper right")
ax1.set_ylim(0, 105)
ax1.axhline(y=50, color=COLORS["muted"], linestyle="--", linewidth=0.7, alpha=0.5)

# ══════════════════════════════════════════════════════════════════════════════
# Chart 2 - Asset WR Comparison (grouped bar)
# ══════════════════════════════════════════════════════════════════════════════
ax2 = axes[0, 1]
assets = ["BTC", "ETH", "SOL"]
whale_asset_wr = [48.4, 60.5, 62.1]
our_asset_wr = [65.7, 60.0, 64.5]

x2 = np.arange(len(assets))
bars3 = ax2.bar(x2 - w / 2, whale_asset_wr, w, label="Whale WR%", color=COLORS["whale"], edgecolor="none", alpha=0.9)
bars4 = ax2.bar(x2 + w / 2, our_asset_wr, w, label="Our WR%", color=COLORS["ours"], edgecolor="none", alpha=0.9)

for bar, val in zip(bars3, whale_asset_wr):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0, f"{val:.1f}%",
             ha="center", va="bottom", fontsize=9, color=COLORS["whale"], fontweight="bold")
for bar, val in zip(bars4, our_asset_wr):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0, f"{val:.1f}%",
             ha="center", va="bottom", fontsize=9, color=COLORS["ours"], fontweight="bold")

ax2.set_xticks(x2)
ax2.set_xticklabels(assets, fontsize=11, fontweight="bold")
ax2.set_ylabel("Win Rate %", color=COLORS["text"])
ax2.set_title("Asset WR Comparison", fontsize=12, fontweight="bold", color="white")
ax2.legend(fontsize=9, loc="upper right")
ax2.set_ylim(0, 85)
ax2.axhline(y=50, color=COLORS["muted"], linestyle="--", linewidth=0.7, alpha=0.5)

# ══════════════════════════════════════════════════════════════════════════════
# Chart 3 - Hourly WR Heatmap (horizontal bar)
# ══════════════════════════════════════════════════════════════════════════════
ax3 = axes[1, 0]
hourly_wr = {
    0: 58.8, 1: 57.9, 2: 57.4, 3: 45.6, 4: 50.4, 5: 59.3, 6: 58.5, 7: 59.6,
    8: 56.6, 9: 55.1, 10: 57.2, 11: 58.8, 12: 56.4, 13: 57.0, 14: 56.2, 15: 56.9,
    16: 63.9, 17: 60.5, 18: 49.7, 19: 55.5, 20: 59.3, 21: 57.8, 22: 58.8, 23: 59.5,
}
our_skip_hours = {1, 8, 14, 15}

hours = list(range(24))
wrs = [hourly_wr[h] for h in hours]
bar_colors = []
for wr_val in wrs:
    if wr_val > 57:
        bar_colors.append(COLORS["green"])
    elif wr_val >= 52:
        bar_colors.append(COLORS["yellow"])
    else:
        bar_colors.append(COLORS["red"])

y_pos = np.arange(24)
hbars = ax3.barh(y_pos, wrs, color=bar_colors, edgecolor="none", alpha=0.85, height=0.7)

for i, (h, wr_val) in enumerate(zip(hours, wrs)):
    label = f" {wr_val:.1f}%"
    if h in our_skip_hours:
        label += "  *SKIP*"
    ax3.text(wr_val + 0.3, i, label, va="center", fontsize=7, color="white", fontweight="bold")

ax3.set_yticks(y_pos)
ax3.set_yticklabels([f"{h:02d}:00" for h in hours], fontsize=7)
ax3.set_xlabel("Whale Win Rate %", color=COLORS["text"])
ax3.set_title("Hourly Whale WR (green >57 | yellow 52-57 | red <52)", fontsize=11, fontweight="bold", color="white")
ax3.set_xlim(40, 70)
ax3.axvline(x=57, color=COLORS["muted"], linestyle="--", linewidth=0.7, alpha=0.5)
ax3.axvline(x=52, color=COLORS["red"], linestyle="--", linewidth=0.7, alpha=0.3)
ax3.invert_yaxis()

# legend for skip hours
skip_patch = mpatches.Patch(facecolor="none", edgecolor="white", label="*SKIP* = our skip hours")
ax3.legend(handles=[skip_patch], fontsize=8, loc="lower right")

# ══════════════════════════════════════════════════════════════════════════════
# Chart 4 - Top 5 Whale Strategies (horizontal bar)
# ══════════════════════════════════════════════════════════════════════════════
ax4 = axes[1, 1]
whale_names = ["0xE594\n(Phantom)", "PBot1\n(Conservative)", "0x8dxd\n(Vol Grinder)", "k9Q2\n(Cheap Entry)", "distinct-baguette\n(Market Maker)"]
whale_strat_wr = [98.9, 94.9, 61.1, 60.7, 46.2]
whale_pnl = ["$110K", "$88K", "$214K", "$269K", "$74K"]
whale_bar_colors = [COLORS["green"], COLORS["green"], COLORS["whale"], COLORS["whale"], COLORS["yellow"]]

y4 = np.arange(len(whale_names))
hbars4 = ax4.barh(y4, whale_strat_wr, color=whale_bar_colors, edgecolor="none", alpha=0.85, height=0.55)

for i, (wr_val, pnl) in enumerate(zip(whale_strat_wr, whale_pnl)):
    ax4.text(wr_val + 0.8, i, f"{wr_val:.1f}%  |  {pnl}", va="center", fontsize=9, color="white", fontweight="bold")

ax4.set_yticks(y4)
ax4.set_yticklabels(whale_names, fontsize=9)
ax4.set_xlabel("Win Rate %", color=COLORS["text"])
ax4.set_title("Top 5 Whale Strategies (crypto_15m)", fontsize=12, fontweight="bold", color="white")
ax4.set_xlim(0, 115)
ax4.axvline(x=50, color=COLORS["muted"], linestyle="--", linewidth=0.7, alpha=0.5)
ax4.invert_yaxis()

# ══════════════════════════════════════════════════════════════════════════════
# Chart 5 - Side Preference (Pie charts side by side)
# ══════════════════════════════════════════════════════════════════════════════
ax5 = axes[2, 0]
ax5.set_title("Side Preference: Whales vs Us", fontsize=12, fontweight="bold", color="white")

# Create two pie charts side by side using inset axes
pie_colors = [COLORS["green"], COLORS["ours"]]

# Our distribution - left pie
ax5_left = ax5.inset_axes([0.02, 0.05, 0.44, 0.85])
ax5_left.set_facecolor(COLORS["card"])
our_sizes = [43, 57]
our_labels = ["UP 43%", "DOWN 57%"]
wedges1, texts1 = ax5_left.pie(
    our_sizes, labels=our_labels, colors=pie_colors,
    startangle=90, textprops={"color": "white", "fontsize": 10, "fontweight": "bold"},
    wedgeprops={"edgecolor": COLORS["card"], "linewidth": 2},
)
ax5_left.set_title("OUR BOT", fontsize=11, color=COLORS["ours"], fontweight="bold", pad=2)

# Whale distribution - right pie
ax5_right = ax5.inset_axes([0.54, 0.05, 0.44, 0.85])
ax5_right.set_facecolor(COLORS["card"])
whale_sizes = [50, 50]
whale_labels = ["UP 50%", "DOWN 50%"]
wedges2, texts2 = ax5_right.pie(
    whale_sizes, labels=whale_labels, colors=pie_colors,
    startangle=90, textprops={"color": "white", "fontsize": 10, "fontweight": "bold"},
    wedgeprops={"edgecolor": COLORS["card"], "linewidth": 2},
)
ax5_right.set_title("WHALES", fontsize=11, color=COLORS["whale"], fontweight="bold", pad=2)

# Hide the parent axes ticks
ax5.set_xticks([])
ax5.set_yticks([])

# ══════════════════════════════════════════════════════════════════════════════
# Chart 6 - Key Recommendations (text box)
# ══════════════════════════════════════════════════════════════════════════════
ax6 = axes[2, 1]
ax6.set_xticks([])
ax6.set_yticks([])
ax6.set_title("Key Recommendations", fontsize=12, fontweight="bold", color="white")

rec_text = (
    "WHALE INSIGHTS FOR V3.7:\n"
    "\n"
    "1. Hour 18 UTC = whale graveyard (49.7% WR) - ADD to skip\n"
    "\n"
    "2. BTC hardest asset (48.4% WR) - tighten BTC filters\n"
    "\n"
    "3. Whales trade UP/DOWN 50/50 - reduce DOWN bias\n"
    "\n"
    "4. <$0.10 = whale paradise (83% WR) but our sample too small\n"
    "\n"
    "5. $0.45-$0.55 whales get 71% WR - we get 50% (reassess cap)\n"
    "\n"
    "6. k9Q2 strategy: only <$0.20 OR >$0.45 (skip middle)"
)

ax6.text(
    0.05, 0.92, rec_text,
    transform=ax6.transAxes,
    fontsize=10.5,
    fontfamily="monospace",
    color=COLORS["accent"],
    verticalalignment="top",
    horizontalalignment="left",
    linespacing=1.3,
    bbox=dict(boxstyle="round,pad=0.6", facecolor="#1a1a2e", edgecolor=COLORS["accent"], linewidth=1.5, alpha=0.9),
)

# ── Final layout ──────────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0, 1, 0.96])

out_path = r"C:\Users\Star\.local\bin\star-polymarket\whale_analysis_charts.png"
fig.savefig(out_path, dpi=150, facecolor=COLORS["bg"], bbox_inches="tight")
plt.close(fig)
print(f"Saved chart to {out_path}")
