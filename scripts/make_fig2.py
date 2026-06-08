import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

judges = ["Claude", "Mistral", "FaithJudge"]
# Table 3 (Claude-annotated 2x2), faithful-rate %
cc = [80, 80, 72]   # content:clean
cs = [48, 50, 39]   # content:scrambled
ac = [81, 67, 77]   # attribution:clean
as_ = [4, 4, 3]     # attribution:scrambled

x = np.arange(len(judges)); w = 0.2
# Okabe-Ito colorblind-safe
col = {"cc": "#56B4E9", "cs": "#0072B2", "ac": "#E69F00", "as": "#D55E00"}
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.bar(x-1.5*w, cc, w, label="content: correct", color=col["cc"], edgecolor="black", linewidth=0.4)
ax.bar(x-0.5*w, cs, w, label="content: misattributed", color=col["cs"], edgecolor="black", linewidth=0.4)
ax.bar(x+0.5*w, ac, w, label="attribution: correct", color=col["ac"], edgecolor="black", linewidth=0.4)
ax.bar(x+1.5*w, as_, w, label="attribution: misattributed", color=col["as"], edgecolor="black", linewidth=0.4)
ax.set_ylabel("Faithful-rate (%)")
ax.set_xticks(x); ax.set_xticklabels(judges)
ax.set_ylim(0, 100)
ax.legend(fontsize=8, ncol=2, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.18))
ax.axhline(50, color="grey", lw=0.5, ls=":")
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig("latex/figures/fig2_attribution.pdf", bbox_inches="tight")
print("wrote latex/figures/fig2_attribution.pdf")
