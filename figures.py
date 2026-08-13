"""
figures.py
==========
All plots. Pure matplotlib (no seaborn dependency). Every function takes a
dataframe and writes a PNG into config.FIG_DIR, and also returns the figure
so you can plt.show() interactively if you prefer.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # comment this out if you want interactive windows
import matplotlib.pyplot as plt
from scipy import stats

import config as C

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10})


def _save(fig, name):
    path = C.FIG_DIR / name
    fig.savefig(path)
    print(f"[fig] wrote {path}")
    return fig


# 1 — distribution of the three mortality outcomes -------------------------
def fig_distributions(merged):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, col, color in zip(axes, ["NMR", "IMR", "U5MR"],
                              ["#4C72B0", "#DD8452", "#C44E52"]):
        v = merged[col].dropna()
        ax.hist(v, bins=30, color=color, alpha=.85, edgecolor="white")
        ax.axvline(v.median(), ls="--", c="k", lw=1)
        ax.set_title(f"{col}  (median {v.median():.1f})")
        ax.set_xlabel("deaths per 1,000")
    axes[0].set_ylabel("districts")
    fig.suptitle("District-level mortality across India (NFHS-5 Bayesian estimates)",
                 fontweight="bold")
    return _save(fig, "01_distributions.png")


# 2 — top burden districts -------------------------------------------------
def fig_top_burden(merged, n=20):
    top = merged.nlargest(n, C.TARGET_OUTCOME)
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(n)[::-1]
    ax.barh(y, top[C.TARGET_OUTCOME], color="#C44E52")
    ax.set_yticks(y)
    ax.set_yticklabels(top["District Names"] + " (" + top["State/UT"].str[:8] + ")",
                       fontsize=8)
    ax.set_xlabel(f"{C.TARGET_OUTCOME} (per 1,000)")
    ax.set_title(f"Top {n} highest-burden districts", fontweight="bold")
    return _save(fig, "02_top_burden.png")


# 3 — correlation of outcome with modifiable levers ------------------------
def fig_lever_correlations(merged):
    from pipeline import _find_col
    rows = []
    for key, (substring, _, _) in C.LEVERS.items():
        col = _find_col(merged, substring)
        r = merged[C.TARGET_OUTCOME].corr(pd.to_numeric(merged[col], errors="coerce"))
        rows.append((C.LEVER_LABELS[key], r))
    df = pd.DataFrame(rows, columns=["lever", "r"]).sort_values("r")
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#4C72B0" if x < 0 else "#C44E52" for x in df["r"]]
    ax.barh(df["lever"], df["r"], color=colors)
    ax.axvline(0, c="k", lw=.8)
    ax.set_xlabel(f"Pearson r with district {C.TARGET_OUTCOME}")
    ax.set_title("Association of modifiable levers with mortality\n"
                 "(ecological — sign of immunization is a confounding artifact)",
                 fontweight="bold", fontsize=10)
    return _save(fig, "03_lever_correlations.png")


# 4 — scatter with fitted line for the strongest social lever --------------
def fig_scatter(merged, substring="Women (age 15-49) who are literate",
                label="Female literacy (%)"):
    from pipeline import _find_col
    col = _find_col(merged, substring)
    x = pd.to_numeric(merged[col], errors="coerce")
    y = merged[C.TARGET_OUTCOME]
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    sl, ic, r, p, _ = stats.linregress(x, y)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(x, y, s=14, alpha=.5, color="#4C72B0")
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, ic + sl * xs, c="k", lw=2)
    ax.set_xlabel(label); ax.set_ylabel(f"{C.TARGET_OUTCOME} (per 1,000)")
    ax.set_title(f"{label} vs {C.TARGET_OUTCOME}   (r = {r:.2f})",
                 fontweight="bold")
    return _save(fig, "04_scatter_literacy.png")


# 5 — progress NFHS-4 -> NFHS-5 by state -----------------------------------
def fig_progress(n4, n5):
    a = n4.groupby("State_name")[C.TARGET_OUTCOME].mean()
    b = n5.groupby("State_name")[C.TARGET_OUTCOME].mean()
    df = pd.concat([a, b], axis=1, keys=["n4", "n5"]).dropna()
    df = df.sort_values("n5")
    fig, ax = plt.subplots(figsize=(7, 9))
    yy = np.arange(len(df))
    ax.hlines(yy, df["n5"], df["n4"], color="#cccccc", lw=2, zorder=1)
    ax.scatter(df["n4"], yy, color="#DD8452", label="NFHS-4", zorder=2, s=22)
    ax.scatter(df["n5"], yy, color="#4C72B0", label="NFHS-5", zorder=2, s=22)
    ax.set_yticks(yy); ax.set_yticklabels(df.index, fontsize=7)
    ax.set_xlabel(f"state-mean {C.TARGET_OUTCOME}")
    ax.set_title(f"Change in {C.TARGET_OUTCOME}, NFHS-4 → NFHS-5", fontweight="bold")
    ax.legend()
    return _save(fig, "05_progress_n4_n5.png")


# 6 — feature importance (Lasso + RF) --------------------------------------
def fig_feature_importance(fi):
    if fi is None:
        return None
    fi = fi.sort_values("rf_importance")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].barh(fi["lever"], fi["lasso_coef"], color="#55A868")
    axes[0].axvline(0, c="k", lw=.8)
    axes[0].set_title("Lasso coefficient (signed)")
    axes[1].barh(fi["lever"], fi["rf_importance"], color="#8172B3")
    axes[1].set_title("Random-forest importance")
    fig.suptitle("Which levers carry signal for "
                 f"{C.TARGET_OUTCOME}?  (ML sanity check)", fontweight="bold")
    return _save(fig, "06_feature_importance.png")


# 7 — coverage-gap heatmap-ish: mean gap by lever --------------------------
def fig_coverage_gaps(actions):
    g = (actions.groupby("lever")["gap"].mean()
         .rename(index=C.LEVER_LABELS).sort_values())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(g.index, g.values, color="#937860")
    ax.set_xlabel("mean coverage gap to target (percentage points)")
    ax.set_title("Average remaining gap to target, by lever", fontweight="bold")
    return _save(fig, "07_coverage_gaps.png")


# 8 — allocation result: spend & benefit by state --------------------------
def fig_allocation(plan):
    by_state = (plan.groupby("State/UT")
                .agg(benefit=("benefit", "sum"), cost=("cost", "sum"))
                .sort_values("benefit").tail(20))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(by_state.index, by_state["benefit"], color="#4C72B0")
    ax.set_xlabel("expected total rate-reduction delivered")
    ax.set_title("Optimised plan: expected benefit by state (top 20)",
                 fontweight="bold")
    return _save(fig, "08_allocation_by_state.png")


# 9 — efficiency frontier (with vs without equity floor) -------------------
def fig_frontier(frontier_eq, frontier_noeq):
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(frontier_noeq["budget"], frontier_noeq["benefit"],
            "-", lw=4, color="#C44E52", alpha=.6, label="pure efficiency")
    ax.plot(frontier_eq["budget"], frontier_eq["benefit"],
            "--o", ms=4, lw=1.6, color="#4C72B0",
            label=f"with {int(C.EQUITY_FLOOR*100)}% equity floor")
    # the two curves nearly coincide here: in this data the most-deprived
    # districts also have the best benefit/cost ratios, so the equity floor is
    # almost 'free'. That overlap is itself a result worth reporting.
    ax.set_xlabel("budget (coverage points)")
    ax.set_ylabel("expected total rate-reduction")
    ax.set_title("Efficiency vs equity frontier", fontweight="bold")
    ax.legend()
    return _save(fig, "09_efficiency_frontier.png")
