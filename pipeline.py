"""
pipeline.py
===========
Data loading + the four analytical stages.

Stage 0  load_and_merge()      join outcomes (y) with covariates (X) by district
Stage 1  build_risk()          district risk surface (uses the Bayesian estimates)
Stage 2  lever_analysis()      coverage gaps + expected mortality-rate reduction
Stage 2b feature_importance()  Lasso + RandomForest sanity check (optional, sklearn)
Stage 3  optimize_allocation() equity-constrained greedy benefit/cost allocation
"""

import numpy as np
import pandas as pd
import warnings
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

import config as C


# --------------------------------------------------------------------------
# Stage 0 — load & merge
# --------------------------------------------------------------------------
def _find_col(df, substring):
    """Return the first column whose name contains `substring` (case-insensitive)."""
    hits = [c for c in df.columns if substring.lower() in str(c).lower()]
    if not hits:
        raise KeyError(f"No column matching {substring!r}")
    return hits[0]


def load_outcomes():
    """District-level Bayesian small-area estimates (NMR/IMR/U5MR/TFR), both rounds."""
    rounds = {}
    for sheet, tag in [("NFHS 4", "n4"), ("NFHS 5 ", "n5")]:
        raw = pd.read_excel(C.OUTCOME_FILE, engine="openpyxl",
                            sheet_name=sheet, header=None)
        hdr = raw.iloc[2].tolist()
        df = raw.iloc[3:].copy()
        df.columns = hdr
        df = df[["State_name", "District", "NMR", "IMR", "U5MR", "TFR"]].copy()
        for col in ["NMR", "IMR", "U5MR", "TFR"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")   # 'NC' -> NaN
        df = df.dropna(subset=["District"])
        df["key"] = (df["State_name"].astype(str).str.strip().str.lower()
                     + "|" + df["District"].astype(str).str.strip().str.lower())
        rounds[tag] = df
    return rounds["n4"], rounds["n5"]


def load_covariates():
    """NFHS-5 district factsheet (the feature matrix X)."""
    df = pd.read_excel(C.FACTSHEET_FILE, engine="xlrd", header=0)
    df["State/UT"] = df["State/UT"].astype(str).str.strip()
    df["District Names"] = df["District Names"].astype(str).str.strip()
    df["key"] = (df["State/UT"].str.lower() + "|"
                 + df["District Names"].str.lower())
    return df


def load_and_merge():
    n4, n5 = load_outcomes()
    cov = load_covariates()

    # district names differ slightly between the two state spellings; match on
    # district token first, fall back to the composite key where possible.
    n5d = n5.assign(d=n5["District"].str.strip().str.lower())
    covd = cov.assign(d=cov["District Names"].str.strip().str.lower())
    merged = covd.merge(
        n5d[["d", "NMR", "IMR", "U5MR", "TFR"]].drop_duplicates("d"),
        on="d", how="inner",
    )
    merged = merged.dropna(subset=[C.TARGET_OUTCOME]).reset_index(drop=True)
    print(f"[load] merged districts: {len(merged)} "
          f"(target = {C.TARGET_OUTCOME})")
    return merged, n4, n5


# --------------------------------------------------------------------------
# Stage 1 — risk surface
# --------------------------------------------------------------------------
def build_risk(merged):
    """
    Use the Bayesian small-area estimates directly as the risk surface and add
    a state-level smoothing column (a light stand-in for a full CAR/ICAR spatial
    prior: shrink each district toward its state mean). Also assign risk
    terciles for stratified reporting.
    """
    out = merged.copy()
    y = out[C.TARGET_OUTCOME]
    state_mean = out.groupby("State/UT")[C.TARGET_OUTCOME].transform("mean")
    # simple James-Stein-style shrink toward the state mean (lambda fixed = 0.3)
    lam = 0.3
    out["risk_smoothed"] = (1 - lam) * y + lam * state_mean
    out["risk_tercile"] = pd.qcut(out["risk_smoothed"], 3,
                                  labels=["Low", "Medium", "High"])
    return out


# --------------------------------------------------------------------------
# Stage 1b — deprivation index for the equity constraint
# --------------------------------------------------------------------------
def add_deprivation(merged):
    out = merged.copy()
    z = np.zeros(len(out))
    for substring, sign in C.DEPRIVATION_COMPONENTS:
        col = _find_col(out, substring)
        v = pd.to_numeric(out[col], errors="coerce")
        v = (v - v.mean()) / v.std(ddof=0)          # z-score
        z = z + sign * v.fillna(0).to_numpy()
    out["deprivation"] = z
    out["deprivation_tercile"] = pd.qcut(out["deprivation"], 3,
                                         labels=["Low", "Medium", "High"])
    return out


# --------------------------------------------------------------------------
# Stage 2 — coverage gaps and expected rate reduction per lever
# --------------------------------------------------------------------------
def lever_analysis(merged):
    """
    For each district d and lever k:
        gap_dk      = max(0, target_k - current_coverage_dk)   [percentage points]
        reduction   = U5MR_d * rrr_k * (gap_dk / 100)          [first-order PAF]
    Returns a long-format table of candidate actions.
    """
    rows = []
    y = merged[C.TARGET_OUTCOME].to_numpy()
    for key, (substring, target, rrr) in C.LEVERS.items():
        col = _find_col(merged, substring)
        cur = pd.to_numeric(merged[col], errors="coerce").fillna(0).to_numpy()
        gap = np.clip(target - cur, 0, None)            # points still to close
        reduction = y * rrr * (gap / 100.0)             # expected drop in rate
        for i in range(len(merged)):
            if gap[i] <= 0:
                continue
            rows.append({
                "key": merged["key"].iloc[i],
                "State/UT": merged["State/UT"].iloc[i],
                "District": merged["District Names"].iloc[i],
                "lever": key,
                "current": cur[i],
                "gap": gap[i],
                "benefit": reduction[i],         # expected reduction in the rate
                "cost": gap[i],                  # cost = points of coverage bought
                "deprivation_tercile": merged["deprivation_tercile"].iloc[i],
            })
    actions = pd.DataFrame(rows)
    actions["ratio"] = actions["benefit"] / actions["cost"].replace(0, np.nan)
    return actions


# --------------------------------------------------------------------------
# Stage 2b — feature importance (optional ML sanity check)
# --------------------------------------------------------------------------
def feature_importance(merged):
    """
    Lasso (interpretable, signed) + RandomForest (nonlinear) on the lever
    columns -> a quick check that the prescriptive levers carry signal.
    Returns None if scikit-learn is unavailable.
    """
    try:
        from sklearn.linear_model import LassoCV
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("[feature_importance] scikit-learn not installed; skipping.")
        return None

    cols, labels = [], []
    for key, (substring, _, _) in C.LEVERS.items():
        cols.append(_find_col(merged, substring))
        labels.append(C.LEVER_LABELS[key])
    X = merged[cols].apply(pd.to_numeric, errors="coerce").fillna(
        merged[cols].apply(pd.to_numeric, errors="coerce").mean())
    y = merged[C.TARGET_OUTCOME].to_numpy()

    Xs = StandardScaler().fit_transform(X)
    lasso = LassoCV(cv=5, random_state=C.RANDOM_SEED).fit(Xs, y)
    rf = RandomForestRegressor(n_estimators=400, random_state=C.RANDOM_SEED).fit(X, y)

    return pd.DataFrame({
        "lever": labels,
        "lasso_coef": lasso.coef_,
        "rf_importance": rf.feature_importances_,
    })


# --------------------------------------------------------------------------
# Stage 3 — equity-constrained greedy allocation
# --------------------------------------------------------------------------
def optimize_allocation(actions, budget=None, equity_floor=None):
    """
    Maximize total expected rate-reduction subject to a budget, with a floor
    reserving part of the budget for the most-deprived ('High') tercile.

    Because actions are treated as independent and divisible-at-the-margin,
    sorting by benefit/cost ratio (greedy) is optimal for this knapsack. The
    equity floor is enforced by spending the reserved share on High-deprivation
    actions first, then filling the rest greedily.
    """
    budget = C.BUDGET if budget is None else budget
    equity_floor = C.EQUITY_FLOOR if equity_floor is None else equity_floor

    a = actions.dropna(subset=["ratio"]).sort_values("ratio", ascending=False)
    reserved = budget * equity_floor

    chosen, spent_eq, spent_total = [], 0.0, 0.0
    # 1) spend the reserved share on the most-deprived districts
    for _, r in a[a["deprivation_tercile"] == "High"].iterrows():
        if spent_eq + r["cost"] > reserved:
            continue
        chosen.append(r.name); spent_eq += r["cost"]; spent_total += r["cost"]
    # 2) fill the remainder greedily from everything not already picked
    for idx, r in a.iterrows():
        if idx in chosen:
            continue
        if spent_total + r["cost"] > budget:
            continue
        chosen.append(idx); spent_total += r["cost"]

    plan = a.loc[chosen].copy()
    print(f"[allocate] budget={budget:.0f}  spent={spent_total:.0f}  "
          f"actions={len(plan)}  expected total rate-reduction="
          f"{plan['benefit'].sum():.1f}  "
          f"(equity tercile share of spend="
          f"{plan.loc[plan.deprivation_tercile=='High','cost'].sum()/spent_total:.0%})")
    return plan


def efficiency_frontier(actions, budgets, equity_floor=None):
    """Total expected benefit as a function of budget (for the frontier plot)."""
    out = []
    for b in budgets:
        plan = optimize_allocation(actions, budget=b, equity_floor=equity_floor)
        out.append((b, plan["benefit"].sum()))
    return pd.DataFrame(out, columns=["budget", "benefit"])
