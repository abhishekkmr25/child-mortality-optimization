"""
main.py
=======
Run the whole pipeline end to end and write every figure to ./figures/.

    python main.py

Requires: pandas, numpy, matplotlib, scipy, xlrd, openpyxl
Optional: scikit-learn  (enables the feature-importance figure)
"""

import numpy as np

import config as C
import pipeline as P
import figures as F


def main():
    np.random.seed(C.RANDOM_SEED)

    # ---- Stage 0: load & merge ------------------------------------------
    merged, n4, n5 = P.load_and_merge()

    # ---- Stage 1: risk surface + deprivation index ----------------------
    merged = P.build_risk(merged)
    merged = P.add_deprivation(merged)

    # ---- Stage 2: lever gaps & expected reductions ----------------------
    actions = P.lever_analysis(merged)
    fi = P.feature_importance(merged)        # optional sklearn

    # ---- Stage 3: optimisation ------------------------------------------
    plan = P.optimize_allocation(actions)
    budgets = np.linspace(500, 8000, 12)
    frontier_eq = P.efficiency_frontier(actions, budgets, equity_floor=C.EQUITY_FLOOR)
    frontier_noeq = P.efficiency_frontier(actions, budgets, equity_floor=0.0)

    # ---- Figures --------------------------------------------------------
    F.fig_distributions(merged)
    F.fig_top_burden(merged)
    F.fig_lever_correlations(merged)
    F.fig_scatter(merged)
    F.fig_progress(n4, n5)
    F.fig_feature_importance(fi)
    F.fig_coverage_gaps(actions)
    F.fig_allocation(plan)
    F.fig_frontier(frontier_eq, frontier_noeq)

    # ---- Save the targeting plan ----------------------------------------
    plan_out = C.DATA_DIR / "targeting_plan.csv"
    plan[["State/UT", "District", "lever", "current", "gap",
          "benefit", "deprivation_tercile"]].to_csv(plan_out, index=False)
    print(f"\nDone. Figures in {C.FIG_DIR}/ and plan in {plan_out}")


if __name__ == "__main__":
    main()
