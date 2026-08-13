# Internship Project — Predict-then-Optimize Targeting of Child-Survival Interventions Across Indian Districts

**Field:** applied data science / health analytics / optimization
**Duration:** 8 weeks (scoped for one intern; extensions noted at the end)
**Prerequisites:** Python, basic statistics, a first course in optimization;
no prior epidemiology assumed.

---

## 1. Background and motivation

Under-five deaths in India are concentrated in a small set of causes and a
small set of places. Two papers frame the problem:

- **Million Death Study (Lancet, 2010)** — *which* deaths to avert. Neonatal:
  prematurity/low birthweight, infections, birth asphyxia. Ages 1–59 months:
  pneumonia and diarrhoea. Burden is strongly gendered and regional.
- **Bango & Ghosh (BMC Public Health, 2023)** — *who* is most at risk. Caste
  and tribe (SC/ST) remain an *independent* risk factor after adjusting for
  socioeconomic status; antenatal care and institutional delivery are
  consistently protective.

No algorithm reduces mortality on its own. What an algorithm *can* do is decide
**where to send a fixed budget of interventions so that the expected number of
averted deaths is maximized, subject to an equity constraint.** That is a
*predict-then-optimize* (prescriptive) problem, and it is the spine of this
project.

## 2. Data

| File | Role | Content |
|---|---|---|
| `hsr271789-sup-0001-supplimentary_tables_final_1_1.xlsx` | outcome `y` | District-level **Bayesian small-area estimates** of NMR, IMR, U5MR, TFR with 95% credible intervals, for NFHS-4 and NFHS-5 (~700 districts). |
| `NFHS_5_India_Districts_Factsheet_Data.xls` | covariates `X` | 707 districts × ~107 indicators (ANC, institutional birth, immunization, sanitation, female literacy, stunting, breastfeeding, …). **Has no mortality column** — which is exactly why the first file is needed. |

The two files join on district and turn into one supervised dataset:
`y` (mortality) + `X` (modifiable + structural covariates).

## 3. Method — the four stages

**Stage 1 — Risk surface.** Use the supplied Bayesian estimates as the district
risk surface and shrink each district toward its state mean (a light stand-in
for a full CAR/ICAR spatial prior; the precision matrix of an ICAR prior *is* a
graph Laplacian — a natural extension for someone with a variational/PDE
background). Output: a smoothed risk value and a risk tercile per district.

**Stage 2 — Lever effects, from the literature not from `X`.** For each
*actionable* lever (ANC-4+, institutional birth, skilled attendance,
immunization, ORS, breastfeeding, sanitation) compute the district coverage gap
to a target and an expected first-order reduction in the rate:
`reduction = U5MR × RRR × gap`. The relative-reduction constants `RRR` come from
the papers and the broader literature — **not** from district correlations,
because those are ecological and confounded. The `03_lever_correlations.png`
figure shows immunization with the *wrong* sign for exactly this reason; keep it
as the motivating example.

**Stage 3 — Equity-constrained allocation.** Choose district-lever actions to
maximize total expected reduction subject to a budget, with a floor reserving a
share of the budget for the most-deprived districts. Because actions are
independent at the margin, a greedy benefit/cost ordering is optimal
(fractional knapsack); the diminishing-returns / submodular version with
interacting levers is the first extension.

**Stage 4 (extension) — Robust allocation.** Carry the credible intervals and
effect-size uncertainty through to a min–max-regret or chance-constrained
allocation. This is the part with genuine research novelty.

## 4. Weekly plan

| Week | Task |
|---|---|
| 1 | Load, clean, join the two files; reproduce `01`–`02` figures. |
| 2 | Exploratory association analysis; build `03`–`04`; write up the ecological-fallacy caveat. |
| 3 | Trend analysis NFHS-4 → NFHS-5 (`05`); state-level narrative. |
| 4 | Lasso + random-forest signal check (`06`); compare with literature signs. |
| 5 | Coverage-gap analysis (`07`); assemble the literature effect-size table with citations. |
| 6 | Implement greedy allocation; produce the plan and `08`. |
| 7 | Efficiency-vs-equity frontier (`09`); sensitivity to `RRR` and `EQUITY_FLOOR`. |
| 8 | Robustness extension + final report. |

## 5. Deliverables

1. Reproducible code (this repo).
2. Nine figures in `figures/`.
3. `targeting_plan.csv` — the chosen district-lever actions.
4. A short report with the four caveats in §6 stated up front.

## 6. Caveats the intern must state up front

- **Ecological fallacy** — both `y` and `X` are district aggregates; a
  district-level association is not an individual causal effect. This is why
  Stage-2 effects are sourced from individual-level/causal studies.
- **Caste is missing at district level** — the factsheet has no SC/ST share, so
  the equity floor uses a deprivation *proxy*. The real version needs a
  Census-2011 district caste layer.
- **Effect sizes are the load-bearing assumption** — the `RRR` constants in
  `config.py` are illustrative; report sensitivity to them.
- **No birth denominator** — the demo optimizes rate-reduction; converting to
  absolute averted deaths needs district birth counts (SRS / Census).

## 7. How to run

```bash
pip install pandas numpy matplotlib scipy xlrd openpyxl scikit-learn
# put the two data files in this folder, then:
python main.py
```

Outputs land in `figures/` and `targeting_plan.csv`. Edit `config.py` to switch
the target outcome (`NMR`/`IMR`/`U5MR`), change the levers/targets/effect sizes,
or move the budget and equity floor.

## 8. Files

```
config.py     paths, levers, illustrative effect sizes, targets, budget
pipeline.py   load/merge + the four stages
figures.py    all nine plots (matplotlib only)
main.py       runs everything
```
