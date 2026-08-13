"""
config.py
=========
Central configuration for the child-survival intervention-targeting project.

Edit the values here rather than hard-coding numbers inside the pipeline.
The two input files must sit in DATA_DIR (by default, the folder that
contains this script). They are exactly the two files you started with:

    NFHS_5_India_Districts_Factsheet_Data.xls          (the covariates X)
    hsr271789-sup-0001-supplimentary_tables_final_1_1.xlsx  (the outcomes y)
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
DATA_DIR   = Path(__file__).resolve().parent       # put the two files here
FIG_DIR    = DATA_DIR / "figures"                  # plots are written here
FIG_DIR.mkdir(exist_ok=True)

FACTSHEET_FILE = DATA_DIR / "NFHS_5_India_Districts_Factsheet_Data.xls"
OUTCOME_FILE   = DATA_DIR / "hsr271789-sup-0001-supplimentary_tables_final_1_1.xlsx"

# Which mortality outcome to target: one of "NMR", "IMR", "U5MR"
TARGET_OUTCOME = "U5MR"

# --------------------------------------------------------------------------
# Modifiable levers
# --------------------------------------------------------------------------
# Each lever maps a short key -> (substring that identifies the factsheet
# column, target coverage we aim to reach, relative reduction in the outcome
# achievable at FULL coverage of this lever).
#
# IMPORTANT — the `rrr` numbers below are ILLUSTRATIVE first-order values,
# loosely anchored to the two papers (Bango & Ghosh adjusted ORs for ANC and
# institutional delivery; MDS cause-attributable fractions for pneumonia /
# diarrhoea / neonatal causes) and to the broad child-survival literature.
# They are the load-bearing assumption of the whole prescriptive layer.
# Replace them with effect sizes from your own causal / RCT / meta-analytic
# review before drawing any policy conclusion.
#
#   key : (column substring,                              target%, rrr)
LEVERS = {
    "anc4":      ("at least 4 antenatal care visits",        90.0, 0.18),
    "institutional": ("Institutional births (in the 5 years", 95.0, 0.20),
    "skilled":   ("Births attended by skilled health",       95.0, 0.16),
    "immun":     ("fully vaccinated based on information from either", 90.0, 0.15),
    "ors":       ("received oral rehydration salts",         80.0, 0.10),
    "ebf":       ("exclusively breastfed",                   70.0, 0.08),
    "sanitation": ("use an improved sanitation facility",    90.0, 0.12),
}

# Human-readable labels for plots
LEVER_LABELS = {
    "anc4": "ANC 4+ visits",
    "institutional": "Institutional birth",
    "skilled": "Skilled birth attendant",
    "immun": "Full immunization (12-23 mo)",
    "ors": "ORS for diarrhoea",
    "ebf": "Exclusive breastfeeding",
    "sanitation": "Improved sanitation",
}

# --------------------------------------------------------------------------
# Deprivation proxy (used for the equity constraint)
# --------------------------------------------------------------------------
# The factsheet has NO caste/tribe share, so a true SC/ST equity floor needs
# an external Census-2011 district layer. As a runnable stand-in we build a
# deprivation index from indicators that ARE present. Each entry is
# (column substring, +1 if higher value = MORE deprived else -1).
DEPRIVATION_COMPONENTS = [
    ("Women (age 15-49) who are literate",                 -1),
    ("use an improved sanitation facility",                -1),
    ("Women age 20-24 years married before age 18",        +1),
    ("Children under 5 years who are stunted",             +1),
    ("clean fuel for cooking",                             -1),
]

# --------------------------------------------------------------------------
# Allocation problem
# --------------------------------------------------------------------------
# Budget is expressed in "coverage points" you are allowed to buy in total
# (sum over chosen district-lever actions of the percentage-point gap closed).
# Tune this to see the efficiency frontier move.
BUDGET = 4000.0

# Fraction of the budget reserved for the most-deprived tercile of districts.
EQUITY_FLOOR = 0.40

RANDOM_SEED = 42
