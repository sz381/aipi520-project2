"""Figure style shared by figures.py: palette, matplotlib defaults, readable feature names."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import RESULTS_DIR

# palette (validated reference instance, light mode)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#a3a29b", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2, "axes.titleweight": "bold",
    "axes.titlesize": 13, "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})

NAMES = {
    "drug__log_prior_trials": "Drug: # earlier trials (any phase)",
    "drug__log_prior_p1": "Drug: # earlier Phase 1 trials",
    "drug__log_prior_p2": "Drug: # earlier Phase 2 trials",
    "drug__log_prior_p3_other_indication": "Drug: # earlier Phase 3 (other indications)",
    "drug__has_prior_p4": "Drug: already in a Phase 4 trial",
    "design__n_secondary_outcomes": "Design: # secondary outcomes",
    "design__n_primary_outcomes": "Design: # primary outcomes",
    "design__n_experimental_arms": "Design: # experimental arms",
    "design__has_placebo": "Design: placebo arm",
    "design__masking_level": "Design: masking level",
    "design__randomized": "Design: randomized",
    "design__number_of_arms": "Design: # arms",
    "design__n_interventions": "Design: # interventions",
    "design__n_drug_interventions": "Design: # drug interventions",
    "design__title_phase2b": "Design: 'Phase 2b' in title",
    "design__title_phase2a": "Design: 'Phase 2a' in title",
    "design__title_extension": "Design: extension / long-term study",
    "geo__log_n_sites": "Geography: # sites",
    "geo__n_countries": "Geography: # countries",
    "geo__has_east_asia": "Geography: East Asian site",
    "geo__has_us": "Geography: US site",
    "disease__log_prior_p3_same_condition": "Disease: # earlier Phase 3 in condition",
    "disease__log_prior_p2_same_condition": "Disease: # earlier Phase 2 in condition",
    "sponsor__log_prior_trials": "Sponsor: # earlier trials",
    "sponsor__log_prior_p2": "Sponsor: # earlier Phase 2",
    "sponsor__log_prior_p3": "Sponsor: # earlier Phase 3",
    "sponsor__known_success_rate": "Sponsor: past Phase 2→3 rate",
    "elig__max_age": "Eligibility: max age",
    "elig__min_age": "Eligibility: min age",
    "elig__n_exclusion_items": "Eligibility: # exclusion criteria",
    "elig__n_inclusion_items": "Eligibility: # inclusion criteria",
    "elig__log_criteria_chars": "Eligibility: criteria length",
    "time__start_year": "Start year",
    "disease__oncology": "Disease: oncology",
    "design__model_crossover": "Design: crossover design",
    "design__has_placebo": "Design: placebo arm",
    "design__number_of_arms": "Design: # arms",
    "drug__log_prior_p2_same_condition": "Drug: # earlier Phase 2 in same condition",
    "drug__age_years": "Drug: years since its first trial",
    "drug__log_recent_2y_trials": "Drug: # trials in the last 2 years",
    "drug__log_n_sponsors_before": "Drug: # sponsors that tested it",
    "drug__log_n_conditions_before": "Drug: # conditions tested",
    "drug__log_prior_own_sponsor": "Drug: # earlier trials by same sponsor",
    "drug__log_prior_other_sponsors": "Drug: # earlier trials by other sponsors",
    "disease__condition_success_rate": "Disease: past Phase 2→3 rate in condition",
    "disease__log_condition_matured_p2": "Disease: # matured Phase 2 in condition",
    "disease__log_recent_p3_same_condition": "Disease: # Phase 3 in condition, last 3 yrs",
    "sponsor__log_recent_3y_trials": "Sponsor: # trials in last 3 years",
    "design__outcome_safety": "Design: primary outcome = safety",
    "design__outcome_hard_efficacy": "Design: primary outcome = hard efficacy",
    "geo__has_europe": "Geography: European site",
    "geo__has_china": "Geography: site in China",
}
GROUP_NAMES = {"design": "Trial design", "sponsor": "Sponsor history", "geo": "Geography",
               "disease": "Disease area", "drug": "Drug history", "elig": "Eligibility",
               "time": "Start year"}
MODEL_NAMES = {"prior_rate": "Base rate (no model)", "logreg": "Logistic regression",
               "rf": "Random forest", "hgb": "Gradient boosting", "text": "Text only (TF-IDF + LR)",
               "blend": "Best model + text blend"}


def pretty(c):
    return NAMES.get(c, c.replace("__", ": ").replace("_", " "))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / name, dpi=200)
    plt.close(fig)
    print("wrote", name)
