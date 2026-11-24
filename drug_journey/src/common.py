"""Shared paths and small helpers for the drug_journey pipeline."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent                        # drug_journey/
REPO = ROOT.parent                       # repository root
RAW_DIR = REPO / "data" / "raw"          # ClinicalTrials.gov AllPublicXML, unzipped
DATA_DIR = ROOT / "data"                 # derived tables (gitignored)
RESULTS_DIR = ROOT / "results"           # figures + metrics
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

TRIALS_PARQUET = DATA_DIR / "trials.parquet"
LABELS_P2_PARQUET = DATA_DIR / "phase2_labels.parquet"
LABELS_P1_PARQUET = DATA_DIR / "phase1_labels.parquet"
LABELS_P3_PARQUET = DATA_DIR / "phase3_labels.parquet"
FEATURES_P2_PARQUET = DATA_DIR / "features_phase2.parquet"
FEATURES_P1_PARQUET = DATA_DIR / "features_phase1.parquet"
FEATURES_P3_PARQUET = DATA_DIR / "features_phase3.parquet"

RANDOM_STATE = 520


def parse_date(text: str | None):
    """ClinicalTrials.gov dates look like 'May 21, 2013' or 'May 2013'."""
    if not text:
        return None
    text = text.strip()
    for fmt in ("%B %d, %Y", "%B %Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def parse_age_years(text: str | None):
    """'18 Years', '6 Months', 'N/A' -> float years (None if missing)."""
    if not text or text.strip() in {"N/A", ""}:
        return None
    m = re.match(r"\s*([\d.]+)\s*(Year|Month|Week|Day|Hour|Minute)", text)
    if not m:
        return None
    val, unit = float(m.group(1)), m.group(2)
    return val / {"Year": 1, "Month": 12, "Week": 52.18, "Day": 365.25,
                  "Hour": 8766, "Minute": 525960}[unit]


# ---------------------------------------------------------------- drug names
# Words that describe formulation / route / dose rather than the molecule.
_FORM_WORDS = (
    r"tablets?|capsules?|injections?|injectable|infusion|oral|solution|suspension|"
    r"intravenous|iv|subcutaneous|sc|intramuscular|im|topical|cream|gel|ointment|"
    r"patch|spray|inhalation|inhaler|powder|extended[- ]release|er|xr|sr|"
    r"film[- ]coated|for injection|dose|doses|dosing|low|high|mg|mcg|ug|µg|ml|kg|"
    r"hydrochloride|hcl|sodium|potassium|mesylate|maleate|sulfate|acetate|"
    r"phosphate|tartrate|citrate|besylate|fumarate|succinate|bromide|calcium"
)
_FORM_RE = re.compile(rf"\b(?:{_FORM_WORDS})\b")
_DOSE_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|ug|µg|g|ml|%|iu|units?|mg/kg|mg/m2)\b")
_PAREN_RE = re.compile(r"\([^)]*\)|\[[^\]]*\]")

# Interventions that are never "the drug being developed".
GENERIC_DRUG_KEYS = {
    "placebo", "placebos", "matchingplacebo", "placebocomparator", "saline", "normalsaline",
    "vehicle", "sham", "standardofcare", "soc", "bestsupportivecare", "usualcare",
    "control", "chemotherapy", "standardchemotherapy", "treatment", "drug", "medication",
    "studydrug", "investigationalproduct", "activecomparator", "comparator", "water",
    "sterilewater", "dextrose", "glucose", "sugarpill", "notreatment", "observation",
    "vaccine", "vaccines", "antibody", "antibodies", "radiotherapy", "radiation", "surgery",
    "pharmacologicalstudy", "laboratorybiomarkeranalysis", "qualityoflifeassessment",
    "questionnaireadministration", "pharmacokineticstudy",
}
_PLACEBO_RE = re.compile(r"placebo|sham|vehicle|saline")


def normalize_drug(name: str | None) -> str | None:
    """Collapse a free-text intervention name into a matching key.

    'ABT-199 (Venetoclax) 400 mg tablets' -> 'abt199'
    Returns None for empty / placebo-like / generic names.
    """
    if not name:
        return None
    s = name.lower()
    if _PLACEBO_RE.search(s):
        return None
    s = _PAREN_RE.sub(" ", s)
    s = _DOSE_RE.sub(" ", s)
    s = _FORM_RE.sub(" ", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    if len(s) < 3 or s in GENERIC_DRUG_KEYS or s.isdigit():
        return None
    return s


def normalize_condition(name: str | None) -> str | None:
    if not name:
        return None
    s = name.lower()
    s = _PAREN_RE.sub(" ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s or None
