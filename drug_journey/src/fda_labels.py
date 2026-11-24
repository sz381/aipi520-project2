"""Step 4: gate-3 label from official FDA approval records (Drugs@FDA).

For every active ingredient / brand name in an NDA or BLA (generics = ANDA are
skipped), take the date of its first original approval.  A Phase 3 trial is
labelled y_fda = 1 if any of its focus drugs got its first FDA approval within
FDA_WINDOW_YEARS after the trial started.  Drugs already approved before the
trial started are flagged (fda_approved_before) so the cohort can drop them.

Input:  data/fda/extracted/*.txt  (Drugs@FDA download, fda.gov, ~6 MB zip)
Out:    data/fda_first_approval.parquet  (key -> first approval date)
        adds y_fda / fda_approved_before / fda_approval_date to a feature table on request
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # run with -I (reads downloaded data)

import pandas as pd

from common import DATA_DIR, normalize_drug

FDA_DIR = DATA_DIR / "fda" / "extracted"
FDA_FIRST_APPROVAL = DATA_DIR / "fda_first_approval.parquet"
FDA_WINDOW_YEARS = 8

# salt / hydrate words FDA appends to the ingredient ("NALDEMEDINE TOSYLATE")
_SALTS = re.compile(
    r"\b(tosylate|mesylate|dimesylate|besylate|camsylate|esylate|edisylate|napsylate|"
    r"hydrochloride|dihydrochloride|monohydrochloride|hcl|hydrobromide|bromide|chloride|"
    r"sulfate|bisulfate|phosphate|diphosphate|nitrate|acetate|diacetate|citrate|fumarate|"
    r"hemifumarate|maleate|tartrate|bitartrate|succinate|lactate|gluconate|malate|benzoate|"
    r"pamoate|xinafoate|trifenatate|oxalate|propionate|dipropionate|valerate|hyclate|"
    r"sodium|disodium|monosodium|potassium|dipotassium|calcium|magnesium|zinc|lysine|"
    r"arginine|meglumine|dimeglumine|tromethamine|olamine|erbumine|"
    r"monohydrate|dihydrate|trihydrate|hemihydrate|sesquihydrate|hydrate|anhydrous)\b")


def fda_key(text):
    if not isinstance(text, str):
        return None
    t = re.sub(r"-[a-z]{4}\b", " ", text.lower())  # biologic suffix: aducanumab-avwa
    return normalize_drug(_SALTS.sub(" ", t))


def build_first_approval():
    rd = lambda f: pd.read_csv(FDA_DIR / f, sep="\t", dtype=str, encoding="latin-1")
    apps, prods, subs = rd("Applications.txt"), rd("Products.txt"), rd("Submissions.txt")
    orig = subs[(subs.SubmissionType == "ORIG") & (subs.SubmissionStatus == "AP")].copy()
    orig["date"] = pd.to_datetime(orig.SubmissionStatusDate, errors="coerce")
    first_by_appl = orig.groupby("ApplNo").date.min()
    p = prods[prods.ApplNo.isin(apps[apps.ApplType.isin(["NDA", "BLA"])].ApplNo)].copy()
    p["date"] = p.ApplNo.map(first_by_appl)
    first = {}
    for name, ing, dt in zip(p.DrugName, p.ActiveIngredient, p.date):
        if pd.isna(dt):
            continue
        keys = {fda_key(name)} | {fda_key(x) for x in str(ing).split(";")}
        for k in keys - {None}:
            if k not in first or dt < first[k]:
                first[k] = dt
    out = pd.DataFrame({"key": list(first), "first_approval": list(first.values())})
    out.to_parquet(FDA_FIRST_APPROVAL, index=False)
    return out


def add_fda_label(df, window_years=FDA_WINDOW_YEARS):
    """Adds fda_approval_date (earliest first approval among focus drugs), y_fda,
    fda_approved_before to a feature table with `focus_keys` and `start_date`."""
    fa = pd.read_parquet(FDA_FIRST_APPROVAL) if FDA_FIRST_APPROVAL.exists() else build_first_approval()
    first = dict(zip(fa.key, fa.first_approval))
    dates = [min((first[k] for k in keys if k in first), default=pd.NaT) for keys in df.focus_keys]
    df = df.copy()
    df["fda_approval_date"] = pd.to_datetime(dates)
    w = pd.Timedelta(days=round(365.25 * window_years))
    df["fda_approved_before"] = df.fda_approval_date < df.start_date
    df["y_fda"] = ((df.fda_approval_date >= df.start_date)
                   & (df.fda_approval_date <= df.start_date + w)).astype(int)
    return df


if __name__ == "__main__":
    fa = build_first_approval()
    print(f"{len(fa):,} approved drug keys -> {FDA_FIRST_APPROVAL}")
    first = dict(zip(fa.key, fa.first_approval))
    for k in ["secukinumab", "semaglutide", "vericiguat", "naldemedine", "valbenazine", "seladelpar",
              "fidaxomicin", "sotagliflozin", "evacetrapib", "solanezumab", "aducanumab"]:
        print(f"  {k:14s} {first[k].date() if k in first else '-'}")
