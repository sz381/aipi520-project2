"""Step 1: parse every ClinicalTrials.gov XML into one parquet table.

Keeps interventional studies only (that's where Phase 1-4 drug trials live).
Free-text fields (summary, eligibility criteria) are kept only for Phase 1-3 rows
to keep the table small.

Run:  python src/parse_trials.py        (~1 min on 10 cores)
Out:  data/trials.parquet
"""
from __future__ import annotations

import glob
import json
import time
import xml.etree.ElementTree as ET
from multiprocessing import Pool

import pandas as pd

from common import RAW_DIR, TRIALS_PARQUET, parse_age_years, parse_date


def _t(root, path):
    el = root.find(path)
    return el.text.strip() if el is not None and el.text else None


def _all(root, path):
    return [e.text.strip() for e in root.findall(path) if e.text and e.text.strip()]


TEXT_PHASES = {"Phase 1", "Phase 1/Phase 2", "Phase 2", "Phase 2/Phase 3", "Phase 3"}
def parse_file(path: str) -> dict | None:
    root = ET.parse(path).getroot()
    if _t(root, "study_type") != "Interventional":
        return None
    phase = _t(root, "phase") or "N/A"

    interventions = []
    for iv in root.findall("intervention"):
        interventions.append({
            "type": _t(iv, "intervention_type"),
            "name": _t(iv, "intervention_name"),
            "other_names": _all(iv, "other_name"),
            "arms": _all(iv, "arm_group_label"),
        })
    arms = [{"label": _t(a, "arm_group_label"), "type": _t(a, "arm_group_type")}
            for a in root.findall("arm_group")]

    enrollment = _t(root, "enrollment")
    enr_el = root.find("enrollment")
    countries = _all(root, "location_countries/country")
    keep_text = phase in TEXT_PHASES

    return {
        "nct_id": _t(root, "id_info/nct_id"),
        "brief_title": _t(root, "brief_title"),
        "official_title": _t(root, "official_title") if keep_text else None,
        "phase": phase,
        "overall_status": _t(root, "overall_status"),
        "why_stopped": _t(root, "why_stopped"),
        "start_date": parse_date(_t(root, "start_date")),
        "primary_completion_date": parse_date(_t(root, "primary_completion_date")),
        "completion_date": parse_date(_t(root, "completion_date")),
        "first_submitted": parse_date(_t(root, "study_first_submitted")),
        "lead_sponsor": _t(root, "sponsors/lead_sponsor/agency"),
        "lead_sponsor_class": _t(root, "sponsors/lead_sponsor/agency_class"),
        "collaborators": _all(root, "sponsors/collaborator/agency"),
        "collaborator_classes": _all(root, "sponsors/collaborator/agency_class"),
        "responsible_party_type": _t(root, "responsible_party/responsible_party_type"),
        "allocation": _t(root, "study_design_info/allocation"),
        "intervention_model": _t(root, "study_design_info/intervention_model"),
        "primary_purpose": _t(root, "study_design_info/primary_purpose"),
        "masking": _t(root, "study_design_info/masking"),
        "number_of_arms": _t(root, "number_of_arms"),
        "enrollment": float(enrollment) if enrollment and enrollment.isdigit() else None,
        "enrollment_type": enr_el.get("type") if enr_el is not None else None,
        "conditions": _all(root, "condition"),
        "condition_mesh": _all(root, "condition_browse/mesh_term"),
        "intervention_mesh": _all(root, "intervention_browse/mesh_term"),
        "interventions_json": json.dumps(interventions),
        "arms_json": json.dumps(arms),
        "keywords": _all(root, "keyword"),
        "n_primary_outcomes": len(root.findall("primary_outcome")),
        "n_secondary_outcomes": len(root.findall("secondary_outcome")),
        "primary_outcome_measures": _all(root, "primary_outcome/measure") if keep_text else [],
        "gender": _t(root, "eligibility/gender"),
        "min_age_years": parse_age_years(_t(root, "eligibility/minimum_age")),
        "max_age_years": parse_age_years(_t(root, "eligibility/maximum_age")),
        "healthy_volunteers": _t(root, "eligibility/healthy_volunteers"),
        "criteria": _t(root, "eligibility/criteria/textblock") if keep_text else None,
        "brief_summary": _t(root, "brief_summary/textblock") if keep_text else None,
        "n_locations": len(root.findall("location")),
        "countries": countries,
        "has_dmc": _t(root, "oversight_info/has_dmc"),
        "has_results": root.find("clinical_results") is not None,
    }


def main():
    files = sorted(glob.glob(str(RAW_DIR / "NCT*" / "*.xml")))
    print(f"{len(files):,} XML files")
    t0 = time.time()
    rows = []
    with Pool() as pool:
        for i, r in enumerate(pool.imap_unordered(parse_file, files, chunksize=1000)):
            if r is not None:
                rows.append(r)
            if i and i % 100_000 == 0:
                print(f"  {i:,} parsed ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows).sort_values("nct_id").reset_index(drop=True)
    df["number_of_arms"] = pd.to_numeric(df["number_of_arms"], errors="coerce")
    df.to_parquet(TRIALS_PARQUET, index=False)
    print(f"{len(df):,} interventional trials -> {TRIALS_PARQUET} ({time.time() - t0:.0f}s)")
    print(df["phase"].value_counts().to_string())


if __name__ == "__main__":
    main()
