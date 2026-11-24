"""Step 3: build the modelling tables (one row per trial) for Phase 1, 2 and 3 cohorts.

Every feature uses only information available when the trial *started*:
history counts (sponsor / drug / condition) only look at trials that started
earlier, and the sponsor's "known success rate" only counts earlier Phase 2s
whose Phase 3 had already started by then.

Column prefixes define the feature groups used for ablation:
  design__  sponsor__  geo__  disease__  drug__  elig__  time__

Out: data/features_phase1.parquet, features_phase2.parquet, features_phase3.parquet
"""
from __future__ import annotations

import bisect
import collections
import json
import re

import numpy as np
import pandas as pd

import labels as lab_mod  # reuse key builders
from common import (FEATURES_P1_PARQUET, FEATURES_P2_PARQUET, FEATURES_P3_PARQUET, LABELS_P1_PARQUET,
                    LABELS_P2_PARQUET, LABELS_P3_PARQUET, TRIALS_PARQUET)

drug_keys_by_intervention = lab_mod.drug_keys_by_intervention
mesh_drug_keys = lab_mod.mesh_drug_keys
condition_keys = lab_mod.condition_keys

COHORT_START_YEARS = (2005, 2019)
EXCLUDED_PURPOSES = {"Diagnostic", "Basic Science", "Screening", "Health Services Research",
                     "Device Feasibility"}

EUROPE = {"Germany", "France", "United Kingdom", "Spain", "Italy", "Poland", "Belgium",
          "Netherlands", "Czechia", "Denmark", "Hungary", "Russia", "Sweden", "Austria",
          "Switzerland", "Bulgaria", "Greece", "Romania", "Norway", "Ukraine", "Finland",
          "Slovakia", "Portugal", "Serbia", "Ireland", "Croatia", "Lithuania", "Latvia",
          "Estonia", "Slovenia", "Belarus", "Georgia", "Moldova", "Iceland", "Luxembourg"}
EAST_ASIA = {"China", "South Korea", "Japan", "Taiwan", "Hong Kong", "Singapore"}

AREAS = {  # therapeutic area -> regex over condition text + condition MeSH
    "oncology": r"cancer|neoplas|tumou?r|carcinom|lymphom|leukemi|leukaemi|myelom|melanom|sarcom|"
                r"glioblastom|gliom|blastom|metasta|myelodysplas|malignan|mesotheliom",
    "cardiovascular": r"heart|cardi|hypertens|coronary|atrial|arrhythm|thromb|stroke|vascular|"
                      r"angina|myocard|embolism|atheroscl",
    "metabolic": r"diabet|obes|lipid|cholesterol|triglycer|metabolic|insulin|glycem|gout|uric",
    "infectious": r"infect|hiv|hepatitis|influenza|virus|viral|bacter|sepsis|tubercul|malaria|"
                  r"covid|sars|fung|pneumonia|vaccin|herpes|cytomegalo|papilloma",
    "neurology": r"alzheimer|parkinson|epilep|seizure|sclerosis|migraine|neuropath|dementia|"
                 r"huntington|amyotroph|neuralgia|spinal muscular|dystroph",
    "psychiatry": r"schizophren|depress|bipolar|anxiety|adhd|attention deficit|autis|"
                  r"psychos|substance|alcohol|opioid|insomnia|ptsd|stress disorder",
    "respiratory": r"asthma|copd|pulmonary disease|cystic fibrosis|bronch|idiopathic pulmonary|"
                   r"respiratory",
    "immunology": r"rheumatoid|psoria|lupus|crohn|colitis|inflammatory bowel|spondyl|"
                  r"atopic|dermatitis|urticaria|sjogren|vasculitis|arthritis",
    "ophthalmology": r"eye|macular|retin|glaucoma|uveitis|ocular|conjunctiv|keratit",
    "gi_hepatic": r"liver|hepat|cirrhos|steatohep|nash|biliary|pancreatit|gastro|bowel syndrome|"
                  r"constipation|reflux",
    "renal_uro": r"kidney|renal|nephr|bladder|urinar|incontinence|prostat",
    "pain": r"\bpain|analges|anesthe|anaesthe",
}


def norm_sponsor(name):
    if not name:
        return None
    s = re.sub(r"[^a-z0-9 ]+", " ", name.lower())
    drop = {"inc", "ltd", "llc", "co", "corp", "corporation", "company", "gmbh", "ag", "sa", "plc",
            "kk", "pharmaceuticals", "pharmaceutical", "pharma", "limited", "the", "and", "usa",
            "us", "global", "research", "development", "r", "d", "bv", "nv", "srl", "spa"}
    toks = [t for t in s.split() if t not in drop]
    return " ".join(toks) or s.strip()


def count_items(text):
    """Rough count of bullet items in the inclusion / exclusion sections."""
    if not text:
        return np.nan, np.nan
    parts = re.split(r"exclusion criteria", text, maxsplit=1, flags=re.I)
    bullet = re.compile(r"^\s*(?:[-*•·]|\d+[.)])\s+", re.M)
    inc = len(bullet.findall(parts[0]))
    exc = len(bullet.findall(parts[1])) if len(parts) > 1 else 0
    return inc, exc


def _s(x):
    """str or '' (pandas gives NaN floats for missing strings)."""
    return x if isinstance(x, str) else ""


def _num(x, default=np.nan):
    return default if x is None or (isinstance(x, float) and np.isnan(x)) else x


class History:
    """Sorted start dates per key -> 'how many started before t'."""

    def __init__(self):
        self.d = collections.defaultdict(list)

    def add(self, key, t):
        self.d[key].append(t)

    def freeze(self):
        for k in self.d:
            self.d[k].sort()

    def before(self, key, t):
        return bisect.bisect_left(self.d.get(key, ()), t)

    def max_before(self, keys, t):
        return max((self.before(k, t) for k in keys), default=0)


def build(jobs):
    """jobs: list of (cohort_labels, out_path, exclude_prior_next_same_condition)."""
    tr = pd.read_parquet(TRIALS_PARQUET)
    tr["t"] = tr["start_date"].fillna(tr["first_submitted"])
    tr = tr[tr["t"].notna()].reset_index(drop=True)
    lab = pd.read_parquet(LABELS_P2_PARQUET)  # Phase 2 outcomes feed the track-record features

    # ------------------------------------------------ history indices (all trials)
    tr["sponsor_key"] = tr["lead_sponsor"].map(norm_sponsor)
    ivkeys = tr["interventions_json"].map(drug_keys_by_intervention)
    raw_keys = [set().union(*iv) | mesh_drug_keys(m) if iv else set()
                for iv, m in zip(ivkeys, tr["intervention_mesh"])]
    ckeys = [condition_keys(c, m) for c, m in zip(tr["conditions"], tr["condition_mesh"])]

    phase_group = tr["phase"].map({"Phase 1": "p1", "Early Phase 1": "p1", "Phase 1/Phase 2": "p1",
                                   "Phase 2": "p2", "Phase 2/Phase 3": "p3", "Phase 3": "p3",
                                   "Phase 4": "p4"}).fillna("na")
    H = {name: History() for name in
         ["sp_all", "sp_p2", "sp_p3", "dr_all", "dr_p1", "dr_p2", "dr_p3", "dr_p4", "co_p2", "co_p3"]}
    for i, (t, s, pg) in enumerate(zip(tr["t"], tr["sponsor_key"], phase_group)):
        if s:
            H["sp_all"].add(s, t)
            if pg in ("p2", "p3"):
                H[f"sp_{pg}"].add(s, t)
        for k in raw_keys[i]:
            H["dr_all"].add(k, t)
            if pg in ("p1", "p2", "p3", "p4"):
                H[f"dr_{pg}"].add(k, t)
        if pg in ("p2", "p3"):
            for k in ckeys[i]:
                H[f"co_{pg}"].add(k, t)
    for h in H.values():
        h.freeze()

    # sponsor's earlier Phase 2 outcomes that were already *visible* at time t.
    # Only Phase 2s that started >= 3 years before t count (younger ones have
    # not had time to reach Phase 3 and would look like failures).
    lab_t = lab.merge(tr[["nct_id", "t", "sponsor_key"]], on="nct_id")
    sp_pairs = collections.defaultdict(list)
    for s, t, p3 in zip(lab_t["sponsor_key"], lab_t["t"], lab_t["first_p3_start"]):
        if s:
            sp_pairs[s].append((t, p3))
    for s in sp_pairs:
        sp_pairs[s].sort(key=lambda x: x[0])
    sp_p2_starts = {s: [x[0] for x in v] for s, v in sp_pairs.items()}
    mature = pd.Timedelta(days=3 * 365)

    def sponsor_track_record(s, t):
        if s not in sp_pairs:
            return 0, 0
        n = bisect.bisect_left(sp_p2_starts[s], t - mature)
        succ = sum(1 for _, p3 in sp_pairs[s][:n] if pd.notna(p3) and p3 < t)
        return n, succ

    # ---- v2: richer drug history (which trials of this drug existed before t?)
    t_list = tr["t"].tolist()
    sp_arr = tr["sponsor_key"].tolist()
    pg_arr = phase_group.tolist()
    key_rows = collections.defaultdict(list)
    for i, ks in enumerate(raw_keys):
        for k in ks:
            key_rows[k].append(i)
    key_t = {}
    for k, rows_ in key_rows.items():
        rows_.sort(key=lambda j: t_list[j])
        key_t[k] = [t_list[j] for j in rows_]
    two_years, three_years = pd.Timedelta(days=730), pd.Timedelta(days=1095)

    def drug_history(focus, t, i):
        prior, first = set(), None
        for k in focus:
            if k not in key_rows:
                continue
            n = bisect.bisect_left(key_t[k], t)
            if n == 0:
                continue
            first = key_t[k][0] if first is None or key_t[k][0] < first else first
            prior.update(key_rows[k][max(0, n - 3000):n])
        prior.discard(i)
        if not prior:
            return dict(n_sponsors=0, n_conditions=0, age=0.0, recent2y=0, same_cond_p2=0,
                        own_sponsor=0, other_sponsor=0)
        s_i, c_i = sp_arr[i], ckeys[i]
        conds = set()
        for j in prior:
            conds |= ckeys[j]
        own = sum(1 for j in prior if s_i and sp_arr[j] == s_i)
        return dict(n_sponsors=len({sp_arr[j] for j in prior if sp_arr[j]}),
                    n_conditions=len(conds), age=(t - first).days / 365.25,
                    recent2y=sum(1 for j in prior if t_list[j] >= t - two_years),
                    same_cond_p2=sum(1 for j in prior if pg_arr[j] == "p2" and ckeys[j] & c_i),
                    own_sponsor=own, other_sponsor=len(prior) - own)

    # ---- v2: condition track record (earlier Phase 2s in this condition, outcome visible by t)
    nct_pos = dict(zip(tr["nct_id"], range(len(tr))))
    co_pairs = collections.defaultdict(list)
    for nct, t0, p3 in zip(lab_t["nct_id"], lab_t["t"], lab_t["first_p3_start"]):
        for k in ckeys[nct_pos[nct]]:
            co_pairs[k].append((t0, p3))
    co_starts, co_p3_sorted = {}, {}
    for k, v in co_pairs.items():
        v.sort(key=lambda x: x[0])
        co_starts[k] = [x[0] for x in v]
        co_p3_sorted[k] = sorted(x[1] for x in v if pd.notna(x[1]))

    def condition_track_record(keys, t):
        best = (0, 0)
        for k in sorted(keys):  # sorted -> ties resolved the same way every run
            if k not in co_pairs:
                continue
            n = bisect.bisect_left(co_starts[k], t - mature)
            if n <= best[0]:
                continue
            # successes visible by t among Phase 2s started before t - 3y:
            # (all with p3 < t) minus (those started in [t-3y, t) with p3 < t)
            lo, hi = n, bisect.bisect_left(co_starts[k], t)
            young = sum(1 for _, p3 in co_pairs[k][lo:hi] if pd.notna(p3) and p3 < t)
            best = (n, bisect.bisect_left(co_p3_sorted[k], t) - young)
        return best

    def window_count(h, keys, t, width):
        return max((h.before(k, t) - h.before(k, t - width) for k in keys), default=0)

    OUTCOME_FLAGS = {
        "safety": r"safety|adverse|tolerab|toxicit|dose[- ]limiting|maximum tolerated",
        "pk": r"pharmacokinetic|\bauc\b|cmax|plasma concentration|half[- ]life|clearance",
        "hard_efficacy": r"surviv|mortality|death|response rate|objective response|remission|relapse|"
                         r"progression|exacerbation|event",
        "change_from_baseline": r"change from baseline|change in",
        "biomarker": r"biomarker|level|concentration|expression|viral load|hba1c|ldl",
    }

    for cohort_lab, out_path, exclude_prior in jobs:
        # ------------------------------------------------ cohort rows
        d = cohort_lab.merge(tr, on="nct_id")
        d = d[d["overall_status"].isin(["Completed", "Terminated"])
              & d["t"].dt.year.between(*COHORT_START_YEARS)
              & ~d["primary_purpose"].isin(EXCLUDED_PURPOSES)
              & ~(d["prior_next_same_condition"] & exclude_prior)].reset_index(drop=True)
        print(f"{out_path.name}: cohort (all sponsors) {len(d):,}")

        tr_index = pd.Series(range(len(tr)), index=tr["nct_id"])
        rows = []
        for r in d.itertuples(index=False):
            i = tr_index[r.nct_id]
            t = r.t
            ivs = json.loads(r.interventions_json)
            arms = json.loads(r.arms_json)
            arm_types = [a.get("type") or "" for a in arms]
            iv_types = [iv.get("type") or "" for iv in ivs]
            names = " ".join(_s(iv.get("name")) for iv in ivs).lower()
            title = f"{_s(r.brief_title)} {_s(r.official_title)}"
            tl = title.lower()
            cond_text = " ".join(list(r.conditions) + list(r.condition_mesh)).lower()
            masking = _s(r.masking).split(" ")[0]
            countries = set(r.countries)
            inc_items, exc_items = count_items(_s(r.criteria))
            focus = set(r.focus_keys)
            n_prior_sp_p2, n_seen_succ = sponsor_track_record(r.sponsor_key, t)
            dh = drug_history(focus, t, i)
            c_n, c_succ = condition_track_record(ckeys[i], t)
            outcome_text = " ".join(r.primary_outcome_measures).lower()
            v2 = {
                "drug__log_n_sponsors_before": np.log1p(dh["n_sponsors"]),
                "drug__log_n_conditions_before": np.log1p(dh["n_conditions"]),
                "drug__age_years": dh["age"],
                "drug__log_recent_2y_trials": np.log1p(dh["recent2y"]),
                "drug__log_prior_p2_same_condition": np.log1p(dh["same_cond_p2"]),
                "drug__log_prior_own_sponsor": np.log1p(dh["own_sponsor"]),
                "drug__log_prior_other_sponsors": np.log1p(dh["other_sponsor"]),
                "disease__condition_success_rate": (c_succ + 5 * 0.2) / (c_n + 5),
                "disease__log_condition_matured_p2": np.log1p(c_n),
                "disease__log_recent_p3_same_condition": np.log1p(window_count(H["co_p3"], ckeys[i], t, three_years)),
                "sponsor__log_recent_3y_trials": np.log1p(window_count(H["sp_all"], [r.sponsor_key], t, three_years)),
                **{f"design__outcome_{k}": float(bool(re.search(p, outcome_text)))
                   for k, p in OUTCOME_FLAGS.items()},
            }

            f = {
                "nct_id": r.nct_id, "y_strict": r.y_strict, "y_drug": r.y_drug,
                "start_date": t, "lead_sponsor": r.lead_sponsor, "sponsor_key": r.sponsor_key,
                "lead_sponsor_class": r.lead_sponsor_class, "overall_status": r.overall_status,
                "focus_keys": list(focus),
                "phase": r.phase, "why_stopped": r.why_stopped, "drug_mode": r.drug_mode,
                "prior_next_any": r.prior_next_any, "has_results": r.has_results,
                "first_next_any_nct": r.first_next_any_nct,
                # ---- design
                "design__randomized": float(r.allocation == "Randomized"),
                "design__model_parallel": float(r.intervention_model == "Parallel Assignment"),
                "design__model_single_group": float(r.intervention_model == "Single Group Assignment"),
                "design__model_crossover": float(r.intervention_model == "Crossover Assignment"),
                "design__masking_level": {"None": 0, "Single": 1, "Double": 2, "Triple": 3,
                                          "Quadruple": 4}.get(masking, np.nan),
                "design__purpose_treatment": float(r.primary_purpose == "Treatment"),
                "design__purpose_prevention": float(r.primary_purpose == "Prevention"),
                "design__number_of_arms": r.number_of_arms,
                "design__n_experimental_arms": arm_types.count("Experimental"),
                "design__has_placebo": float("Placebo Comparator" in arm_types or "placebo" in names),
                "design__has_active_comparator": float("Active Comparator" in arm_types),
                "design__n_primary_outcomes": r.n_primary_outcomes,
                "design__n_secondary_outcomes": r.n_secondary_outcomes,
                "design__has_dmc": {"Yes": 1.0, "No": 0.0}.get(r.has_dmc, np.nan),
                "design__n_interventions": len(ivs),
                "design__n_drug_interventions": sum(x in ("Drug", "Biological") for x in iv_types),
                "design__has_biological": float("Biological" in iv_types),
                "design__has_non_drug": float(any(x not in ("Drug", "Biological") for x in iv_types)),
                "design__all_drugs_established": float(r.drug_mode == "established"),
                "design__title_phase2b": float(bool(re.search(r"phase\s*(2|ii)\s*b\b", tl))),
                "design__title_phase2a": float(bool(re.search(r"phase\s*(2|ii)\s*a\b", tl))),
                "design__title_dose": float(bool(re.search(r"dose[- ]?(ranging|finding|response)", tl))),
                "design__title_pilot_poc": float(bool(re.search(r"pilot|proof[- ]of[- ]concept|exploratory", tl))),
                "design__title_extension": float(bool(re.search(r"extension|long[- ]term", tl))),
                "design__pi_responsible": float(r.responsible_party_type in ("Principal Investigator",
                                                                            "Sponsor-Investigator")),
                # ---- sponsor
                "sponsor__class_industry": float(r.lead_sponsor_class == "Industry"),
                "sponsor__class_nih_fed": float(r.lead_sponsor_class in ("NIH", "U.S. Fed")),
                "sponsor__n_collaborators": len(r.collaborators),
                "sponsor__has_industry_collab": float("Industry" in list(r.collaborator_classes)),
                "sponsor__log_prior_trials": np.log1p(H["sp_all"].before(r.sponsor_key, t)),
                "sponsor__log_prior_p2": np.log1p(H["sp_p2"].before(r.sponsor_key, t)),
                "sponsor__log_prior_p3": np.log1p(H["sp_p3"].before(r.sponsor_key, t)),
                "sponsor__known_success_rate": (n_seen_succ + 5 * 0.2) / (n_prior_sp_p2 + 5),
                # ---- geography (site list as recorded)
                "geo__log_n_sites": np.log1p(r.n_locations),
                "geo__n_countries": len(countries),
                "geo__has_us": float("United States" in countries),
                "geo__has_europe": float(bool(countries & EUROPE)),
                "geo__has_east_asia": float(bool(countries & EAST_ASIA)),
                "geo__has_china": float("China" in countries),
                # ---- disease
                **{f"disease__{a}": float(bool(re.search(p, cond_text))) for a, p in AREAS.items()},
                "disease__n_conditions": len(r.conditions),
                "disease__log_prior_p3_same_condition": np.log1p(H["co_p3"].max_before(ckeys[i], t)),
                "disease__log_prior_p2_same_condition": np.log1p(H["co_p2"].max_before(ckeys[i], t)),
                # ---- drug history (focus drug, before t)
                "drug__log_prior_trials": np.log1p(H["dr_all"].max_before(focus, t)),
                "drug__log_prior_p1": np.log1p(H["dr_p1"].max_before(focus, t)),
                "drug__log_prior_p2": np.log1p(H["dr_p2"].max_before(focus, t)),
                "drug__log_prior_p3_other_indication": np.log1p(H["dr_p3"].max_before(focus, t)),
                "drug__has_prior_p4": float(H["dr_p4"].max_before(focus, t) > 0),
                # ---- eligibility
                "elig__log_criteria_chars": np.log1p(len(_s(r.criteria))),
                "elig__n_inclusion_items": inc_items,
                "elig__n_exclusion_items": exc_items,
                "elig__min_age": _num(r.min_age_years),
                "elig__max_age": _num(r.max_age_years, 100.0),
                "elig__pediatric": float(_num(r.max_age_years, 100.0) < 18),
                "elig__gender_all": float(r.gender == "All"),
                "elig__healthy_volunteers": float(r.healthy_volunteers == "Accepts Healthy Volunteers"
                                                  or r.healthy_volunteers == "Yes"),
                # ---- time
                "time__start_year": t.year + (t.month - 1) / 12,
                "sponsor__n_matured_prior_p2": np.log1p(n_prior_sp_p2),
                **v2,
                # ---- text
                "text": " ".join([title, _s(r.brief_summary), " ".join(r.primary_outcome_measures)]),
            }
            rows.append(f)
        F = pd.DataFrame(rows)
        F.to_parquet(out_path, index=False)
        feats = [c for c in F.columns if "__" in c]
        print(f"{len(F):,} rows, {len(feats)} features -> {out_path}")
        print(F.groupby("lead_sponsor_class")["y_strict"].agg(["mean", "size"]).round(3).to_string())


def main():
    # Phase 2 table drops trials whose drug already had a Phase 3 in the same condition
    build([(pd.read_parquet(LABELS_P2_PARQUET), FEATURES_P2_PARQUET, True),
           (pd.read_parquet(LABELS_P1_PARQUET), FEATURES_P1_PARQUET, False),
           (pd.read_parquet(LABELS_P3_PARQUET), FEATURES_P3_PARQUET, False)])


if __name__ == "__main__":
    main()
