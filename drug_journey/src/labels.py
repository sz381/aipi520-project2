"""Step 2: label drug trials by "did the drug progress to the next phase?".

y_strict = 1  if a Phase 3 (or Phase 2/3) trial that shares the Phase 2's
              *focus drug* AND at least one condition started within
              LABEL_WINDOW_YEARS after the Phase 2 started.
y_drug   = 1  same, but ignoring the condition (sensitivity analysis).

Focus drug: a non-placebo Drug/Biological intervention that was NOT already
"established" (>= ESTABLISHED_MIN_P3 Phase 3 trials started before this
Phase 2).  Background drugs like cisplatin or metformin are therefore not
used for matching, otherwise "new drug X + cisplatin" would match every
later cisplatin trial.  If every drug in the trial is established (a
repurposing / new-indication study), the Phase 3 must contain *all* of them.

Drug synonyms (code name <-> INN <-> brand, e.g. MK-3475 = pembrolizumab =
Keytruda) are learned from interventions that list `other_name`s.  Only
strong, repeated pairs are kept and they are expanded ONE hop (no
transitive closure -- that chains combination regimens into one giant
component).

The same labeller also builds Phase 1 -> Phase 2+ and Phase 3 -> Phase 4 labels
(Phase 4 = post-marketing study). Phase 1 labels feed gate 1; Phase 3 -> Phase 4 is only used
for the "no Phase 4 yet" flag that defines the gate-3 cohort (gate 3 itself is labelled with FDA data).

Out: data/phase2_labels.parquet, data/phase1_labels.parquet, data/phase3_labels.parquet
"""
from __future__ import annotations

import bisect
import collections
import itertools
import json
from datetime import timedelta

import numpy as np
import pandas as pd

from common import (LABELS_P1_PARQUET, LABELS_P2_PARQUET, LABELS_P3_PARQUET, TRIALS_PARQUET,
                    normalize_condition, normalize_drug)

LABEL_WINDOW_YEARS = 6
ESTABLISHED_MIN_P3 = 10
PHASE3 = {"Phase 3", "Phase 2/Phase 3"}

# MeSH intervention terms that are drug *classes*, not drugs.
_CLASS_WORDS = ("agents", "inhibitors", "antibodies", "vaccines", "drugs", "combinations",
                "hormones", "antagonists", "agonists", "products", "preparations",
                "compounds", "analogs", "derivatives", "blockers", "modulators")
# Condition keys too broad to say "same indication".
GENERIC_CONDITIONS = {
    "neoplasms", "neoplasm", "cancer", "cancers", "tumor", "tumors", "tumour", "carcinoma",
    "adenocarcinoma", "solid tumor", "solid tumors", "solid tumour", "advanced solid tumors",
    "advanced solid tumor", "malignant solid tumor", "neoplasm metastasis", "metastatic cancer",
    "recurrence", "agnosia", "syndrome", "disease", "diseases", "infection", "infections",
    "inflammation", "healthy", "healthy volunteers", "healthy volunteer", "healthy subjects",
    "chronic disease", "disease progression", "pathologic processes", "critical illness",
    "neoplasms by site", "neoplasms second primary", "lymphoma", "leukemia", "pain",
}


def drug_keys_by_intervention(interventions_json):
    """List of key-sets, one per Drug/Biological intervention."""
    out = []
    for iv in json.loads(interventions_json):
        if iv.get("type") not in ("Drug", "Biological"):
            continue
        keys = {k for k in [normalize_drug(iv.get("name"))] +
                [normalize_drug(o) for o in iv.get("other_names", [])] if k}
        if keys:
            out.append(keys)
    return out


def mesh_drug_keys(mesh_terms):
    keys = set()
    for m in mesh_terms:
        if any(w in m.lower() for w in _CLASS_WORDS):
            continue
        k = normalize_drug(m)
        if k:
            keys.add(k)
    return keys


def condition_keys(conditions, mesh):
    keys = {normalize_condition(x) for x in list(conditions) + list(mesh)}
    return {k for k in keys if k and k not in GENERIC_CONDITIONS}


def learn_synonyms(per_trial_ivkeys):
    """One-hop synonym map from name/other_name co-listing."""
    freq, pair = collections.Counter(), collections.Counter()
    for ivkeys in per_trial_ivkeys:
        seen_keys, seen_pairs = set(), set()
        for keys in ivkeys:
            seen_keys |= keys
            ks = sorted(k for k in keys if len(k) >= 5)
            if 2 <= len(ks) <= 4:
                seen_pairs |= set(itertools.combinations(ks, 2))
        freq.update(seen_keys)
        pair.update(seen_pairs)
    syn = collections.defaultdict(set)
    for (a, b), w in pair.items():
        if w >= 2 and w / min(freq[a], freq[b]) >= 0.3:
            syn[a].add(b)
            syn[b].add(a)
    return syn


def load_context():
    """Everything the labeller needs, built once over all interventional trials."""
    cols = ["nct_id", "phase", "start_date", "first_submitted", "interventions_json",
            "intervention_mesh", "conditions", "condition_mesh"]
    df = pd.read_parquet(TRIALS_PARQUET, columns=cols)
    df["t"] = df["start_date"].fillna(df["first_submitted"])
    df = df[df["t"].notna()].reset_index(drop=True)

    ivkeys = df["interventions_json"].map(drug_keys_by_intervention)
    syn = learn_synonyms(ivkeys)
    print(f"synonym map: {len(syn):,} keys with >=1 synonym")
    raw_keys = [set().union(*iv) | mesh_drug_keys(m) if iv else set()
                for iv, m in zip(ivkeys, df["intervention_mesh"])]
    cond_keys = [condition_keys(c, m) for c, m in zip(df["conditions"], df["condition_mesh"])]
    t_arr = df["t"].to_numpy()
    ctx = {"df": df, "ivkeys": ivkeys, "syn": syn, "raw_keys": raw_keys, "cond_keys": cond_keys,
           "t_arr": t_arr}
    # "established" is always judged by the number of earlier Phase 3 trials of the drug
    p3_rows = index_phase(ctx, PHASE3)
    ctx["p3_dates_by_key"] = {k: sorted(t_arr[i] for i in rows) for k, rows in p3_rows.items()}
    return ctx


def index_phase(ctx, phases):
    """drug key -> row indices of trials in `phases`."""
    is_target = ctx["df"]["phase"].isin(phases).to_numpy()
    rows_by_key = collections.defaultdict(list)
    for i in np.flatnonzero(is_target):
        for k in ctx["raw_keys"][i]:
            rows_by_key[k].append(i)
    return rows_by_key


def label_progression(ctx, from_phases, to_phases, window_years=LABEL_WINDOW_YEARS):
    """Label every drug trial in `from_phases` by whether a `to_phases` trial of the
    same focus drug (y_drug) / same drug + condition (y_strict) started within the window."""
    df, ivkeys, syn = ctx["df"], ctx["ivkeys"], ctx["syn"]
    raw_keys, cond_keys, t_arr = ctx["raw_keys"], ctx["cond_keys"], ctx["t_arr"]
    p3_dates_by_key = ctx["p3_dates_by_key"]
    next_rows_by_key = index_phase(ctx, to_phases)

    def n_prior_p3(key, t):
        return bisect.bisect_left(p3_dates_by_key.get(key, []), t)

    window = pd.Timedelta(days=round(365.25 * window_years))
    out = []
    src_idx = df.index[df["phase"].isin(from_phases)]
    for i in src_idx:
        if not ivkeys[i]:
            continue  # no Drug/Biological intervention
        t = t_arr[i]
        # drug groups = each raw key + its one-hop synonyms
        groups = {}
        for k in raw_keys[i]:
            g = {k} | syn.get(k, set())
            est = max(n_prior_p3(x, t) for x in g)
            groups[k] = (g, est >= ESTABLISHED_MIN_P3)
        focus = [g for g, est in groups.values() if not est]
        mode = "novel" if focus else "established"
        if not focus:
            focus = [g for g, _ in groups.values()]

        # candidate next-phase rows
        cand = set()
        if mode == "novel":
            for g in focus:
                for k in g:
                    cand.update(next_rows_by_key.get(k, ()))
        else:
            sets = []
            for g in focus:
                s = set()
                for k in g:
                    s.update(next_rows_by_key.get(k, ()))
                sets.append(s)
            cand = set.intersection(*sets) if sets else set()
        cand.discard(i)

        ck = cond_keys[i]
        strict, drug_only, prior_same, prior_any = [], [], False, False
        for j in cand:
            same_cond = bool(ck & cond_keys[j]) if ck else (mode == "novel")
            if t_arr[j] < t:
                prior_any = True
                if same_cond:
                    prior_same = True
                continue
            if t_arr[j] > t + window:
                continue
            drug_only.append(j)
            if same_cond:
                strict.append(j)
        strict.sort(key=lambda j: (t_arr[j], j))  # ties broken by row -> deterministic
        drug_only.sort(key=lambda j: (t_arr[j], j))
        out.append({
            "nct_id": df.at[i, "nct_id"],
            "drug_mode": mode,
            "n_drug_keys": len(groups),
            "focus_keys": sorted({k for g in focus for k in g})[:20],
            "prior_next_same_condition": prior_same,
            "prior_next_any": prior_any,
            "y_strict": int(bool(strict)),
            "y_drug": int(bool(drug_only)),
            "first_next_start": t_arr[strict[0]] if strict else pd.NaT,
            "first_next_nct": df.at[strict[0], "nct_id"] if strict else None,
            "first_next_any_nct": df.at[drug_only[0], "nct_id"] if drug_only else None,
            "matched_next": [df.at[j, "nct_id"] for j in strict[:5]],
        })
    return pd.DataFrame(out)


def summarize(lab, name):
    print(f"{name}: {len(lab):,} drug trials | y_strict {lab.y_strict.mean():.3f} y_drug {lab.y_drug.mean():.3f}"
          f" | established {(lab.drug_mode == 'established').mean():.3f}"
          f" | prior next-phase same condition {lab.prior_next_same_condition.mean():.3f}"
          f" any {lab.prior_next_any.mean():.3f}")


def main():
    ctx = load_context()
    # main task: Phase 2 -> Phase 3 (column aliases keep the downstream scripts unchanged)
    lab = label_progression(ctx, {"Phase 2"}, PHASE3)
    lab = lab.assign(prior_p3_same_condition=lab.prior_next_same_condition,
                     first_p3_start=lab.first_next_start, first_p3_nct=lab.first_next_nct,
                     matched_p3=lab.matched_next)
    lab.to_parquet(LABELS_P2_PARQUET, index=False)
    summarize(lab, "Phase 2 -> Phase 3")
    # extra topics: Phase 1 -> Phase 2 (or later), Phase 3 -> Phase 4 (post-marketing = approved)
    lab1 = label_progression(ctx, {"Phase 1"}, {"Phase 2", "Phase 2/Phase 3", "Phase 3"})
    lab1.to_parquet(LABELS_P1_PARQUET, index=False)
    summarize(lab1, "Phase 1 -> Phase 2+")
    lab3 = label_progression(ctx, {"Phase 3"}, {"Phase 4"})
    lab3.to_parquet(LABELS_P3_PARQUET, index=False)
    summarize(lab3, "Phase 3 -> Phase 4")


if __name__ == "__main__":
    main()
