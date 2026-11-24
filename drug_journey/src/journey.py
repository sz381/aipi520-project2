"""Step 5: "a new drug's journey" -- one pipeline, three gates.

  Stage 1  Phase 1 -> Phase 2+   industry Phase 1 of a drug never in Phase 2+ before
                                 label: a Phase 2/3 trial of the drug starts within 6 years
  Stage 2  Phase 2 -> Phase 3    industry Phase 2 of a drug never in Phase 3 before
                                 label: a Phase 3 trial of the drug starts within 6 years
  Stage 3  Phase 3 -> FDA        industry Phase 3 (US site, not a vaccine) of a drug not yet
                                 FDA-approved and never in Phase 4
                                 label: first FDA approval (Drugs@FDA) within 8 years

Every stage gets the same battery, all choices made on VAL, test scored once:
  label sanity, LR / RF / GB (+ TF-IDF text blend), bootstrap CI, hit rates,
  permutation importance, feature-group ablation, random-vs-temporal split, oncology subgroup.

Run:  python -I src/journey.py
Out:  results/journey.json, results/journey_*.csv, results/stage2_vs_top.json (if TOP files exist),
      results/fig1-5_*.png and slide_fig*.png (via figures.py)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold

import fda_labels as FDA
import modeling as T
from common import (DATA_DIR, FEATURES_P1_PARQUET, FEATURES_P2_PARQUET, FEATURES_P3_PARQUET, RANDOM_STATE,
                    RESULTS_DIR)


GRIDS = {
    "logreg": {"C": [0.01, 0.03, 0.1, 0.3]},
    "rf": {"min_samples_leaf": [5, 20], "max_features": ["sqrt", 0.3]},
    "hgb": {"learning_rate": [0.03, 0.1], "max_iter": [150, 400], "max_leaf_nodes": [7, 15],
            "min_samples_leaf": [40, 100], "l2_regularization": [1.0]},
}
STOP = {"efficacy / futility": r"futil|efficac|lack of (benefit|response|effect)|did not meet|endpoint|interim",
        "safety": r"safety|toxic|adverse",
        "business / strategic": r"business|strateg|sponsor decision|commercial|portfolio|company decision|"
                                r"funding|financial|priorit",
        "slow enrollment": r"enrol|accrual|recruit"}


def stop_reason(s):
    if not isinstance(s, str):
        return "not given"
    for k, p in STOP.items():
        if re.search(p, s.lower()):
            return k
    return "other"


def first_in_class(df):
    """Drug never seen in Phase 3 / Phase 4 before (and not an established background drug)."""
    return df[(df.drug_mode == "novel") & (df.drug__log_prior_p3_other_indication == 0)
              & (df.drug__has_prior_p4 == 0)]


def is_vaccine(df):
    return (df.focus_keys.map(lambda ks: any("vaccin" in x for x in ks))
            | df.text.str.lower().str.contains("vaccin"))


def stages():
    P1, P2, P3 = (pd.read_parquet(p) for p in (FEATURES_P1_PARQUET, FEATURES_P2_PARQUET, FEATURES_P3_PARQUET))
    s1 = first_in_class(P1[(P1.lead_sponsor_class == "Industry") & (P1.phase == "Phase 1")
                              & ~P1.prior_next_any]).assign(y=lambda d: d.y_drug)
    s2 = first_in_class(P2[(P2.lead_sponsor_class == "Industry")]).assign(y=lambda d: d.y_drug)
    s3 = P3[(P3.lead_sponsor_class == "Industry") & (P3.phase == "Phase 3") & (P3.drug_mode == "novel")
            & ~P3.prior_next_any & (P3.drug__has_prior_p4 == 0) & (P3.start_date.dt.year <= 2017)
            & (P3.geo__has_us == 1)]
    s3 = FDA.add_fda_label(s3)
    s3 = s3[~s3.fda_approved_before & ~is_vaccine(s3)].assign(y=lambda d: d.y_fda)
    return {"stage1_P1_to_P2": s1, "stage2_P2_to_P3": s2, "stage3_P3_to_FDA": s3}


def tune(kind, tr, va, cols):
    best = (-1, None)
    for p in ParameterGrid(GRIDS[kind]):
        m = T.make_model(kind, p).fit(tr[cols], tr.y)
        a = roc_auc_score(va.y, m.predict_proba(va[cols])[:, 1])
        if a > best[0]:
            best = (a, p)
    return best


def text_probs(train, target, C):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=50000, sublinear_tf=True,
                          stop_words="english")
    m = LogisticRegression(C=C, max_iter=3000).fit(vec.fit_transform(train.text), train.y)
    return m.predict_proba(vec.transform(target.text))[:, 1]


def hit_rates(y, p):
    o = np.argsort(-p)
    n = len(p)
    q = np.array_split(np.asarray(y)[o], 5)
    return {"top10": float(np.asarray(y)[o[: n // 10]].mean()),
            "bottom20": float(np.asarray(y)[o[-n // 5:]].mean()),
            "by_quintile": [float(x.mean()) for x in q]}


def analyse(name, d):
    print(f"\n=== {name}: n={len(d):,}, pos={d.y.mean():.3f}", flush=True)
    tr, va, te = T.split(d)
    trva = pd.concat([tr, va])
    cols = T.feature_cols(d)
    R = {"n": len(d), "n_train": len(tr), "n_val": len(va), "n_test": len(te),
         "pos_rate": float(d.y.mean()), "pos_rate_test": float(te.y.mean()), "n_features": len(cols)}

    # label sanity
    st = d.groupby("overall_status").y.agg(["mean", "size"])
    R["label_by_status"] = {k: {"rate": float(v["mean"]), "n": int(v["size"])} for k, v in st.iterrows()}
    term = d[d.overall_status == "Terminated"].assign(reason=lambda x: x.why_stopped.map(stop_reason))
    R["terminated_by_reason"] = {k: {"rate": float(v["mean"]), "n": int(v["size"])}
                                 for k, v in term.groupby("reason").y.agg(["mean", "size"]).iterrows()}

    # models (val-tuned), refit on train+val, score test
    val_p, test_p, params = {}, {}, {}
    for kind in ["logreg", "rf", "hgb"]:
        a, p = tune(kind, tr, va, cols)
        params[kind] = p
        val_p[kind] = T.make_model(kind, p).fit(tr[cols], tr.y).predict_proba(va[cols])[:, 1]
        test_p[kind] = T.make_model(kind, p).fit(trva[cols], trva.y).predict_proba(te[cols])[:, 1]
    best_c = max([0.03, 0.1, 0.3], key=lambda C: roc_auc_score(va.y, text_probs(tr, va, C)))
    val_p["text"], test_p["text"] = text_probs(tr, va, best_c), text_probs(trva, te, best_c)
    base = max(["rf", "hgb"], key=lambda k: roc_auc_score(va.y, val_p[k]))
    w = max([0.0, 0.2, 0.3, 0.4], key=lambda w: roc_auc_score(va.y, (1 - w) * val_p[base] + w * val_p["text"]))
    val_p["final"] = (1 - w) * val_p[base] + w * val_p["text"]
    test_p["final"] = (1 - w) * test_p[base] + w * test_p["text"]
    R["final_model"] = f"{base} + text (w={w})" if w else base
    R["models"] = {}
    for k in ["logreg", "rf", "hgb", "text", "final"]:
        R["models"][k] = {"val_auc": roc_auc_score(va.y, val_p[k]), "test_auc": roc_auc_score(te.y, test_p[k]),
                          "test_pr_auc": average_precision_score(te.y, test_p[k]),
                          "brier": brier_score_loss(te.y, np.clip(test_p[k], 0, 1))}
    R["models"]["final"].update(T.bootstrap_ci(te.y.values, test_p["final"]))
    R["hit_rates"] = hit_rates(te.y.values, test_p["final"])
    print(f"  final {R['final_model']}: test AUC {R['models']['final']['test_auc']:.3f} "
          f"[{R['models']['final']['roc_auc_ci'][0]:.3f}-{R['models']['final']['roc_auc_ci'][1]:.3f}] "
          f"| LR {R['models']['logreg']['test_auc']:.3f} RF {R['models']['rf']['test_auc']:.3f} "
          f"GB {R['models']['hgb']['test_auc']:.3f} text {R['models']['text']['test_auc']:.3f} "
          f"| top10% {R['hit_rates']['top10']:.2f} vs base {te.y.mean():.2f}", flush=True)

    # interpretation: permutation importance + group ablation on the structured model
    m = T.make_model(base, params[base]).fit(trva[cols], trva.y)
    pi = permutation_importance(m, te[cols], te.y, scoring="roc_auc", n_repeats=10,
                                random_state=RANDOM_STATE, n_jobs=-1)
    imp = pd.DataFrame({"feature": cols, "importance": pi.importances_mean, "std": pi.importances_std}) \
        .sort_values("importance", ascending=False)
    imp.to_csv(RESULTS_DIR / f"journey_{name}_importance.csv", index=False)
    R["top_features"] = imp.head(10).to_dict("records")
    full = R["models"][base]["test_auc"]
    R["ablation_change_when_removed"] = {}
    for g in T.GROUPS:
        c = T.feature_cols(d, [x for x in T.GROUPS if x != g])
        p_ = T.make_model(base, params[base]).fit(trva[c], trva.y).predict_proba(te[c])[:, 1]
        R["ablation_change_when_removed"][g] = roc_auc_score(te.y, p_) - full

    # leakage check: random 5-fold CV vs temporal test
    data = pd.concat([tr, va, te])
    oof = np.zeros(len(data))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE).split(data, data.y):
        oof[b] = T.make_model(base, params[base]).fit(data[cols].iloc[a], data.y.iloc[a]) \
            .predict_proba(data[cols].iloc[b])[:, 1]
    R["random_cv_auc"], R["temporal_auc_structured"] = roc_auc_score(data.y, oof), full

    # drugs never seen in train/val: rules out memorising drug names / shared drug-level labels
    seen = {k_ for ks in trva.focus_keys for k_ in ks}
    new = ~te.focus_keys.map(lambda ks: any(k_ in seen for k_ in ks)).values
    R["unseen_drugs"] = {"n_test": int(new.sum()), "pos_rate": float(te.y[new].mean()),
                         "auc_final": roc_auc_score(te.y[new], test_p["final"][new]),
                         "auc_structured": roc_auc_score(te.y[new], test_p[base][new])}

    # oncology subgroup
    onc = te.disease__oncology.values == 1
    R["oncology"] = {"n_test": int(onc.sum()), "pos_rate": float(te.y[onc].mean()),
                     "auc": roc_auc_score(te.y[onc], test_p["final"][onc])}
    R["non_oncology"] = {"n_test": int((~onc).sum()), "pos_rate": float(te.y[~onc].mean()),
                         "auc": roc_auc_score(te.y[~onc], test_p["final"][~onc])}
    print(f"  unseen drugs: n={R['unseen_drugs']['n_test']} AUC {R['unseen_drugs']['auc_final']:.3f} "
          f"(structured {R['unseen_drugs']['auc_structured']:.3f})", flush=True)
    print(f"  random CV {R['random_cv_auc']:.3f} vs temporal {full:.3f} | oncology {R['oncology']['auc']:.3f} "
          f"(n={onc.sum()}) | top features: {', '.join(imp.feature.head(5))}", flush=True)
    pd.DataFrame({"nct_id": te.nct_id.values, "y": te.y.values, "final": test_p["final"]}) \
        .to_csv(RESULTS_DIR / f"journey_{name}_test_predictions.csv", index=False)
    return R


def compare_stage2_with_top(s2):
    """Stage-2 labels vs the TOP benchmark (HINT paper) trial-success labels, if the files exist."""
    from sklearn.metrics import cohen_kappa_score
    top_dir = DATA_DIR / "top_benchmark"
    files = [top_dir / f"phase_II_{s}.csv" for s in ["train", "valid", "test"]]
    if not all(f.exists() for f in files):
        print("TOP benchmark files not found -> skipping the stage-2 comparison")
        return
    top = pd.concat([pd.read_csv(f, usecols=["nctid", "label", "phase"]) for f in files])
    top = top[top.phase == "phase 2"].drop_duplicates("nctid").rename(columns={"nctid": "nct_id", "label": "top"})
    m = s2.merge(top, on="nct_id")
    out = {"overlap": len(m), "accuracy": float((m.top == m.y).mean()),
           "kappa": float(cohen_kappa_score(m.top, m.y)),
           "P(top=1|ours=1)": float(m.top[m.y == 1].mean()), "P(top=1|ours=0)": float(m.top[m.y == 0].mean())}
    for st in ["Completed", "Terminated"]:
        s = m[m.overall_status == st]
        out[st] = {"n": len(s), "P(top=1|ours=1)": float(s.top[s.y == 1].mean()),
                   "P(top=1|ours=0)": float(s.top[s.y == 0].mean())}
    json.dump(out, open(RESULTS_DIR / "stage2_vs_top.json", "w"), indent=2)
    print(f"stage 2 vs TOP: overlap {out['overlap']:,}, P(TOP success | ours passed) {out['P(top=1|ours=1)']:.2f}")


def main():
    st = stages()
    out = {name: analyse(name, d) for name, d in st.items()}
    json.dump(out, open(RESULTS_DIR / "journey.json", "w"), indent=2, default=float)
    compare_stage2_with_top(st["stage2_P2_to_P3"])
    import figures
    figures.main()
    figures.slide_versions()


if __name__ == "__main__":
    main()
