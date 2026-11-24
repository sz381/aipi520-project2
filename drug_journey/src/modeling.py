"""Shared modelling utilities: feature groups, temporal split, model factory, bootstrap CI."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import RANDOM_STATE

GROUPS = ["design", "sponsor", "geo", "disease", "drug", "elig", "time"]
TRAIN_END, VAL_END, TEST_END = 2012, 2014, 2019


# --------------------------------------------------------------------- utils
def feature_cols(df, groups=GROUPS, extra=()):
    cols = [c for c in df.columns if c.split("__")[0] in groups and "__" in c]
    cols += [c for c in extra if c not in cols]
    return [c for c in cols if df[c].nunique(dropna=True) > 1]


def split(df):
    y = df["start_date"].dt.year
    return df[y <= TRAIN_END], df[(y > TRAIN_END) & (y <= VAL_END)], df[(y > VAL_END) & (y <= TEST_END)]


def make_model(kind, params):
    if kind == "logreg":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             LogisticRegression(max_iter=2000, **params))
    if kind == "rf":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                             RandomForestClassifier(n_estimators=500, n_jobs=-1,
                                                    random_state=RANDOM_STATE, **params))
    if kind == "hgb":
        return HistGradientBoostingClassifier(random_state=RANDOM_STATE, **params)
    raise ValueError(kind)




def bootstrap_ci(y, p, n=1000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    aucs, aps = [], []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
        aps.append(average_precision_score(y[idx], p[idx]))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"roc_auc_ci": q(aucs), "pr_auc_ci": q(aps)}
