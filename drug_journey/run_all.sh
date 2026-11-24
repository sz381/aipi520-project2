#!/usr/bin/env bash
# Rebuild the whole "new drug's journey" analysis. Run from the drug_journey/ folder.
# Needs: ../data/raw (ClinicalTrials.gov XML) and data/fda/extracted (Drugs@FDA); see README.md.
set -euo pipefail
PY=${PY:-../.venv/bin/python}
$PY src/parse_trials.py      # ~1 min   XML -> data/trials.parquet
$PY src/labels.py            # ~30 s    progression labels for Phase 1/2/3 trials
$PY src/features.py          # ~40 s    feature tables for Phase 1/2/3 cohorts
$PY -I src/fda_labels.py     # ~5 s     FDA first-approval lookup
$PY -I src/journey.py        # ~6-8 min three gates: models, checks, figures
