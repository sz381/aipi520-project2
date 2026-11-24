# A new drug’s journey: predicting clinical-trial success one gate at a time

AIPI 520 · Project 2 · Team 8. This folder holds the code, results, figures and slides for our final
framing. It documents *how to run and where things are*; the course writeup is written separately by the team.

Given an industry-sponsored clinical trial of a drug that has not yet passed the next gate, each model
outputs the probability that the drug passes it:

| Gate | Cohort (industry sponsors, trials started 2005–2019) | Label = 1 if… | n | pass rate | Test ROC-AUC [95% CI] |
|---|---|---|---|---|---|
| Phase 1 → Phase 2 | Phase 1 trials of drugs never in Phase 2+ | a Phase 2 or 3 trial of the drug starts within 6 years | 6,769 | 41% | **0.718** [0.698–0.737] |
| Phase 2 → Phase 3 | Phase 2 trials of drugs never in Phase 3 | a Phase 3 trial of the drug starts within 6 years | 7,041 | 33% | **0.722** [0.699–0.743] |
| Phase 3 → FDA approval | Phase 3 trials with a US site, drug not yet approved, no vaccines (start ≤2017) | the drug’s first FDA approval (Drugs@FDA) within 8 years | 3,114 | 37% | **0.807** [0.778–0.837] |

Temporal split for every gate: train on trials started ≤2012, choose models on 2013–14, test once on 2015+.

## Folder layout

```
drug_journey/
├── run_all.sh              rebuild everything (≈10 min)
├── src/
│   ├── common.py           paths + drug/condition name normalisation
│   ├── parse_trials.py     ClinicalTrials.gov XML -> data/trials.parquet
│   ├── labels.py           drug synonyms, background drugs, next-phase progression labels
│   ├── features.py         ~80 time-safe features per trial (Phase 1/2/3 tables)
│   ├── fda_labels.py       first FDA approval date per drug (Drugs@FDA)
│   ├── modeling.py         feature groups, temporal split, model factory, bootstrap CI
│   ├── journey.py          the three gates: cohorts, models, checks (writes results/)
│   ├── figures.py          fig1–fig5 + larger-type slide versions
│   └── style.py            figure palette and readable feature names
├── results/                journey.json (all numbers), per-gate importance / test predictions, figures
├── notes/                  Chinese walkthroughs for the team (background, full walkthrough)
├── slides/                 link to the deck + its source files
└── data/                   derived tables and downloads (gitignored)
```

## How to run

Requirements: the repo’s `requirements.txt` (Python 3.13, pandas, scikit-learn, pyarrow, matplotlib) installed in `../.venv`.

Data:
1. ClinicalTrials.gov full dump unzipped into `../data/raw/` (`AllPublicXML.zip`, one `NCT*/NCT*.xml` file per trial).
2. Drugs@FDA (official FDA approval records, ~6 MB), for gate 3:
   ```bash
   mkdir -p data/fda/extracted && curl -sSfL -o data/fda/drugsatfda.zip https://www.fda.gov/media/89850/download && unzip -q -o data/fda/drugsatfda.zip -d data/fda/extracted
   ```
3. Optional, for the label check against an expert benchmark: the TOP Phase II files from the HINT paper
   (`phase_II_train/valid/test.csv` from github.com/futianfan/clinical-trial-outcome-prediction, `data/`; non-commercial / academic use only) in `data/top_benchmark/`.

Then, from this folder:

```bash
./run_all.sh
```

Scripts that read downloaded files run with `python -I` (isolated mode). All steps are deterministic; a rerun reproduces the numbers above.

## Method in brief

- **Parsing:** 613,113 XML records → 467,911 interventional trials in one table.
- **Drug names:** synonyms (e.g. MK-3475 = pembrolizumab = Keytruda) learned from `other_name` fields, kept only if seen together in ≥2 trials, expanded one hop (no transitive closure). A drug with ≥10 earlier Phase 3 trials is a background drug and is not used for matching.
- **Labels:** gates 1–2 = a later trial of the same focus drug in the next phase; gate 3 = first NDA/BLA approval in Drugs@FDA (salt words and 4-letter biologic suffixes stripped before matching).
- **Features:** design, sponsor history, geography, disease area, drug history, eligibility, start year, plus TF-IDF text. Every history count only uses trials that started before the trial in question. Post-hoc fields (actual enrollment, status, posted results) are never used.
- **Models:** logistic regression, random forest, gradient boosting, and a TF-IDF logistic regression; the final model blends the best tree model with the text model. Hyper-parameters and blend weight are chosen on the validation years only.
- **Checks:** completed vs terminated pass rates, agreement with the TOP benchmark, drugs unseen in training, random vs temporal split, feature-group ablation, oncology subgroup.

## Key results (`results/journey.json`)

| | Phase 1 → 2 | Phase 2 → 3 | Phase 3 → FDA |
|---|---|---|---|
| Logistic regression / RF / GB | 0.706 / 0.712 / 0.707 | 0.717 / 0.722 / 0.720 | 0.762 / 0.802 / 0.804 |
| Final model (tree + text) | **0.718** | **0.722** | **0.807** |
| Pass rate, top 20% / bottom 20% of predictions | 68% / 18% | 61% / 14% | 70% / 4% |
| Completed vs terminated pass rate (label check) | 44% vs 20% | 35% vs 23% | 41% vs 18% |
| AUC on drugs unseen in training | 0.698 | 0.717 | 0.795 |
| Random 5-fold CV vs temporal split | 0.709 vs 0.712 | 0.747 vs 0.720 | 0.883 vs 0.802 |
| AUC change when the drug-history group is removed | −0.11 | −0.07 | −0.13 |

Gate-2 labels vs the TOP benchmark (1,536 shared trials): when our label says “passed”, TOP says the trial succeeded 77% of the time (82% for completed trials); κ = 0.22.

Figures: `fig1_gate_auc` (AUC per gate), `fig2_hit_rates` (pass rate by fifth of predicted probability), `fig3_drivers` (permutation importance), `fig4_label_check` (completed vs terminated), `fig5_split_check` (random vs temporal split), `fig6_all_framings` (the 18 problem framings explored before choosing this one; produced during exploration).

## Where the rest is

The exploration that led here (a first single-task Phase 2 model, 18 alternative framings, leakage and model-side experiments) lives in a local `claude_trials/` folder that is not committed. This folder contains only what the final framing needs.
