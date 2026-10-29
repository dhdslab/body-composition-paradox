# Body composition paradox

Analysis code and aggregated results for the manuscript
*"Body Composition Paradox: Shifting Reference Standards and Limited Added Value of Estimates in
Disease Prediction"* (Kim *et al.*, under review).

The study uses the **Korea National Health and Nutrition Examination Survey (KNHANES) 2008–2024** to
ask two questions about body composition estimated from routine measurements:

1. Does a DXA-based estimation model developed in 2008–2010 remain accurate in 2024, and is the DXA
   reference itself comparable between the two periods?
2. Does adding the estimates to the measurements they were calculated from improve the prediction of
   25 current diseases?

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)

---

## The study in brief

| Analysis | Development | Test | Script |
|---|---|---|---|
| Body composition models (4 DXA targets × 3 predictor tiers) | 2008–2009, tuned in 2010, refitted in 2008–2010 and frozen | 2024, adults aged ≥40 years (n = 3,253) | `02`, `03` |
| Comparability of the DXA reference (period comparison, calibration alignment, BIA triangulation) | – | 2008–2010 vs 2024; BIA 2022–2024 | `05`, `07` |
| Disease models A–D (25 endpoints) | 2011–2016 fit, 2017–2018 tuning, refitted in 2011–2018 | 2019–2021 (n = 17,573); 2022–2024 for 3 endpoints | `04`, `08`–`10` |
| Disease models E–H with measured DXA (exploratory, 13 endpoints) | 2008–2009 | 2010; 2024 transport and cross-validation | `06` |

Disease models: **A** anthropometry; **B** A + four estimated body composition values; **C** A + 12 blood
measurements; **D** C + the four estimates. Because the estimates in D are calculated from predictors
already in C, D versus C asks whether a transformation of existing inputs improves prediction.

---

## What is in this repository

| Path | Contents |
|---|---|
| `pipeline/common.py` | Paths, predictor sets, model specifications, survey-weighted metrics and the Rao–Wu bootstrap shared by every step |
| `pipeline/01_prepare_data.py` … `13_supplementary_data.py` | The analysis, one numbered step per script (see [Pipeline](#pipeline)) |
| `run_all.py` | Runs the steps in order |
| `check_reproduction.py` | Compares a regenerated results directory with `results/`, cell by cell |
| `results/tables/` | Every aggregated result (CSV) behind the manuscript and Supplementary Information |
| `results/figures/` | Figures 1–6 and Supplementary Figures 1–4 as drawn by the code |
| `results/source_data/` | The values plotted in each figure panel |
| `results/Supplementary_Data_1.xlsx`, `results/Source_Data.xlsx` | The two data files submitted with the manuscript |
| `docs/` | [Where each reported result comes from](docs/RESULTS_MAP.md) and [reproducibility notes](docs/REPRODUCIBILITY.md) |

No participant-level data and no fitted model objects are included. Steps write these to `work/`,
which git ignores.

---

## Data

KNHANES raw data are **not** distributed here; the KDCA terms of use do not allow redistribution.
Register at the Korea Disease Control and Prevention Agency (https://knhanes.kdca.go.kr), download the
SAS files, and place them under `data/KNHANES/`, which git ignores:

```
data/KNHANES/  hn08_all.sas7bdat ... hn24_all.sas7bdat   (annual files, 2008-2024)
               hn08_dxa.sas7bdat  hn09_dxa.sas7bdat  hn10_dxa.sas7bdat   (DXA, merged by ID within year)
```

The DXA variables of 2024 are in `hn24_all`. Another location can be set with `BCP_DATA`.

---

## Install

**Python 3.12** with the pinned environment of the study:

```bash
pip install -r requirements.txt
```

The figures use Arial.

---

## Run

```bash
python run_all.py                 # all steps, from the raw data
python run_all.py --from 2        # from step 02, once work/analysis_dataset.pkl exists
python run_all.py --only 12       # one step
```

To regenerate everything without touching the committed results and then compare:

```bash
BCP_RESULTS=results_regen python run_all.py        # PowerShell: $env:BCP_RESULTS = "results_regen"
python check_reproduction.py results_regen
```

Steps 02–13 take about six minutes on a desktop computer, most of it in step 04. Step 01 reads the 17
annual files and takes a few minutes more.

---

## Pipeline

| Step | Script | What it does | Main outputs (`results/tables/`) |
|---|---|---|---|
| 01 | `01_prepare_data.py` | Reads the raw files, applies eligibility, impossible-value screens, the survey design and the case/control coding of every current-disease item | `cohort_flow`, `predictor_missingness`, `disease_catalogue`; `work/analysis_dataset.pkl` |
| 02 | `02_body_models.py` | Historical body composition models: family selection in 2010, refit in 2008–2010, frozen, tested in 2024 | `body_tuning`, `body_validation`, `body_subgroups` |
| 03 | `03_contemporary_body_models.py` | 2024 models, fivefold PSU-held-out cross-validation | `contemporary_body_validation` |
| 04 | `04_disease_models.py` | Disease models A–D, 2019–2021 and 2022–2024, paired differences with Benjamini–Hochberg q values | `disease_validation`, `disease_paired_comparisons`, `disease_subgroups`, `disease_tuning` |
| 05 | `05_sensitivity.py` | Estimates against BIA and complete-laboratory DXA; model C without diagnosis-related biomarkers | `body_sensitivity`, `BIA_DXA_agreement_2024`, `disease_diagnostic_lab_sensitivity` |
| 06 | `06_measured_dxa_disease_models.py` | Exploratory models E–H with measured DXA | `measured_body_disease_*` |
| 07 | `07_dxa_comparability.py` | Period comparison of measured DXA, calibration alignment, BMI-binned means, low appendicular lean mass index, BIA triangulation | `era_difference_*`, `era_binned_by_bmi`, `low_almi_*`, `bias_triangulation`, `contemporary_body_subgroups` |
| 08 | `08_calibration.py` | Weighted calibration by decile of predicted probability | `calibration_deciles*` |
| 09 | `09_estimates_only.py` | Disease models with only the four estimates (and with age and sex only) | `disease_estimates_only` |
| 10 | `10_same_family.py` | B vs A and D vs C with both models in the same model family | `disease_same_family` |
| 11 | `11_tables.py` | Tables 1 and 2 | `Table_1_characteristics`, `Table_2_disease_models` |
| 12 | `12_figures.py` | Figures 1–6 and Supplementary Figures 1–4, and the plotted values | `results/figures/`, `results/source_data/` |
| 13 | `13_supplementary_data.py` | Supplementary Data 1 and the Source Data file | `results/*.xlsx` |

[docs/RESULTS_MAP.md](docs/RESULTS_MAP.md) lists, for every figure, table and Supplementary Table, the
file and the step that produce it.

---

## Methods, as implemented

### Populations and survey design

- Adults aged ≥19 years, not known to be pregnant, with complete anthropometry (age, sex, BMI, height,
  weight, waist circumference) and a positive examination weight.
- DXA weights `wt_ex1` (2008), `wt_dw` (2009) and `wt_itvex` (2010, 2024); examination weight `wt_itvex`
  for the disease models. Stratum `kstrata1` for 2008 DXA data and `kstrata` otherwise. PSU and stratum
  keys are prefixed with the survey year.
- Within 2008–2010, DXA weights are multiplied by 108/307 and 199/307 (2008, 2009) for the tuning fit and
  by 0.2, 0.4 and 0.4 (2008–2010) for the final refit and the period comparison.

### Body composition models (`02`, `03`)

- Targets: whole-body fat percentage (primary), fat mass, fat-free mass including bone, and appendicular
  lean soft tissue mass (limb lean mass minus limb bone mineral content).
- Predictor tiers: age, sex and BMI; + height, weight and waist circumference; + 12 blood measurements
  (`common.TIERS`).
- Ridge regression with quadratic terms versus histogram gradient boosting; the family with the lower
  weighted RMSE in 2010 is refitted in 2008–2010 and frozen before any 2024 prediction. Hyperparameters
  are fixed in `common.reg`.
- The 2024 models reuse the historical families without reselection.

### Disease models (`04`, `06`, `09`, `10`)

- **Case**: physician diagnosis and current disease. **Control**: no current disease, or no physician
  diagnosis with a current-disease response of yes, no or not applicable; before 2013 only, both items
  not applicable. **Missing**: every other combination, including unknown responses
  (`01_prepare_data.harmonise_disease`).
- An item is evaluated if it has ≥100 development and ≥30 test cases, including ≥50 in the fitting and
  ≥15 in the tuning subset. 25 of 49 items qualify.
- Spline logistic regression versus gradient boosting; the family with the lower weighted log loss in
  2017–2018 is refitted in 2011–2018 (`common.select_and_fit`). Hyperparameters are fixed in
  `common.clf`. Because the family is selected per model, compared models can use different algorithms;
  `10_same_family.py` repeats B vs A and D vs C with the comparator's family.
- Models E–H are spline logistic regressions with fixed specifications.

### Uncertainty and comparisons

- 500 Rao–Wu rescaled bootstrap replicates: in each stratum with *n* PSUs, *n* − 1 PSUs are drawn with
  replacement and weights are multiplied by *n*/(*n* − 1) (`common.replicate_factors`, seed 927).
  Predictions are held fixed, so intervals do not include model-fitting uncertainty.
- Paired differences use the same replicates. P values come from a normal approximation and are adjusted
  by Benjamini–Hochberg within comparison and metric. The differences for models E–H are not adjusted
  for multiple comparisons.
- Metrics are survey-weighted: RMSE, MAE, R², bias and limits of agreement; AUROC, average precision,
  Brier score, log loss, calibration intercept (slope fixed at 1) and calibration slope.

### Comparability of the DXA reference (`07`)

- Adults aged ≥40 years. Each measured DXA target is regressed on a 2024 indicator, age, (age − 55)²,
  BMI, height, weight and waist circumference, with female sex and its interactions with **age, BMI,
  height and waist circumference** (`07_dxa_comparability.design`). Each period has equal total weight.
- Calibration alignment: the released 2008–2011 values use Hologic's classic calibration and the 2024
  values the NHANES calibration. 2008–2010 lean soft tissue mass is multiplied by 0.946 and the difference
  added to fat mass (`common.K_NHANES`); for comparison with the historical models, 2024 values are
  converted back by the inverse transformation.
- BIA (InBody 970) is used as an independent reference in 2022–2023 and in the 3,186 participants with
  both BIA and DXA in 2024.

---

## Limitations

- KNHANES current-disease items are self-reported physician diagnoses; the evaluated endpoints do not
  cover all diseases or undiagnosed disease.
- The 2024 DXA examination included only adults aged ≥40 years, and the 2024 software version and
  scanner cross-calibration data were not available. Calibration alignment uses a single published
  coefficient.
- The comparability analyses, the 2024 refitting, models E–H and the analyses in steps 09 and 10 were
  post hoc.
- All analyses are cross-sectional. The predicted probabilities are not causal effects or incidence.

---

## Citation

Please cite the manuscript above. Citation details will be added on publication.
