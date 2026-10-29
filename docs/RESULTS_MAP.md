# Where each reported result comes from

Every file below is in `results/` and is written by the step shown. Tables are CSV files in
`results/tables/`; figure panels are also given as CSV files in `results/source_data/`. The same
values, with more decimal places, are collected in `results/Supplementary_Data_1.xlsx`.

## Numbers in the abstract

| Result | File | Rows |
|---|---|---|
| Historical fat-percentage estimates against 2024 DXA: RMSE 7.55, bias −6.74 points | `body_validation.csv` (02) | `target = body_fat_pct`, `tier = laboratory`, `period = 2024_external` |
| Against concurrent bioimpedance: RMSE 3.66, mean bias −0.76 points | `bias_triangulation.csv` (07) | `Historical model minus 2024 BIA`, `body_fat_pct` |
| Released DXA fat percentage 6.7 points higher in 2024 at identical anthropometry | `era_difference_measured_dxa.csv` (07) | `group = All`, `body_fat_pct`, `adjusted_difference` |
| 3.0 points after aligning the calibration | `era_difference_recalibrated.csv` (07) | `group = All`, `body_fat_pct`, `adjusted_difference` |
| Below the AWGS low-muscle-mass threshold, aged 40–64: 19.0% → 46.2% (men), 21.3% → 66.5% (women) | `low_almi_by_era.csv` (07) | `Men 40-64`, `Women 40-64` |
| The same 2008–2010 proportions after calibration alignment: 34.6% and 38.3% | `low_almi_recalibrated.csv` (07) | `Men 40-64`, `Women 40-64`, `era = 2008-2010` |
| Estimated composition alone: AUROC > 0.60 for 20 of 25 diseases | `disease_estimates_only.csv` (09) | `model = estimates_only_laboratory` |
| Adding the estimates to their laboratory inputs: median AUROC change 0.0001, none significant after correction | `disease_paired_comparisons.csv` (04) | `D_laboratory_body minus C_laboratory`, `metric = auc` (`difference`, `q_value`) |

## Main text

| Item | Files | Step |
|---|---|---|
| Fig. 1 | `cohort_flow.csv`; counts of DXA and BIA availability from the analysis dataset | 01, 12 |
| Fig. 2 | `body_validation.csv` (`2024_external`), `contemporary_body_validation.csv` | 02, 03 |
| Fig. 3a | `bias_triangulation.csv` | 07 |
| Fig. 3b,c | `era_binned_by_bmi.csv` (`body_fat_pct`) | 07 |
| Fig. 3d | `era_difference_measured_dxa.csv` (`group = All`) | 07 |
| Fig. 3e | `low_almi_by_era.csv` | 07 |
| Fig. 4 | `disease_validation.csv` (`2019_2021`), `disease_paired_comparisons.csv` | 04 |
| Fig. 5 | `calibration_deciles_recurring.csv`, `disease_validation.csv` | 04, 08 |
| Fig. 6 | `measured_body_disease_validation.csv`, `measured_body_disease_comparisons.csv` | 06 |
| Table 1 | `Table_1_characteristics.csv` | 11 |
| Table 2 | `Table_2_disease_models.csv` | 04, 11 |
| Same-family comparison (Results) | `disease_same_family.csv` | 10 |
| Bias against 2024 DXA after calibration alignment, −3.25 points (Results) | `bias_triangulation.csv` (`… (classic calibration)` rows) | 07 |

The submitted Fig. 1 was laid out from the draft in `results/figures/Fig1_study_design.png`; the
numbers are the same.

## Supplementary Information

| Item | Files | Step |
|---|---|---|
| Supplementary Table 1 (TRIPOD+AI checklist), Table 2 (variable definitions) | – (text; definitions implemented in `01_prepare_data.py`) | 01 |
| Supplementary Table 3, screening of the 49 current-disease items | `disease_catalogue.csv` | 01, 04 |
| Supplementary Table 4, body composition models in 2024 | `body_validation.csv`, `contemporary_body_validation.csv` | 02, 03 |
| Supplementary Table 5, by sex and age | `body_subgroups.csv`, `contemporary_body_subgroups.csv` | 02, 07 |
| Supplementary Table 6, alternative references | `bias_triangulation.csv`, `body_sensitivity.csv` | 05, 07 |
| Supplementary Table 7, measured DXA, 2024 minus 2008–2010 | `era_difference_measured_dxa.csv`, `era_difference_recalibrated.csv` | 07 |
| Supplementary Table 8, disease models, 25 endpoints | `disease_validation.csv` (`2019_2021`) | 04 |
| Supplementary Table 9, model C by sex and age | `disease_subgroups.csv` | 04 |
| Supplementary Table 10, model C without diagnosis-related biomarkers | `disease_diagnostic_lab_sensitivity.csv` | 05 |
| Supplementary Table 11, 2022–2024 | `disease_validation.csv` (`2022_2024`) | 04 |
| Supplementary Table 12, eligibility for models E–H | `measured_body_disease_catalogue.csv` | 06 |
| Supplementary Tables 13–15, models E–H | `measured_body_disease_validation.csv` (`2010_temporal`, `2024_transport`, `2024_internal_psu_cv`) | 06 |
| Supplementary Fig. 1, Bland–Altman | `body_validation.csv` and the participant-level estimates | 02, 12 |
| Supplementary Fig. 2, other compartments by BMI | `era_binned_by_bmi.csv` | 07 |
| Supplementary Fig. 3, calibration, 25 endpoints | `calibration_deciles.csv` | 08 |
| Supplementary Fig. 4, calibration of model F in 2024 | `calibration_deciles_measured.csv` | 08 |
| Supplementary Data 1 | `results/Supplementary_Data_1.xlsx` (sheet list in its README sheet) | 13 |
| Source Data | `results/Source_Data.xlsx` | 12, 13 |

## Column conventions

| Column | Meaning |
|---|---|
| `rmse`, `mae`, `r2`, `bias` | Survey-weighted RMSE, MAE, R² and mean of predicted − reference (percentage points for `body_fat_pct`, kg otherwise) |
| `loa_low`, `loa_high` | Bias ± 1.96 survey-weighted SD of the differences (descriptive) |
| `*_lo`, `*_hi`, `ci_low`, `ci_high` | 95% percentile interval from 500 Rao–Wu bootstrap replicates, predictions held fixed |
| `auc`, `ap`, `brier`, `logloss` | Survey-weighted AUROC, average precision, Brier score and log loss |
| `calibration_intercept`, `calibration_slope` | Calibration-in-the-large with the slope fixed at 1; slope with a free intercept |
| `q_value` | Benjamini–Hochberg adjusted P value within comparison and metric |
| `tier` | `minimal` = age, sex, BMI; `anthropometry` = + height, weight, waist; `laboratory` = + 12 blood measurements |
| `n`, `train_n`, `test_n`, `*_events` | Unweighted counts |
| `label` | Item label from the KNHANES user guide (Korean); English names are in `pipeline/common.py` (`DISEASE_NAMES`) |
