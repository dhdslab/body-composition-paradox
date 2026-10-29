"""Step 11. Main-text tables.

Outputs (results/tables/):
    Table_1_characteristics.csv  survey-weighted means (SD) or percentages of the four analytic populations (Table 1)
    Table_2_disease_models.csv   models A-D in 2019-2021 for the three endpoints available through 2022-2024 (Table 2)
"""
from common import DISEASE_NAMES, TABLES, TARGETS, load_dataset, np, pd

d = load_dataset()
dxa_ok = d[TARGETS].notna().all(axis=1) & d.body_weight.gt(0)
cohorts = {'DXA development': d[d.year.le(2010) & dxa_ok], 'Disease development': d[d.year.between(2011, 2018)],
           'Disease temporal test': d[d.year.between(2019, 2021)], 'DXA 2024 test': d[d.year.eq(2024) & dxa_ok]}
rows = []
for name, g in cohorts.items():
    w = g.body_weight.copy() if name.startswith('DXA') else g.disease_weight.copy()
    if name == 'DXA development':
        w *= g.year.map({2008: .2, 2009: .4, 2010: .4})
    rows.append(dict(cohort=name, variable='N', value=str(len(g))))
    for var in ['age', 'HE_BMI', 'HE_ht', 'HE_wt', 'HE_wc', 'HE_glu', 'HE_crea']:
        ok = g[var].notna(); mu = np.average(g.loc[ok, var], weights=w[ok])
        sd = np.sqrt(np.average((g.loc[ok, var] - mu) ** 2, weights=w[ok]))
        rows.append(dict(cohort=name, variable=var, value=f'{mu:.1f} ({sd:.1f})'))
    rows.append(dict(cohort=name, variable='Women %', value=f'{100 * np.average(g.sex.eq(2), weights=w):.1f}'))
pd.DataFrame(rows).pivot(index='variable', columns='cohort', values='value').to_csv(TABLES / 'Table_1_characteristics.csv')

dv = pd.read_csv(TABLES / 'disease_validation.csv')
fmt = lambda x, k: f'{x:.{k}f}'.replace('-', '−')
table2 = []
for c in ['DE1_pr', 'DI1_pr', 'DI2_pr']:
    for _, r in dv[dv.disease.eq(c) & dv.period.eq('2019_2021')].iterrows():
        table2.append({'Endpoint': DISEASE_NAMES[c], 'Model': r.model[0], 'AUROC (95% CI)': f'{fmt(r.auc, 3)} ({fmt(r.auc_lo, 3)}–{fmt(r.auc_hi, 3)})',
                       'Brier score': fmt(r.brier, 3), 'Calibration intercept': fmt(r.calibration_intercept, 2),
                       'Calibration slope': fmt(r.calibration_slope, 2)})
pd.DataFrame(table2).to_csv(TABLES / 'Table_2_disease_models.csv', index=False, encoding='utf-8-sig')
print('DONE')
