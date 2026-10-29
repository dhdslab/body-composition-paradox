"""Step 05. Sensitivity analyses of the body composition and disease models.

1. Frozen historical estimates against alternative references: 2024 DXA restricted to complete laboratory data,
   and BIA (InBody 970) in 2022-2023 (all adults) and 2024 (aged >=40 years); plus 2024 BIA minus 2024 DXA.
2. Model C refitted without diagnosis-related biomarkers: glucose for diabetes; total cholesterol, HDL
   cholesterol and triglycerides for dyslipidaemia. The model family is reselected in 2017-2018.

Outputs:
    results/tables/body_sensitivity.csv
    results/tables/BIA_DXA_agreement_2024.csv
    results/tables/disease_diagnostic_lab_sensitivity.csv
"""
from threadpoolctl import threadpool_limits

from common import (BASIC, LABS, TABLES, TARGETS, WORK, add_ci, class_boot, class_metrics, load_dataset, pd, reg_metrics,
                    replicate_factors, select_and_fit, weights_for)

threadpool_limits(4)
d = load_dataset()
bp = pd.read_pickle(WORK / 'body_predictions.pkl')
BIA = {'body_fat_pct': ['BIA_PBF'], 'body_fat_kg': ['BIA_BFM'], 'fat_free_kg': ['BIA_FFM'],
       'appendicular_lean_kg': ['BIA_LRA', 'BIA_LLA', 'BIA_LRL', 'BIA_LLL']}

# ----------------------------------------------------------------- 1. alternative references
sensitivity, agreement = [], []
for analysis, mask in [('2024_DXA_complete_labs', d.year.eq(2024) & d.age.ge(40)), ('2022_2023_BIA', d.year.between(2022, 2023)),
                       ('2024_BIA', d.year.eq(2024) & d.age.ge(40))]:
    for target in TARGETS:
        if 'DXA' in analysis:
            z = d[mask & d[TARGETS].notna().all(axis=1) & d[LABS].notna().all(axis=1) & d.body_weight.gt(0)].copy()
            y, w = z[target], z.body_weight
        else:
            cols = BIA[target]
            z = d[mask & d[cols].notna().all(axis=1)].copy()
            y = z[cols].sum(axis=1); w = z.wt_itvex.where(z.year.ne(2022), z.wt_bia)  # BIA weight wt_bia in 2022
            ok = w.gt(0) & y.gt(0); z, y, w = z[ok], y[ok], w[ok]
        for tier in ['anthropometry', 'laboratory']:
            sensitivity.append(dict(analysis=analysis, target=target, tier=tier, n=len(z), **reg_metrics(y, bp.loc[z.index, target + '_' + tier], w)))
        if analysis == '2024_BIA':  # BIA minus DXA in the same participants
            ok = z[target].notna()
            agreement.append(dict(target=target, n=int(ok.sum()), **reg_metrics(z.loc[ok, target], y[ok], w[ok])))
pd.DataFrame(sensitivity).to_csv(TABLES / 'body_sensitivity.csv', index=False)
pd.DataFrame(agreement).to_csv(TABLES / 'BIA_DXA_agreement_2024.csv', index=False)

# ----------------------------------------------------------------- 2. omission of diagnosis-related biomarkers (post hoc)
results = []
for disease, exclude in [('DE1_pr', ['HE_glu']), ('DI2_pr', ['HE_chol', 'HE_HDL_st2', 'HE_TG'])]:
    cols = BASIC + [c for c in LABS if c not in exclude]
    tr = d[d.year.between(2011, 2018) & d[disease].notna()]
    m, family, _ = select_and_fit(cols, disease, tr[tr.year.le(2016)], tr[tr.year.ge(2017)], tr)
    for period, lo, hi in [('2019_2021', 2019, 2021), ('2022_2024', 2022, 2024)]:
        frame = d[d.year.between(lo, hi)]; z = frame[frame[disease].notna()]
        p = m.predict_proba(z[cols])[:, 1]; W = weights_for(z, z.disease_weight, replicate_factors(frame))
        results.append(add_ci(dict(disease=disease, period=period, excluded=';'.join(exclude), family=family, n=len(z),
                                   **class_metrics(z[disease].values, p, z.disease_weight)), class_boot(z[disease], p, W)))
pd.DataFrame(results).to_csv(TABLES / 'disease_diagnostic_lab_sensitivity.csv', index=False)
print('DONE')
