"""Step 04. Disease models A-D: development in 2011-2018 and temporal test in 2019-2021 (and 2022-2024).

    A  anthropometric predictors
    B  A + the four estimates of the frozen laboratory-assisted body composition models (step 02)
    C  A + the original laboratory measurements
    D  C + the four estimates
For each endpoint and model, spline logistic regression and gradient boosting are fitted in 2011-2016, the family
with the lower weighted log loss in 2017-2018 is selected and refitted in 2011-2018. Endpoints whose items continue
after 2021 are also evaluated in 2022-2024 without updating. Paired differences use the same bootstrap replicates;
P values come from a normal approximation and are adjusted by Benjamini-Hochberg within comparison and metric.

Outputs:
    results/tables/disease_catalogue.csv          adds the disposition of every item
    results/tables/disease_tuning.csv             log loss of both families in 2017-2018
    results/tables/disease_validation.csv         all metrics with 95% CIs, 2019-2021 and 2022-2024
    results/tables/disease_paired_comparisons.csv B-A, C-B, D-C for AUROC and Brier score, with q values
    results/tables/disease_subgroups.csv          2019-2021 by sex and age group
    work/disease_test_predictions.pkl             2019-2021 predictions of models A-D
    work/models/disease_<item>_<model>.joblib
"""
import joblib
from scipy.stats import norm as normal_dist
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

from common import (DISEASE_SETS, MODELS, TABLES, WORK, add_ci, add_estimates, class_boot, class_metrics, load_dataset, np, pd,
                    replicate_factors, select_and_fit, weights_for)

threadpool_limits(4)
d = add_estimates(load_dataset())
catalogue = pd.read_csv(TABLES / 'disease_catalogue.csv')
development, test, later = d[d.year.between(2011, 2018)], d[d.year.between(2019, 2021)], d[d.year.between(2022, 2024)]
rep, rep_later = replicate_factors(test), replicate_factors(later)
SUBGROUPS = lambda te: [('men', te.sex.eq(1)), ('women', te.sex.eq(2)), ('age19_39', te.age.lt(40)),
                        ('age40_64', te.age.between(40, 64)), ('age65plus', te.age.ge(65))]


def disposition(r):
    """Reason for not evaluating an item, or '' if it passes the prespecified screens."""
    if r.variable.startswith(('DC11', 'DC12', 'DC13')): return 'Other-cancer slot is heterogeneous, not a disease-specific outcome'
    if r.variable == 'DN1_pr': return 'Definition changed from renal failure to kidney disease'
    if r.train_n == 0 or r.test_n == 0: return 'No disjoint development and temporal-test outcome data'
    if r.train_events < 100 or r.test_events < 30: return 'Below 100 development or 30 test events'
    return ''


rows, tuning, diffs, subgroups, predictions = [], [], [], [], []
for ix, r in catalogue.iterrows():
    c = r.variable
    reason = disposition(r)
    if reason:
        catalogue.loc[ix, 'status'] = reason; continue
    tr = development[development[c].notna()].copy(); te = test[test[c].notna()].copy(); nt = later[later[c].notna()].copy()
    fit_part, tune_part = tr[tr.year.le(2016)], tr[tr.year.ge(2017)]
    if fit_part[c].sum() < 50 or tune_part[c].sum() < 15:
        catalogue.loc[ix, 'status'] = 'Insufficient events for temporal tuning'; continue
    catalogue.loc[ix, 'status'] = 'Evaluated'
    catalogue.loc[ix, 'actual_train_events'] = tr[c].sum(); catalogue.loc[ix, 'actual_test_events'] = te[c].sum()

    w = te.disease_weight; W = weights_for(te, w, rep); boots, probs = {}, {}
    common_info = dict(disease=c, label=r.label)
    for model, cols in DISEASE_SETS.items():
        m, family, losses = select_and_fit(cols, c, fit_part, tune_part, tr)
        tuning += [dict(disease=c, model=model, family=f, logloss=ll, fit_n=len(fit_part), tune_n=len(tune_part)) for ll, f in losses]
        joblib.dump(dict(model=m, features=cols), MODELS / f'disease_{c}_{model}.joblib')

        p = m.predict_proba(te[cols])[:, 1]; probs[model] = p; boots[model] = class_boot(te[c], p, W)
        rows.append(add_ci(dict(**common_info, model=model, family=family, period='2019_2021', train_n=len(tr), train_events=int(tr[c].sum()),
                                test_n=len(te), test_events=int(te[c].sum()), **class_metrics(te[c].values, p, w)), boots[model]))
        for group, mask in SUBGROUPS(te):
            events = te.loc[mask, c].sum()
            if events >= 10 and mask.sum() - events >= 10:
                subgroups.append(dict(disease=c, model=model, group=group, n=int(mask.sum()), events=int(events),
                                      **class_metrics(te.loc[mask, c].values, p[mask], w[mask])))
        if len(nt) > 0 and nt[c].sum() >= 30:
            pn = m.predict_proba(nt[cols])[:, 1]; wn = nt.disease_weight
            rows.append(add_ci(dict(**common_info, model=model, family=family, period='2022_2024', train_n=len(tr), train_events=int(tr[c].sum()),
                                    test_n=len(nt), test_events=int(nt[c].sum()), **class_metrics(nt[c].values, pn, wn)),
                               class_boot(nt[c], pn, weights_for(nt, wn, rep_later))))

    point = {x['model']: x for x in rows if x['disease'] == c and x['period'] == '2019_2021'}
    for alt, base in [('B_inferred_body', 'A_anthropometry'), ('C_laboratory', 'B_inferred_body'), ('D_laboratory_body', 'C_laboratory')]:
        for metric in ['auc', 'brier']:
            a = boots[alt][metric] - boots[base][metric]; lo, hi = np.quantile(a, [.025, .975])
            diff = point[alt][metric] - point[base][metric]; se = np.std(a, ddof=1)
            diffs.append(dict(**common_info, comparison=alt + ' minus ' + base, metric=metric, difference=diff, ci_low=lo, ci_high=hi,
                              p_value=2 * normal_dist.sf(abs(diff / se)) if se else 1))

    out = te[['ID', 'year', 'age', 'sex', 'psu_key', 'disease_stratum_key', 'disease_weight', c]].copy().rename(columns={c: 'outcome'})
    out['disease'] = c
    for model, p in probs.items(): out[model] = p
    predictions.append(out)
    print(c, 'AUROC A-D', *[round(point[x]['auc'], 3) for x in DISEASE_SETS], flush=True)

diffs = pd.DataFrame(diffs)
for _, g in diffs.groupby(['comparison', 'metric']):
    diffs.loc[g.index, 'q_value'] = multipletests(g.p_value, method='fdr_bh')[1]
pd.DataFrame(rows).to_csv(TABLES / 'disease_validation.csv', index=False, encoding='utf-8-sig')
diffs.to_csv(TABLES / 'disease_paired_comparisons.csv', index=False, encoding='utf-8-sig')
catalogue.to_csv(TABLES / 'disease_catalogue.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(tuning).to_csv(TABLES / 'disease_tuning.csv', index=False)
pd.DataFrame(subgroups).to_csv(TABLES / 'disease_subgroups.csv', index=False)
pd.concat(predictions).to_pickle(WORK / 'disease_test_predictions.pkl')
print('DONE')
