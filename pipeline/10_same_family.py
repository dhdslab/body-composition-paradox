"""Step 10. Same-family sensitivity analysis of the incremental comparisons (post hoc).

The model family was selected separately for each disease model, so for some endpoints model B and model A, or
model D and model C, used different algorithms. Here model B or D is refitted with the family selected for its
comparator (A or C) whenever the two differ; every other step is unchanged (2011-2018 fit, 2019-2021 test, the same
bootstrap replicates, normal-approximation P values and Benjamini-Hochberg adjustment).

Output: results/tables/disease_same_family.csv
"""
from scipy.stats import norm as normal_dist
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

from common import (DISEASE_SETS, TABLES, WORK, add_estimates, class_boot, class_metrics, clf, fit, load_dataset, np, pd,
                    replicate_factors, weights_for)

threadpool_limits(4)
d = add_estimates(load_dataset())
validation = pd.read_csv(TABLES / 'disease_validation.csv')
family = validation[validation.period.eq('2019_2021')].set_index(['disease', 'model']).family
pred = pd.read_pickle(WORK / 'disease_test_predictions.pkl')
catalogue = pd.read_csv(TABLES / 'disease_catalogue.csv')
development, test = d[d.year.between(2011, 2018)], d[d.year.between(2019, 2021)]
rep = replicate_factors(test)

rows = []
for _, r in catalogue[catalogue.status.eq('Evaluated')].iterrows():
    c = r.variable
    tr = development[development[c].notna()]; te = test[test[c].notna()]
    pc = pred[pred.disease.eq(c)]
    assert (pc[['ID', 'year']].values == te[['ID', 'year']].values).all()
    w = te.disease_weight; W = weights_for(te, w, rep)
    for alt, base in [('B_inferred_body', 'A_anthropometry'), ('D_laboratory_body', 'C_laboratory')]:
        fb, fa = family[(c, base)], family[(c, alt)]
        pb = pc[base].values
        cols = DISEASE_SETS[alt]
        pa = pc[alt].values if fa == fb else fit(clf(fb), tr[cols], tr[c], tr.disease_weight).predict_proba(te[cols])[:, 1]
        a = class_boot(te[c], pa, W)['auc'] - class_boot(te[c], pb, W)['auc']
        diff = class_metrics(te[c].values, pa, w)['auc'] - class_metrics(te[c].values, pb, w)['auc']
        se = np.std(a, ddof=1)
        rows.append(dict(disease=c, label=r.label, comparison=alt + ' minus ' + base, comparator_family=fb, original_family=fa,
                         refitted_same_family=fa != fb, difference=diff, ci_low=np.quantile(a, .025), ci_high=np.quantile(a, .975),
                         p_value=2 * normal_dist.sf(abs(diff / se)) if se else 1.0))
res = pd.DataFrame(rows)
for _, g in res.groupby('comparison'):
    res.loc[g.index, 'q_value'] = multipletests(g.p_value, method='fdr_bh')[1]
res.to_csv(TABLES / 'disease_same_family.csv', index=False, encoding='utf-8-sig')
for comparison, g in res.groupby('comparison'):
    print(comparison, 'refitted', int(g.refitted_same_family.sum()), 'median %.4f' % g.difference.median(),
          'q<0.05:', list(g.disease[g.q_value < .05]))
print('DONE')
