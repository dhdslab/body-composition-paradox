"""Step 09. Disease discrimination of the estimated body composition values alone (post hoc).

Same endpoints, periods and model selection as step 04, with only the four estimates as predictors, without the
anthropometric or laboratory inputs from which they were derived. Repeated with the estimates of the second-tier
(anthropometric) body composition models, and compared with models including only age and sex.

Output: results/tables/disease_estimates_only.csv
"""
from threadpoolctl import threadpool_limits

from common import (TABLES, TARGETS, add_ci, add_estimates, class_boot, class_metrics, load_dataset, pd, replicate_factors,
                    select_and_fit, weights_for)

threadpool_limits(4)
d = add_estimates(load_dataset(), tiers=('laboratory', 'anthropometry'))
SETS = {f'estimates_only_{tier}': [f'{t}_{tier}' for t in TARGETS] for tier in ['laboratory', 'anthropometry']}
SETS['age_sex'] = ['age', 'sex']

catalogue = pd.read_csv(TABLES / 'disease_catalogue.csv')
development, test = d[d.year.between(2011, 2018)], d[d.year.between(2019, 2021)]
rep = replicate_factors(test)
rows = []
for _, r in catalogue[catalogue.status.eq('Evaluated')].iterrows():
    c = r.variable
    tr = development[development[c].notna()]; te = test[test[c].notna()]
    w = te.disease_weight; W = weights_for(te, w, rep)
    for model, cols in SETS.items():
        m, family, _ = select_and_fit(cols, c, tr[tr.year.le(2016)], tr[tr.year.ge(2017)], tr)
        p = m.predict_proba(te[cols])[:, 1]
        rows.append(add_ci(dict(disease=c, label=r.label, model=model, family=family, period='2019_2021', train_n=len(tr), train_events=int(tr[c].sum()),
                                test_n=len(te), test_events=int(te[c].sum()), **class_metrics(te[c].values, p, w)), class_boot(te[c], p, W)))
    print(c, *[f"{x['model']}={x['auc']:.3f}" for x in rows[-len(SETS):]], flush=True)
pd.DataFrame(rows).to_csv(TABLES / 'disease_estimates_only.csv', index=False, encoding='utf-8-sig')
print('DONE')
