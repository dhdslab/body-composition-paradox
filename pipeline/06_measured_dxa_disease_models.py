"""Step 06. Exploratory disease models using measured DXA body composition (models E-H, post hoc).

    E  age and sex                 F  age, sex and the four measured DXA targets
    G  anthropometric predictors   H  G + the four measured DXA targets
All models are spline logistic regressions with fixed specifications (no outcome-guided family selection).
Historical models are developed in 2008-2009, tested in 2010 and applied without modification to 2024; endpoints
need at least 100 development and 30 test cases. Separate 2024 models (diabetes, hypertension, dyslipidaemia) are
evaluated by fivefold PSU-held-out cross-validation and then refitted in the full 2024 sample.

Outputs:
    results/tables/measured_body_disease_catalogue.csv
    results/tables/measured_body_disease_validation.csv   2010 temporal, 2024 transport, 2024 cross-validation
    results/tables/measured_body_disease_comparisons.csv  F-E and H-G paired AUROC and Brier differences, 2010
    work/measured_body_disease_oof.pkl
    work/models/measured_*.joblib, measured_disease_manifest.json
"""
import hashlib
import json

import joblib
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from common import (BASIC, MODELS, SEED, TABLES, TARGETS, WORK, add_ci, class_boot, class_metrics, clf, fit, load_dataset, np, pd,
                    replicate_factors, weights_for)

threadpool_limits(4)
d = load_dataset()
b = d[d[TARGETS].notna().all(axis=1) & d.body_weight.gt(0)].copy()
catalogue = pd.read_csv(TABLES / 'disease_catalogue.csv')
SETS = {'age_sex': ['age', 'sex'], 'measured_body': ['age', 'sex'] + TARGETS, 'anthropometry': BASIC, 'anthropometry_measured': BASIC + TARGETS}
dev, test10, now = b[b.year.isin([2008, 2009])], b[b.year.eq(2010)], b[b.year.eq(2024) & b.age.ge(40)]
rep10 = replicate_factors(d[d.year.eq(2010)], 'stratum_key'); rep24 = replicate_factors(d[d.year.eq(2024)], 'stratum_key')


def save_model(model, cols, name, **info):
    path = MODELS / f'{name}.joblib'; joblib.dump(dict(model=model, features=cols), path)
    manifest.append(dict(**info, features=cols, model_file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))


rows, status, manifest, oof, diffs = [], [], [], [], []
# ----------------------------------------------------------------- historical models: 2008-2009 -> 2010 -> 2024
for _, r in catalogue.iterrows():
    c = r.variable; t = dev[dev[c].notna()]; v = test10[test10[c].notna()]
    reason = 'Evaluated'
    if c.startswith(('DC11', 'DC12', 'DC13')) or c == 'DN1_pr': reason = 'Heterogeneous or changing definition'
    elif t[c].sum() < 100 or v[c].sum() < 30: reason = 'Fewer than 100 development or 30 temporal-test cases, or outcome unavailable'
    status.append(dict(disease=c, label=r.label, status=reason, train_n=len(t), train_events=int(t[c].sum()), test_n=len(v), test_events=int(v[c].sum())))
    if reason != 'Evaluated':
        continue
    W = weights_for(v, v.body_weight, rep10); boot, point = {}, {}
    for name, cols in SETS.items():
        m = fit(clf('spline'), t[cols], t[c], t.body_weight * t.year.map({2008: 108 / 307, 2009: 199 / 307}))
        p = m.predict_proba(v[cols])[:, 1]; boot[name] = class_boot(v[c], p, W)
        point[name] = class_metrics(v[c].values, p, v.body_weight)
        rows.append(add_ci(dict(disease=c, model=name, period='2010_temporal', train_n=len(t), test_n=len(v), test_events=int(v[c].sum()), **point[name]), boot[name]))
        if name == 'measured_body':
            save_model(m, cols, f'measured_historical_{c}', disease=c, era='historical')
            manifest[-1].update(validation='2010 temporal', age_min=19)
        n = now[now[c].notna()]
        if n[c].sum() >= 30:
            pn = m.predict_proba(n[cols])[:, 1]
            rows.append(add_ci(dict(disease=c, model=name, period='2024_transport', train_n=len(t), test_n=len(n), test_events=int(n[c].sum()),
                                    **class_metrics(n[c].values, pn, n.body_weight)), class_boot(n[c], pn, weights_for(n, n.body_weight, rep24))))
    for alt, base in [('measured_body', 'age_sex'), ('anthropometry_measured', 'anthropometry')]:
        for metric in ['auc', 'brier']:
            a = boot[alt][metric] - boot[base][metric]
            diffs.append(dict(disease=c, comparison=alt + ' minus ' + base, metric=metric, difference=point[alt][metric] - point[base][metric],
                              ci_low=np.quantile(a, .025), ci_high=np.quantile(a, .975)))
    print(c, 'historical models done', flush=True)

# ----------------------------------------------------------------- contemporary models: 2024 PSU cross-validation
splits = list(GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(now, groups=now.psu_key))  # common folds for all endpoints
for c in ['DE1_pr', 'DI1_pr', 'DI2_pr']:
    n = now[now[c].notna()]; WN = weights_for(n, n.body_weight, rep24)
    for name, cols in SETS.items():
        pp = pd.Series(np.nan, index=now.index)
        for tr_idx, te_idx in splits:
            tt, vv = now.iloc[tr_idx], now.iloc[te_idx]; tt, vv = tt[tt[c].notna()], vv[vv[c].notna()]
            assert set(tt.psu_key).isdisjoint(vv.psu_key)
            pp.loc[vv.index] = fit(clf('spline'), tt[cols], tt[c], tt.body_weight).predict_proba(vv[cols])[:, 1]
        p = pp.loc[n.index].values; assert np.isfinite(p).all()
        rows.append(add_ci(dict(disease=c, model=name, period='2024_internal_psu_cv', train_n=len(n), test_n=len(n), test_events=int(n[c].sum()),
                                **class_metrics(n[c].values, p, n.body_weight)), class_boot(n[c], p, WN)))
        oof.append(pd.DataFrame(dict(ID=n.ID, year=n.year, disease=c, model=name, outcome=n[c], prediction=p, weight=n.body_weight)))
        if name == 'measured_body':
            save_model(fit(clf('spline'), n[cols], n[c], n.body_weight), cols, f'measured_contemporary_{c}', disease=c, era='contemporary')
            manifest[-1].update(validation='2024 internal PSU cross-validation only', age_min=40)
    print(c, 'contemporary models done', flush=True)

pd.DataFrame(rows).to_csv(TABLES / 'measured_body_disease_validation.csv', index=False)
pd.DataFrame(status).to_csv(TABLES / 'measured_body_disease_catalogue.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(diffs).to_csv(TABLES / 'measured_body_disease_comparisons.csv', index=False)
pd.concat(oof).to_pickle(WORK / 'measured_body_disease_oof.pkl')
(MODELS / 'measured_disease_manifest.json').write_text(json.dumps(manifest, indent=2))
print('DONE')
