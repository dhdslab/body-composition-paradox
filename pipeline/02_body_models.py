"""Step 02. Historical body composition models: development in 2008-2010 and temporal test in 2024.

For each DXA target and predictor tier, ridge regression and gradient boosting are fitted in 2008-2009 and the
family with the lower weighted RMSE in 2010 is selected. The selected family is refitted in 2008-2010, frozen,
and applied without modification to adults aged >=40 years in 2024.

Outputs:
    results/tables/body_tuning.csv       2010 RMSE of both families (model selection)
    results/tables/body_validation.csv   2010 (age >=40) and 2024 performance with 95% CIs
    results/tables/body_subgroups.csv    2024 performance by sex and age group
    work/body_predictions.pkl            estimates of every target and tier for every participant
    work/models/body_*.joblib, body_model_manifest.json
"""
import hashlib
import json

import joblib
from threadpoolctl import threadpool_limits

from common import (MODELS, TABLES, TARGETS, TIERS, WORK, add_ci, fit, load_dataset, pd, reg, reg_boot, reg_metrics,
                    replicate_factors, weights_for)

threadpool_limits(4)
d = load_dataset()
dxa_ok = d[TARGETS].notna().all(axis=1) & d.body_weight.gt(0)
body = d[d.year.le(2010) & dxa_ok].copy()
train, tune = body[body.year.le(2009)], body[body.year.eq(2010)]
test = d[d.year.eq(2024) & d.age.ge(40) & dxa_ok].copy()
W = weights_for(test, test.body_weight, replicate_factors(d[d.year.eq(2024)], 'stratum_key'))

# Within-period factors: 108/307 and 199/307 for 2008 and 2009 in the tuning fit; 0.2, 0.4 and 0.4 for 2008-2010
# in the final refit (Supplementary Methods).
w_train = train.body_weight * train.year.map({2008: 108 / 307, 2009: 199 / 307})
w_body = body.body_weight * body.year.map({2008: .2, 2009: .4, 2010: .4})

tuning, validation, subgroups, manifest = [], [], [], []
predictions = d[['ID', 'year', 'age', 'sex', 'psu_key']].copy()
for target in TARGETS:
    for tier, cols in TIERS.items():
        scores = []
        for family in ['ridge', 'boosting']:
            m = fit(reg(family), train[cols], train[target], w_train)
            row = dict(target=target, tier=tier, family=family, n=len(tune), **reg_metrics(tune[target], m.predict(tune[cols]), tune.body_weight))
            tuning.append(row); scores.append((row['rmse'], family, m))
        _, family, m = sorted(scores, key=lambda x: x[0])[0]
        older = tune[tune.age.ge(40)]
        validation.append(dict(target=target, tier=tier, family=family, period='2010_tuning_age40', n=len(older),
                               **reg_metrics(older[target], m.predict(older[cols]), older.body_weight)))

        # Refit in 2008-2010 and freeze before any 2024 prediction is made.
        final = fit(reg(family), body[cols], body[target], w_body)
        path = MODELS / f'body_{target}_{tier}.joblib'; joblib.dump(final, path)
        manifest.append(dict(target=target, tier=tier, family=family, features=cols, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

        p = final.predict(test[cols])
        row = add_ci(dict(target=target, tier=tier, family=family, period='2024_external', n=len(test),
                          **reg_metrics(test[target], p, test.body_weight)), reg_boot(test[target], p, W))
        validation.append(row)
        predictions[f'{target}_{tier}'] = final.predict(d[cols])
        for group, mask in [('men', test.sex.eq(1)), ('women', test.sex.eq(2)), ('age40_64', test.age.lt(65)), ('age65plus', test.age.ge(65))]:
            subgroups.append(add_ci(dict(target=target, tier=tier, group=group, n=int(mask.sum()),
                                         **reg_metrics(test.loc[mask, target], p[mask], test.loc[mask, 'body_weight'])),
                                    reg_boot(test.loc[mask, target], p[mask], W[:, mask])))
        print(target, tier, family, '2024 RMSE', round(row['rmse'], 3), flush=True)

pd.DataFrame(tuning).to_csv(TABLES / 'body_tuning.csv', index=False)
pd.DataFrame(validation).to_csv(TABLES / 'body_validation.csv', index=False)
pd.DataFrame(subgroups).to_csv(TABLES / 'body_subgroups.csv', index=False)
predictions.to_pickle(WORK / 'body_predictions.pkl')
(MODELS / 'body_model_manifest.json').write_text(json.dumps(manifest, indent=2))
print('DONE')
