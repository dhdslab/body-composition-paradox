"""Step 03. Contemporary body composition models refitted in 2024 (post hoc internal validation).

The model family selected in the historical data is reused without reselection. Out-of-fold predictions come
from fivefold cross-validation with primary sampling units (PSUs) as groups, so no PSU contributes to both
training and testing within a fold. The models are then refitted in the full 2024 sample.

Outputs:
    results/tables/contemporary_body_validation.csv
    work/contemporary_oof_predictions.pkl
    work/models/contemporary_*.joblib, contemporary_manifest.json
"""
import hashlib
import json

import joblib
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from common import (MODELS, SEED, TABLES, TARGETS, WORK, add_ci, fit, load_dataset, np, pd, reg, reg_boot, reg_metrics,
                    replicate_factors, weights_for)

threadpool_limits(4)
d = load_dataset()
z = d[d.year.eq(2024) & d.age.ge(40) & d[TARGETS].notna().all(axis=1) & d.body_weight.gt(0)].copy()
W = weights_for(z, z.body_weight, replicate_factors(d[d.year.eq(2024)], 'stratum_key'))
historical = json.loads((MODELS / 'body_model_manifest.json').read_text())
cv = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)

rows, manifest = [], []
oof = z[['ID', 'year', 'age', 'sex', 'psu_key', 'body_weight'] + TARGETS].copy()
for t in TARGETS:
    for tier in ['minimal', 'anthropometry', 'laboratory']:
        cfg = next(x for x in historical if x['target'] == t and x['tier'] == tier)
        cols, family = cfg['features'], cfg['family']
        p = np.zeros(len(z))
        for fold, (tr, te) in enumerate(cv.split(z, groups=z.psu_key)):
            assert not set(z.iloc[tr].psu_key) & set(z.iloc[te].psu_key)
            m = fit(reg(family), z.iloc[tr][cols], z.iloc[tr][t], z.iloc[tr].body_weight)
            p[te] = m.predict(z.iloc[te][cols]); oof.loc[z.index[te], 'fold'] = fold
        row = add_ci(dict(target=t, tier=tier, family=family, n=len(z), **reg_metrics(z[t], p, z.body_weight)), reg_boot(z[t], p, W))
        rows.append(row); oof[t + '_' + tier] = p

        final = fit(reg(family), z[cols], z[t], z.body_weight)
        path = MODELS / f'contemporary_{t}_{tier}.joblib'; joblib.dump(final, path)
        manifest.append(dict(**{**cfg, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
                             validation='2024 fivefold PSU internal cross-validation', training_age_min=40))
        print(t, tier, 'RMSE', round(row['rmse'], 3), flush=True)

pd.DataFrame(rows).to_csv(TABLES / 'contemporary_body_validation.csv', index=False)
oof.to_pickle(WORK / 'contemporary_oof_predictions.pkl')
(MODELS / 'contemporary_manifest.json').write_text(json.dumps(manifest, indent=2))
print('DONE')
