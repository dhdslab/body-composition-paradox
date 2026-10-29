"""Step 08. Weighted calibration by decile of predicted probability (descriptive; no recalibration in the test data).

Outputs (results/tables/):
    calibration_deciles.csv            models B and C, 2019-2021, all 25 endpoints (Supplementary Fig. 3)
    calibration_deciles_recurring.csv  models A-D, 2019-2021 and 2022-2024, three endpoints (Fig. 5a-c)
    calibration_deciles_measured.csv   model F in 2024, transported and cross-validated (Fig. 6d, Supplementary Fig. 4)
"""
import joblib

from common import DISEASE_NAMES, MODELS, TABLES, TARGETS, WORK, add_estimates, load_dataset, np, pd, roc_auc_score

d = add_estimates(load_dataset())
test = pd.read_pickle(WORK / 'disease_test_predictions.pkl')
LETTER = {'A_anthropometry': 'A', 'B_inferred_body': 'B', 'C_laboratory': 'C', 'D_laboratory_body': 'D'}

# ----------------------------------------------------------------- models B and C, 2019-2021, quantile bins
cal25 = []
for c in sorted(test.disease.unique()):
    z = test[test.disease == c].copy()
    for model in ['B_inferred_body', 'C_laboratory']:
        z['bin'] = pd.qcut(z[model], 10, labels=False, duplicates='drop')
        for bn, g in z.groupby('bin'):
            cal25.append(dict(disease=c, model=model, bin=bn, n=len(g), predicted=np.average(g[model], weights=g.disease_weight),
                              observed=np.average(g.outcome, weights=g.disease_weight)))
pd.DataFrame(cal25).to_csv(TABLES / 'calibration_deciles.csv', index=False)


def deciles(y, p, w):
    """Weighted mean prediction and observed prevalence within deciles of predicted probability (ties broken by order)."""
    q = pd.qcut(pd.Series(p).rank(method='first'), 10, labels=False).values
    out = []
    for k in range(10):
        m = q == k
        out.append(dict(decile=k + 1, n=int(m.sum()), predicted=np.average(p[m], weights=w[m]), observed=np.average(y[m], weights=w[m])))
    return out


# ----------------------------------------------------------------- models A-D over time, three recurring endpoints
dv = pd.read_csv(TABLES / 'disease_validation.csv')
recurring = []
for c in ['DE1_pr', 'DI1_pr', 'DI2_pr']:
    nt = d[d.year.between(2022, 2024) & d[c].notna()]
    z = test[test.disease.eq(c)]
    for model, letter in LETTER.items():
        obj = joblib.load(MODELS / f'disease_{c}_{model}.joblib')
        p22 = obj['model'].predict_proba(nt[obj['features']])[:, 1]
        auc = roc_auc_score(nt[c], p22, sample_weight=nt.disease_weight)
        assert abs(auc - dv.query("disease==@c and model==@model and period=='2022_2024'").auc.iloc[0]) < 1e-9, (c, model)
        for period, y, p, w in [('2019-2021', z.outcome.values, z[model].values, z.disease_weight.values),
                                ('2022-2024', nt[c].values, p22, nt.disease_weight.values)]:
            recurring += [dict(disease=c, endpoint=DISEASE_NAMES[c], model=letter, period=period, **r) for r in deciles(y, p, w)]
pd.DataFrame(recurring).to_csv(TABLES / 'calibration_deciles_recurring.csv', index=False)

# ----------------------------------------------------------------- model F (age, sex and measured DXA) in 2024
oof = pd.read_pickle(WORK / 'measured_body_disease_oof.pkl')
now = d[d.year.eq(2024) & d.age.ge(40) & d[TARGETS].notna().all(axis=1) & d.body_weight.gt(0)]
mv = pd.read_csv(TABLES / 'measured_body_disease_validation.csv')
measured = []
for c in ['DE1_pr', 'DI1_pr', 'DI2_pr']:
    z = oof[(oof.disease == c) & (oof.model == 'measured_body')]
    measured += [dict(disease=c, endpoint=DISEASE_NAMES[c], analysis='2024 internal PSU cross-validation', **r)
                 for r in deciles(z.outcome.values.astype(float), z.prediction.values, z.weight.values)]
    obj = joblib.load(MODELS / f'measured_historical_{c}.joblib'); n = now[now[c].notna()]
    p = obj['model'].predict_proba(n[obj['features']])[:, 1]
    auc = roc_auc_score(n[c], p, sample_weight=n.body_weight)
    assert abs(auc - mv.query("disease==@c and model=='measured_body' and period=='2024_transport'").auc.iloc[0]) < 1e-9
    measured += [dict(disease=c, endpoint=DISEASE_NAMES[c], analysis='Historical model transported to 2024', **r)
                 for r in deciles(n[c].values, p, n.body_weight.values)]
pd.DataFrame(measured).to_csv(TABLES / 'calibration_deciles_measured.csv', index=False)
print('DONE')
