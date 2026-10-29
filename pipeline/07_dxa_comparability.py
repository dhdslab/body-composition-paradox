"""Step 07. Comparability of the DXA reference across survey periods (post hoc), adults aged >=40 years.

1. Period comparison: each measured DXA target regressed on a 2024 indicator, age, (age - 55)^2, BMI, height, weight
   and waist circumference, with female sex and its interactions with age, BMI, height and waist circumference
   (weighted least squares; each period contributes equal total weight). Repeated after recalibrating 2008-2010 to
   the NHANES calibration (lean soft tissue x 0.946, the difference added to fat mass).
2. Weighted means by BMI category, with 95% percentile intervals from the same bootstrap replicates (Fig. 3b,c).
3. Proportion below the AWGS 2019 DXA thresholds for low appendicular lean mass index (7.0 / 5.4 kg/m^2).
4. Frozen historical laboratory-assisted estimates against 2010 DXA, 2022-2023 BIA and, in the same 2024
   participants, BIA and DXA (also with 2024 DXA converted back to the classic calibration).
5. Subgroup performance of the contemporary 2024 models.

Outputs (results/tables/): era_difference_measured_dxa.csv, era_difference_recalibrated.csv, era_binned_by_bmi.csv,
low_almi_by_era.csv, low_almi_recalibrated.csv, bias_triangulation.csv, contemporary_body_subgroups.csv
"""
from threadpoolctl import threadpool_limits

from common import (K_NHANES, LABS, BASIC, N_BOOT, TABLES, TARGETS, WORK, add_ci, add_estimates, fit, load_dataset, np, pd, reg,
                    reg_boot, reg_metrics, replicate_factors, weights_for)

threadpool_limits(4)
d = add_estimates(load_dataset())
for t in TARGETS:
    d[t + '_hist_lab'] = d[t + '_laboratory']
dxa_ok = d[TARGETS].notna().all(axis=1) & d.body_weight.gt(0)


def wls_boot(X, y, w, W):
    """Weighted least squares point estimate and replicate estimates."""
    X = np.asarray(X, float); y = np.asarray(y, float); w = np.asarray(w, float)
    beta = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * y))
    reps = np.empty((W.shape[0], X.shape[1]))
    for b in range(W.shape[0]):
        wb = W[b]
        reps[b] = np.linalg.solve(X.T @ (wb[:, None] * X), X.T @ (wb * y))
    return beta, reps


def design(g, by_sex=True):
    """Design matrix of the period comparison. Sex interactions: age, BMI, height and waist circumference only."""
    X = pd.DataFrame(dict(const=1.0, era2024=g.year.eq(2024).astype(float), age=g.age, age2=(g.age - 55) ** 2,
                          bmi=g.HE_BMI, ht=g.HE_ht, wt=g.HE_wt, wc=g.HE_wc), index=g.index)
    if by_sex:
        f = g.sex.eq(2).astype(float)
        X['female'] = f; X['age_f'] = g.age * f; X['bmi_f'] = g.HE_BMI * f; X['wc_f'] = g.HE_wc * f; X['ht_f'] = g.HE_ht * f
    return X


# ----------------------------------------------------------------- 1. period comparison
frame = d[d.year.isin([2008, 2009, 2010, 2024])]
reps_era = replicate_factors(frame, 'stratum_key', N_BOOT)  # generated jointly for the four survey years
era = d[d.year.isin([2008, 2009, 2010, 2024]) & dxa_ok & d.age.ge(40)].copy()
era['era'] = np.where(era.year.eq(2024), '2024', '2008-2010')
# Period factors within 2008-2010 as in model development; each era then normalised to equal total weight.
era['w'] = era.body_weight * era.year.map({2008: .2, 2009: .4, 2010: .4, 2024: 1.0})
era['w'] = era.w / era.groupby('era').w.transform('sum')
Wera = weights_for(era, era.w, reps_era)
groups = [('All', np.ones(len(era), bool), True), ('Men', era.sex.eq(1).values, False), ('Women', era.sex.eq(2).values, False),
          ('Age 40-64', era.age.lt(65).values, True), ('Age 65+', era.age.ge(65).values, True)]


def era_rows(frame):
    rows = []
    for gname, mask, by_sex in groups:
        g = frame[mask]; Wg = Wera[:, mask]; X = design(g, by_sex)
        for t in TARGETS:
            beta, rb = wls_boot(X, g[t], g.w, Wg)
            j = list(X.columns).index('era2024')
            m0 = np.average(g.loc[g.era.eq('2008-2010'), t], weights=g.loc[g.era.eq('2008-2010'), 'w'])
            m1 = np.average(g.loc[g.era.eq('2024'), t], weights=g.loc[g.era.eq('2024'), 'w'])
            e0 = g.era.eq('2008-2010').values; e1 = ~e0
            raw = (Wg[:, e1] @ g[t].values[e1]) / Wg[:, e1].sum(1) - (Wg[:, e0] @ g[t].values[e0]) / Wg[:, e0].sum(1)
            rows.append(dict(group=gname, target=t, n_2008_2010=int(e0.sum()), n_2024=int(e1.sum()),
                             mean_2008_2010=m0, mean_2024=m1, unadjusted_difference=m1 - m0,
                             unadjusted_lo=np.quantile(raw, .025), unadjusted_hi=np.quantile(raw, .975),
                             adjusted_difference=beta[j], adjusted_lo=np.quantile(rb[:, j], .025), adjusted_hi=np.quantile(rb[:, j], .975)))
    return pd.DataFrame(rows)


era_rows(era).to_csv(TABLES / 'era_difference_measured_dxa.csv', index=False)

# Released 2008-2011 whole-body values use Hologic's classic calibration and the 2024 values the NHANES calibration.
# Recalibrate 2008-2010: lean soft tissue mass x 0.946, the difference added to fat mass so total mass is unchanged.
recal = era.copy(); old = recal.era.eq('2008-2010')
lst = (recal.DW_WBT_LN - recal.DW_WBT_BMC) / 1000; shift = (1 - K_NHANES) * lst
recal.loc[old, 'fat_free_kg'] = recal.fat_free_kg - shift
recal.loc[old, 'body_fat_kg'] = recal.body_fat_kg + shift
recal.loc[old, 'body_fat_pct'] = 100 * recal.body_fat_kg / (recal.DW_WBT_MS / 1000)
recal.loc[old, 'appendicular_lean_kg'] = recal.appendicular_lean_kg * K_NHANES
era_rows(recal).to_csv(TABLES / 'era_difference_recalibrated.csv', index=False)

# ----------------------------------------------------------------- 2. weighted means by BMI category (Fig. 3b,c)
era['bmi_bin'] = pd.cut(era.HE_BMI, [15, 19, 21, 23, 25, 27, 29, 31, 35])
binned = []
for (e, s, bb), g in era.groupby(['era', 'sex', 'bmi_bin'], observed=True):
    if len(g) < 20: continue
    Wg = Wera[:, era.index.get_indexer(g.index)]
    for t in TARGETS:
        mu = np.average(g[t], weights=g.w)
        with np.errstate(invalid='ignore', divide='ignore'): rb = (Wg @ g[t].values) / Wg.sum(1)
        binned.append(dict(era=e, sex=int(s), bmi_bin=str(bb), bmi_mid=bb.mid, target=t, n=len(g), mean=mu,
                           ci_lo=np.nanquantile(rb, .025), ci_hi=np.nanquantile(rb, .975), empty_replicates=int(np.isnan(rb).sum())))
pd.DataFrame(binned).to_csv(TABLES / 'era_binned_by_bmi.csv', index=False)


# ----------------------------------------------------------------- 3. low appendicular lean mass index (AWGS 2019)
def low_almi(frame):
    frame = frame.copy()
    frame['almi'] = frame.appendicular_lean_kg / (frame.HE_ht / 100) ** 2
    frame['low_almi'] = np.where(frame.sex.eq(1), frame.almi < 7.0, frame.almi < 5.4).astype(float)
    low = []
    for gname, mask in [('Men 40-64', frame.sex.eq(1) & frame.age.lt(65)), ('Men 65+', frame.sex.eq(1) & frame.age.ge(65)),
                        ('Women 40-64', frame.sex.eq(2) & frame.age.lt(65)), ('Women 65+', frame.sex.eq(2) & frame.age.ge(65))]:
        for e in ['2008-2010', '2024']:
            m = (mask & frame.era.eq(e)).values
            g = frame[m]; Wg = Wera[:, m]
            p = np.average(g.low_almi, weights=g.w); pb = (Wg @ g.low_almi.values) / Wg.sum(1)
            low.append(dict(group=gname, era=e, n=int(m.sum()), mean_almi=np.average(g.almi, weights=g.w), low_almi_pct=100 * p,
                            low_almi_lo=100 * np.quantile(pb, .025), low_almi_hi=100 * np.quantile(pb, .975)))
    return pd.DataFrame(low)


low_almi(era).to_csv(TABLES / 'low_almi_by_era.csv', index=False)
low_almi(recal).to_csv(TABLES / 'low_almi_recalibrated.csv', index=False)

# ----------------------------------------------------------------- 4. triangulation of the reference measurement
bia_cols = {'body_fat_pct': ['BIA_PBF'], 'body_fat_kg': ['BIA_BFM'], 'fat_free_kg': ['BIA_FFM'],
            'appendicular_lean_kg': ['BIA_LRA', 'BIA_LLA', 'BIA_LRL', 'BIA_LLL']}
for t, c in bia_cols.items():
    d[t + '_bia'] = d[c].sum(axis=1, min_count=len(c)).where(d[c].notna().all(axis=1))
    d.loc[d[t + '_bia'].le(0), t + '_bia'] = np.nan
d['bia_weight'] = d.wt_itvex.where(d.year.ne(2022), d.wt_bia)
tri = []
# (i) 2010 DXA, age >=40, with the tuning-stage model fitted in 2008-2009 (reproduces the published 2010 RMSE).
validation = pd.read_csv(TABLES / 'body_validation.csv')
fam = validation.query("tier=='laboratory' and period=='2024_external'").set_index('target').family
published = validation.query("tier=='laboratory' and period=='2010_tuning_age40'").set_index('target')
cols = BASIC + LABS
body = d[d.year.le(2010) & dxa_ok]; tr = body[body.year.le(2009)]; va = body[body.year.eq(2010) & body.age.ge(40)]
W10 = weights_for(va, va.body_weight, replicate_factors(d[d.year.eq(2010)], 'stratum_key', N_BOOT))
for t in TARGETS:
    m = fit(reg(fam[t]), tr[cols], tr[t], tr.body_weight * tr.year.map({2008: 108 / 307, 2009: 199 / 307}))
    p = m.predict(va[cols]); r = reg_metrics(va[t], p, va.body_weight)
    assert abs(r['rmse'] - published.loc[t, 'rmse']) < 1e-6, (t, r['rmse'], published.loc[t, 'rmse'])
    tri.append(add_ci(dict(comparison='Historical model minus 2010 DXA', setting='2010, age 40+', target=t, n=len(va), **r), reg_boot(va[t], p, W10)))
# (ii) 2022-2023 BIA, age >=40.
rep2223 = replicate_factors(d[d.year.between(2022, 2023)], 'stratum_key', N_BOOT)
for t in TARGETS:
    z = d[d.year.between(2022, 2023) & d.age.ge(40) & d[t + '_bia'].notna() & d.bia_weight.gt(0)]
    W = weights_for(z, z.bia_weight, rep2223)
    tri.append(add_ci(dict(comparison='Historical model minus BIA', setting='2022-2023, age 40+', target=t, n=len(z),
                           **reg_metrics(z[t + '_bia'], z[t + '_hist_lab'], z.bia_weight)), reg_boot(z[t + '_bia'], z[t + '_hist_lab'], W)))
# (iii) 2024 participants with both BIA and DXA: same people, three comparisons.
rep24 = replicate_factors(d[d.year.eq(2024)], 'stratum_key', N_BOOT)
pair = d[d.year.eq(2024) & d.age.ge(40) & dxa_ok & d[[t + '_bia' for t in TARGETS]].notna().all(axis=1)]
W24 = weights_for(pair, pair.body_weight, rep24)
for t in TARGETS:
    for name, ref, pred in [('Historical model minus 2024 DXA', t, t + '_hist_lab'), ('Historical model minus 2024 BIA', t + '_bia', t + '_hist_lab'),
                            ('2024 BIA minus 2024 DXA', t, t + '_bia')]:
        tri.append(add_ci(dict(comparison=name, setting='2024 paired BIA and DXA, age 40+', target=t, n=len(pair),
                               **reg_metrics(pair[ref], pair[pred], pair.body_weight)), reg_boot(pair[ref], pair[pred], W24)))
# Same paired sample, 2024 DXA converted back to the classic calibration of the 2008-2010 training data.
pair = pair.copy(); lst_n = (pair.DW_WBT_LN - pair.DW_WBT_BMC) / 1000; dl = lst_n / K_NHANES - lst_n
pair['body_fat_kg_classic'] = pair.body_fat_kg - dl; pair['fat_free_kg_classic'] = pair.fat_free_kg + dl
pair['body_fat_pct_classic'] = 100 * pair.body_fat_kg_classic / (pair.DW_WBT_MS / 1000)
pair['appendicular_lean_kg_classic'] = pair.appendicular_lean_kg / K_NHANES
for t in TARGETS:
    for name, ref, pred in [('Historical model minus 2024 DXA (classic calibration)', t + '_classic', t + '_hist_lab'),
                            ('2024 BIA minus 2024 DXA (classic calibration)', t + '_classic', t + '_bia')]:
        tri.append(add_ci(dict(comparison=name, setting='2024 paired BIA and DXA, age 40+', target=t, n=len(pair),
                               **reg_metrics(pair[ref], pair[pred], pair.body_weight)), reg_boot(pair[ref], pair[pred], W24)))
pd.DataFrame(tri).to_csv(TABLES / 'bias_triangulation.csv', index=False)

# ----------------------------------------------------------------- 5. contemporary model subgroups
oof = pd.read_pickle(WORK / 'contemporary_oof_predictions.pkl')
Wc = weights_for(oof, oof.body_weight, rep24)
subgroups = []
for t in TARGETS:
    for tier in ['minimal', 'anthropometry', 'laboratory']:
        p = oof[t + '_' + tier].values
        for group, mask in [('men', oof.sex.eq(1)), ('women', oof.sex.eq(2)), ('age40_64', oof.age.lt(65)), ('age65plus', oof.age.ge(65))]:
            mk = mask.values
            subgroups.append(add_ci(dict(target=t, tier=tier, group=group, n=int(mk.sum()),
                                         **reg_metrics(oof.loc[mask, t], p[mk], oof.loc[mask, 'body_weight'])),
                                    reg_boot(oof.loc[mask, t], p[mk], Wc[:, mk])))
pd.DataFrame(subgroups).to_csv(TABLES / 'contemporary_body_subgroups.csv', index=False)
print('DONE')
