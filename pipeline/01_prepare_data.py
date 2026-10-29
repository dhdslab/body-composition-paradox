"""Step 01. Build the analysis dataset from the KNHANES raw files.

Inputs (not distributed, see the README):
    data/KNHANES/hn08_all.sas7bdat ... hn24_all.sas7bdat
    data/KNHANES/hn08_dxa.sas7bdat ... hn10_dxa.sas7bdat
Outputs:
    work/analysis_dataset.pkl            eligible adults of all survey years (participant level)
    work/variable_inventory.csv          variables and labels of every raw file
    results/tables/cohort_flow.csv       participants by year and eligibility step (Fig. 1)
    results/tables/predictor_missingness.csv
    results/tables/disease_catalogue.csv every current-disease item with development and test counts
"""
import numpy as np
import pandas as pd
import pyreadstat

from common import BASIC, DATA, LABS, TABLES, WORK

YEARS = range(2008, 2025)
DESIGN = ['ID', 'ID_fam', 'year', 'psu', 'kstrata', 'kstrata1', 'wt_itvex', 'wt_ex', 'wt_ex1', 'wt_dw', 'wt_bia', 'HE_prg', 'HE_fst']
# Conservative impossible-value screens, not data-driven trimming (Supplementary Methods).
BOUNDS = {'HE_BMI': (10, 80), 'HE_ht': (100, 230), 'HE_wt': (20, 250), 'HE_wc': (40, 200), 'HE_glu': (20, 1000), 'HE_chol': (20, 1000),
          'HE_HDL_st2': (1, 200), 'HE_TG': (5, 5000), 'HE_ast': (1, 10000), 'HE_alt': (1, 10000), 'HE_HB': (2, 25), 'HE_HCT': (10, 75),
          'HE_BUN': (1, 200), 'HE_crea': (.1, 30), 'HE_WBC': (.1, 100), 'HE_RBC': (1, 10)}
DXA_NOT_MEASURED = [999.9, 9999.9, 9999.99, 9999.999, 99999.9, 99999.99, 999999.9]
TARGET_BOUNDS = {'body_fat_kg': (1, 100), 'fat_free_kg': (15, 120), 'body_fat_pct': (1, 70), 'appendicular_lean_kg': (3, 60)}
LIMBS = ['Lrm', 'Rrm', 'Llg', 'Rlg']


def inventory():
    """Variables and labels of every annual (_all) and DXA (_dxa) file."""
    rows = []
    for p in sorted(DATA.glob('*.sas7bdat')):
        if '_all' not in p.name.lower() and '_dxa' not in p.name.lower():
            continue
        _, meta = pyreadstat.read_sas7bdat(str(p), metadataonly=True, encoding='cp949')
        rows += [dict(file=p.name.lower(), variable=c, label=l) for c, l in zip(meta.column_names, meta.column_labels)]
    return pd.DataFrame(rows)


def harmonise_disease(d, item, year):
    """Case, control and missing coding of one current-disease item (Methods; Supplementary Methods).

    <item>_pr is current disease and <item>_dg physician diagnosis; 0 = no, 1 = yes, 8 = not applicable, 9 = unknown.
      case     current disease yes and diagnosis yes
      control  current disease no; or diagnosis no with current disease yes, no or not applicable;
               or, before 2013 only, both items not applicable (the skip pattern changed in 2013)
      missing  every other combination, including unknown responses
    """
    v = d[item]; y = v.where(v.isin([0, 1]))
    dg = item.replace('_pr', '_dg')
    if dg in d:
        y = y.mask(v.isin([0, 1, 8]) & d[dg].eq(0), 0)
        if year <= 2012:
            y = y.mask(v.eq(8) & d[dg].eq(8), 0)
        y = y.mask(v.eq(1) & ~d[dg].isin([0, 1]), np.nan)
    if item == 'DC5_pr':  # cervical cancer, women only
        y = y.where(d.sex.eq(2))
    return y


inv = inventory()
inv.to_csv(WORK / 'variable_inventory.csv', index=False, encoding='utf-8-sig')
DISEASES = sorted(inv.loc[inv.variable.str.match(r'D[A-Z][0-9]+_pr$'), 'variable'].unique())
wanted = set(BASIC + LABS + DESIGN + DISEASES + [x.replace('_pr', '_dg') for x in DISEASES])

records, flow = [], []
for yr in YEARS:
    name = f'hn{yr % 100:02d}_all.sas7bdat'
    cols = inv.loc[inv.file == name, 'variable'].tolist()
    use = [c for c in cols if c in wanted or c.startswith(('DW_', 'BIA_')) or c.lower() in ['id', 'id_fam']]
    d, _ = pyreadstat.read_sas7bdat(str(DATA / name), usecols=use, encoding='cp949')
    d = d.rename(columns={c: c.upper() for c in d if c.lower() == 'id'})
    assert not d.ID.duplicated().any()
    n_raw = len(d)

    # 2008-2010 whole-body and regional DXA values are released in separate files.
    if yr <= 2010:
        dx, _ = pyreadstat.read_sas7bdat(str(DATA / f'hn{yr % 100:02d}_dxa.sas7bdat'), encoding='cp949')
        dx = dx.rename(columns={c: c.upper() for c in dx if c.lower() == 'id'})
        assert not dx.ID.duplicated().any()
        keep = [c for c in dx if c == 'ID' or c.startswith('DW_') or (c in ['wt_dw', 'kstrata1'] and c not in d)]
        d = d.merge(dx[keep], on='ID', how='left', validate='one_to_one')

    d['year'] = yr
    for c in BASIC + LABS + DESIGN:
        if c not in d:
            d[c] = np.nan
    for c, (lo, hi) in BOUNDS.items():
        d[c] = d[c].where(d[c].between(lo, hi))
    for c in [x for x in d if x.startswith('DW_')]:
        d[c] = d[c].mask(d[c].isin(DXA_NOT_MEASURED))

    # Survey design: DXA weights (wt_ex1 2008, wt_dw 2009, wt_itvex 2010 and 2024) and examination weights for disease;
    # kstrata1 for 2008 DXA data and kstrata otherwise; PSU and strata keys prefixed with the year.
    d['body_weight'] = d['wt_ex1'] if yr == 2008 else d['wt_dw'] if yr == 2009 else d['wt_itvex']
    d['design_stratum'] = d['kstrata1'] if yr == 2008 else d['kstrata']
    d['disease_weight'] = d['wt_itvex']
    d['stratum_key'] = str(yr) + '_' + d['design_stratum'].astype(str)
    d['disease_stratum_key'] = str(yr) + '_' + d['kstrata'].astype(str)
    d['psu_key'] = str(yr) + '_' + d.psu.astype(str)

    # DXA targets. Regional lean mass includes bone mineral content, which is subtracted for appendicular lean soft tissue.
    d['body_fat_kg'] = d.get('DW_WBT_FT', np.nan) / 1000
    d['fat_free_kg'] = d.get('DW_WBT_LN', np.nan) / 1000
    d['body_fat_pct'] = d.get('DW_WBT_pFT', np.nan)
    if all('DW_' + x + '_LN' in d for x in LIMBS):
        d['appendicular_lean_kg'] = sum(d['DW_' + x + '_LN'] - d['DW_' + x + '_BMC'] for x in LIMBS) / 1000
    else:
        d['appendicular_lean_kg'] = np.nan
    for c, (lo, hi) in TARGET_BOUNDS.items():
        d[c] = d[c].where(d[c].between(lo, hi))

    for item in DISEASES:
        if item not in d:
            d[item] = np.nan
        d[item] = harmonise_disease(d, item, yr)

    # Eligibility: adults, not known to be pregnant, complete anthropometry and a positive examination weight.
    adult = d.age.ge(19); not_pregnant = ~d.HE_prg.eq(1)
    complete = d[BASIC].notna().all(axis=1) & d.sex.isin([1, 2]); weighted = d.disease_weight.gt(0)
    eligible = adult & not_pregnant & complete & weighted
    dxa_complete = eligible & d[list(TARGET_BOUNDS)].notna().all(axis=1) & d.body_weight.gt(0)
    flow.append(dict(year=yr, raw=n_raw, adult=int(adult.sum()), nonpregnant_adult=int((adult & not_pregnant).sum()),
                     eligible=int(eligible.sum()), dxa_complete=int(dxa_complete.sum())))
    records.append(d[eligible].copy())
    print(yr, flow[-1], flush=True)

data = pd.concat(records, ignore_index=True)
data.to_pickle(WORK / 'analysis_dataset.pkl')
pd.DataFrame(flow).to_csv(TABLES / 'cohort_flow.csv', index=False)
data.groupby('year')[BASIC + LABS].agg(lambda x: x.isna().mean()).to_csv(TABLES / 'predictor_missingness.csv')

catalogue = []
for item in DISEASES:
    z = inv[inv.variable.eq(item)]
    dev = data.loc[data.year.between(2011, 2018), item]; test = data.loc[data.year.between(2019, 2021), item]
    catalogue.append(dict(variable=item, label=z.label.iloc[-1], available_files=';'.join(z.file), train_n=int(dev.notna().sum()),
                          train_events=int(dev.sum()), test_n=int(test.notna().sum()), test_events=int(test.sum())))
pd.DataFrame(catalogue).to_csv(TABLES / 'disease_catalogue.csv', index=False, encoding='utf-8-sig')
print('DONE', data.shape)
