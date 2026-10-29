"""Step 12. Main and Supplementary Figures (183 mm wide, Arial) and the data plotted in each panel.

Outputs:
    results/figures/Fig1_study_design.png ... Fig6_measured_composition.png
    results/figures/Supplementary_Fig1_bland_altman.png ... Supplementary_Fig4_measured_calibration.png
    results/source_data/<panel>.csv       aggregated values plotted in each panel (collated by step 13)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image

from common import DISEASE_NAMES as NAMES, FIGURES, SOURCE_DATA, TABLES, TARGETS, TARGET_NAMES as TN, WORK, load_dataset, np, pd

TIER = {'minimal': 'Age, sex, BMI', 'anthropometry': 'Anthropometry', 'laboratory': 'Anthropometry\n+ blood tests'}
# Reference categorical palette (validated order): slot 1 blue, slot 2 orange, slot 3 aqua, slot 4 yellow.
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, GRID, NEUTRAL = '#0b0b0b', '#52514e', '#e4e3df', '#a8a7a1'
SEQ = LinearSegmentedColormap.from_list('blue', ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'axes.titlesize': 7.5, 'axes.titleweight': 'bold', 'axes.labelsize': 7,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5, 'axes.linewidth': .6,
                     'xtick.major.width': .6, 'ytick.major.width': .6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.labelcolor': INK, 'text.color': INK,
                     # Opaque, borderless legend background so grid or reference lines never run through legend text.
                     'legend.frameon': True, 'legend.facecolor': 'white', 'legend.edgecolor': 'none', 'legend.framealpha': 1, 'legend.fancybox': False,
                     'savefig.dpi': 400, 'figure.dpi': 100, 'lines.linewidth': 1.2})
MM = 1 / 25.4


def label(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='left')


def grid(ax, axis='y'):
    ax.grid(axis=axis, color=GRID, lw=.5); ax.set_axisbelow(True)


SAVED = []


def save(fig, name):
    SAVED.append(name)
    fig.savefig(FIGURES / name, bbox_inches='tight', facecolor='white', pad_inches=.04)
    plt.close(fig)


def src(df, name):
    df.to_csv(SOURCE_DATA / f'{name}.csv', index=False)


flow = pd.read_csv(TABLES / 'cohort_flow.csv')
bv = pd.read_csv(TABLES / 'body_validation.csv'); cv = pd.read_csv(TABLES / 'contemporary_body_validation.csv')
dv = pd.read_csv(TABLES / 'disease_validation.csv'); diff = pd.read_csv(TABLES / 'disease_paired_comparisons.csv')
tri = pd.read_csv(TABLES / 'bias_triangulation.csv'); era = pd.read_csv(TABLES / 'era_difference_measured_dxa.csv')
binned = pd.read_csv(TABLES / 'era_binned_by_bmi.csv'); low = pd.read_csv(TABLES / 'low_almi_by_era.csv')
calr = pd.read_csv(TABLES / 'calibration_deciles_recurring.csv'); calm = pd.read_csv(TABLES / 'calibration_deciles_measured.csv')
mv = pd.read_csv(TABLES / 'measured_body_disease_validation.csv'); mc = pd.read_csv(TABLES / 'measured_body_disease_comparisons.csv')

# =============================================================== Fig. 1 participant flow (CONSORT-style)
d0 = load_dataset()
fl = flow.set_index('year')


def counts(a, b):
    g = fl.loc[a:b]
    return dict(raw=int(g.raw.sum()), age=int((g.raw - g.adult).sum()), preg=int((g.adult - g.nonpregnant_adult).sum()),
                anth=int((g.nonpregnant_adult - g.eligible).sum()), elig=int(g.eligible.sum()), dxa=int(g.dxa_complete.sum()))


dxa_ok0 = d0[TARGETS].notna().all(axis=1) & d0.body_weight.gt(0)
h0 = d0[d0.year.le(2010)]; n_nodxa_h = int(h0[TARGETS].isna().all(axis=1).sum()); n_bad_h = int((~dxa_ok0[h0.index]).sum()) - n_nodxa_h
z0 = d0[d0.year.eq(2024)]; n_young24 = int(z0.age.lt(40).sum()); n_nodxa24 = int((z0.age.ge(40) & ~dxa_ok0[z0.index]).sum())
bia = ['BIA_PBF', 'BIA_BFM', 'BIA_FFM', 'BIA_LRA', 'BIA_LLA', 'BIA_LRL', 'BIA_LLL']
n_pair = int((z0.age.ge(40) & dxa_ok0[z0.index] & z0[bia].notna().all(axis=1) & z0[bia].gt(0).all(axis=1)).sum())
H, D24, DEV, FIT, TUN, TST, LATE = counts(2008, 2010), counts(2024, 2024), counts(2011, 2018), counts(2011, 2016), counts(2017, 2018), counts(2019, 2021), counts(2022, 2024)
assert H['elig'] - n_nodxa_h - n_bad_h == H['dxa'] and D24['elig'] - n_young24 - n_nodxa24 == D24['dxa'] == 3253 and n_pair == 3186
n = lambda v: f'{v:,}'

FW, FH = 183, 214
fig = plt.figure(figsize=(FW * MM, FH * MM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, FW); ax.set_ylim(FH, 0); ax.axis('off')
LINE = '#4a4a4a'
STY = {'src': ('white', INK, .8), 'exc': ('white', '#9b9a95', .6), 'coh': ('white', INK, .8), 'dev': ('#cde2fb', BLUE, .9),
       'tune': ('#f0efec', '#8a8984', .9), 'test': ('#fde3d7', ORANGE, .9), 'note': ('white', '#9b9a95', .6)}
flow_rows = []


def box(x, y, w, text, kind, ha='center', size=6.2, h=None, weight='normal'):
    lines = text.count('\n') + 1; h = h or 2.55 * lines + 2.2
    fc, ec, lw = STY[kind]; ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, ls='--' if kind == 'note' else '-', zorder=2))
    ax.text(x + 1.3 if ha == 'left' else x + w / 2, y + h / 2, text, ha=ha, va='center', fontsize=size, linespacing=1.18, zorder=3, fontweight=weight)
    flow_rows.append(dict(box=text.replace('\n', ' '), role=kind)); return x, y, w, h


def seg(x0, y0, x1, y1, **k):
    ax.plot([x0, x1], [y0, y1], color=LINE, lw=.7, solid_capstyle='butt', zorder=1, **k)


def lab(x, y, s, size, ha='left', va='center', bold=False, italic=False, color=INK, linespacing=1.2):
    ax.text(x, y, s, fontsize=size, ha=ha, va=va, fontweight='bold' if bold else 'normal', style='italic' if italic else 'normal', color=color, linespacing=linespacing)


def stream(cx0, cx1, y, src_text, excl, coh_text, coh_kind, exc_h=None):
    W = cx1 - cx0; trunk = cx0 + .29 * W; sw = .56 * W
    _, _, _, hs = box(trunk - sw / 2, y, sw, src_text, 'src'); y += hs + 4
    ex_x = trunk + 4.5; top = y - 4
    for t in excl:
        _, by, _, bh = box(ex_x, y, cx1 - ex_x, t, 'exc', ha='left', size=5.9, h=exc_h); seg(trunk, by + bh / 2, ex_x, by + bh / 2); y += bh + 1.8
    y += 3; _, cy, cw, ch = box(trunk - sw / 2, y, sw, coh_text, coh_kind, weight='bold'); seg(trunk, top, trunk, cy)
    return trunk, cy + ch


def children(trunk, ybot, ytop, specs):
    """specs: list of (x, w, text, kind); draws a bus from the parent to each child."""
    mid = (ybot + ytop) / 2; cs = [x + w / 2 for x, w, _, _ in specs]
    seg(trunk, ybot, trunk, mid); seg(min(cs + [trunk]), mid, max(cs + [trunk]), mid)
    out = []
    for (x, w, t, k), c in zip(specs, cs):
        seg(c, mid, c, ytop); out.append(box(x, ytop, w, t, k))
    return out


# ---- panel a: body composition models
lab(1, 3.5, 'a', 9, bold=True); lab(6, 3.5, 'Body composition models (DXA reference)', 7.5, bold=True)
ta, ya = stream(2, 90, 9, f'KNHANES 2008–2010\nsurvey participants (n = {n(H["raw"])})',
                [f'Aged <19 years (n = {n(H["age"])})', f'Known pregnancy (n = {n(H["preg"])})',
                 f'Incomplete anthropometry or nonpositive\nexamination weight (n = {n(H["anth"])})', f'No DXA examination (n = {n(n_nodxa_h)})',
                 f'Incomplete or implausible DXA values (n = {n(n_bad_h)})'],
                f'DXA cohort, 2008–2010\n(n = {n(H["dxa"])})', 'coh', exc_h=7.3)
tb, yb = stream(95, 181, 9, f'KNHANES 2024\nsurvey participants (n = {n(D24["raw"])})',
                [f'Aged <19 years (n = {n(D24["age"])})', f'Known pregnancy (n = {n(D24["preg"])})',
                 f'Incomplete anthropometry or nonpositive\nexamination weight (n = {n(D24["anth"])})', f'Aged 19–39 years, not eligible\nfor DXA in 2024 (n = {n(n_young24)})',
                 f'No DXA examination (n = {n(n_nodxa24)})'],
                f'2024 DXA test cohort, aged ≥40 years\n(n = {n(D24["dxa"])})', 'test', exc_h=7.3)
ych = max(ya, yb) + 7
dev_b, tun_b = children(ta, ya, ych, [(4, 34, f'Development, 2008–2009\n(n = {n(counts(2008, 2009)["dxa"])})', 'dev'), (42, 30, f'Tuning, 2010\n(n = {n(counts(2010, 2010)["dxa"])})', 'tune')])
yfz = ych + dev_b[3] + 7
mid = (ych + dev_b[3] + yfz) / 2
for x, y, w, h in (dev_b, tun_b): seg(x + w / 2, y + h, x + w / 2, mid)
seg(dev_b[0] + dev_b[2] / 2, mid, tun_b[0] + tun_b[2] / 2, mid); fz = box(4, yfz, 68, 'Models refitted in 2008–2010 and frozen', 'dev', weight='bold')
seg(38, mid, 38, yfz)
kids = children(tb, yb, yfz, [(96, 26.5, 'Frozen historical\nmodels tested', 'test'), (124.5, 27, 'Refitted 2024 models,\nfivefold PSU CV', 'test'),
                              (153.5, 27.5, f'Paired BIA and DXA\n(n = {n(n_pair)})', 'test')])
yy = yfz + fz[3] / 2
ax.annotate('', xy=(kids[0][0] - .3, yy), xytext=(fz[0] + fz[2] + .3, yy), arrowprops=dict(arrowstyle='-|>', color=LINE, lw=.8, ls='--', mutation_scale=7), zorder=1)
lab((fz[0] + fz[2] + kids[0][0]) / 2, yy - 1.4, 'applied without\nmodification', 5.4, ha='center', va='bottom', color=INK2, linespacing=1.1)
ypb = yfz + max(fz[3], kids[0][3]) + 12

# ---- panel b: disease models using estimated body composition
lab(1, ypb, 'b', 9, bold=True); lab(6, ypb, 'Disease models using estimated body composition (models A–D)', 7.5, bold=True)
y0 = ypb + 6
ex3 = lambda c: [f'Aged <19 years (n = {n(c["age"])})', f'Known pregnancy (n = {n(c["preg"])})', f'Incomplete anthropometry or\nnonpositive examination weight\n(n = {n(c["anth"])})']
t1, b1 = stream(2, 61.5, y0, f'KNHANES 2011–2018\nparticipants (n = {n(DEV["raw"])})', ex3(DEV), f'Development cohort\n(n = {n(DEV["elig"])})', 'dev')
t2, b2 = stream(62.5, 122, y0, f'KNHANES 2019–2021\nparticipants (n = {n(TST["raw"])})', ex3(TST), f'Temporal test cohort\n(n = {n(TST["elig"])})', 'test')
t3, b3 = stream(123, 181, y0, f'KNHANES 2022–2024\nparticipants (n = {n(LATE["raw"])})', ex3(LATE), f'Later test cohort\n(n = {n(LATE["elig"])})', 'test')
yk = max(b1, b2, b3) + 7
children(t1, b1, yk, [(2.5, 25, f'Fitting, 2011–2016\n(n = {n(FIT["elig"])})', 'dev'), (29.5, 25, f'Tuning, 2017–2018\n(n = {n(TUN["elig"])})', 'tune')])
children(t2, b2, yk, [(t2 - 17, 34, '25 of 49 current-disease\nitems evaluated', 'test')])
children(t3, b3, yk, [(t3 - 17, 34, 'Diabetes, hypertension and\ndyslipidaemia (3 endpoints)', 'test')])
yend = yk + 2.55 * 2 + 2.2
# legend
hand = [Rectangle((0, 0), 1, 1, fc=STY[k][0], ec=STY[k][1], lw=.8) for k in ['dev', 'tune', 'test', 'exc']]
fig.legend(hand, ['Development', 'Tuning', 'Testing', 'Exclusion'], loc='upper right', bbox_to_anchor=(.995, .998), ncol=4, handlelength=1.3, columnspacing=1.2, fontsize=6.2)
ax.set_ylim(yend + 2, 0); fig.set_size_inches(FW * MM, (yend + 2) * MM)
save(fig, 'Fig1_study_design.png')
src(pd.DataFrame(flow_rows), 'Fig1')
src(flow[['year', 'raw', 'adult', 'nonpregnant_adult', 'eligible', 'dxa_complete']], 'Fig1_by_year')

# =============================================================== Fig. 2 body composition RMSE
fig, axes = plt.subplots(2, 2, figsize=(183 * MM, 112 * MM))
s2 = []
for ax, t, L in zip(axes.flat, TARGETS, 'abcd'):
    for i, tier in enumerate(['minimal', 'anthropometry', 'laboratory']):
        a = bv[(bv.target == t) & (bv.tier == tier) & (bv.period == '2024_external')].iloc[0]; b = cv[(cv.target == t) & (cv.tier == tier)].iloc[0]
        for xx, r, col, nm in [(i - .12, a, ORANGE, 'Historical model applied to 2024'), (i + .12, b, BLUE, '2024 model, PSU cross-validation')]:
            ax.errorbar(xx, r.rmse, yerr=[[r.rmse - r.rmse_lo], [r.rmse_hi - r.rmse]], fmt='o', color=col, ms=4.5, capsize=2.2, elinewidth=.9, mec='white', mew=.5)
            s2.append(dict(panel=L, target=TN[t], predictors=TIER[tier].replace('\n', ' '), evaluation=nm, rmse=r.rmse, ci_low=r.rmse_lo, ci_high=r.rmse_hi, bias=r.bias, r2=r.r2))
    ax.set_title(TN[t]); ax.set_xticks(range(3), [TIER[k] for k in TIER]); ax.set_xlim(-.5, 2.5)
    ax.set_ylabel('RMSE (percentage points)' if t.endswith('pct') else 'RMSE (kg)'); ax.set_ylim(bottom=0); grid(ax); label(ax, L)
fig.legend([Line2D([], [], marker='o', ls='', color=ORANGE, ms=4.5), Line2D([], [], marker='o', ls='', color=BLUE, ms=4.5)],
           ['Frozen 2008–2010 models applied to 2024 DXA', '2024 models, fivefold PSU-held-out cross-validation'], loc='lower center', ncol=2, bbox_to_anchor=(.5, -.01))
fig.tight_layout(rect=[0, .05, 1, 1], h_pad=1.6, w_pad=2.5); save(fig, 'Fig2_body_composition_error.png'); src(pd.DataFrame(s2), 'Fig2')

# =============================================================== Fig. 3 measurement comparability
fig = plt.figure(figsize=(183 * MM, 128 * MM))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=.55, wspace=.42)
ax = fig.add_subplot(gs[0, 0])
order = [('Historical model minus 2010 DXA', '2010, age 40+', 'Model − 2010 DXA'), ('Historical model minus BIA', '2022-2023, age 40+', 'Model − 2022–2023 BIA'),
         ('Historical model minus 2024 BIA', None, 'Model − 2024 BIA*'), ('Historical model minus 2024 DXA', None, 'Model − 2024 DXA*'), ('2024 BIA minus 2024 DXA', None, '2024 BIA − DXA*')]
s3a = []
for i, (cmp, setting, lab) in enumerate(order):
    r = tri[(tri.comparison == cmp) & (tri.target == 'body_fat_pct')].iloc[0]
    col = ORANGE if 'DXA*' in lab else BLUE
    ax.errorbar(r.bias, i, xerr=[[r.bias - r.bias_lo], [r.bias_hi - r.bias]], fmt='o', color=col, ms=4.5, capsize=2, elinewidth=.9, mec='white', mew=.5)
    s3a.append(dict(comparison=lab.replace('*', ' (paired 2024 sample)'), n=int(r.n), mean_difference=r.bias, ci_low=r.bias_lo, ci_high=r.bias_hi, rmse=r.rmse))
ax.axvline(0, color=INK2, lw=.6, ls='--'); ax.set_yticks(range(len(order)), [o[2] for o in order]); ax.invert_yaxis()
ax.set_xlabel('Mean difference in body fat\n(percentage points, 95% CI)'); grid(ax, 'x'); ax.set_xlim(-8, 1.5)
ax.set_title('Agreement across references', loc='left'); label(ax, 'a', x=-.62)
s3b = []
for j, (sx, sname, L) in enumerate([(1, 'Men', 'b'), (2, 'Women', 'c')]):
    ax = fig.add_subplot(gs[0, 1 + j])
    for e, col in [('2008-2010', BLUE), ('2024', ORANGE)]:
        z = binned[(binned.era == e) & (binned.sex == sx) & (binned.target == 'body_fat_pct')]
        ax.errorbar(z.bmi_mid, z['mean'], yerr=[z['mean'] - z.ci_lo, z.ci_hi - z['mean']], fmt='o-', color=col, ms=3.5, capsize=1.5, elinewidth=.8, lw=1.1, mec='white', mew=.4,
                    label='2008–2010 DXA' if e != '2024' else '2024 DXA')
        s3b += [dict(panel=L, sex=sname, era=e, bmi_bin=r.bmi_bin, n=int(r.n), mean_body_fat_pct=r['mean'], ci_low=r.ci_lo, ci_high=r.ci_hi) for _, r in z.iterrows()]
    ax.set_title(sname, loc='left'); ax.set_xlabel('BMI (kg m$^{-2}$)'); ax.set_ylabel('Measured body fat (%)' if j == 0 else '')
    ax.set_ylim(10, 52); grid(ax); label(ax, L, x=-.2)
    if j == 0: ax.legend(loc='upper left')
ax = fig.add_subplot(gs[1, 0])
s3d = []
for i, t in enumerate(['body_fat_kg', 'fat_free_kg', 'appendicular_lean_kg']):
    r = era[(era.group == 'All') & (era.target == t)].iloc[0]
    ax.errorbar(r.unadjusted_difference, i - .14, xerr=[[r.unadjusted_difference - r.unadjusted_lo], [r.unadjusted_hi - r.unadjusted_difference]], fmt='o', mfc='white', color=INK2, ms=4, capsize=2, elinewidth=.9)
    ax.errorbar(r.adjusted_difference, i + .14, xerr=[[r.adjusted_difference - r.adjusted_lo], [r.adjusted_hi - r.adjusted_difference]], fmt='o', color=ORANGE, ms=4.5, capsize=2, elinewidth=.9, mec='white', mew=.5)
    s3d.append(dict(target=TN[t], unadjusted=r.unadjusted_difference, unadjusted_lo=r.unadjusted_lo, unadjusted_hi=r.unadjusted_hi, adjusted=r.adjusted_difference, adjusted_lo=r.adjusted_lo, adjusted_hi=r.adjusted_hi))
fp = era[(era.group == 'All') & (era.target == 'body_fat_pct')].iloc[0]
s3d.append(dict(target='Body fat percentage (percentage points)', unadjusted=fp.unadjusted_difference, unadjusted_lo=fp.unadjusted_lo, unadjusted_hi=fp.unadjusted_hi, adjusted=fp.adjusted_difference, adjusted_lo=fp.adjusted_lo, adjusted_hi=fp.adjusted_hi))
ax.axvline(0, color=INK2, lw=.6, ls='--'); ax.set_yticks(range(3), ['Fat mass', 'Fat-free mass', 'Appendicular lean\nsoft tissue mass']); ax.set_ylim(2.6, -1.7)
ax.set_xlabel('2024 minus 2008–2010 (kg, 95% CI)'); grid(ax, 'x'); ax.set_title('Era difference, age ≥40 y', loc='left'); label(ax, 'd', x=-.62)
ax.legend([Line2D([], [], marker='o', ls='', mfc='white', color=INK2, ms=4), Line2D([], [], marker='o', ls='', color=ORANGE, ms=4.5)], ['Unadjusted', 'Adjusted for anthropometry'], loc='upper left', frameon=True, facecolor='white', edgecolor='none', framealpha=1)
ax = fig.add_subplot(gs[1, 1:])
grp = ['Men 40-64', 'Men 65+', 'Women 40-64', 'Women 65+']
for k, (e, col) in enumerate([('2008-2010', BLUE), ('2024', ORANGE)]):
    z = low[low.era == e].set_index('group').loc[grp]
    x = np.arange(4) + (k - .5) * .34
    ax.bar(x, z.low_almi_pct, width=.32, color=col, label='2008–2010 DXA' if k == 0 else '2024 DXA')
    ax.errorbar(x, z.low_almi_pct, yerr=[z.low_almi_pct - z.low_almi_lo, z.low_almi_hi - z.low_almi_pct], fmt='none', ecolor=INK, capsize=1.8, elinewidth=.7)
ax.set_xticks(range(4), ['Men 40–64 y', 'Men ≥65 y', 'Women 40–64 y', 'Women ≥65 y']); ax.set_ylim(0, 100)
ax.set_ylabel('Below AWGS 2019 DXA\nlow-muscle-mass threshold (%)'); grid(ax); ax.legend(loc='upper left', ncol=2)
ax.set_title('Appendicular lean mass index below 7.0 (men) or 5.4 (women) kg m$^{-2}$', loc='left'); label(ax, 'e', x=-.1)
save(fig, 'Fig3_measurement_comparability.png')
src(pd.DataFrame(s3a), 'Fig3a'); src(pd.DataFrame(s3b), 'Fig3b_c'); src(pd.DataFrame(s3d), 'Fig3d'); src(low, 'Fig3e')

# =============================================================== Fig. 4 disease discrimination
v = dv[dv.period == '2019_2021'].copy(); v['name'] = v.disease.map(NAMES)
piv = v.pivot_table(index='disease', columns='model', values='auc')[['A_anthropometry', 'B_inferred_body', 'C_laboratory', 'D_laboratory_body']]
piv = piv.sort_values('C_laboratory', ascending=False); order_d = list(piv.index)
fig = plt.figure(figsize=(183 * MM, 150 * MM))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1, 1], wspace=.08)
ax = fig.add_subplot(gs[0]); y = np.arange(len(order_d))
im = ax.imshow(piv.values, aspect='auto', cmap=SEQ, vmin=.55, vmax=.95)
for i in range(piv.shape[0]):
    for j in range(4):
        val = piv.values[i, j]; ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=5.4, color='white' if val > .80 else INK)
ax.set_xticks(range(4), ['A', 'B', 'C', 'D']); ax.xaxis.tick_top(); ax.set_yticks(y, [NAMES[c] for c in order_d])
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
ax.set_title('AUROC, 2019–2021', pad=14); label(ax, 'a', x=-.55, y=1.02)
cax = ax.inset_axes([0.1, -0.075, 0.8, 0.012]); cb = fig.colorbar(im, cax=cax, orientation='horizontal'); cb.set_label('AUROC', fontsize=6); cb.ax.tick_params(labelsize=5.5, length=2); cb.outline.set_linewidth(.4)
s4 = []
for k, (cmp, title, L) in enumerate([('B_inferred_body minus A_anthropometry', 'B − A: estimated composition\nadded to anthropometry', 'b'),
                                     ('D_laboratory_body minus C_laboratory', 'D − C: estimated composition added\nto anthropometry and blood tests', 'c')]):
    ax = fig.add_subplot(gs[1 + k])
    z = diff[(diff.metric == 'auc') & (diff.comparison == cmp)].set_index('disease').loc[order_d]
    sig = z.q_value < .05
    ax.errorbar(z.difference, y, xerr=[z.difference - z.ci_low, z.ci_high - z.difference], fmt='none', ecolor=BLUE, elinewidth=.9, capsize=1.6)
    ax.scatter(z.difference[~sig], y[~sig.values], s=16, facecolor='white', edgecolor=BLUE, lw=.9, zorder=3)
    ax.scatter(z.difference[sig], y[sig.values], s=16, facecolor=BLUE, edgecolor='white', lw=.4, zorder=3)
    ax.axvline(0, color=INK2, ls='--', lw=.6); ax.set_ylim(len(y) - .5, -.5); ax.set_yticks(y, ['' for _ in y]); ax.tick_params(axis='y', length=0)
    ax.set_xlim(-.05, .07); ax.set_xlabel('Difference in AUROC (95% CI)'); ax.set_title(title, pad=14); grid(ax, 'x'); label(ax, L, x=-.04, y=1.02)
    s4 += [dict(panel=L, endpoint=NAMES[c], comparison=cmp.replace('_', ' '), difference=r.difference, ci_low=r.ci_low, ci_high=r.ci_high, q_value=r.q_value) for c, r in z.iterrows()]
ax.legend([Line2D([], [], marker='o', ls='', mfc=BLUE, mec='white', color=BLUE, ms=4.5), Line2D([], [], marker='o', ls='', mfc='white', mec=BLUE, ms=4.5)],
          ['FDR q < 0.05', 'q ≥ 0.05'], loc='upper center', bbox_to_anchor=(.5, -.09), ncol=2)
save(fig, 'Fig4_disease_discrimination.png')
src(v[['name', 'model', 'test_events', 'test_n', 'auc', 'auc_lo', 'auc_hi']].rename(columns={'name': 'endpoint'}), 'Fig4a'); src(pd.DataFrame(s4), 'Fig4b_c')

# =============================================================== Fig. 5 calibration over time
fig, axes = plt.subplots(2, 2, figsize=(183 * MM, 140 * MM))
for ax, c, L in zip(axes.flat[:3], ['DE1_pr', 'DI1_pr', 'DI2_pr'], 'abc'):
    mx = 0
    for per, col in [('2019-2021', BLUE), ('2022-2024', ORANGE)]:
        z = calr[(calr.disease == c) & (calr.model == 'C') & (calr.period == per)]
        ax.plot(z.predicted, z.observed, 'o-', color=col, ms=3.5, lw=1.1, mec='white', mew=.4, label=per.replace('-', '–'))
        mx = max(mx, z.predicted.max(), z.observed.max())
        r = dv[(dv.disease == c) & (dv.model == 'C_laboratory') & (dv.period == per.replace('-', '_'))].iloc[0]
        ax.text(.97, .05 + .08 * (per == '2019-2021'), f'{per.replace("-", "–")}: observed {100 * r.prevalence:.1f}%, mean predicted {100 * r.predicted_prevalence:.1f}%',
                transform=ax.transAxes, fontsize=5.6, color=col, ha='right', va='bottom')
    lim = mx * 1.05; ax.plot([0, lim], [0, lim], '--', color=INK2, lw=.6); ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Observed prevalence'); ax.set_title(NAMES[c] + ' (model C)', loc='left'); grid(ax, 'both'); label(ax, L, x=-.18)
    if L == 'a': ax.legend(loc='upper left', title='Test period', title_fontsize=6)
ax = axes.flat[3]
rowsd = []; yy = 0; ticks = []; tl = []
for c in ['DE1_pr', 'DI1_pr', 'DI2_pr']:
    for m, mm in [('A_anthropometry', 'A'), ('B_inferred_body', 'B'), ('C_laboratory', 'C'), ('D_laboratory_body', 'D')]:
        for per, col, off in [('2019_2021', BLUE, -.15), ('2022_2024', ORANGE, .15)]:
            r = dv[(dv.disease == c) & (dv.model == m) & (dv.period == per)].iloc[0]
            ax.plot(r.calibration_intercept, yy + off, 'o', color=col, ms=4, mec='white', mew=.4)
            rowsd.append(dict(endpoint=NAMES[c], model=mm, period=per.replace('_', '-'), calibration_intercept=r.calibration_intercept, calibration_slope=r.calibration_slope,
                              observed=r.prevalence, mean_predicted=r.predicted_prevalence))
        ticks.append(yy); tl.append(f'{NAMES[c]} {mm}'); yy += 1
    yy += .6
ax.axvline(0, color=INK2, ls='--', lw=.6); ax.set_yticks(ticks, tl); ax.invert_yaxis(); grid(ax, 'x')
ax.set_xlabel('Calibration intercept (0 = ideal; >0 = underprediction)'); ax.set_title('Calibration-in-the-large, all models', loc='left'); label(ax, 'd', x=-.42)
ax.legend([Line2D([], [], marker='o', ls='', color=BLUE, ms=4), Line2D([], [], marker='o', ls='', color=ORANGE, ms=4)], ['2019–2021', '2022–2024'], loc='center right')
fig.tight_layout(h_pad=2, w_pad=3); save(fig, 'Fig5_calibration_over_time.png')
src(calr[calr.model == 'C'], 'Fig5a_c'); src(pd.DataFrame(rowsd), 'Fig5d')

# =============================================================== Fig. 6 measured-composition extension
names13 = mv[mv.period == '2010_temporal'].disease.unique()
fF = mv[(mv.period == '2010_temporal') & (mv.model == 'measured_body')].set_index('disease')
fE = mv[(mv.period == '2010_temporal') & (mv.model == 'age_sex')].set_index('disease')
o13 = list(fF.sort_values('auc', ascending=False).index)
fig, axes = plt.subplots(2, 2, figsize=(183 * MM, 130 * MM)); y = np.arange(len(o13))
ax = axes[0, 0]; s6a = []
for i, c in enumerate(o13):
    ax.plot([fE.loc[c, 'auc'], fF.loc[c, 'auc']], [i, i], color=GRID, lw=2, zorder=1)
    s6a.append(dict(endpoint=NAMES[c], auc_E_age_sex=fE.loc[c, 'auc'], auc_F_age_sex_dxa=fF.loc[c, 'auc'], F_ci_low=fF.loc[c, 'auc_lo'], F_ci_high=fF.loc[c, 'auc_hi'], test_cases=int(fF.loc[c, 'test_events'])))
ax.scatter(fE.loc[o13, 'auc'], y, s=18, facecolor='white', edgecolor=INK2, lw=.9, zorder=2, label='E: age and sex')
ax.scatter(fF.loc[o13, 'auc'], y, s=20, color=BLUE, edgecolor='white', lw=.4, zorder=3, label='F: age, sex and measured DXA')
ax.set_yticks(y, [NAMES[c] for c in o13]); ax.set_ylim(len(y) - .5, -3.4); ax.set_xlabel('AUROC, 2010 temporal test'); grid(ax, 'x'); ax.legend(loc='upper left'); ax.set_xlim(.55, .95)
ax.set_title('Historical development (2008–2009)', loc='left'); label(ax, 'a', x=-.5)
ax = axes[0, 1]; s6b = []
for k, (cmp, col, nm) in enumerate([('measured_body minus age_sex', BLUE, 'F − E: DXA added to age and sex'), ('anthropometry_measured minus anthropometry', ORANGE, 'H − G: DXA added to anthropometry')]):
    z = mc[(mc.metric == 'auc') & (mc.comparison == cmp)].set_index('disease').loc[o13]
    ax.errorbar(z.difference, y + (k - .5) * .3, xerr=[z.difference - z.ci_low, z.ci_high - z.difference], fmt='o', color=col, ms=3.8, capsize=1.5, elinewidth=.8, mec='white', mew=.4, label=nm)
    s6b += [dict(endpoint=NAMES[c], comparison=nm, difference=r.difference, ci_low=r.ci_low, ci_high=r.ci_high) for c, r in z.iterrows()]
ax.axvline(0, color=INK2, ls='--', lw=.6); ax.set_yticks(y, ['' for _ in y]); ax.set_ylim(len(y) - .5, -3.4); ax.set_xlabel('Difference in AUROC (95% CI)'); grid(ax, 'x')
ax.legend(loc='upper left'); ax.set_title('Incremental discrimination, 2010', loc='left'); label(ax, 'b', x=-.06)
ax = axes[1, 0]; s6c = []
models = [('age_sex', 'E'), ('measured_body', 'F'), ('anthropometry', 'G'), ('anthropometry_measured', 'H')]
for k, (per, col, nm) in enumerate([('2024_transport', ORANGE, 'Historical model transported to 2024'), ('2024_internal_psu_cv', BLUE, '2024 model, PSU cross-validation')]):
    xs = []; vals = []; lo = []; hi = []
    for i, c in enumerate(['DE1_pr', 'DI1_pr', 'DI2_pr']):
        for j, (m, mm) in enumerate(models):
            r = mv[(mv.disease == c) & (mv.model == m) & (mv.period == per)].iloc[0]
            xs.append(i * 5 + j + (k - .5) * .3); vals.append(r.auc); lo.append(r.auc_lo); hi.append(r.auc_hi)
            s6c.append(dict(endpoint=NAMES[c], model=mm, analysis=nm, auc=r.auc, ci_low=r.auc_lo, ci_high=r.auc_hi))
    vals = np.array(vals); ax.errorbar(xs, vals, yerr=[vals - np.array(lo), np.array(hi) - vals], fmt='o', color=col, ms=3.8, capsize=1.5, elinewidth=.8, mec='white', mew=.4, label=nm)
ax.set_xticks([i * 5 + j for i in range(3) for j in range(4)], ['E', 'F', 'G', 'H'] * 3)
for i, c in enumerate(['DE1_pr', 'DI1_pr', 'DI2_pr']): ax.text(i * 5 + 1.5, -.16, NAMES[c], transform=ax.get_xaxis_transform(), ha='center', fontsize=6.5)
ax.set_ylabel('AUROC in 2024 (95% CI)'); ax.set_ylim(.6, .87); grid(ax); ax.legend(loc='upper left', ncol=1); ax.set_title('Contemporary evaluation, age ≥40 y', loc='left'); label(ax, 'c', x=-.15)
ax = axes[1, 1]; s6d = []
for i, c in enumerate(['DE1_pr', 'DI1_pr', 'DI2_pr']):
    t = mv[(mv.disease == c) & (mv.model == 'measured_body') & (mv.period == '2024_transport')].iloc[0]
    q = mv[(mv.disease == c) & (mv.model == 'measured_body') & (mv.period == '2024_internal_psu_cv')].iloc[0]
    for k, (val, col, nm) in enumerate([(100 * t.prevalence, '#cbcac4', 'Observed'), (100 * t.predicted_prevalence, ORANGE, 'Predicted, transported model'), (100 * q.predicted_prevalence, BLUE, 'Predicted, 2024 model')]):
        ax.bar(i + (k - 1) * .26, val, width=.24, color=col, label=nm if i == 0 else None)
        s6d.append(dict(endpoint=NAMES[c], quantity=nm, percent=val))
ax.set_xticks(range(3), [NAMES[c] for c in ['DE1_pr', 'DI1_pr', 'DI2_pr']]); ax.set_ylabel('Weighted prevalence, 2024 (%)'); grid(ax); ax.legend(loc='upper left')
ax.set_ylim(0, 52); ax.set_title('Calibration-in-the-large, model F', loc='left'); label(ax, 'd', x=-.15)
fig.tight_layout(h_pad=2.2, w_pad=2.5); save(fig, 'Fig6_measured_composition.png')
src(pd.DataFrame(s6a), 'Fig6a'); src(pd.DataFrame(s6b), 'Fig6b'); src(pd.DataFrame(s6c), 'Fig6c'); src(pd.DataFrame(s6d), 'Fig6d')

# =============================================================== Supplementary Fig. 1 Bland-Altman
d = load_dataset(); bp = pd.read_pickle(WORK / 'body_predictions.pkl')
z = d[d.year.eq(2024) & d[TARGETS].notna().all(axis=1) & d.body_weight.gt(0)]
fig, axes = plt.subplots(2, 2, figsize=(183 * MM, 130 * MM)); sba = []
for ax, t, L in zip(axes.flat, TARGETS, 'abcd'):
    pr = bp.loc[z.index, t + '_laboratory']; me = z[t]; a = bv[(bv.target == t) & (bv.tier == 'laboratory') & (bv.period == '2024_external')].iloc[0]
    hb = ax.hexbin((pr + me) / 2, pr - me, gridsize=40, cmap=SEQ, mincnt=1, linewidths=.1)
    for yv, ls in [(a.bias, '-'), (a.loa_low, '--'), (a.loa_high, '--')]: ax.axhline(yv, color=ORANGE, ls=ls, lw=.9)
    ax.axhline(0, color=INK2, lw=.5)
    ax.set_title(TN[t], loc='left'); ax.set_xlabel('Mean of measured and predicted' + (' (%)' if t.endswith('pct') else ' (kg)')); ax.set_ylabel('Predicted − measured' + (' (points)' if t.endswith('pct') else ' (kg)'))
    label(ax, L, x=-.15); sba.append(dict(target=TN[t], n=len(z), bias=a.bias, loa_low=a.loa_low, loa_high=a.loa_high))
fig.tight_layout(h_pad=2, w_pad=3); save(fig, 'Supplementary_Fig1_bland_altman.png'); src(pd.DataFrame(sba), 'SupplementaryFig1')

# =============================================================== Supplementary Fig. 2 other compartments by BMI and era
fig, axes = plt.subplots(2, 3, figsize=(183 * MM, 112 * MM), sharex=True); ss2 = []
for i, (sx, sname) in enumerate([(1, 'Men'), (2, 'Women')]):
    for j, t in enumerate(['body_fat_kg', 'fat_free_kg', 'appendicular_lean_kg']):
        ax = axes[i, j]
        for e, col in [('2008-2010', BLUE), ('2024', ORANGE)]:
            zz = binned[(binned.era == e) & (binned.sex == sx) & (binned.target == t)]
            ax.errorbar(zz.bmi_mid, zz['mean'], yerr=[zz['mean'] - zz.ci_lo, zz.ci_hi - zz['mean']], fmt='o-', color=col, ms=3, capsize=1.3, elinewidth=.7, lw=1, mec='white', mew=.3, label=e.replace('-', '–') + ' DXA')
            ss2 += [dict(sex=sname, target=TN[t], era=e, bmi_bin=r.bmi_bin, n=int(r.n), mean=r['mean'], ci_low=r.ci_lo, ci_high=r.ci_hi) for _, r in zz.iterrows()]
        ax.set_title(f'{sname}: ' + {'body_fat_kg': 'fat mass', 'fat_free_kg': 'fat-free mass', 'appendicular_lean_kg': 'appendicular lean soft tissue mass'}[t], loc='left')
        ax.set_ylabel('kg'); grid(ax); label(ax, 'abcdef'[i * 3 + j], x=-.2)
        if i == 1: ax.set_xlabel('BMI (kg m$^{-2}$)')
        if i == 0 and j == 0: ax.legend(loc='upper left')
fig.tight_layout(h_pad=1.6, w_pad=2.2); save(fig, 'Supplementary_Fig2_era_by_bmi.png'); src(pd.DataFrame(ss2), 'SupplementaryFig2')

# =============================================================== Supplementary Fig. 3 calibration, 25 endpoints
cal25 = pd.read_csv(TABLES / 'calibration_deciles.csv')
fig, axes = plt.subplots(5, 5, figsize=(183 * MM, 200 * MM))
for ax, c in zip(axes.flat, order_d):
    mx = 0
    for m, col, nm in [('B_inferred_body', ORANGE, 'B'), ('C_laboratory', BLUE, 'C')]:
        zz = cal25[(cal25.disease == c) & (cal25.model == m)]
        ax.plot(zz.predicted, zz.observed, 'o-', color=col, ms=2.2, lw=.8, mec='white', mew=.2, label=f'Model {nm}'); mx = max(mx, zz.predicted.max(), zz.observed.max())
    ax.plot([0, mx * 1.05], [0, mx * 1.05], '--', color=INK2, lw=.5); ax.set_xlim(0, mx * 1.05); ax.set_ylim(0, mx * 1.05)
    ax.set_title(NAMES[c], fontsize=6.5, loc='left'); ax.tick_params(labelsize=5)
for ax in axes[-1]: ax.set_xlabel('Predicted', fontsize=6)
for ax in axes[:, 0]: ax.set_ylabel('Observed', fontsize=6)
fig.legend([Line2D([], [], color=ORANGE, marker='o', ms=3), Line2D([], [], color=BLUE, marker='o', ms=3)], ['Model B: anthropometry + estimated composition', 'Model C: anthropometry + blood tests'], loc='lower center', ncol=2, bbox_to_anchor=(.5, -.005))
fig.tight_layout(rect=[0, .025, 1, 1], h_pad=1.0, w_pad=.8); save(fig, 'Supplementary_Fig3_calibration_25_endpoints.png')
src(cal25.assign(endpoint=cal25.disease.map(NAMES)), 'SupplementaryFig3')

# =============================================================== Supplementary Fig. 4 measured-composition calibration in 2024
fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 62 * MM))
for ax, c, L in zip(axes, ['DE1_pr', 'DI1_pr', 'DI2_pr'], 'abc'):
    mx = 0
    for an, col in [('2024 internal PSU cross-validation', BLUE), ('Historical model transported to 2024', ORANGE)]:
        zz = calm[(calm.disease == c) & (calm.analysis == an)]
        ax.plot(zz.predicted, zz.observed, 'o-', color=col, ms=3, lw=1, mec='white', mew=.3, label=an); mx = max(mx, zz.predicted.max(), zz.observed.max())
    ax.plot([0, mx * 1.05], [0, mx * 1.05], '--', color=INK2, lw=.5); ax.set_xlim(0, mx * 1.05); ax.set_ylim(0, mx * 1.05)
    ax.set_title(NAMES[c] + ' (model F)', loc='left'); ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Observed prevalence' if L == 'a' else ''); grid(ax, 'both'); label(ax, L, x=-.2)
hh, ll = axes[0].get_legend_handles_labels()
fig.legend(hh, ll, loc='lower center', ncol=2, bbox_to_anchor=(.5, -.01))
fig.tight_layout(rect=[0, .09, 1, 1], w_pad=2); save(fig, 'Supplementary_Fig4_measured_calibration.png'); src(calm, 'SupplementaryFig4')

for name in SAVED:
    p = FIGURES / name; im = Image.open(p); im.load(); im.save(p, dpi=(400, 400)); print(p.name, im.size)
print('DONE')
