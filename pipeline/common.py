"""Shared paths, constants, model specifications and survey-weighted statistics.

Every step of the pipeline imports from this module, so the model families, the bootstrap
and the metrics are defined in one place.
"""
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

# ----------------------------------------------------------------------------- paths
REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('BCP_DATA', REPO / 'data' / 'KNHANES'))   # raw KNHANES files (not distributed)
WORK = Path(os.environ.get('BCP_WORK', REPO / 'work'))               # participant-level intermediates (git-ignored)
MODELS = WORK / 'models'                                              # fitted model objects (git-ignored)
RESULTS = Path(os.environ.get('BCP_RESULTS', REPO / 'results'))      # aggregated results (committed)
TABLES = RESULTS / 'tables'
FIGURES = RESULTS / 'figures'
SOURCE_DATA = RESULTS / 'source_data'
for _p in (WORK, MODELS, TABLES, FIGURES, SOURCE_DATA):
    _p.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------- study constants
SEED = 927           # used by every model and by the bootstrap
N_BOOT = 500         # Rao-Wu rescaled bootstrap replicates

# Whole-body DXA targets (Methods, Key variables).
TARGETS = ['body_fat_pct', 'body_fat_kg', 'fat_free_kg', 'appendicular_lean_kg']
TARGET_NAMES = {'body_fat_pct': 'Body fat percentage', 'body_fat_kg': 'Fat mass', 'fat_free_kg': 'Fat-free mass',
                'appendicular_lean_kg': 'Appendicular lean soft tissue mass'}

# Predictor tiers. "basic" are the anthropometric predictors of disease model A.
BASIC = ['age', 'sex', 'HE_BMI', 'HE_ht', 'HE_wt', 'HE_wc']
LABS = ['HE_glu', 'HE_chol', 'HE_HDL_st2', 'HE_TG', 'HE_ast', 'HE_alt', 'HE_HB', 'HE_HCT', 'HE_BUN', 'HE_crea', 'HE_WBC', 'HE_RBC']
TIERS = {'minimal': ['age', 'sex', 'HE_BMI'], 'anthropometry': BASIC, 'laboratory': BASIC + LABS}

# KNHANES current-disease items (<item>_pr) and the English names used in tables and figures.
DISEASE_NAMES = {
    'DC1_pr': 'Gastric cancer', 'DC2_pr': 'Liver cancer', 'DC3_pr': 'Colorectal cancer', 'DC4_pr': 'Breast cancer',
    'DC5_pr': 'Cervical cancer', 'DC6_pr': 'Lung cancer', 'DC7_pr': 'Thyroid cancer', 'DC11_pr': 'Other cancer (item 1)',
    'DC12_pr': 'Other cancer (item 2)', 'DC13_pr': 'Other cancer (item 3)', 'DE1_pr': 'Diabetes', 'DE2_pr': 'Thyroid disease',
    'DF2_pr': 'Depression', 'DH1_pr': 'Cataract or glaucoma', 'DH2_pr': 'Cataract', 'DH3_pr': 'Glaucoma', 'DH4_pr': 'Otitis media',
    'DH6_pr': 'Macular degeneration', 'DH7_pr': 'Diabetic retinopathy', 'DI1_pr': 'Hypertension', 'DI2_pr': 'Dyslipidaemia',
    'DI3_pr': 'Stroke', 'DI4_pr': 'Coronary heart disease', 'DI5_pr': 'Myocardial infarction', 'DI6_pr': 'Angina',
    'DI8_pr': 'Haemorrhoids', 'DJ1_pr': 'Tuberculosis', 'DJ2_pr': 'Pulmonary tuberculosis', 'DJ3_pr': 'Extrapulmonary tuberculosis',
    'DJ4_pr': 'Asthma', 'DJ5_pr': 'COPD', 'DJ6_pr': 'Sinusitis', 'DJ7_pr': 'Bronchiectasis', 'DJ8_pr': 'Allergic rhinitis',
    'DK2_pr': 'Peptic ulcer', 'DK4_pr': 'Cirrhosis', 'DK7_pr': 'Temporomandibular disorder', 'DK8_pr': 'Hepatitis B',
    'DK9_pr': 'Hepatitis C', 'DL1_pr': 'Atopic dermatitis', 'DL2_pr': 'Anaemia', 'DM1_pr': 'Arthritis', 'DM2_pr': 'Osteoarthritis',
    'DM3_pr': 'Rheumatoid arthritis', 'DM4_pr': 'Osteoporosis', 'DM6_pr': 'Low back pain', 'DM8_pr': 'Gout',
    'DN1_pr': 'Renal failure or kidney disease', 'DN2_pr': 'Urinary incontinence'}

# Disease models A-D (Methods, Disease prediction models). The estimates are those of the frozen
# laboratory-assisted body composition models.
ESTIMATES = [t + '_laboratory' for t in TARGETS]
DISEASE_SETS = {'A_anthropometry': BASIC, 'B_inferred_body': BASIC + ESTIMATES,
                'C_laboratory': BASIC + LABS, 'D_laboratory_body': BASIC + LABS + ESTIMATES}

# Hologic classic-to-NHANES calibration: lean soft tissue mass x 0.946 (Hong et al., J Bone Metab 2026).
K_NHANES = 0.946


def load_dataset():
    """Eligible adults from all survey years (written by step 01)."""
    return pd.read_pickle(WORK / 'analysis_dataset.pkl')


def add_estimates(d, tiers=('laboratory',)):
    """Attach the frozen historical body composition estimates (step 02) as <target>_<tier> columns."""
    bp = pd.read_pickle(WORK / 'body_predictions.pkl')
    assert (d[['ID', 'year']].values == bp[['ID', 'year']].values).all()
    cols = [f'{t}_{tier}' for tier in tiers for t in TARGETS]
    return pd.concat([d, pd.DataFrame({c: bp[c].values for c in cols}, index=d.index)], axis=1)


# ----------------------------------------------------------------------------- models
def norm(w):
    """Normalise sample weights to a mean of 1 for model fitting."""
    return np.asarray(w, dtype=float) / np.mean(w)


def reg(family):
    """Body composition model: ridge regression with quadratic terms, or gradient boosting."""
    if family == 'ridge':
        return make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(),
                             PolynomialFeatures(2, include_bias=False), Ridge(alpha=100))
    return make_pipeline(SimpleImputer(strategy='median', add_indicator=True),
                         HistGradientBoostingRegressor(max_iter=160, max_leaf_nodes=15, l2_regularization=10, learning_rate=.06,
                                                       min_samples_leaf=30, early_stopping=False, random_state=SEED))


def clf(family):
    """Disease model: spline logistic regression, or gradient boosting."""
    if family == 'spline':
        return make_pipeline(SimpleImputer(strategy='median', add_indicator=True), SplineTransformer(n_knots=4, degree=3, include_bias=False),
                             StandardScaler(), LogisticRegression(C=.1, max_iter=1000, solver='lbfgs'))
    return make_pipeline(SimpleImputer(strategy='median', add_indicator=True),
                         HistGradientBoostingClassifier(max_iter=140, max_leaf_nodes=7, l2_regularization=20, learning_rate=.05,
                                                        min_samples_leaf=60, early_stopping=False, random_state=SEED))


def fit(model, X, y, w):
    """Fit a pipeline with normalised sample weights passed to its final estimator."""
    name = model.steps[-1][0]
    model.fit(X, y, **{name + '__sample_weight': norm(w)})
    return model


def select_and_fit(cols, outcome, fit_frame, tune_frame, final_frame, weight='disease_weight', families=('spline', 'boosting')):
    """Choose the disease model family by weighted log loss in the tuning period, then refit it in the full development period.

    Returns the fitted model, the selected family and the tuning log losses.
    """
    losses = []
    for fam in families:
        m = fit(clf(fam), fit_frame[cols], fit_frame[outcome], fit_frame[weight])
        p = m.predict_proba(tune_frame[cols])[:, 1]
        losses.append((log_loss(tune_frame[outcome], p, sample_weight=tune_frame[weight], labels=[0, 1]), fam))
    fam = min(losses)[1]
    return fit(clf(fam), final_frame[cols], final_frame[outcome], final_frame[weight]), fam, losses


# ----------------------------------------------------------------------------- metrics
def reg_metrics(y, p, w):
    """Survey-weighted RMSE, MAE, R², mean bias (predicted - reference) and descriptive limits of agreement."""
    y = np.asarray(y); p = np.asarray(p); w = np.asarray(w); e = p - y
    mse = np.average(e ** 2, weights=w); bias = np.average(e, weights=w)
    sd = np.sqrt(np.average((e - bias) ** 2, weights=w))
    return dict(rmse=np.sqrt(mse), mae=np.average(abs(e), weights=w),
                r2=1 - mse / np.average((y - np.average(y, weights=w)) ** 2, weights=w),
                bias=bias, loa_low=bias - 1.96 * sd, loa_high=bias + 1.96 * sd)


def calibration(y, p, w):
    """Weighted calibration intercept (slope fixed at 1) and calibration slope (with a free intercept)."""
    x = logit(np.clip(p, 1e-6, 1 - 1e-6)); X = np.column_stack([np.ones(len(x)), x]); w = norm(w); b = np.array([0., 1.])
    for _ in range(40):  # Newton-Raphson for logit(P) = b0 + b1 * logit(p)
        mu = expit(X @ b); v = w * mu * (1 - mu)
        step = np.linalg.solve(X.T @ (v[:, None] * X) + np.eye(2) * 1e-8, X.T @ (w * (y - mu))); b += step
        if np.max(abs(step)) < 1e-7: break
    a = 0.
    for _ in range(40):  # Newton-Raphson for logit(P) = a + logit(p)
        mu = expit(x + a); delta = np.sum(w * (y - mu)) / max(np.sum(w * mu * (1 - mu)), 1e-8); a += delta
        if abs(delta) < 1e-7: break
    return a, b[1]


def class_metrics(y, p, w):
    """Survey-weighted AUROC, average precision, Brier score, log loss, prevalence and calibration."""
    ci, cs = calibration(y, p, w)
    return dict(auc=roc_auc_score(y, p, sample_weight=w), ap=average_precision_score(y, p, sample_weight=w),
                brier=np.average((y - p) ** 2, weights=w), logloss=log_loss(y, p, sample_weight=w, labels=[0, 1]),
                prevalence=np.average(y, weights=w), predicted_prevalence=np.average(p, weights=w),
                calibration_intercept=ci, calibration_slope=cs)


# ----------------------------------------------------------------------------- survey bootstrap
def replicate_factors(frame, stratum='disease_stratum_key', B=N_BOOT):
    """Rao-Wu rescaled bootstrap factors for the PSUs of `frame`.

    Within each stratum with n PSUs, n - 1 PSUs are drawn with replacement and each draw is multiplied
    by n / (n - 1). PSU and stratum keys are prefixed with the survey year. The full sampling frame is
    passed, so PSUs without valid responses for a given outcome are not dropped.
    Returns (factors [B x PSU], PSU index, number of single-PSU strata).
    """
    ps = frame[['psu_key', stratum]].drop_duplicates(); assert not ps.psu_key.duplicated().any()
    keys = ps.psu_key.tolist(); idx = {k: i for i, k in enumerate(keys)}
    f = np.ones((B, len(keys)), np.float32); rng = np.random.default_rng(SEED); single = 0
    for _, g in ps.groupby(stratum):
        ii = [idx[k] for k in g.psu_key]; n = len(ii)
        if n < 2:
            single += 1; continue
        f[:, ii] = rng.multinomial(n - 1, np.ones(n) / n, size=B) * n / (n - 1)
    return f, idx, single


def weights_for(d, w, reps):
    """Replicate weights (B x rows) for the rows of `d`."""
    f, ix, _ = reps
    return f[:, [ix[k] for k in d.psu_key]] * np.asarray(w)[None, :]


def reg_boot(y, p, W):
    """Replicate RMSE, MAE, R² and bias with predictions held fixed."""
    y = np.asarray(y); p = np.asarray(p); s = W.sum(1); e = p - y
    mse = (W @ (e ** 2)) / s; mu = (W @ y) / s; var = (W @ (y * y)) / s - mu * mu
    return dict(rmse=np.sqrt(mse), mae=(W @ abs(e)) / s, r2=1 - mse / var, bias=(W @ e) / s)


def class_boot(y, p, W):
    """Replicate AUROC, Brier score and prevalence with predictions held fixed."""
    y = np.asarray(y); p = np.asarray(p); s = W.sum(1)
    order = np.argsort(p, kind='stable'); yp = y[order]; wp = W[:, order]
    starts = np.r_[0, np.flatnonzero(np.diff(p[order])) + 1]  # tied predictions form one group
    pos = np.add.reduceat(wp * yp, starts, axis=1); neg = np.add.reduceat(wp * (1 - yp), starts, axis=1)
    auc = np.sum(pos * (np.cumsum(neg, axis=1) - .5 * neg), axis=1) / (pos.sum(1) * neg.sum(1))
    return dict(auc=auc, brier=W @ ((y - p) ** 2) / s, prevalence=W @ y / s)


def add_ci(row, boot):
    """Add 95% percentile intervals (<metric>_lo, <metric>_hi) from replicate estimates."""
    for k, a in boot.items():
        row[k + '_lo'], row[k + '_hi'] = np.nanquantile(a, [.025, .975])
    return row
