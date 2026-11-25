"""Descriptive calibration and paired clustered-loss inference for revision v5.

The smoother is a diagnostic estimated on assessment outcomes, never an
evaluated recalibrator. Input probabilities and realized calibration fits stay
fixed during its patient bootstrap. Three knots imply two predictor degrees of
freedom plus an intercept, not three predictor degrees of freedom.
"""
from __future__ import annotations
import numpy as np


def _expit(x):
    x = np.clip(np.asarray(x, dtype=float), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def _design(x, knots):
    a, b, c = np.asarray(knots, dtype=float)
    if not (a < b < c):
        raise ValueError('Three distinct score-defined knots are required')
    x = np.asarray(x, dtype=float)
    h = (np.maximum(x-a, 0)**3
         - (c-a)/(c-b)*np.maximum(x-b, 0)**3
         + (b-a)/(c-b)*np.maximum(x-c, 0)**3) / (c-a)**2
    return np.column_stack([np.ones(len(x)), x, h])


def _fit(X, y, w, start=None, max_iter=60, penalty=1e-6):
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    total = w.sum()
    events = np.dot(w, y)
    if total <= 0 or events <= 0 or events >= total:
        return None, 'single_class_or_empty'
    beta = (np.array([_logit([events/total])[0], 0., 0.])
            if start is None else np.asarray(start, dtype=float).copy())
    ridge = np.diag([0., penalty, penalty])

    def objective(coef):
        eta = X @ coef
        return float(np.dot(w, np.logaddexp(0., eta)-y*eta)
                     + .5*coef @ ridge @ coef)

    old = objective(beta)
    for _ in range(max_iter):
        mu = _expit(X @ beta)
        score = X.T @ (w*(y-mu)) - ridge @ beta
        curv = w*np.maximum(mu*(1-mu), 1e-10)
        H = X.T @ (X*curv[:, None]) + ridge
        try:
            step = np.linalg.solve(H, score)
        except np.linalg.LinAlgError:
            return None, 'singular_information'
        factor = 1.
        accepted = False
        for _ in range(25):
            candidate = beta + factor*step
            new = objective(candidate)
            if np.isfinite(new) and new <= old + 1e-9:
                accepted = True
                break
            factor *= .5
        if not accepted:
            return None, 'line_search_failed'
        beta = candidate
        old = new
        if np.max(np.abs(factor*step)) < 1e-7:
            return beta, 'converged'
    return None, 'iteration_limit'


def paired_cluster_mean_ci(subject, differences, *, draws=1000, seed=20261123,
                           patient_weighted=False):
    """CI for fixed per-stay paired losses, including average-over-fit losses."""
    subject = np.asarray(subject)
    d = np.asarray(differences, dtype=float)
    was_vector = d.ndim == 1
    if was_vector:
        d = d[:, None]
    if len(subject) != len(d) or not np.isfinite(d).all():
        raise ValueError('Require exact, finite paired losses for every row')
    ids, inverse = np.unique(subject, return_inverse=True)
    n = len(ids)
    if n < 2:
        raise ValueError('At least two independent patient identifiers required')
    counts = np.bincount(inverse, minlength=n).astype(float)
    sums = np.column_stack([np.bincount(inverse, weights=d[:, j], minlength=n)
                            for j in range(d.shape[1])])
    if patient_weighted:
        sums = sums/counts[:, None]
        counts = np.ones(n)
    point = sums.sum(axis=0)/counts.sum()
    rng = np.random.default_rng(seed)
    boot = np.empty((draws, d.shape[1]))
    for k in range(draws):
        weights = rng.multinomial(n, np.full(n, 1/n))
        boot[k] = weights @ sums / np.dot(weights, counts)
    low, high = np.quantile(boot, [.025, .975], axis=0)
    def out(a):
        return float(a[0]) if was_vector else a.tolist()
    return dict(point=out(point), low=out(low), high=out(high),
                n_patients=n, n_stays=len(d), bootstrap_draws=draws,
                valid_draws=draws, weighting='patient' if patient_weighted else 'stay',
                scope='patient-cluster percentile conditional on frozen per-stay losses')


def calibration_curve_cluster(subject, y, p, *, draws=1000, seed=20261123,
                              grid_points=61, min_valid_fraction=.9,
                              progress=None):
    subject, y, p = np.asarray(subject), np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if not (len(subject) == len(y) == len(p)) or not np.isfinite(p).all():
        raise ValueError('Require exact finite prediction/label/patient rows')
    if ((p < 0) | (p > 1)).any():
        raise ValueError('Prediction probabilities must lie within [0,1]')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Binary outcome required')
    ids, inverse = np.unique(subject, return_inverse=True)
    if len(np.unique(p)) < 4:
        return dict(status='constant_or_insufficient_score_support', n_patients=len(ids),
                    n_stays=len(y), events=int(y.sum()), mean_predicted=float(p.mean()),
                    observed=float(y.mean()), grid=[], curve=[], low=[], high=[],
                    valid_bootstrap=0, invalid_bootstrap=draws)
    x = _logit(p)
    knots = np.quantile(x, [.1, .5, .9])
    if len(np.unique(knots)) < 3:
        return dict(status='degenerate_knots', n_patients=len(ids), n_stays=len(y),
                    events=int(y.sum()), mean_predicted=float(p.mean()), observed=float(y.mean()),
                    grid=[], curve=[], low=[], high=[], valid_bootstrap=0, invalid_bootstrap=draws)
    X = _design(x, knots)
    support = np.quantile(p, [.01, .99])
    grid = _expit(np.linspace(_logit([support[0]])[0], _logit([support[1]])[0], grid_points))
    G = _design(_logit(grid), knots)
    beta, status = _fit(X, y, np.ones(len(y)))
    if beta is None:
        return dict(status=status, knots_logit=knots.tolist(), grid=grid.tolist(),
                    curve=[], low=[], high=[], valid_bootstrap=0, invalid_bootstrap=draws)
    fitted = _expit(G @ beta)
    n = len(ids)
    rng = np.random.default_rng(seed)
    boot, failure = [], {}
    for k in range(draws):
        cluster_weights = rng.multinomial(n, np.full(n, 1/n))
        b, state = _fit(X, y, cluster_weights[inverse], start=beta)
        if b is None:
            failure[state] = failure.get(state, 0) + 1
        else:
            boot.append(_expit(G @ b))
        if progress is not None and (k+1) % 100 == 0:
            progress(k+1, draws, len(boot))
    enough = len(boot) >= int(np.ceil(draws*min_valid_fraction))
    low, high = (np.quantile(np.asarray(boot), [.025, .975], axis=0)
                 if enough else (np.full(grid_points, np.nan), np.full(grid_points, np.nan)))
    qedges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
    bins = np.minimum(np.searchsorted(qedges, p, side='right')-1, len(qedges)-2)
    risk_bins = []
    for k in range(max(0, len(qedges)-1)):
        m = bins == k
        if m.any():
            risk_bins.append(dict(bin=k, n_stays=int(m.sum()), events=int(y[m].sum()),
                                  mean_predicted=float(p[m].mean()), observed=float(y[m].mean())))
    return dict(status='ok' if enough else 'insufficient_valid_bootstrap',
                n_stays=len(y), n_patients=n, events=int(y.sum()),
                mean_predicted=float(p.mean()), observed=float(y.mean()),
                knots_logit=knots.tolist(), predictor_df=2, coefficient_count=3,
                beta=beta.tolist(), grid=grid.tolist(), curve=fitted.tolist(),
                low=low.tolist(), high=high.tolist(), valid_bootstrap=len(boot),
                invalid_bootstrap=draws-len(boot), failures=failure, risk_bins=risk_bins,
                probability_support=support.tolist(), bootstrap_draws=draws,
                scope='pointwise patient-cluster diagnostic-smoother intervals; fixed realized predictor and knots')


def verify_analytic_invariants():
    """Deterministic numerical checks only; no synthetic publication results."""
    p = np.repeat([.1, .3, .7, .9], 100)
    y = np.concatenate([np.r_[np.ones(k), np.zeros(100-k)] for k in [10, 30, 70, 90]])
    x = _logit(p)
    knots = np.quantile(x, [.1, .5, .9])
    b, state = _fit(_design(x, knots), y, np.ones(len(y)))
    if state != 'converged' or not np.allclose(b, [0, 1, 0], atol=1e-4):
        raise AssertionError('Identity-calibration analytic case failed')
    result = paired_cluster_mean_ci(np.repeat(np.arange(100), 2), np.full(200, .125), draws=100)
    if not (result['point'] == result['low'] == result['high'] == .125):
        raise AssertionError('Constant paired loss invariant failed')
    return dict(status='passed', identity_beta=b.tolist(), constant_difference_ci=result,
                purpose='internal deterministic numeric verification, not study data')
