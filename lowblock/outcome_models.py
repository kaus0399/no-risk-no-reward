"""Cross-fitted gradient boosting for the outcome models (and a tabular choice model), five folds by match."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from . import config as C

OUTCOMES = [("rest_xg", "reg"), ("opp_xg20", "reg"), ("turn5", "clf")]


def gbm(kind, n):
    kw = dict(learning_rate=0.1, max_iter=200, min_samples_leaf=50, l2_regularization=1.0, random_state=C.SEED)
    if n >= 2000:
        kw.update(early_stopping=True, validation_fraction=0.1, n_iter_no_change=10)
    else:
        kw.update(early_stopping=False, max_iter=100)
    return HistGradientBoostingClassifier(**kw) if kind == "clf" else HistGradientBoostingRegressor(**kw)


def crossfit_choice(X, A, K, fold):
    """Out-of-fold choice probabilities [n, K]."""
    P = np.zeros((len(A), K))
    for f in range(C.N_FOLDS):
        tr, te = fold != f, fold == f
        m = gbm("clf", int(tr.sum())).fit(X[tr], A[tr])
        P[np.ix_(np.where(te)[0], m.classes_)] = m.predict_proba(X[te])
    return P


def crossfit_outcome(X, A, Y, K, fold, kind):
    """Out-of-fold predicted outcome under each option [n, K], one model per option fitted on the rows that took it."""
    Mu = np.zeros((len(A), K))
    for f in range(C.N_FOLDS):
        tr, te = fold != f, fold == f
        for a in range(K):
            sel = tr & (A == a)
            y = Y[sel]
            if sel.sum() < 20 or (kind == "clf" and len(np.unique(y)) < 2):
                Mu[te, a] = y.mean() if sel.sum() else Y[tr].mean()
                continue
            if kind == "clf":
                Mu[te, a] = gbm("clf", int(sel.sum())).fit(X[sel], y.astype(int)).predict_proba(X[te])[:, 1]
            else:
                Mu[te, a] = gbm("reg", int(sel.sum())).fit(X[sel], y).predict(X[te])
    return Mu
