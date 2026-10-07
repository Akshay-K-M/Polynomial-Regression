"""
Automated polynomial regression: degree + model selection by K-fold CV.

For each problem (var1, var2) and each polynomial degree d:
    features = all monomials of total degree <= d (standardised)
    models   = OLS (closed form), Ridge (closed form), Lasso, Elastic Net
    score    = mean K-fold validation MSE (same folds for every model/degree)
The (degree, model, hyper-parameters) with the lowest CV MSE is refit on ALL
training rows and used to predict the test file.

Usage:
    python poly_regression.py --data-dir . --roll IMT2024014 --out-dir out
Expects  <roll>_train_var1.csv, <roll>_test_var1.csv, ... in --data-dir.
Writes   <roll>_pred_var1.csv, <roll>_pred_var2.csv, cv_results_var*.csv and
         cv_mse_vs_degree_var*.png in --out-dir.
"""
import argparse
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ----------------------------------------------------------------- settings
DEGREES = {"var1": range(1, 11), "var2": range(1, 21)}   # up to 10 / up to 20
K = 5
SEED = 0
RIDGE_ALPHAS = np.logspace(-4, 4, 41)                    # ridge penalty grid
ENET_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]                  # L1 share for elastic net
N_ALPHAS = 40                                            # lambda grid size (lasso/enet)
MAX_ITER = 5000
PATIENCE = 3      # stop raising the degree after this many degrees without a new best CV MSE
TOL = 0.01        # prefer the lowest degree whose CV MSE is within 1% of the overall best


# ----------------------------------------------------------------- features
def make_features(X_tr, X_te, degree):
    """All monomials with total degree <= degree, standardised using train stats."""
    pf = PolynomialFeatures(degree, include_bias=False)
    P_tr, P_te = pf.fit_transform(X_tr), pf.transform(X_te)
    sc = StandardScaler().fit(P_tr)
    return sc.transform(P_tr), sc.transform(P_te)


# ------------------------------------------- closed-form OLS / Ridge (lecture 7)
def ridge_fit(X, y, alpha):
    """w = (X^T X + alpha I)^-1 X^T y on centred data, via SVD (works for p > n).
    alpha = N*lambda in the lecture's notation. Returns (coef, intercept)."""
    xm, ym = X.mean(0), y.mean()
    U, s, Vt = np.linalg.svd(X - xm, full_matrices=False)
    coef = Vt.T @ (s / (s ** 2 + alpha) * (U.T @ (y - ym)))
    return coef, ym - xm @ coef


def ols_fit(X, y):
    """Least squares (normal equation / pseudo-inverse solution)."""
    xm, ym = X.mean(0), y.mean()
    coef = np.linalg.lstsq(X - xm, y - ym, rcond=None)[0]
    return coef, ym - xm @ coef


def ridge_cv(X, y, folds, alphas):
    """Mean validation MSE for every alpha (one SVD per fold)."""
    sse = np.zeros(len(alphas))
    for tr, va in folds:
        xm, ym = X[tr].mean(0), y[tr].mean()
        U, s, Vt = np.linalg.svd(X[tr] - xm, full_matrices=False)
        Uty = U.T @ (y[tr] - ym)
        Xv = (X[va] - xm) @ Vt.T                                   # (n_val, r)
        W = (s[:, None] / (s[:, None] ** 2 + alphas[None, :])) * Uty[:, None]   # (r, n_alphas)
        pred = Xv @ W + ym
        sse += ((y[va][:, None] - pred) ** 2).sum(0)
    return sse / len(y)


def ols_cv(X, y, folds):
    sse = 0.0
    for tr, va in folds:
        coef, b = ols_fit(X[tr], y[tr])
        sse += ((y[va] - (X[va] @ coef + b)) ** 2).sum()
    return sse / len(y)


# ---------------------------------------------------------------- main search
def evaluate_degree(X, y, folds, n_train_fold):
    """Return list of dicts: best CV result of each model family at this degree."""
    out = []
    p = X.shape[1]

    if p < n_train_fold:                                   # OLS only if determined
        out.append(dict(model="ols", cv_mse=ols_cv(X, y, folds), params={}, n_features=p))

    mse = ridge_cv(X, y, folds, RIDGE_ALPHAS)
    i = int(np.argmin(mse))
    out.append(dict(model="ridge", cv_mse=mse[i], params={"alpha": float(RIDGE_ALPHAS[i])}, n_features=p))

    for name, ratios in (("lasso", [1.0]), ("elasticnet", ENET_RATIOS)):
        best = None
        for r in ratios:
            # lambda grid: from the smallest alpha that zeroes every coef, down 1000x
            a_max = np.max(np.abs(X.T @ (y - y.mean()))) / (len(y) * r)
            grid = a_max * np.logspace(0, -3, N_ALPHAS)
            m = ElasticNetCV(l1_ratio=r, alphas=grid, cv=folds, max_iter=MAX_ITER,
                             tol=1e-4, n_jobs=-1, random_state=SEED).fit(X, y)
            cv = m.mse_path_.mean(axis=1)                  # (n_alphas,)
            j = int(np.argmin(cv))
            if best is None or cv[j] < best["cv_mse"]:
                best = dict(model=name, cv_mse=cv[j],
                            params={"alpha": float(m.alphas_[j]), "l1_ratio": r}, n_features=p)
        out.append(best)
    return out


def fit_predict(model, params, X_tr, y_tr, X_te):
    if model == "ols":
        coef, b = ols_fit(X_tr, y_tr)
    elif model == "ridge":
        coef, b = ridge_fit(X_tr, y_tr, params["alpha"])
    else:
        m = ElasticNet(alpha=params["alpha"], l1_ratio=params["l1_ratio"],
                       max_iter=20000, tol=1e-6).fit(X_tr, y_tr)
        coef, b = m.coef_, m.intercept_
    return X_te @ coef + b, int(np.sum(np.abs(coef) > 1e-12))


COLORS = {"ols": "#7a7a7a", "ridge": "#0072B2", "lasso": "#D55E00", "elasticnet": "#009E73"}
MARKERS = {"ols": "s", "ridge": "o", "lasso": "^", "elasticnet": "D"}


def pick_winner(res):
    """Lowest degree whose best CV MSE is within TOL of the overall best; then best model there."""
    ok = res[res["cv_mse"] <= res["cv_mse"].min() * (1 + TOL)]
    ok = ok[ok["degree"] == ok["degree"].min()]
    return ok.loc[ok["cv_mse"].idxmin()]


def print_summary(res, var, best):
    """One row per degree: the best model at that degree, with CV MSE and CV R2."""
    top = res.loc[res.groupby("degree")["cv_mse"].idxmin(),
                  ["degree", "n_features", "model", "cv_mse", "cv_r2"]].copy()
    top["chosen"] = np.where(top["degree"] == best["degree"], "<== chosen", "")
    top = top.rename(columns={"n_features": "terms", "model": "best_model",
                              "cv_mse": "CV_MSE", "cv_r2": "CV_R2"})
    print(f"\n===== {var}: best model at each degree ({K}-fold CV) =====")
    print(top.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def plot_curves(res, var, chosen_degree, path):
    """CV MSE (log scale) and CV R2 against polynomial degree, one line per model family."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, label in zip(axes, ["cv_mse", "cv_r2"], [f"{K}-fold CV MSE (log scale)", f"{K}-fold CV R$^2$"]):
        for model, g in res.groupby("model"):
            g = g.sort_values("degree")
            ax.plot(g["degree"], g[col], marker=MARKERS[model], ms=5, lw=1.6,
                    color=COLORS[model], label=model)
        ax.axvline(chosen_degree, color="k", ls="--", lw=1, label=f"chosen (degree {chosen_degree})")
        ax.set_xlabel("polynomial degree"); ax.set_ylabel(label)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yscale("log")
    axes[1].set_ylim(0, 1.0)            # unregularised fits that blow up (R2 < 0) fall off the plot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.suptitle(f"{var}: model selection by {K}-fold cross-validation")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_problem(var, args):
    tr = pd.read_csv(os.path.join(args.data_dir, f"{args.roll}_train_{var}.csv"))
    te = pd.read_csv(os.path.join(args.data_dir, f"{args.roll}_test_{var}.csv"))
    feats = [c for c in tr.columns if c != "y"]
    X_tr, y_tr, X_te = tr[feats].values, tr["y"].values, te[feats].values
    folds = list(KFold(K, shuffle=True, random_state=SEED).split(X_tr))
    n_fold_train = len(folds[0][0])
    var_y = y_tr.var()

    rows, best_so_far, stale = [], np.inf, 0
    for d in DEGREES[var]:
        t0 = time.time()
        P_tr, _ = make_features(X_tr, X_te, d)
        for r in evaluate_degree(P_tr, y_tr, folds, n_fold_train):
            r["degree"] = d
            r["cv_r2"] = 1 - r["cv_mse"] / var_y
            rows.append(r)
        top = min((r for r in rows if r["degree"] == d), key=lambda r: r["cv_mse"])
        print(f"[{var}] degree {d:2d} ({P_tr.shape[1]:5d} terms)  best={top['model']:<10s} "
              f"CV MSE={top['cv_mse']:.4f}  CV R2={top['cv_r2']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if top["cv_mse"] < best_so_far:
            best_so_far, stale = top["cv_mse"], 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"[{var}] no improvement for {PATIENCE} degrees -> stopping sweep", flush=True)
                break

    res = pd.DataFrame(rows)
    res["params"] = res["params"].astype(str)
    res.to_csv(os.path.join(args.out_dir, f"cv_results_{var}.csv"), index=False)

    # ---- winner -> plots/summary -> refit on ALL training rows -> predict test
    best = pick_winner(res)
    d = int(best["degree"])
    print_summary(res, var, best)
    plot_curves(res, var, d, os.path.join(args.out_dir, f"cv_curves_{var}.png"))

    params = next(r["params"] for r in rows if r["degree"] == d and r["model"] == best["model"])
    P_tr, P_te = make_features(X_tr, X_te, d)
    pred, nnz = fit_predict(best["model"], params, P_tr, y_tr, P_te)
    pd.DataFrame({"y": pred}).to_csv(os.path.join(args.out_dir, f"{args.roll}_pred_{var}.csv"),
                                     index=False)
    print(f"\n[{var}] FINAL MODEL: degree={d}, model={best['model']}, params={params}\n"
          f"       CV MSE={best['cv_mse']:.4f}, CV R2={best['cv_r2']:.4f}, "
          f"non-zero coefficients={nnz}/{P_tr.shape[1]}\n", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--roll", default="IMT2024014")
    ap.add_argument("--vars", nargs="+", default=["var1", "var2"])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for v in args.vars:
        run_problem(v, args)