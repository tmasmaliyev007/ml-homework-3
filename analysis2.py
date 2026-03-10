import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model        import LinearRegression, Ridge, Lasso
from sklearn.preprocessing       import StandardScaler, PolynomialFeatures
from sklearn.decomposition       import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection     import KFold, LeaveOneOut, cross_val_score, train_test_split
from sklearn.metrics             import mean_squared_error, r2_score
from sklearn.pipeline            import Pipeline

sns.set_theme(style="whitegrid", palette="muted")

PAL = {"cv5":   "#4C72B0", "cv10":  "#DD8452", "loocv": "#55A868",
       "boot":  "#C44E52", "ridge": "#9467BD", "lasso": "#8C564B"}

REGRESSION_TARGET = "balance"

# ════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════

def savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def prepare_regression_data(df, feature_cols, target=REGRESSION_TARGET,
                             sample_n=4000, random_state=42):
    """
    Sub-sample df, keep numeric-origin features (avoid dummy explosion in poly),
    standardise and return (X_scaled, y, feature_names, scaler).
    """
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(df), size=min(sample_n, len(df)), replace=False)
    sub = df.iloc[idx].reset_index(drop=True)

    preferred = ["age", "day", "duration", "campaign",
                 "pdays", "previous", "default",
                 "housing", "loan", "pdays_contacted"]
    num_feats = [c for c in preferred if c in feature_cols]
    if len(num_feats) < 4:
        num_feats = feature_cols[:10]

    X = sub[num_feats].values.astype(float)
    y = sub[target].values.astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, y, list(num_feats), scaler


# ════════════════════════════════════════════════════════════════════════
# 1.  RESAMPLING
# ════════════════════════════════════════════════════════════════════════

# ── 1a  K-Fold CV ────────────────────────────────────────────────────────

def cv_mse_vs_degree(X, y, degrees=range(1, 7), k_values=(5, 10),
                     random_state=42):
    """Returns {k: {degree: mean_cv_mse}}."""
    results = {}
    for k in k_values:
        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
        d_map = {}
        for d in degrees:
            pipe = Pipeline([
                ("poly",   PolynomialFeatures(degree=d, include_bias=False)),
                ("scaler", StandardScaler()),
                ("lr",     LinearRegression()),
            ])
            scores = cross_val_score(pipe, X, y, cv=kf,
                                     scoring="neg_mean_squared_error")
            d_map[d] = float(-scores.mean())
        results[k] = d_map
    return results


def loocv_mse_vs_degree(X, y, degrees=range(1, 5)):
    """LOOCV (K=n) — call on a small subsample for tractability."""
    loo = LeaveOneOut()
    d_map = {}
    for d in degrees:
        pipe = Pipeline([
            ("poly",   PolynomialFeatures(degree=d, include_bias=False)),
            ("scaler", StandardScaler()),
            ("lr",     LinearRegression()),
        ])
        scores = cross_val_score(pipe, X, y, cv=loo,
                                 scoring="neg_mean_squared_error")
        d_map[d] = float(-scores.mean())
    return d_map


def plot_cv_mse_vs_degree(cv_results, loocv_results, degrees, plot_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    deg_list = list(degrees)
    for k, col, marker, lbl in [
            (5,  PAL["cv5"],   "o-",  "CV  K = 5"),
            (10, PAL["cv10"],  "s-",  "CV  K = 10")]:
        mses = [cv_results[k][d] for d in deg_list]
        ax.plot(deg_list, mses, marker, color=col, lw=2, markersize=7, label=lbl)

    loo_degs = sorted(loocv_results)
    loo_vals = [loocv_results[d] for d in loo_degs]
    ax.plot(loo_degs, loo_vals, "^--", color=PAL["loocv"], lw=2,
            markersize=7, label="LOOCV (K = n)")

    ax.set_xlabel("Polynomial Degree", fontsize=12)
    ax.set_ylabel("Mean Squared Error  (€²)", fontsize=12)
    ax.set_title("Cross-Validation: MSE vs Polynomial Degree\n"
                 "(Regression target: account balance)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    savefig(fig, f"{plot_dir}/resample_cv_mse_degree.png")


# ── 1b  Bootstrap ────────────────────────────────────────────────────────

def bootstrap_mse_vs_degree(X, y, degrees=range(1, 7),
                              n_boot=200, random_state=42):
    """Returns {degree: (mean_mse, std_mse)} using OOB samples."""
    rng = np.random.default_rng(random_state)
    n   = len(X)
    out = {}
    for d in degrees:
        poly = PolynomialFeatures(degree=d, include_bias=False)
        sc   = StandardScaler()
        lr   = LinearRegression()
        mses = []
        for _ in range(n_boot):
            b_idx = rng.integers(0, n, size=n)
            oob   = np.setdiff1d(np.arange(n), b_idx)
            if len(oob) < 5:
                continue
            try:
                Xb   = sc.fit_transform(poly.fit_transform(X[b_idx]))
                Xoob = sc.transform(poly.transform(X[oob]))
                lr.fit(Xb, y[b_idx])
                mses.append(mean_squared_error(y[oob], lr.predict(Xoob)))
            except Exception:
                continue
        out[d] = (float(np.mean(mses)) if mses else np.nan,
                  float(np.std(mses))  if mses else np.nan)
    return out


def bootstrap_mse_samples(X, y, degree=1, n_boot=300, random_state=42):
    """List of per-iteration OOB MSEs for a single degree (for histogram)."""
    rng  = np.random.default_rng(random_state)
    n    = len(X)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    sc   = StandardScaler()
    lr   = LinearRegression()
    mses = []
    for _ in range(n_boot):
        b_idx = rng.integers(0, n, size=n)
        oob   = np.setdiff1d(np.arange(n), b_idx)
        if len(oob) < 5:
            continue
        try:
            Xb   = sc.fit_transform(poly.fit_transform(X[b_idx]))
            Xoob = sc.transform(poly.transform(X[oob]))
            lr.fit(Xb, y[b_idx])
            mses.append(mean_squared_error(y[oob], lr.predict(Xoob)))
        except Exception:
            continue
    return mses


def plot_bootstrap_mse_vs_degree(boot_results, cv_results, degrees, plot_dir):
    deg_list  = [d for d in degrees
                 if d in boot_results and not np.isnan(boot_results[d][0])]
    means     = [boot_results[d][0] for d in deg_list]
    stds      = [boot_results[d][1] for d in deg_list]
    cv5_means = [cv_results[5][d]   for d in deg_list]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(deg_list, means, yerr=stds, fmt="o-", color=PAL["boot"],
                lw=2, markersize=7, capsize=5, label="Bootstrap  (mean ± 1 SD)")
    ax.plot(deg_list, cv5_means, "s--", color=PAL["cv5"],
            lw=2, markersize=7, label="CV  K = 5")
    ax.set_xlabel("Polynomial Degree", fontsize=12)
    ax.set_ylabel("Mean Squared Error  (€²)", fontsize=12)
    ax.set_title("Bootstrap vs K-Fold CV: MSE vs Polynomial Degree",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    savefig(fig, f"{plot_dir}/resample_bootstrap_mse_degree.png")


def plot_bootstrap_distribution(mse_samples, plot_dir):
    mu = np.mean(mse_samples)
    ci = np.percentile(mse_samples, [2.5, 97.5])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(mse_samples, bins=40, color=PAL["boot"], edgecolor="white", alpha=0.85)
    ax.axvline(mu,    color="black", lw=2, linestyle="--",
               label=f"Mean = {mu:,.0f}")
    ax.axvline(ci[0], color="gray",  lw=1.5, linestyle=":",
               label=f"95% CI [{ci[0]:,.0f},  {ci[1]:,.0f}]")
    ax.axvline(ci[1], color="gray",  lw=1.5, linestyle=":")
    ax.set_xlabel("OOB MSE  (€²)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Bootstrap MSE Distribution — Degree-1 Model\n"
                 "Sampling distribution of the test-error estimate",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    savefig(fig, f"{plot_dir}/resample_bootstrap_dist.png")


# ════════════════════════════════════════════════════════════════════════
# 2.  MODEL SELECTION
# ════════════════════════════════════════════════════════════════════════

# ── 2a  Forward Stepwise Subset Selection ───────────────────────────────

def forward_stepwise_selection(X, y, feature_names):
    n, p = X.shape
    lr_full  = LinearRegression().fit(X, y)
    rss_full = float(np.sum((y - lr_full.predict(X)) ** 2))
    sigma2_f = rss_full / max(n - p - 1, 1)

    remaining, selected = list(range(p)), []
    rows = []

    for k in range(1, p + 1):
        best_rss, best_feat = np.inf, None
        for feat in remaining:
            combo = selected + [feat]
            lr    = LinearRegression().fit(X[:, combo], y)
            rss   = float(np.sum((y - lr.predict(X[:, combo])) ** 2))
            if rss < best_rss:
                best_rss, best_feat = rss, feat

        selected.append(best_feat)
        remaining.remove(best_feat)

        Xs    = X[:, selected]
        lr    = LinearRegression().fit(Xs, y)
        y_hat = lr.predict(Xs)
        rss   = float(np.sum((y - y_hat) ** 2))
        tss   = float(np.sum((y - y.mean()) ** 2))
        r2    = 1 - rss / tss
        adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - k - 1, 1)
        cp    = (rss + 2 * k * sigma2_f) / n
        aic   = n * np.log(rss / n) + 2 * (k + 1)
        bic   = n * np.log(rss / n) + np.log(n) * (k + 1)
        rows.append({
            "n_features": k,
            "features":   tuple(feature_names[i] for i in selected),
            "RSS": rss, "R2": r2, "Adj_R2": adj_r2,
            "Cp": cp, "AIC": aic, "BIC": bic,
        })
    return pd.DataFrame(rows)


def cv_mse_by_n_features(X, y, max_features, k=5, random_state=42):
    """
    Honest k-fold CV MSE for each forward-stepwise model size.
    Reruns forward selection inside each fold.
    Returns {n_features: cv_mse}.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    cv_mses = {}
    for size in range(1, max_features + 1):
        fold_mses = []
        for tr_idx, te_idx in kf.split(X):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_te, y_te = X[te_idx], y[te_idx]
            rem, sel = list(range(X.shape[1])), []
            for _ in range(size):
                best_r, best_f = np.inf, None
                for f in rem:
                    combo = sel + [f]
                    lr    = LinearRegression().fit(X_tr[:, combo], y_tr)
                    r_    = mean_squared_error(y_tr, lr.predict(X_tr[:, combo]))
                    if r_ < best_r:
                        best_r, best_f = r_, f
                sel.append(best_f); rem.remove(best_f)
            lr = LinearRegression().fit(X_tr[:, sel], y_tr)
            fold_mses.append(mean_squared_error(y_te, lr.predict(X_te[:, sel])))
        cv_mses[size] = float(np.mean(fold_mses))
    return cv_mses


def plot_subset_selection_criteria(subset_df, cv_mses, plot_dir):
    ns   = subset_df["n_features"].values
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    specs = [
        ("Cp",    "Mallow's Cₚ",  True,  "#4C72B0"),
        ("AIC",   "AIC",           True,  "#DD8452"),
        ("BIC",   "BIC",           True,  "#55A868"),
        ("Adj_R2","Adjusted R²",  False, "#C44E52"),
    ]
    for i, (col, label, low_better, color) in enumerate(specs):
        vals = subset_df[col].values
        best = int(np.argmin(vals) if low_better else np.argmax(vals))
        axes[i].plot(ns, vals, "o-", color=color, lw=2, markersize=7)
        axes[i].scatter(ns[best], vals[best], color="red", zorder=5,
                        s=120, label=f"Optimal = {ns[best]}")
        axes[i].set_xlabel("Number of Predictors", fontsize=10)
        axes[i].set_ylabel(label, fontsize=10)
        axes[i].set_title(f"{'↓' if low_better else '↑'} {label}",
                          fontweight="bold")
        axes[i].legend(fontsize=9)

    cv_ns   = sorted(cv_mses)
    cv_vals = [cv_mses[k] for k in cv_ns]
    best_cv = cv_ns[int(np.argmin(cv_vals))]
    axes[4].plot(cv_ns, cv_vals, "o-", color="#9467BD", lw=2, markersize=7)
    axes[4].scatter(best_cv, cv_mses[best_cv], color="red", zorder=5,
                    s=120, label=f"Optimal = {best_cv}")
    axes[4].set_xlabel("Number of Predictors", fontsize=10)
    axes[4].set_ylabel("5-Fold CV MSE", fontsize=10)
    axes[4].set_title("↓ CV MSE", fontweight="bold")
    axes[4].legend(fontsize=9)
    axes[5].set_visible(False)

    plt.suptitle("Forward Stepwise Selection — Model Selection Criteria",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    savefig(fig, f"{plot_dir}/subset_selection_criteria.png")


# ── 2b  Ridge & Lasso ───────────────────────────────────────────────────

def ridge_lasso_analysis(X, y, alphas=None, cv=5):
    """
    Full Ridge/Lasso analysis.
    Returns dict with MSE curves, best alphas, coefficient paths.
    """
    if alphas is None:
        alphas = np.logspace(-4, 4, 80)

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    ridge_mses, lasso_mses = [], []
    ridge_paths, lasso_paths = [], []

    for a in alphas:
        rm, lm = [], []
        for tr, te in kf.split(X):
            r_ = Ridge(alpha=a).fit(X[tr], y[tr])
            l_ = Lasso(alpha=a, max_iter=5000).fit(X[tr], y[tr])
            rm.append(mean_squared_error(y[te], r_.predict(X[te])))
            lm.append(mean_squared_error(y[te], l_.predict(X[te])))
        ridge_mses.append(float(np.mean(rm)))
        lasso_mses.append(float(np.mean(lm)))
        ridge_paths.append(Ridge(alpha=a).fit(X, y).coef_)
        lasso_paths.append(Lasso(alpha=a, max_iter=5000).fit(X, y).coef_)

    ridge_mses = np.array(ridge_mses)
    lasso_mses = np.array(lasso_mses)
    return {
        "alphas":            alphas,
        "ridge_mses":        ridge_mses,
        "lasso_mses":        lasso_mses,
        "ridge_best_alpha":  alphas[int(np.argmin(ridge_mses))],
        "lasso_best_alpha":  alphas[int(np.argmin(lasso_mses))],
        "ridge_coef_paths":  np.array(ridge_paths),
        "lasso_coef_paths":  np.array(lasso_paths),
    }


def compute_bias_variance(X, y, alphas, model_type="ridge",
                           n_boot=80, random_state=42):
    """
    Bootstrap bias-variance decomposition for Ridge or Lasso.
    Returns (bias², variance, total_mse) arrays per alpha.
    """
    rng    = np.random.default_rng(random_state)
    n      = len(X)
    te_idx = rng.choice(n, size=max(int(0.2 * n), 20), replace=False)
    tr_mask = np.ones(n, dtype=bool); tr_mask[te_idx] = False
    X_tr, y_tr = X[tr_mask], y[tr_mask]
    X_te, y_te = X[te_idx],  y[te_idx]
    MC = Ridge if model_type == "ridge" else Lasso

    biases, variances, totals = [], [], []
    for a in alphas:
        preds = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(X_tr), size=len(X_tr))
            try:
                m = MC(alpha=a, max_iter=3000).fit(X_tr[idx], y_tr[idx])
                preds.append(m.predict(X_te))
            except Exception:
                continue
        if len(preds) < 5:
            biases.append(np.nan); variances.append(np.nan); totals.append(np.nan)
            continue
        P    = np.array(preds)
        mu_p = P.mean(axis=0)
        b2   = float(np.mean((mu_p - y_te) ** 2))
        var  = float(np.mean(P.var(axis=0)))
        biases.append(b2); variances.append(var); totals.append(b2 + var)

    return np.array(biases), np.array(variances), np.array(totals)


def plot_ridge_lasso_cv_mse(res, plot_dir):
    alphas = res["alphas"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, mses, best, label, col in zip(
            axes,
            [res["ridge_mses"], res["lasso_mses"]],
            [res["ridge_best_alpha"], res["lasso_best_alpha"]],
            ["Ridge", "Lasso"],
            [PAL["ridge"], PAL["lasso"]]):
        ax.semilogx(alphas, mses, color=col, lw=2)
        ax.axvline(best, color="red", linestyle="--", lw=1.5,
                   label=f"Optimal λ = {best:.4f}")
        ax.scatter(best, mses[np.argmin(mses)], color="red", zorder=5, s=80)
        ax.set_xlabel("λ  (regularisation strength)", fontsize=11)
        ax.set_ylabel("CV MSE  (€²)", fontsize=11)
        ax.set_title(f"{label}:  CV MSE vs λ", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
    plt.tight_layout()
    savefig(fig, f"{plot_dir}/shrinkage_cv_mse.png")


def plot_bias_variance_tradeoff(alphas, br, vr, tr_, bl, vl, tl, plot_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, b, v, t, label in zip(
            axes, [br, bl], [vr, vl], [tr_, tl], ["Ridge", "Lasso"]):
        mask = ~np.isnan(t)
        ax.semilogx(alphas[mask], b[mask], "--", lw=2,   color="#2196F3", label="Bias²")
        ax.semilogx(alphas[mask], v[mask], ":",  lw=2,   color="#FF9800", label="Variance")
        ax.semilogx(alphas[mask], t[mask], "-",  lw=2.5, color="#9C27B0", label="Total MSE")
        best_i = int(np.nanargmin(t))
        ax.axvline(alphas[best_i], color="black", linestyle=":", lw=1.5,
                   label=f"Min MSE  λ={alphas[best_i]:.4f}")
        ax.set_xlabel("λ", fontsize=11); ax.set_ylabel("Error  (€²)", fontsize=11)
        ax.set_title(f"{label}:  Bias–Variance Tradeoff", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
    plt.tight_layout()
    savefig(fig, f"{plot_dir}/shrinkage_bias_variance.png")


def plot_coefficient_paths(res, feature_names, plot_dir):
    alphas = res["alphas"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, paths, label in zip(
            axes,
            [res["ridge_coef_paths"], res["lasso_coef_paths"]],
            ["Ridge", "Lasso"]):
        cmap = plt.cm.tab20
        for j in range(paths.shape[1]):
            fname = feature_names[j] if j < len(feature_names) else f"F{j}"
            ax.semilogx(alphas, paths[:, j], lw=1.3, alpha=0.85,
                        color=cmap(j % 20), label=fname)
        ax.axhline(0, color="black", lw=0.5, linestyle="--")
        ax.set_xlabel("λ", fontsize=11); ax.set_ylabel("Coefficient", fontsize=11)
        ax.set_title(f"{label} Coefficient Path", fontsize=12, fontweight="bold")
        if paths.shape[1] <= 12:
            ax.legend(fontsize=7, loc="upper right", ncol=2)
    plt.tight_layout()
    savefig(fig, f"{plot_dir}/shrinkage_coef_paths.png")


# ── 2c  PCA Regression & PLS ────────────────────────────────────────────

def pca_regression_mse(X_tr, y_tr, X_te, y_te, max_components=None):
    mc = min(max_components or X_tr.shape[1],
             X_tr.shape[1], X_tr.shape[0] - 1)
    pca_full = PCA(n_components=mc).fit(X_tr)
    evr      = pca_full.explained_variance_ratio_
    tr_mses, te_mses = [], []
    for k in range(1, mc + 1):
        pca   = PCA(n_components=k)
        Xp_tr = pca.fit_transform(X_tr)
        Xp_te = pca.transform(X_te)
        lr    = LinearRegression().fit(Xp_tr, y_tr)
        tr_mses.append(mean_squared_error(y_tr, lr.predict(Xp_tr)))
        te_mses.append(mean_squared_error(y_te, lr.predict(Xp_te)))
    return (np.arange(1, mc + 1),
            np.array(tr_mses), np.array(te_mses),
            evr[:mc], pca_full)


def pls_regression_mse(X_tr, y_tr, X_te, y_te, max_components=None):
    mc = min(max_components or X_tr.shape[1],
             X_tr.shape[1], X_tr.shape[0] - 1)
    tr_mses, te_mses = [], []
    for k in range(1, mc + 1):
        pls = PLSRegression(n_components=k, max_iter=500).fit(X_tr, y_tr)
        tr_mses.append(mean_squared_error(y_tr, pls.predict(X_tr).ravel()))
        te_mses.append(mean_squared_error(y_te, pls.predict(X_te).ravel()))
    return (np.arange(1, mc + 1),
            np.array(tr_mses), np.array(te_mses))


def plot_pca_mse(comps, tr_mse, te_mse, evr, plot_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    best_te = int(np.argmin(te_mse))
    axes[0].plot(comps, tr_mse, "o-", color="steelblue", lw=2, label="Train MSE")
    axes[0].plot(comps, te_mse, "s-", color="coral",     lw=2, label="Test MSE")
    axes[0].scatter(comps[best_te], te_mse[best_te], color="red",
                    zorder=5, s=120, label=f"Min Test @ {comps[best_te]} comp.")
    axes[0].set_xlabel("Number of Principal Components", fontsize=11)
    axes[0].set_ylabel("MSE  (€²)", fontsize=11)
    axes[0].set_title("PCA Regression: MSE vs Components", fontweight="bold")
    axes[0].legend()

    cum = np.cumsum(evr) * 100
    axes[1].bar(comps, evr * 100, color="steelblue", edgecolor="white",
                alpha=0.75, label="Individual")
    ax2 = axes[1].twinx()
    ax2.plot(comps, cum, "r--o", lw=2, markersize=5, label="Cumulative")
    ax2.set_ylabel("Cumulative Explained Variance (%)", fontsize=10, color="red")
    ax2.tick_params(axis="y", colors="red"); ax2.set_ylim(0, 105)
    axes[1].set_xlabel("Principal Component", fontsize=11)
    axes[1].set_ylabel("Explained Variance (%)", fontsize=11)
    axes[1].set_title("Scree Plot & Cumulative Variance", fontweight="bold")
    axes[1].legend(loc="upper left"); ax2.legend(loc="center right")
    plt.tight_layout()
    savefig(fig, f"{plot_dir}/pca_mse_variance.png")


def plot_pca_vs_pls(pca_comps, pca_te, pls_comps, pls_te,
                    baseline_mse, plot_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pca_comps, pca_te, "o-", color=PAL["cv5"],  lw=2, markersize=7,
            label="PCR  (PCA + OLS)")
    ax.plot(pls_comps, pls_te, "s-", color=PAL["cv10"], lw=2, markersize=7,
            label="PLS Regression")
    ax.axhline(baseline_mse, color="gray", lw=2, linestyle="--",
               label=f"Full OLS (all features)  MSE = {baseline_mse:,.0f}")
    ax.set_xlabel("Number of Components", fontsize=12)
    ax.set_ylabel("Test MSE  (€²)", fontsize=12)
    ax.set_title("PCR vs PLS vs Full OLS — Test MSE vs Components",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    savefig(fig, f"{plot_dir}/pca_vs_pls.png")


def plot_pca_biplot(pca_model, X_tr, y_tr, feature_names, plot_dir, n_arrows=8):
    X_proj = pca_model.transform(X_tr)
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(X_proj[:, 0], X_proj[:, 1],
                    c=y_tr, cmap="coolwarm", alpha=0.35, s=6)
    plt.colorbar(sc, ax=ax, label="balance  (€)")

    load  = pca_model.components_[:2].T
    mag   = np.sqrt(load[:, 0] ** 2 + load[:, 1] ** 2)
    top   = np.argsort(mag)[-n_arrows:]
    scale = 2.5 * np.abs(X_proj[:, :2]).max()

    for i in top:
        ax.annotate("", xy=(load[i, 0] * scale, load[i, 1] * scale),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
        ax.text(load[i, 0] * scale * 1.13, load[i, 1] * scale * 1.13,
                feature_names[i] if i < len(feature_names) else f"F{i}",
                fontsize=8, ha="center", fontweight="bold")

    ax.set_xlabel(f"PC1  ({pca_model.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2  ({pca_model.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
    ax.set_title("PCA Biplot: PC1 vs PC2  (coloured by balance)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, f"{plot_dir}/pca_biplot.png")


# ── Final model summary ──────────────────────────────────────────────────

def build_model_summary(X_tr, y_tr, X_te, y_te,
                         ridge_alpha, lasso_alpha,
                         pca_n_best, pls_n_best):
    rows = []
    lr = LinearRegression().fit(X_tr, y_tr)
    rows.append({"Model": "OLS  (all features)",
                 "MSE": mean_squared_error(y_te, lr.predict(X_te)),
                 "R²":  r2_score(y_te, lr.predict(X_te))})

    r_ = Ridge(alpha=ridge_alpha).fit(X_tr, y_tr)
    rows.append({"Model": f"Ridge  (λ={ridge_alpha:.4f})",
                 "MSE": mean_squared_error(y_te, r_.predict(X_te)),
                 "R²":  r2_score(y_te, r_.predict(X_te))})

    l_ = Lasso(alpha=lasso_alpha, max_iter=5000).fit(X_tr, y_tr)
    rows.append({"Model": f"Lasso  (λ={lasso_alpha:.4f})",
                 "MSE": mean_squared_error(y_te, l_.predict(X_te)),
                 "R²":  r2_score(y_te, l_.predict(X_te))})

    pca  = PCA(n_components=pca_n_best)
    lr_p = LinearRegression()
    Xp_tr = pca.fit_transform(X_tr); Xp_te = pca.transform(X_te)
    lr_p.fit(Xp_tr, y_tr)
    rows.append({"Model": f"PCR  ({pca_n_best} components)",
                 "MSE": mean_squared_error(y_te, lr_p.predict(Xp_te)),
                 "R²":  r2_score(y_te, lr_p.predict(Xp_te))})

    pls = PLSRegression(n_components=pls_n_best, max_iter=500).fit(X_tr, y_tr)
    rows.append({"Model": f"PLS  ({pls_n_best} components)",
                 "MSE": mean_squared_error(y_te, pls.predict(X_te).ravel()),
                 "R²":  r2_score(y_te, pls.predict(X_te).ravel())})

    return pd.DataFrame(rows)
