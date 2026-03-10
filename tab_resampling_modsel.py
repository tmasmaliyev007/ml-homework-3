import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import os, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
import analysis2 as a2

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def _show(fig, fname=None):
    st.pyplot(fig)
    if fname:
        fig.savefig(os.path.join(PLOT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════
# TAB 7 — RESAMPLING
# ════════════════════════════════════════════════════════════════════════

def render_resampling_tab(df, feature_cols, random_seed):
    st.header("🔁 Resampling Methods")
    st.markdown(
        "**Regression target:** `balance` (average yearly account balance, €).  \n"
        "We compare K-fold CV (K=5, K=10, LOOCV) and Bootstrap for estimating "
        "test error across polynomial degrees of a linear model."
    )

    # ── Configuration ──────────────────────────────────────────────────
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        sample_n   = st.slider("Sample size for analysis", 500, 5000, 3000, step=500,
                                key="rs_sample_n",
                                help="Subset of bank-full data to keep computations fast")
        loocv_n    = st.slider("LOOCV subsample size", 100, 600, 300, step=50,
                                key="rs_loocv_n",
                                help="LOOCV is O(n²) — keep small")
    with col_cfg2:
        max_degree = st.slider("Max polynomial degree", 2, 6, 5, key="rs_maxdeg")
        n_boot     = st.slider("Bootstrap iterations", 50, 500, 200, step=50, key="rs_nboot")
    with col_cfg3:
        st.markdown("**Methods included**")
        st.markdown("✅ K-Fold CV  (K=5 and K=10)  \n✅ LOOCV (K=n)  \n✅ Bootstrap (OOB)")

    run_btn = st.button("▶  Run Resampling Analysis", type="primary", key="rs_run")

    if not run_btn:
        st.info("Configure parameters above and click **Run Resampling Analysis**.")
        return

    degrees_full = range(1, max_degree + 1)
    degrees_loo  = range(1, min(max_degree, 4) + 1)   # LOOCV capped at deg 4

    with st.spinner("Preparing data …"):
        X, y, feat_names, _ = a2.prepare_regression_data(
            df, feature_cols, sample_n=sample_n, random_state=int(random_seed))
        X_loo = X[:loocv_n]; y_loo = y[:loocv_n]

    # ── 1a. K-Fold CV ──────────────────────────────────────────────────
    st.subheader("1a · Cross-Validation: K = 5, K = 10, LOOCV")
    st.markdown(
        r"""
**Theory:**  K-fold CV splits data into K folds, trains on K-1, evaluates on the held-out fold,
and averages:

$$\text{CV}_{(K)} = \frac{1}{K}\sum_{k=1}^{K} \text{MSE}_k$$

**LOOCV** is the special case K=n. It has low bias but high variance and is computationally
expensive — O(n) model fits.  We demonstrate it on a subsample of size **"""
        + str(loocv_n) + "**."
    )

    with st.spinner(f"Running K-fold CV (degrees 1–{max_degree}) …"):
        cv_res = a2.cv_mse_vs_degree(X, y, degrees=degrees_full, k_values=(5, 10),
                                      random_state=int(random_seed))
    with st.spinner(f"Running LOOCV on {loocv_n} samples (degrees 1–{max(degrees_loo)}) …"):
        loocv_res = a2.loocv_mse_vs_degree(X_loo, y_loo, degrees=degrees_loo)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        fig, ax = plt.subplots(figsize=(8, 5))
        deg_list = list(degrees_full)
        for k, col, marker, lbl in [
                (5,  a2.PAL["cv5"],   "o-",  "CV  K=5"),
                (10, a2.PAL["cv10"],  "s-",  "CV  K=10")]:
            ax.plot(deg_list, [cv_res[k][d] for d in deg_list],
                    marker, color=col, lw=2, markersize=7, label=lbl)
        loo_degs = sorted(loocv_res)
        ax.plot(loo_degs, [loocv_res[d] for d in loo_degs],
                "^--", color=a2.PAL["loocv"], lw=2, markersize=7,
                label=f"LOOCV (n={loocv_n})")
        ax.set_xlabel("Polynomial Degree", fontsize=11)
        ax.set_ylabel("Mean Squared Error  (€²)", fontsize=11)
        ax.set_title("MSE vs Polynomial Degree — CV Comparison", fontweight="bold")
        ax.legend(fontsize=10)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.tight_layout()
        _show(fig, "resample_cv_mse_degree.png")

    with col_b:
        cv_rows = []
        for d in degrees_full:
            row = {"Degree": d,
                   "CV K=5":  f"{cv_res[5][d]:,.0f}",
                   "CV K=10": f"{cv_res[10][d]:,.0f}"}
            if d in loocv_res:
                row["LOOCV"] = f"{loocv_res[d]:,.0f}"
            cv_rows.append(row)
        st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, height=280)
        best5   = min(degrees_full, key=lambda d: cv_res[5][d])
        best10  = min(degrees_full, key=lambda d: cv_res[10][d])
        st.success(f"**Optimal degree:**  K=5 → {best5}   |   K=10 → {best10}")

    # ── 1b. Bootstrap ──────────────────────────────────────────────────
    st.subheader("1b · Bootstrap Resampling")
    st.markdown(
        r"""
The **bootstrap** draws B samples of size n *with replacement*.  Out-of-Bag (OOB)
observations (~36.8% of the data on average) serve as the test set:

$$\hat{\text{Err}}^{\text{boot}} = \frac{1}{B}\sum_{b=1}^{B} \frac{1}{|C^{-b}|}
\sum_{i \in C^{-b}} L(y_i,\, \hat{f}^{*b}(x_i))$$

where $C^{{-b}}$ is the OOB set for bootstrap sample $b$.
        """
    )
    with st.spinner(f"Running {n_boot} bootstrap iterations …"):
        boot_res = a2.bootstrap_mse_vs_degree(
            X, y, degrees=degrees_full, n_boot=n_boot,
            random_state=int(random_seed))
        boot_samples = a2.bootstrap_mse_samples(
            X, y, degree=1, n_boot=n_boot, random_state=int(random_seed))

    col_c, col_d = st.columns(2)
    with col_c:
        # MSE vs degree comparison
        deg_list2 = [d for d in degrees_full
                     if d in boot_res and not np.isnan(boot_res[d][0])]
        means = [boot_res[d][0] for d in deg_list2]
        stds  = [boot_res[d][1] for d in deg_list2]
        cv5_m = [cv_res[5][d]   for d in deg_list2]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(deg_list2, means, yerr=stds, fmt="o-",
                    color=a2.PAL["boot"], lw=2, markersize=7, capsize=5,
                    label="Bootstrap  (mean ± 1 SD)")
        ax.plot(deg_list2, cv5_m, "s--", color=a2.PAL["cv5"],
                lw=2, markersize=7, label="CV  K=5")
        ax.set_xlabel("Polynomial Degree", fontsize=11)
        ax.set_ylabel("MSE  (€²)", fontsize=11)
        ax.set_title("Bootstrap vs CV K=5: MSE vs Degree", fontweight="bold")
        ax.legend(fontsize=10)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.tight_layout()
        _show(fig, "resample_bootstrap_mse_degree.png")

    with col_d:
        # Bootstrap distribution
        mu = np.mean(boot_samples)
        ci = np.percentile(boot_samples, [2.5, 97.5])
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(boot_samples, bins=35, color=a2.PAL["boot"],
                edgecolor="white", alpha=0.85)
        ax.axvline(mu,    color="black", lw=2, linestyle="--",
                   label=f"Mean = {mu:,.0f}")
        ax.axvline(ci[0], color="gray",  lw=1.5, linestyle=":",
                   label=f"95% CI [{ci[0]:,.0f}, {ci[1]:,.0f}]")
        ax.axvline(ci[1], color="gray",  lw=1.5, linestyle=":")
        ax.set_xlabel("OOB MSE  (€²)", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title("Bootstrap MSE Sampling Distribution\n(Degree-1 model)",
                     fontweight="bold")
        ax.legend(fontsize=9)
        plt.tight_layout()
        _show(fig, "resample_bootstrap_dist.png")

    # Summary table
    st.subheader("📊 Resampling Summary")
    summary_rows = []
    for d in degrees_full:
        r = {"Degree": d,
             "CV K=5 MSE":  f"{cv_res[5][d]:,.0f}",
             "CV K=10 MSE": f"{cv_res[10][d]:,.0f}"}
        if d in loocv_res:
            r["LOOCV MSE"] = f"{loocv_res[d]:,.0f}"
        if d in boot_res and not np.isnan(boot_res[d][0]):
            r["Bootstrap MSE"] = f"{boot_res[d][0]:,.0f} ± {boot_res[d][1]:,.0f}"
        summary_rows.append(r)
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    st.markdown("""
**Key takeaways:**
- K=5 CV typically has higher bias but lower variance than LOOCV.
- K=10 CV is a good compromise; preferred in practice (lower variance than LOOCV, lower bias than K=5).
- Bootstrap produces similar estimates to K-fold CV, and also gives a full sampling distribution of the error estimate.
- As polynomial degree increases beyond the optimal, all methods detect overfitting via rising MSE.
    """)


# ════════════════════════════════════════════════════════════════════════
# TAB 8 — MODEL SELECTION
# ════════════════════════════════════════════════════════════════════════

def render_model_selection_tab(df, feature_cols, random_seed):
    st.header("🎯 Model Selection Methods")
    st.markdown(
        "**Regression target:** `balance` (€).  \n"
        "We apply three model selection frameworks: "
        "subset selection, shrinkage (Ridge/Lasso), and dimensionality reduction (PCA/PLS)."
    )

    ms_sample = st.sidebar.slider("Model Selection sample size", 500, 6000, 4000,
                                   step=500, key="ms_sample")
    run_ms = st.button("▶  Run Model Selection Analysis", type="primary", key="ms_run")
    if not run_ms:
        st.info("Click **Run Model Selection Analysis** to begin (may take ~30–60 seconds).")
        return

    with st.spinner("Preparing data …"):
        X, y, feat_names, _ = a2.prepare_regression_data(
            df, feature_cols, sample_n=ms_sample, random_state=int(random_seed))
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, random_state=int(random_seed))
    p = X.shape[1]

    # ── 2a. Subset Selection ───────────────────────────────────────────
    st.subheader("2a · Forward Stepwise Subset Selection")
    st.markdown(r"""
Forward stepwise adds one predictor at a time, choosing the one that most reduces RSS:

$$\mathcal{M}_k = \arg\min_{\mathcal{M}: |\mathcal{M}|=k} \text{RSS}(\mathcal{M})$$

Selection criteria penalise model complexity:

| Criterion | Formula | Best |
|:---|:---|:---|
| **Mallow's Cₚ** | $(RSS + 2k\hat\sigma^2) / n$ | ↓ |
| **AIC** | $n\ln(RSS/n) + 2(k+1)$ | ↓ |
| **BIC** | $n\ln(RSS/n) + \ln(n)(k+1)$ | ↓ |
| **Adj R²** | $1 - \frac{RSS/(n-k-1)}{TSS/(n-1)}$ | ↑ |
    """)

    with st.spinner("Running forward stepwise selection …"):
        sub_df = a2.forward_stepwise_selection(X, y, feat_names)
        max_fs = min(p, 10)
        cv_mses_sub = a2.cv_mse_by_n_features(
            X, y, max_features=max_fs, k=5, random_state=int(random_seed))

    ns = sub_df["n_features"].values
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    specs = [
        ("Cp",    "Mallow's Cₚ",  True,  "#4C72B0"),
        ("AIC",   "AIC",           True,  "#DD8452"),
        ("BIC",   "BIC",           True,  "#55A868"),
        ("Adj_R2","Adjusted R²",  False, "#C44E52"),
    ]
    for i, (col, label, low_better, color) in enumerate(specs):
        vals = sub_df[col].values
        best = int(np.argmin(vals) if low_better else np.argmax(vals))
        axes[i].plot(ns, vals, "o-", color=color, lw=2, markersize=7)
        axes[i].scatter(ns[best], vals[best], color="red", zorder=5, s=120,
                        label=f"Optimal = {ns[best]}")
        axes[i].set_xlabel("Number of Predictors", fontsize=10)
        axes[i].set_ylabel(label, fontsize=10)
        axes[i].set_title(f"{'↓' if low_better else '↑'} {label}", fontweight="bold")
        axes[i].legend(fontsize=9)

    cv_ns   = sorted(cv_mses_sub)
    cv_vals = [cv_mses_sub[k] for k in cv_ns]
    best_cv = cv_ns[int(np.argmin(cv_vals))]
    axes[4].plot(cv_ns, cv_vals, "o-", color="#9467BD", lw=2, markersize=7)
    axes[4].scatter(best_cv, cv_mses_sub[best_cv], color="red", zorder=5,
                    s=120, label=f"Optimal = {best_cv}")
    axes[4].set_xlabel("Number of Predictors", fontsize=10)
    axes[4].set_ylabel("5-Fold CV MSE", fontsize=10)
    axes[4].set_title("↓ CV MSE", fontweight="bold")
    axes[4].legend(fontsize=9)
    axes[5].set_visible(False)
    plt.suptitle("Forward Stepwise Subset Selection — Model Selection Criteria",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    _show(fig, "subset_selection_criteria.png")

    # Criteria summary
    best_cp  = sub_df.loc[sub_df["Cp"].idxmin(),    "n_features"]
    best_aic = sub_df.loc[sub_df["AIC"].idxmin(),   "n_features"]
    best_bic = sub_df.loc[sub_df["BIC"].idxmin(),   "n_features"]
    best_ar2 = sub_df.loc[sub_df["Adj_R2"].idxmax(),"n_features"]
    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
    col_s1.metric("Best Cₚ",      f"{best_cp} vars")
    col_s2.metric("Best AIC",     f"{best_aic} vars")
    col_s3.metric("Best BIC",     f"{best_bic} vars")
    col_s4.metric("Best Adj R²",  f"{best_ar2} vars")
    col_s5.metric("Best CV MSE",  f"{best_cv} vars")

    with st.expander("📋 Full Stepwise Selection Table"):
        st.dataframe(
            sub_df[["n_features","features","Adj_R2","Cp","AIC","BIC"]].style.format({
                "Adj_R2": "{:.4f}", "Cp": "{:.2f}", "AIC": "{:.2f}", "BIC": "{:.2f}"
            }), use_container_width=True)

    # ── 2b. Ridge & Lasso ──────────────────────────────────────────────
    st.subheader("2b · Shrinkage Methods: Ridge & Lasso")
    st.markdown(r"""
**Ridge** adds an L2 penalty:  $\min_\beta \|y - X\beta\|^2 + \lambda\|\beta\|^2$

**Lasso** adds an L1 penalty:  $\min_\beta \|y - X\beta\|^2 + \lambda\|\beta\|_1$

Key difference: Lasso drives coefficients exactly to zero (feature selection);
Ridge shrinks all coefficients continuously. Optimal λ is selected via 5-fold CV.
    """)

    col_rl1, col_rl2 = st.columns(2)
    with col_rl1:
        alpha_lo = st.number_input("λ min (log10)", value=-4.0, step=0.5, key="rl_alo")
        alpha_hi = st.number_input("λ max (log10)", value=4.0,  step=0.5, key="rl_ahi")
    with col_rl2:
        n_alphas = st.slider("Number of λ values", 20, 100, 60, key="rl_nalpha")
        bv_boot  = st.slider("Bootstrap iterations (Bias-Variance)", 20, 100, 40, key="rl_bvboot")

    with st.spinner("Fitting Ridge and Lasso across λ grid …"):
        alphas = np.logspace(alpha_lo, alpha_hi, n_alphas)
        rl_res = a2.ridge_lasso_analysis(X, y, alphas=alphas, cv=5)

    ridge_best = rl_res["ridge_best_alpha"]
    lasso_best = rl_res["lasso_best_alpha"]
    col_rb, col_lb = st.columns(2)
    col_rb.metric("Ridge  Optimal λ", f"{ridge_best:.5f}")
    col_lb.metric("Lasso  Optimal λ", f"{lasso_best:.5f}")

    # CV MSE curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, mses, best, label, col in zip(
            axes,
            [rl_res["ridge_mses"], rl_res["lasso_mses"]],
            [ridge_best, lasso_best],
            ["Ridge", "Lasso"],
            [a2.PAL["ridge"], a2.PAL["lasso"]]):
        ax.semilogx(alphas, mses, color=col, lw=2)
        ax.axvline(best, color="red", linestyle="--", lw=1.5,
                   label=f"Optimal λ = {best:.5f}")
        ax.scatter(best, mses[np.argmin(mses)], color="red", zorder=5, s=80)
        ax.set_xlabel("λ  (regularisation strength)", fontsize=11)
        ax.set_ylabel("CV MSE  (€²)", fontsize=11)
        ax.set_title(f"{label}:  CV MSE vs λ", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
    plt.tight_layout()
    _show(fig, "shrinkage_cv_mse.png")

    # Bias-Variance tradeoff
    with st.spinner("Computing bias-variance decomposition …"):
        bv_alphas = np.logspace(alpha_lo, alpha_hi, min(n_alphas, 30))
        br, vr, tr_ = a2.compute_bias_variance(
            X, y, bv_alphas, "ridge", n_boot=bv_boot, random_state=int(random_seed))
        bl, vl, tl  = a2.compute_bias_variance(
            X, y, bv_alphas, "lasso", n_boot=bv_boot, random_state=int(random_seed))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, b, v, t, label in zip(
            axes, [br, bl], [vr, vl], [tr_, tl], ["Ridge", "Lasso"]):
        mask = ~np.isnan(t)
        ax.semilogx(bv_alphas[mask], b[mask], "--", lw=2,   color="#2196F3", label="Bias²")
        ax.semilogx(bv_alphas[mask], v[mask], ":",  lw=2,   color="#FF9800", label="Variance")
        ax.semilogx(bv_alphas[mask], t[mask], "-",  lw=2.5, color="#9C27B0", label="Total MSE")
        if np.any(mask):
            best_i = int(np.nanargmin(t))
            ax.axvline(bv_alphas[best_i], color="black", linestyle=":", lw=1.5,
                       label=f"Min MSE  λ={bv_alphas[best_i]:.4f}")
        ax.set_xlabel("λ", fontsize=11); ax.set_ylabel("Error  (€²)", fontsize=11)
        ax.set_title(f"{label}:  Bias–Variance Tradeoff", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
    plt.tight_layout()
    _show(fig, "shrinkage_bias_variance.png")

    # Coefficient paths
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, paths, label in zip(
            axes,
            [rl_res["ridge_coef_paths"], rl_res["lasso_coef_paths"]],
            ["Ridge", "Lasso"]):
        cmap = plt.cm.tab20
        for j in range(paths.shape[1]):
            ax.semilogx(alphas, paths[:, j], lw=1.3, alpha=0.85,
                        color=cmap(j % 20),
                        label=feat_names[j] if j < len(feat_names) else f"F{j}")
        ax.axhline(0, color="black", lw=0.5, linestyle="--")
        ax.set_xlabel("λ", fontsize=11); ax.set_ylabel("Coefficient", fontsize=11)
        ax.set_title(f"{label} Coefficient Path", fontsize=12, fontweight="bold")
        if paths.shape[1] <= 12:
            ax.legend(fontsize=7, loc="upper right", ncol=2)
    plt.tight_layout()
    _show(fig, "shrinkage_coef_paths.png")

    lasso_zero = np.sum(
        Lasso(alpha=lasso_best, max_iter=5000).fit(X_tr, y_tr).coef_ == 0)
    st.info(
        f"At optimal λ = {lasso_best:.5f}, Lasso sets **{lasso_zero} / {p}** "
        f"coefficients exactly to zero (automatic feature selection).")

    # ── 2c. PCA & PLS ──────────────────────────────────────────────────
    st.subheader("2c · Dimensionality Reduction: PCA Regression & PLS")
    st.markdown(r"""
**Principal Component Regression (PCR)** first projects X onto its top k principal
components $Z_k = XV_k$, then fits OLS on $Z_k$.

**Partial Least Squares (PLS)** finds directions that maximise covariance between X
and y simultaneously — PLS components carry more predictive signal per component than PCA.

Both are compared against full OLS (all features) on held-out test MSE.
    """)

    max_comp = st.slider("Max components to evaluate", 2, p, min(p, 10), key="pca_maxcomp")

    with st.spinner("Fitting PCA regression and PLS …"):
        pca_comps, pca_tr, pca_te, evr, pca_full = a2.pca_regression_mse(
            X_tr, y_tr, X_te, y_te, max_components=max_comp)
        pls_comps, pls_tr, pls_te = a2.pls_regression_mse(
            X_tr, y_tr, X_te, y_te, max_components=max_comp)

    ols_mse = mean_squared_error(y_te, LinearRegression().fit(X_tr, y_tr).predict(X_te))

    pca_best_n = int(pca_comps[np.argmin(pca_te)])
    pls_best_n = int(pls_comps[np.argmin(pls_te)])

    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.metric("OLS Test MSE (full)", f"{ols_mse:,.0f}")
    col_p2.metric(f"PCR Best MSE ({pca_best_n} comp.)", f"{pca_te.min():,.0f}")
    col_p3.metric(f"PLS Best MSE ({pls_best_n} comp.)", f"{pls_te.min():,.0f}")

    # PCA MSE + scree
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    best_te = int(np.argmin(pca_te))
    axes[0].plot(pca_comps, pca_tr, "o-", color="steelblue", lw=2, label="Train MSE")
    axes[0].plot(pca_comps, pca_te, "s-", color="coral",     lw=2, label="Test MSE")
    axes[0].scatter(pca_comps[best_te], pca_te[best_te], color="red",
                    zorder=5, s=120, label=f"Min Test @ {pca_comps[best_te]} comp.")
    axes[0].set_xlabel("Number of Principal Components", fontsize=11)
    axes[0].set_ylabel("MSE  (€²)", fontsize=11)
    axes[0].set_title("PCA Regression: MSE vs Components", fontweight="bold")
    axes[0].legend()

    cum = np.cumsum(evr) * 100
    axes[1].bar(pca_comps, evr * 100, color="steelblue", edgecolor="white",
                alpha=0.75, label="Individual")
    ax2 = axes[1].twinx()
    ax2.plot(pca_comps, cum, "r--o", lw=2, markersize=5, label="Cumulative")
    ax2.set_ylabel("Cumulative Variance (%)", color="red", fontsize=10)
    ax2.tick_params(axis="y", colors="red"); ax2.set_ylim(0, 105)
    axes[1].set_xlabel("Principal Component", fontsize=11)
    axes[1].set_ylabel("Explained Variance (%)", fontsize=11)
    axes[1].set_title("Scree Plot & Cumulative Variance", fontweight="bold")
    axes[1].legend(loc="upper left"); ax2.legend(loc="center right")
    plt.tight_layout()
    _show(fig, "pca_mse_variance.png")

    # PCR vs PLS comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pca_comps, pca_te, "o-", color=a2.PAL["cv5"],  lw=2, markersize=7,
            label="PCR  (PCA + OLS)")
    ax.plot(pls_comps, pls_te, "s-", color=a2.PAL["cv10"], lw=2, markersize=7,
            label="PLS Regression")
    ax.axhline(ols_mse, color="gray", lw=2, linestyle="--",
               label=f"Full OLS MSE = {ols_mse:,.0f}")
    ax.set_xlabel("Number of Components", fontsize=12)
    ax.set_ylabel("Test MSE  (€²)", fontsize=12)
    ax.set_title("PCR vs PLS vs Full OLS — Test MSE", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    _show(fig, "pca_vs_pls.png")

    # PCA biplot
    if pca_full.components_.shape[0] >= 2:
        st.subheader("PCA Biplot (PC1 vs PC2)")
        fig, ax = plt.subplots(figsize=(9, 6))
        X_proj = pca_full.transform(X_tr)
        sc = ax.scatter(X_proj[:, 0], X_proj[:, 1],
                        c=y_tr, cmap="coolwarm", alpha=0.35, s=6)
        plt.colorbar(sc, ax=ax, label="balance  (€)")
        load  = pca_full.components_[:2].T
        mag   = np.sqrt(load[:, 0] ** 2 + load[:, 1] ** 2)
        top   = np.argsort(mag)[-min(8, len(feat_names)):]
        scale = 2.5 * np.abs(X_proj[:, :2]).max()
        for i in top:
            ax.annotate("", xy=(load[i,0]*scale, load[i,1]*scale),
                        xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
            ax.text(load[i,0]*scale*1.13, load[i,1]*scale*1.13,
                    feat_names[i] if i < len(feat_names) else f"F{i}",
                    fontsize=8, ha="center", fontweight="bold")
        ax.set_xlabel(f"PC1  ({pca_full.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
        ax.set_ylabel(f"PC2  ({pca_full.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
        ax.set_title("PCA Biplot: PC1 vs PC2  (coloured by balance)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        _show(fig, "pca_biplot.png")

    # ── Final model summary ────────────────────────────────────────────
    st.subheader("📊 Final Model Comparison")
    with st.spinner("Fitting all final models …"):
        summary_df = a2.build_model_summary(
            X_tr, y_tr, X_te, y_te,
            ridge_alpha=ridge_best, lasso_alpha=lasso_best,
            pca_n_best=pca_best_n, pls_n_best=pls_best_n)

    st.dataframe(
        summary_df.style
            .format({"MSE": "{:,.0f}", "R²": "{:.4f}"})
            .highlight_min(subset=["MSE"], color="#c6efce")
            .highlight_max(subset=["R²"],  color="#c6efce"),
        use_container_width=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors_bar = ["#4C72B0","#9467BD","#8C564B","#55A868","#DD8452"]
    axes[0].barh(summary_df["Model"], summary_df["MSE"],
                 color=colors_bar, edgecolor="white")
    axes[0].set_xlabel("Test MSE  (€²)", fontsize=11)
    axes[0].set_title("Test MSE by Model", fontweight="bold")
    axes[1].barh(summary_df["Model"], summary_df["R²"],
                 color=colors_bar, edgecolor="white")
    axes[1].set_xlabel("R²", fontsize=11)
    axes[1].set_title("R² by Model", fontweight="bold")
    plt.tight_layout()
    _show(fig, "model_selection_final_comparison.png")

    st.markdown("""
**Key takeaways:**
- **Subset selection** uses interpretable criteria (BIC tends to select the most parsimonious model).
- **Ridge** keeps all features but shrinks coefficients; best when all features contribute.
- **Lasso** performs automatic feature selection; best when only a few features are relevant.
- **PCR** maximises explained variance in X — components may not predict y well.
- **PLS** maximises explained covariance between X and y — typically needs fewer components than PCR.
    """)
