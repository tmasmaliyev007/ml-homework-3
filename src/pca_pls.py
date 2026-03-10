import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.pipeline import Pipeline

from src.data_utils import load_covtype_data, get_X_y, split_and_scale, RANDOM_STATE

MAX_ITER = 2000


# ── PCA analysis ─────────────────────────────────────────────────────────
def pca_component_analysis(X_train, y_train, X_test, y_test, max_comp=None):
    """
    For n_components = 1 … max_comp, fit PCA + Logistic Regression and
    record 5‑fold CV error and test error.
    """
    if max_comp is None:
        max_comp = X_train.shape[1]

    cv_errors = []
    test_errors = []
    explained_vars = []

    pca_full = PCA(n_components=max_comp, random_state=RANDOM_STATE)
    pca_full.fit(X_train)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)

    for nc in range(1, max_comp + 1):
        pipe = Pipeline([
            ("pca", PCA(n_components=nc, random_state=RANDOM_STATE)),
            ("lr", LogisticRegression(max_iter=MAX_ITER, random_state=RANDOM_STATE)),
        ])
        cv_acc = cross_val_score(pipe, X_train, y_train, cv=5,
                                 scoring="accuracy", n_jobs=-1).mean()
        cv_errors.append(1.0 - cv_acc)

        pipe.fit(X_train, y_train)
        test_errors.append(1.0 - pipe.score(X_test, y_test))
        explained_vars.append(cum_var[nc - 1])

        print(f"  PCA nc={nc:2d}  CV_err={cv_errors[-1]:.4f}  "
              f"Test_err={test_errors[-1]:.4f}  Cum_var={explained_vars[-1]:.4f}")

    return cv_errors, test_errors, explained_vars


# ── PLS analysis ─────────────────────────────────────────────────────────
def pls_component_analysis(X_train, y_train, X_test, y_test, max_comp=None):
    """
    PLS: project X,y jointly, then classify with Logistic Regression.
    """
    if max_comp is None:
        max_comp = min(X_train.shape[1], 10)

    lb = LabelBinarizer()
    Y_train_bin = lb.fit_transform(y_train)

    cv_errors = []
    test_errors = []

    for nc in range(1, max_comp + 1):
        pls = PLSRegression(n_components=nc, max_iter=500)
        pls.fit(X_train, Y_train_bin)

        X_train_pls = pls.transform(X_train)
        X_test_pls = pls.transform(X_test)

        lr = LogisticRegression(max_iter=MAX_ITER, random_state=RANDOM_STATE)
        cv_acc = cross_val_score(lr, X_train_pls, y_train, cv=5,
                                 scoring="accuracy", n_jobs=-1).mean()
        cv_errors.append(1.0 - cv_acc)

        lr.fit(X_train_pls, y_train)
        test_errors.append(1.0 - lr.score(X_test_pls, y_test))

        print(f"  PLS nc={nc:2d}  CV_err={cv_errors[-1]:.4f}  "
              f"Test_err={test_errors[-1]:.4f}")

    return cv_errors, test_errors


# ── No‑reduction baseline ────────────────────────────────────────────────
def baseline_accuracy(X_train, y_train, X_test, y_test):
    lr = LogisticRegression(max_iter=MAX_ITER, random_state=RANDOM_STATE)
    cv_acc = cross_val_score(lr, X_train, y_train, cv=5,
                             scoring="accuracy", n_jobs=-1).mean()
    lr.fit(X_train, y_train)
    test_acc = lr.score(X_test, y_test)
    return 1.0 - cv_acc, 1.0 - test_acc


# ── Plotting ─────────────────────────────────────────────────────────────
def plot_pca_pls(pca_cv, pca_test, pls_cv, pls_test, explained_vars,
                 baseline_cv, baseline_test,
                 save_path="task2c_pca_pls.png"):
    n_pca = range(1, len(pca_cv) + 1)
    n_pls = range(1, len(pls_cv) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- (a) Explained variance ---
    ax = axes[0]
    ax.bar(n_pca, explained_vars, color="steelblue", alpha=0.7)
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA — Cumulative Explained Variance")
    ax.axhline(0.95, color="red", ls="--", label="95 % threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- (b) PCA error vs components ---
    ax = axes[1]
    ax.plot(list(n_pca), pca_cv, "b-o", markersize=4, label="PCA CV Error")
    ax.plot(list(n_pca), pca_test, "b--s", markersize=4, label="PCA Test Error")
    ax.axhline(baseline_cv, color="gray", ls=":", label=f"Baseline CV={baseline_cv:.4f}")
    ax.axhline(baseline_test, color="gray", ls="--", alpha=0.5,
               label=f"Baseline Test={baseline_test:.4f}")
    opt_pca = np.argmin(pca_cv) + 1
    ax.axvline(opt_pca, color="red", ls=":", alpha=0.7, label=f"Optimal PCA nc={opt_pca}")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Misclassification Error")
    ax.set_title("PCA — Error vs Number of Components")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (c) PCA vs PLS comparison ---
    ax = axes[2]
    n_common = min(len(pca_cv), len(pls_cv))
    ax.plot(range(1, n_common + 1), pca_cv[:n_common], "b-o", markersize=4, label="PCA CV Error")
    ax.plot(range(1, n_common + 1), pls_cv[:n_common], "r-s", markersize=4, label="PLS CV Error")
    ax.plot(range(1, n_common + 1), pca_test[:n_common], "b--o", markersize=3, alpha=0.5,
            label="PCA Test Error")
    ax.plot(range(1, n_common + 1), pls_test[:n_common], "r--s", markersize=3, alpha=0.5,
            label="PLS Test Error")
    ax.axhline(baseline_cv, color="gray", ls=":", label="Baseline CV")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Misclassification Error")
    ax.set_title("PCA vs PLS — Comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Task 2c — PCA & PLS Dimension Reduction (Forest Cover Type)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Plot saved to {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Task 2c: PCA and PLS")
    print("=" * 60)

    df = load_covtype_data()
    X, y, _ = get_X_y(df, continuous_only=True)
    X_train, X_test, y_train, y_test, _ = split_and_scale(X, y)
    n_feat = X_train.shape[1]
    print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

    # Baseline (all features, no reduction)
    print("[0] Baseline (no dimensionality reduction) …")
    bl_cv, bl_test = baseline_accuracy(X_train, y_train, X_test, y_test)
    print(f"  Baseline CV error = {bl_cv:.4f}, Test error = {bl_test:.4f}\n")

    # PCA
    print("[1] PCA Component Analysis …")
    pca_cv, pca_test, expl_var = pca_component_analysis(
        X_train, y_train, X_test, y_test, max_comp=n_feat
    )

    # PLS
    max_pls = min(n_feat, 10)
    print(f"\n[2] PLS Component Analysis (up to {max_pls} components) …")
    pls_cv, pls_test = pls_component_analysis(
        X_train, y_train, X_test, y_test, max_comp=max_pls
    )

    # Plot
    print("\n[3] Plotting …")
    plot_pca_pls(pca_cv, pca_test, pls_cv, pls_test, expl_var, bl_cv, bl_test)

    # Summary
    print("\n" + "─" * 60)
    opt_pca = np.argmin(pca_cv) + 1
    opt_pls = np.argmin(pls_cv) + 1
    print(f"  Baseline          : CV err = {bl_cv:.4f}   Test err = {bl_test:.4f}")
    print(f"  Best PCA (nc={opt_pca:2d})  : CV err = {pca_cv[opt_pca-1]:.4f}   "
          f"Test err = {pca_test[opt_pca-1]:.4f}")
    print(f"  Best PLS (nc={opt_pls:2d})  : CV err = {pls_cv[opt_pls-1]:.4f}   "
          f"Test err = {pls_test[opt_pls-1]:.4f}")

    if pls_cv[opt_pls - 1] < pca_cv[opt_pca - 1]:
        print("  → PLS outperforms PCA (supervised reduction leverages label info)")
    else:
        print("  → PCA performs comparably / better than PLS on this data")


if __name__ == "__main__":
    main()