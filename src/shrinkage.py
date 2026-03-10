import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import mode as sp_mode
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from .data_utils import (load_credit_data, preprocess, get_X_y,
                         split_and_scale, RANDOM_STATE)

MAX_ITER = 3000
N_BOOT = 50

# C = 1/λ ; small C → strong regularisation
C_VALUES = np.logspace(-3, 3, 20)
LAMBDA_VALUES = 1.0 / C_VALUES


def bias_variance_decomposition(X_train, y_train, X_test, y_test,
                                C_values, penalty="l2", n_boot=N_BOOT):
    """
    Bootstrap bias‑variance decomposition for classification.
    bias  ≈ P(mode_prediction ≠ true)
    var   ≈ mean disagreement with mode
    error ≈ mean misclassification
    """
    rng = np.random.RandomState(RANDOM_STATE)
    n_train = len(X_train)
    n_test = len(X_test)

    bias_arr, var_arr, err_arr = [], [], []

    for ci, C in enumerate(C_values):
        preds = np.zeros((n_boot, n_test), dtype=int)
        for b in range(n_boot):
            idx = rng.randint(0, n_train, n_train)
            lr = LogisticRegression(C=C, penalty=penalty, solver="saga",
                                    max_iter=MAX_ITER,
                                    random_state=RANDOM_STATE)
            lr.fit(X_train[idx], y_train[idx])
            preds[b] = lr.predict(X_test)

        mode_preds = sp_mode(preds, axis=0, keepdims=False).mode
        bias = np.mean(mode_preds != y_test)
        variance = np.mean(preds != mode_preds[np.newaxis, :])
        total_err = np.mean(preds != y_test[np.newaxis, :])

        bias_arr.append(bias)
        var_arr.append(variance)
        err_arr.append(total_err)

        if (ci + 1) % 5 == 0 or ci == 0:
            print(f"    λ={1/C:10.4f}  bias={bias:.4f}  "
                  f"var={variance:.4f}  err={total_err:.4f}")

    return np.array(bias_arr), np.array(var_arr), np.array(err_arr)


def find_optimal_lambda(X, y, penalty="l2"):
    """5‑fold CV to find optimal λ."""
    best_C, best_acc = None, -1
    for C in C_VALUES:
        lr = LogisticRegression(C=C, penalty=penalty, solver="saga",
                                max_iter=MAX_ITER, random_state=RANDOM_STATE)
        acc = cross_val_score(lr, X, y, cv=5, scoring="accuracy",
                              n_jobs=-1).mean()
        if acc > best_acc:
            best_acc = acc
            best_C = C
    return 1.0 / best_C, best_C


def plot_bias_variance(lambdas, ridge_bv, lasso_bv,
                       ridge_opt, lasso_opt,
                       save_path="task2b_shrinkage.png"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    for ax, (bias, var, err), name, opt_lam in zip(
        axes, [ridge_bv, lasso_bv],
        ["Ridge (L2)", "Lasso (L1)"], [ridge_opt, lasso_opt],
    ):
        ax.plot(np.log10(lambdas), bias, "b-o", ms=3, label="Bias²")
        ax.plot(np.log10(lambdas), var, "r-s", ms=3, label="Variance")
        ax.plot(np.log10(lambdas), err, "k--^", ms=3, label="Total Error")
        ax.axvline(np.log10(opt_lam), color="green", ls=":", lw=2,
                   label=f"Optimal λ={opt_lam:.4f}")
        ax.set_xlabel("log₁₀(λ)")
        ax.set_ylabel("Error / Component")
        ax.set_title(f"{name} — Bias‑Variance Tradeoff")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Task 2b — Shrinkage: Bias‑Variance Tradeoff (Credit Approval)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Plot saved → {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Task 2b: Shrinkage Methods (Credit Approval)")
    print("=" * 60)

    df = preprocess(load_credit_data())
    X, y, _ = get_X_y(df, continuous_only=False)
    X_train, X_test, y_train, y_test, _ = split_and_scale(X, y)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

    print("[1] Optimal λ via 5‑fold CV …")
    ridge_lam, ridge_C = find_optimal_lambda(X_train, y_train, "l2")
    print(f"  Ridge optimal λ = {ridge_lam:.6f}  (C = {ridge_C:.4f})")
    lasso_lam, lasso_C = find_optimal_lambda(X_train, y_train, "l1")
    print(f"  Lasso optimal λ = {lasso_lam:.6f}  (C = {lasso_C:.4f})")

    print("\n[2] Bias‑Variance decomposition (Ridge) …")
    ridge_bv = bias_variance_decomposition(
        X_train, y_train, X_test, y_test, C_VALUES, penalty="l2")

    print("\n[3] Bias‑Variance decomposition (Lasso) …")
    lasso_bv = bias_variance_decomposition(
        X_train, y_train, X_test, y_test, C_VALUES, penalty="l1")

    print("\n[4] Plotting …")
    plot_bias_variance(LAMBDA_VALUES, ridge_bv, lasso_bv,
                       ridge_lam, lasso_lam)

    print("\n" + "─" * 50)
    print(f"  Ridge optimal λ = {ridge_lam:.6f}")
    print(f"  Lasso optimal λ = {lasso_lam:.6f}")
    mr = np.argmin(ridge_bv[2])
    ml = np.argmin(lasso_bv[2])
    print(f"  Ridge min error: {ridge_bv[2][mr]:.4f} at λ={LAMBDA_VALUES[mr]:.4f}")
    print(f"  Lasso min error: {lasso_bv[2][ml]:.4f} at λ={LAMBDA_VALUES[ml]:.4f}")