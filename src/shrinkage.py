import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from src.data_utils import load_covtype_data, get_X_y, split_and_scale, RANDOM_STATE

MAX_ITER = 3000
N_BOOT = 100  # bootstrap rounds for bias‑variance decomposition

# Regularization grid
C_VALUES = np.logspace(-4, 4, 30)
LAMBDA_VALUES = 1.0 / C_VALUES


# ── Bias‑Variance decomposition via bootstrap ────────────────────────────
def bias_variance_decomposition(X_train, y_train, X_test, y_test,
                                C_values, penalty="l2", n_boot=N_BOOT):
    """
    For each C, repeat bootstrap training n_boot times.  On the test set
    compute:
      • bias²  = E[(mode_prediction − true)²]  (≈ fraction where the
                  most‑frequent predicted class ≠ true class)
      • variance = 1 − (frequency of mode prediction)   (disagreement)
      • total error = mean misclassification rate

    Returns arrays of shape (len(C_values),) for bias, variance, total_error.
    """
    rng = np.random.RandomState(RANDOM_STATE)
    n_train = len(X_train)
    n_test = len(X_test)

    bias_arr = []
    var_arr = []
    err_arr = []

    for ci, C in enumerate(C_values):
        preds = np.zeros((n_boot, n_test), dtype=int)  # predictions per bootstrap
        for b in range(n_boot):
            idx = rng.randint(0, n_train, n_train)
            model = LogisticRegression(
                C=C, penalty=penalty, solver="saga", max_iter=MAX_ITER,
                random_state=RANDOM_STATE, multi_class="multinomial",
            )
            model.fit(X_train[idx], y_train[idx])
            preds[b] = model.predict(X_test)

        # Mode prediction per test sample
        from scipy.stats import mode as sp_mode
        mode_preds = sp_mode(preds, axis=0, keepdims=False).mode

        # Bias: fraction where mode ≠ true
        bias = np.mean(mode_preds != y_test)

        # Variance: average disagreement with mode
        variance = np.mean(preds != mode_preds[np.newaxis, :])

        # Total error: average misclassification across bootstraps
        total_err = np.mean(preds != y_test[np.newaxis, :])

        bias_arr.append(bias)
        var_arr.append(variance)
        err_arr.append(total_err)

        if (ci + 1) % 10 == 0 or ci == 0:
            print(f"    λ={1/C:10.4f}  (C={C:.4e})  bias={bias:.4f}  "
                  f"var={variance:.4f}  err={total_err:.4f}")

    return np.array(bias_arr), np.array(var_arr), np.array(err_arr)


# ── Optimal λ by cross‑validation ────────────────────────────────────────
def find_optimal_lambda(X, y, penalty="l2"):
    model = LogisticRegressionCV(
        Cs=C_VALUES, penalty=penalty, solver="saga", max_iter=MAX_ITER,
        cv=5, random_state=RANDOM_STATE, multi_class="multinomial",
        scoring="accuracy",
    )
    model.fit(X, y)
    best_C = model.C_[0]
    best_lambda = 1.0 / best_C
    return best_lambda, best_C, model


# ── Plotting ──────────────────────────────────────────────────────────────
def plot_bias_variance(lambdas, ridge_bv, lasso_bv,
                       ridge_opt, lasso_opt,
                       save_path="task2b_shrinkage.png"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    for ax, (bias, var, err), name, opt_lam in zip(
        axes,
        [ridge_bv, lasso_bv],
        ["Ridge (L2)", "Lasso (L1)"],
        [ridge_opt, lasso_opt],
    ):
        ax.plot(np.log10(lambdas), bias, "b-o", markersize=3, label="Bias²")
        ax.plot(np.log10(lambdas), var, "r-s", markersize=3, label="Variance")
        ax.plot(np.log10(lambdas), err, "k--^", markersize=3, label="Total Error")
        ax.axvline(np.log10(opt_lam), color="green", linestyle=":", lw=2,
                   label=f"Optimal λ={opt_lam:.4f}")
        ax.set_xlabel("log₁₀(λ)")
        ax.set_ylabel("Error / Component")
        ax.set_title(f"{name} — Bias‑Variance Tradeoff")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Task 2b — Shrinkage: Bias‑Variance Tradeoff", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Plot saved to {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Task 2b: Shrinkage Methods (Ridge & Lasso)")
    print("=" * 60)

    df = load_covtype_data()
    X, y, _ = get_X_y(df, continuous_only=True)
    X_train, X_test, y_train, y_test, _ = split_and_scale(X, y)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

    # ── Optimal lambdas via CV ──
    print("[1] Finding optimal λ by 5‑fold CV …")
    ridge_lam, ridge_C, _ = find_optimal_lambda(X_train, y_train, penalty="l2")
    print(f"  Ridge optimal λ = {ridge_lam:.6f}  (C = {ridge_C:.4f})")
    lasso_lam, lasso_C, _ = find_optimal_lambda(X_train, y_train, penalty="l1")
    print(f"  Lasso optimal λ = {lasso_lam:.6f}  (C = {lasso_C:.4f})")

    # ── Bias‑variance decomposition ──
    print("\n[2] Bias‑Variance decomposition (Ridge) …")
    ridge_bv = bias_variance_decomposition(
        X_train, y_train, X_test, y_test, C_VALUES, penalty="l2"
    )

    print("\n[3] Bias‑Variance decomposition (Lasso) …")
    lasso_bv = bias_variance_decomposition(
        X_train, y_train, X_test, y_test, C_VALUES, penalty="l1"
    )

    # ── Plot ──
    print("\n[4] Plotting …")
    plot_bias_variance(LAMBDA_VALUES, ridge_bv, lasso_bv, ridge_lam, lasso_lam)

    # ── Summary ──
    print("\n" + "─" * 50)
    print(f"  Ridge  →  optimal λ = {ridge_lam:.6f}")
    print(f"  Lasso  →  optimal λ = {lasso_lam:.6f}")
    min_ridge = np.argmin(ridge_bv[2])
    min_lasso = np.argmin(lasso_bv[2])
    print(f"  Ridge min total error: {ridge_bv[2][min_ridge]:.4f} at λ={LAMBDA_VALUES[min_ridge]:.4f}")
    print(f"  Lasso min total error: {lasso_bv[2][min_lasso]:.4f} at λ={LAMBDA_VALUES[min_lasso]:.4f}")


if __name__ == "__main__":
    main()