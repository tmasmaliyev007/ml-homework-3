import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.pipeline import make_pipeline

from .data_utils import (load_credit_data, preprocess, get_X_y, RANDOM_STATE,
                         CONTINUOUS_COLS)

# ── Settings ─────────────────────────────────────────────────────────────
DEGREES = range(1, 6)   # polynomial degrees 1–5
K_VALUES = [5, 10]       # plus LOOCV
N_BOOTSTRAP = 50         # bootstrap iterations
MAX_ITER = 2000


def _make_pipeline(degree):
    """Poly features → scale → logistic regression."""
    return make_pipeline(
        PolynomialFeatures(degree, interaction_only=False, include_bias=False),
        StandardScaler(),
        LogisticRegression(max_iter=MAX_ITER, random_state=RANDOM_STATE,
                           solver="lbfgs"),
    )


def cross_validation_errors(X, y):
    """Returns dict: {label: [error_per_degree]}"""
    results = {}
    for k in K_VALUES:
        errs = []
        for d in DEGREES:
            pipe = _make_pipeline(d)
            acc = cross_val_score(pipe, X, y, cv=k, scoring="accuracy",
                                  n_jobs=-1)
            errs.append(1.0 - acc.mean())
        results[f"K={k}"] = errs
        print(f"  CV K={k:>2}  errors: {[f'{e:.4f}' for e in errs]}")

    # LOOCV — feasible because n=690
    loo = LeaveOneOut()
    errs = []
    for d in DEGREES:
        pipe = _make_pipeline(d)
        acc = cross_val_score(pipe, X, y, cv=loo, scoring="accuracy",
                              n_jobs=-1)
        errs.append(1.0 - acc.mean())
        print(f"    LOOCV degree {d} — error: {errs[-1]:.4f}")
    results["LOOCV"] = errs
    return results


def bootstrap_errors(X, y):
    """OOB bootstrap estimate of error for each degree."""
    rng = np.random.RandomState(RANDOM_STATE)
    n = len(X)
    errs_by_degree = []
    for d in DEGREES:
        boot_errs = []
        for _ in range(N_BOOTSTRAP):
            idx_train = rng.randint(0, n, n)
            idx_oob = np.setdiff1d(np.arange(n), idx_train)
            if len(idx_oob) == 0:
                continue
            pipe = _make_pipeline(d)
            pipe.fit(X[idx_train], y[idx_train])
            acc = pipe.score(X[idx_oob], y[idx_oob])
            boot_errs.append(1.0 - acc)
        errs_by_degree.append(np.mean(boot_errs))
        print(f"  Bootstrap degree {d} — mean error: {errs_by_degree[-1]:.4f}")
    return errs_by_degree


def plot_results(cv_results, boot_errors, save_path="task1_resampling.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    ax = axes[0]
    for label, errs in cv_results.items():
        ax.plot(list(DEGREES), errs, marker="o", label=label)
    ax.set_xlabel("Polynomial Degree")
    ax.set_ylabel("Misclassification Error Rate")
    ax.set_title("Cross‑Validation Error vs Polynomial Degree")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(list(DEGREES), boot_errors, marker="s", color="tab:red",
            label="Bootstrap (OOB)")
    ax.set_xlabel("Polynomial Degree")
    ax.set_title("Bootstrap Error vs Polynomial Degree")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Task 1 — Resampling Methods (Credit Approval)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Plot saved → {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Task 1: Resampling Methods (Credit Approval)")
    print("=" * 60)

    df = preprocess(load_credit_data())
    # Use continuous features only for polynomial expansion
    X, y, cols = get_X_y(df, continuous_only=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Data: {X.shape[0]} samples, {len(cols)} continuous features "
          f"({cols}), binary target\n")

    print("[1a] Cross‑Validation (K=5, K=10, LOOCV) …")
    cv_results = cross_validation_errors(X_scaled, y)

    print("\n[1b] Bootstrap …")
    boot_errors = bootstrap_errors(X_scaled, y)

    print("\n[1c] Plotting …")
    plot_results(cv_results, boot_errors)

    # Summary table
    print("\n" + "─" * 58)
    print(f"{'Degree':<8}", end="")
    for label in cv_results:
        print(f"{label:<12}", end="")
    print(f"{'Bootstrap':<12}")
    print("─" * 58)
    for i, d in enumerate(DEGREES):
        print(f"{d:<8}", end="")
        for label in cv_results:
            print(f"{cv_results[label][i]:<12.4f}", end="")
        print(f"{boot_errors[i]:<12.4f}")
