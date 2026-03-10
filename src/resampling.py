import numpy as np

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data_utils import load_covtype_data, get_X_y, RANDOM_STATE

# ── Settings ──────────────────────────────────────────────────────────────
DEGREES = range(1, 6)  # polynomial degrees 1 through 5
K_VALUES = [5, 10]     # plus LOOCV (K=n)
N_BOOTSTRAP = 200      # bootstrap iterations
MAX_ITER = 2000        # logistic‑regression convergence

# For LOOCV we use a smaller subset to keep runtime feasible
LOOCV_SUBSET = 500


# ── Helpers ───────────────────────────────────────────────────────────────
def _make_pipeline(degree):
    """Poly features → scale → logistic regression."""
    return make_pipeline(
        PolynomialFeatures(degree, interaction_only=False, include_bias=False),
        StandardScaler(),
        LogisticRegression(max_iter=MAX_ITER, random_state=RANDOM_STATE, solver="lbfgs"),
    )


def cross_validation_errors(X, y):
    """
    Returns dict: {k_or_'LOOCV': [error_deg1, error_deg2, ...]}
    """
    results = {}
    for k in K_VALUES:
        errs = []
        for d in DEGREES:
            pipe = _make_pipeline(d)
            acc = cross_val_score(pipe, X, y, cv=k, scoring="accuracy", n_jobs=-1)
            errs.append(1.0 - acc.mean())
        results[f"K={k}"] = errs
        print(f"  CV K={k}  done — errors: {[f'{e:.4f}' for e in errs]}")

    # LOOCV on subset
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X), size=min(LOOCV_SUBSET, len(X)), replace=False)
    X_sub, y_sub = X[idx], y[idx]
    loo = LeaveOneOut()
    errs = []
    for d in DEGREES:
        pipe = _make_pipeline(d)
        acc = cross_val_score(pipe, X_sub, y_sub, cv=loo, scoring="accuracy", n_jobs=-1)
        errs.append(1.0 - acc.mean())
        print(f"    LOOCV degree {d} done — error: {errs[-1]:.4f}")
    results["LOOCV"] = errs
    return results


def bootstrap_errors(X, y):
    """
    Returns list of errors, one per degree.
    """
    rng = np.random.RandomState(RANDOM_STATE)
    n = len(X)
    errs_by_degree = []
    for d in DEGREES:
        boot_errs = []
        for b in range(N_BOOTSTRAP):
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
    """Plot error rate vs polynomial degree for CV and Bootstrap."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # --- Left: Cross‑Validation ---
    ax = axes[0]
    for label, errs in cv_results.items():
        ax.plot(list(DEGREES), errs, marker="o", label=label)
    ax.set_xlabel("Polynomial Degree")
    ax.set_ylabel("Misclassification Error Rate")
    ax.set_title("Cross‑Validation Error vs Polynomial Degree")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Right: Bootstrap ---
    ax = axes[1]
    ax.plot(list(DEGREES), boot_errors, marker="s", color="tab:red", label="Bootstrap")
    ax.set_xlabel("Polynomial Degree")
    ax.set_title("Bootstrap Error vs Polynomial Degree")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Task 1 — Resampling Methods (Forest Cover Type)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Plot saved to {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Task 1: Resampling Methods")
    print("=" * 60)

    df = load_covtype_data()
    X, y, cols = get_X_y(df, continuous_only=True)  # use 10 continuous features
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes\n")

    # Scale once (poly features will be re‑scaled inside pipeline)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("[1a] Cross‑Validation (K=5, K=10, LOOCV) …")
    cv_results = cross_validation_errors(X_scaled, y)

    print("\n[1b] Bootstrap …")
    boot_errors = bootstrap_errors(X_scaled, y)

    print("\n[1c] Plotting …")
    plot_results(cv_results, boot_errors)

    # Summary table
    print("\n" + "─" * 55)
    print(f"{'Degree':<8}", end="")
    for label in cv_results:
        print(f"{label:<12}", end="")
    print(f"{'Bootstrap':<12}")
    print("─" * 55)
    for i, d in enumerate(DEGREES):
        print(f"{d:<8}", end="")
        for label in cv_results:
            print(f"{cv_results[label][i]:<12.4f}", end="")
        print(f"{boot_errors[i]:<12.4f}")