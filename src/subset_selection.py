import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from .data_utils import load_credit_data, preprocess, get_X_y, RANDOM_STATE

MAX_ITER = 2000


# ── Information criteria helpers ─────────────────────────────────────────
def _log_likelihood(model, X, y):
    proba = np.clip(model.predict_proba(X), 1e-15, 1.0)
    ll = 0.0
    for i, c in enumerate(y):
        col = np.where(model.classes_ == c)[0][0]
        ll += np.log(proba[i, col])
    return ll


def _null_log_likelihood(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    ll = sum(np.log(probs[np.where(classes == c)[0][0]]) for c in y)
    return ll


def compute_criteria(model, X, y, p):
    n = len(y)
    ll = _log_likelihood(model, X, y)
    ll0 = _null_log_likelihood(y)
    deviance = -2.0 * ll

    aic = -2 * ll + 2 * p
    bic = -2 * ll + np.log(n) * p
    cp = deviance / n - 2 + 2 * (p + 1) / n
    r2_mcf = 1.0 - ll / ll0
    adj_r2 = 1.0 - ((1.0 - r2_mcf) * (n - 1) / max(n - p - 1, 1))

    return {"AIC": aic, "BIC": bic, "Cp": cp, "Adj_R2": adj_r2}


# ── Forward stepwise selection ───────────────────────────────────────────
def forward_stepwise(X, y, feature_names, max_features=None):
    n_features = X.shape[1]
    if max_features is None:
        max_features = n_features

    selected = []
    remaining = list(range(n_features))
    history = []
    scaler = StandardScaler()

    for step in range(1, max_features + 1):
        best_score = -np.inf
        best_feat = None
        for f in remaining:
            candidate = selected + [f]
            X_sub = scaler.fit_transform(X[:, candidate])
            model = LogisticRegression(max_iter=MAX_ITER,
                                       random_state=RANDOM_STATE)
            cv_acc = cross_val_score(model, X_sub, y, cv=5,
                                     scoring="accuracy", n_jobs=-1).mean()
            if cv_acc > best_score:
                best_score = cv_acc
                best_feat = f

        selected.append(best_feat)
        remaining.remove(best_feat)

        X_sub = scaler.fit_transform(X[:, selected])
        model = LogisticRegression(max_iter=MAX_ITER,
                                   random_state=RANDOM_STATE)
        model.fit(X_sub, y)

        crit = compute_criteria(model, X_sub, y, p=len(selected))
        crit["CV_Error"] = 1.0 - best_score
        crit["n_features"] = step
        crit["features"] = [feature_names[i] for i in selected]
        history.append(crit)
        print(f"  Step {step:2d} | +{feature_names[best_feat]:<6} | "
              f"CV_Err={crit['CV_Error']:.4f}  AIC={crit['AIC']:.1f}  "
              f"BIC={crit['BIC']:.1f}  Adj_R²={crit['Adj_R2']:.4f}")

    return history


# ── Plotting ─────────────────────────────────────────────────────────────
def plot_subset_selection(history, save_path="task2a_subset_selection.png"):
    n_feat = [h["n_features"] for h in history]
    metrics = {
        "Cp": [h["Cp"] for h in history],
        "AIC": [h["AIC"] for h in history],
        "BIC": [h["BIC"] for h in history],
        "Adjusted R²": [h["Adj_R2"] for h in history],
        "CV Error": [h["CV_Error"] for h in history],
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()

    for idx, (name, vals) in enumerate(metrics.items()):
        ax = axes[idx]
        ax.plot(n_feat, vals, marker="o", linewidth=1.5)
        opt = np.argmax(vals) if name == "Adjusted R²" else np.argmin(vals)
        ax.axvline(n_feat[opt], color="red", ls="--", alpha=0.6)
        ax.scatter([n_feat[opt]], [vals[opt]], color="red", zorder=5, s=80,
                   label=f"Optimal p={n_feat[opt]}")
        ax.set_xlabel("Number of Predictors")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].axis("off")
    plt.suptitle("Task 2a — Subset Selection (Credit Approval)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Plot saved → {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Task 2a: Subset Selection (Credit Approval)")
    print("=" * 60)

    df = preprocess(load_credit_data())
    X, y, feature_names = get_X_y(df, continuous_only=False)
    print(f"Using all {X.shape[1]} features, {X.shape[0]} samples\n")

    print("Forward Stepwise Selection (5‑fold CV):")
    history = forward_stepwise(X, y, feature_names)

    print("\n" + "─" * 55)
    for metric in ["CV_Error", "AIC", "BIC", "Cp"]:
        opt = min(history, key=lambda h: h[metric])
        print(f"  Best by {metric:<10}: p={opt['n_features']}  "
              f"{metric}={opt[metric]:.4f}")
    opt_r2 = max(history, key=lambda h: h["Adj_R2"])
    print(f"  Best by Adj_R²    : p={opt_r2['n_features']}  "
          f"Adj_R²={opt_r2['Adj_R2']:.4f}")
    print(f"  Features (Adj_R²) : {opt_r2['features']}")

    print("\nPlotting …")
    plot_subset_selection(history)