import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import expit

from sklearn.linear_model import LogisticRegression, LinearRegression, PoissonRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    mean_squared_error, r2_score,
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted")

# ─── Column metadata ──────────────────────────────────────────────────────────

CATEGORICAL_COLS = ["job", "marital", "education", "contact", "month", "poutcome"]
BINARY_COLS      = ["default", "housing", "loan"]   # yes/no encoded to 0/1
NUMERIC_COLS     = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
TARGET_COL       = "y"

# For Poisson regression target (campaign contact count — natural count variable)
POISSON_TARGET   = "campaign"

# ─── Data loading & preprocessing ────────────────────────────────────────────

def load_and_preprocess(path: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Returns
    -------
    df_raw   : raw dataframe (for EDA display)
    df       : encoded dataframe ready for modelling
    features : list of feature column names
    """
    df_raw = pd.read_csv(path, sep=";")

    df = df_raw.copy()

    # Encode binary yes/no columns
    for col in BINARY_COLS + [TARGET_COL]:
        df[col] = (df[col] == "yes").astype(int)

    # One-hot encode categoricals (drop_first to avoid multicollinearity)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True, dtype=int)

    # pdays == -1 means "not previously contacted" → replace with 0,
    # add indicator flag
    df["pdays_contacted"] = (df["pdays"] >= 0).astype(int)
    df["pdays"] = df["pdays"].replace(-1, 0)

    feature_cols = [c for c in df.columns if c not in [TARGET_COL]]
    return df_raw, df, feature_cols


def split_and_scale(df: pd.DataFrame, features: list[str],
                    target: str = TARGET_COL,
                    test_size: float = 0.20,
                    random_state: int = 42):
    X = df[features]
    y = df[target]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(X_tr)
    Xs_te = scaler.transform(X_te)
    return X_tr, X_te, y_tr, y_te, Xs_tr, Xs_te, scaler


# ─── Logistic regression with inference stats ─────────────────────────────────

def fit_logistic_with_stats(Xs_train, y_train, feature_names: list[str]):
    """
    Fit LR and return (model, stats_df) where stats_df has
    Coefficient, Std_Error, Z_Statistic, P_Value, Odds_Ratio for each feature + intercept.
    """
    model = LogisticRegression(max_iter=2000, random_state=42, solver="lbfgs", C=1.0)
    model.fit(Xs_train, y_train)

    coef      = model.coef_[0]
    intercept = model.intercept_[0]
    all_coef  = np.concatenate([[intercept], coef])

    # Fisher information → covariance matrix → SE
    Xb    = np.column_stack([np.ones(len(Xs_train)), Xs_train])
    p_hat = expit(Xb @ all_coef)
    W     = p_hat * (1 - p_hat)
    try:
        H   = (Xb * W[:, None]).T @ Xb        # X^T W X
        cov = np.linalg.inv(H)
        se  = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        se = np.full(len(all_coef), np.nan)

    z   = all_coef / se
    pv  = 2 * (1 - stats.norm.cdf(np.abs(z)))
    or_ = np.exp(all_coef)

    result = pd.DataFrame({
        "Feature":      ["Intercept"] + list(feature_names),
        "Coefficient":  all_coef,
        "Std_Error":    se,
        "Z_Statistic":  z,
        "P_Value":      pv,
        "Odds_Ratio":   or_,
    })
    return model, result


# ─── Confounding analysis ─────────────────────────────────────────────────────

def confounding_analysis(Xs_train, y_train, feature_names: list[str],
                         exposure: str, confounder: str):
    """
    Compare coefficient of `exposure` with vs without `confounder` in the model.
    Returns dict with both coefs and % change.
    """
    names = list(feature_names)
    if exposure not in names or confounder not in names:
        return None

    _, stats_full = fit_logistic_with_stats(Xs_train, y_train, names)
    coef_full = stats_full.loc[stats_full["Feature"] == exposure, "Coefficient"].values[0]

    # Remove confounder column
    idx_conf  = names.index(confounder)
    Xs_no_conf = np.delete(Xs_train, idx_conf, axis=1)
    names_no_c = [n for n in names if n != confounder]
    _, stats_red = fit_logistic_with_stats(Xs_no_conf, y_train, names_no_c)
    coef_red  = stats_red.loc[stats_red["Feature"] == exposure, "Coefficient"].values[0]

    change_pct = abs(coef_full - coef_red) / (abs(coef_red) + 1e-9) * 100
    return {"full": coef_full, "reduced": coef_red, "change_pct": change_pct}


# ─── Evaluation helpers ───────────────────────────────────────────────────────

def evaluate_binary(model, Xs_test, y_test, threshold=0.5):
    proba = model.predict_proba(Xs_test)[:, 1]
    pred  = (proba >= threshold).astype(int)
    return {
        "accuracy":  accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall":    recall_score(y_test, pred, zero_division=0),
        "f1":        f1_score(y_test, pred, zero_division=0),
        "auc":       roc_auc_score(y_test, proba),
        "proba":     proba,
        "pred":      pred,
    }


def youden_threshold(y_test, proba):
    fpr, tpr, thresholds = roc_curve(y_test, proba)
    j       = tpr - fpr
    best_i  = np.argmax(j)
    return thresholds[best_i], fpr, tpr


# ─── Plot helpers ──────────────────────────────────────────────────────────────

def savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── Individual plot functions (called from app.py & standalone script) ───────

def plot_eda_distributions(df_raw, numeric_cols, plot_dir):
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols[:9]):
        axes[i].hist(df_raw[col].dropna(), bins=35, color="steelblue",
                     edgecolor="white", alpha=0.85)
        axes[i].set_title(col, fontsize=11)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Numeric Feature Distributions – Bank Marketing Dataset",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig(fig, f"{plot_dir}/eda_distributions.png")


def plot_eda_categoricals(df_raw, cat_cols, plot_dir):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        vc = df_raw[col].value_counts()
        axes[i].bar(vc.index, vc.values, color="steelblue", edgecolor="white")
        axes[i].set_title(col, fontsize=11)
        axes[i].tick_params(axis="x", rotation=30)
    plt.suptitle("Categorical Feature Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig(fig, f"{plot_dir}/eda_categoricals.png")


def plot_eda_target(df_raw, plot_dir):
    vc = df_raw["y"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(["No (88%)", "Yes (12%)"], vc.values,
                color=["steelblue", "coral"], edgecolor="white")
    axes[0].set_title("Target Distribution (Subscribed?)", fontweight="bold")
    axes[0].set_ylabel("Count")
    # Subscription rate by job
    rates = df_raw.groupby("job")["y"].apply(lambda s: (s == "yes").mean()).sort_values()
    axes[1].barh(rates.index, rates.values * 100, color="steelblue", edgecolor="white")
    axes[1].set_xlabel("Subscription Rate (%)")
    axes[1].set_title("Subscription Rate by Job", fontweight="bold")
    plt.tight_layout()
    savefig(fig, f"{plot_dir}/eda_target.png")


def plot_eda_correlation(df, numeric_cols, plot_dir):
    cols = [c for c in numeric_cols if c in df.columns] + ["y"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax, square=True)
    ax.set_title("Correlation Heatmap (Numeric Features + Target)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, f"{plot_dir}/eda_correlation.png")


def plot_eda_duration_balance(df_raw, plot_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for val, col, lbl in zip(["no", "yes"], ["steelblue", "coral"], ["No", "Yes"]):
        sub = df_raw[df_raw["y"] == val]["duration"]
        axes[0].hist(sub, bins=40, alpha=0.6, color=col, label=lbl, edgecolor="white")
    axes[0].set_title("Call Duration by Outcome", fontweight="bold")
    axes[0].set_xlabel("Duration (seconds)"); axes[0].legend()
    df_raw.boxplot(column="balance", by="y", ax=axes[1],
                   boxprops=dict(color="steelblue"), medianprops=dict(color="red"))
    axes[1].set_title("Balance by Subscription Outcome"); axes[1].set_xlabel("Subscribed?")
    plt.suptitle("")
    plt.tight_layout()
    savefig(fig, f"{plot_dir}/eda_duration_balance.png")


def plot_lr_coefficients(lr_stats, plot_dir, top_n=20):
    feat_df = lr_stats[lr_stats["Feature"] != "Intercept"].copy()
    feat_df["abs_coef"] = feat_df["Coefficient"].abs()
    feat_df = feat_df.nlargest(top_n, "abs_coef").sort_values("Coefficient")
    colors  = ["coral" if v > 0 else "steelblue" for v in feat_df["Coefficient"]]

    fig, ax = plt.subplots(figsize=(9, max(5, len(feat_df) * 0.38)))
    ax.barh(feat_df["Feature"], feat_df["Coefficient"], color=colors, edgecolor="white")
    ax.errorbar(feat_df["Coefficient"], feat_df["Feature"],
                xerr=1.96 * feat_df["Std_Error"], fmt="none",
                color="black", capsize=3, linewidth=1)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Top {top_n} LR Coefficients with 95% CI\n(Standardized Features)",
                 fontweight="bold")
    ax.set_xlabel("Coefficient")
    plt.tight_layout()
    savefig(fig, f"{plot_dir}/lr_coefficients.png")


def plot_lr_pvalues(lr_stats, plot_dir, top_n=20):
    feat_df = lr_stats[lr_stats["Feature"] != "Intercept"].copy()
    feat_df["abs_coef"] = feat_df["Coefficient"].abs()
    feat_df = feat_df.nlargest(top_n, "abs_coef").sort_values("P_Value", ascending=False)
    colors  = ["green" if p < 0.05 else "gray" for p in feat_df["P_Value"]]

    fig, ax = plt.subplots(figsize=(9, max(5, len(feat_df) * 0.38)))
    ax.barh(feat_df["Feature"], -np.log10(feat_df["P_Value"].clip(1e-300)),
            color=colors, edgecolor="white")
    ax.axvline(-np.log10(0.05), color="red", linestyle="--", label="p = 0.05")
    ax.set_title(f"Feature Significance (−log₁₀ p-value) – Top {top_n}",
                 fontweight="bold")
    ax.set_xlabel("−log₁₀(p-value)"); ax.legend()
    plt.tight_layout()
    savefig(fig, f"{plot_dir}/lr_pvalues.png")


def plot_confounding(result: dict, exposure: str, confounder: str, plot_dir):
    labels = [f"Full model\n(with {confounder})", f"Reduced model\n(without {confounder})"]
    vals   = [result["full"], result["reduced"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, vals, color=["steelblue", "coral"], edgecolor="white", width=0.45)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(
        f"Confounding Analysis: '{exposure}' coefficient\n"
        f"Change when removing '{confounder}': {result['change_pct']:.1f}%",
        fontweight="bold"
    )
    ax.set_ylabel("Coefficient (standardized)")
    plt.tight_layout()
    savefig(fig, f"{plot_dir}/lr_confounding.png")


def plot_multiclass_cm(y_test, y_pred, class_names, title, fname, plot_dir):
    cm  = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    savefig(fig, f"{plot_dir}/{fname}")


def plot_lda_roc_threshold(y_test, proba, plot_dir, label="LDA"):
    fpr, tpr, thresholds = roc_curve(y_test, proba)
    roc_auc_val = auc(fpr, tpr)
    j           = tpr - fpr
    best_i      = np.argmax(j)
    best_t      = thresholds[best_i]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(fpr, tpr, color="steelblue", lw=2,
                 label=f"{label} (AUC = {roc_auc_val:.3f})")
    axes[0].scatter(fpr[best_i], tpr[best_i], color="red", zorder=5,
                    label=f"Optimal threshold = {best_t:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"{label} ROC Curve", fontweight="bold"); axes[0].legend()

    thr_plot = thresholds[thresholds <= 1.0]
    axes[1].plot(thr_plot, tpr[:len(thr_plot)], label="TPR (Sensitivity)", color="green")
    axes[1].plot(thr_plot, fpr[:len(thr_plot)], label="FPR (1-Specificity)", color="red")
    axes[1].plot(thr_plot, j[:len(thr_plot)],   label="Youden's J", color="steelblue",
                 linestyle="--")
    axes[1].axvline(best_t, color="black", linestyle=":", lw=1.5,
                    label=f"Best = {best_t:.3f}")
    axes[1].set_xlabel("Threshold")
    axes[1].set_title("Threshold vs TPR / FPR / J", fontweight="bold")
    axes[1].legend()
    plt.tight_layout()
    savefig(fig, f"{plot_dir}/lda_roc_threshold.png")
    return roc_auc_val, best_t


def plot_lda_projection(lda_model, Xs_test, y_test, plot_dir):
    proj = lda_model.transform(Xs_test)
    fig, ax = plt.subplots(figsize=(9, 4))
    for cls, col, lbl in zip([0, 1], ["steelblue", "coral"], ["Not Subscribed", "Subscribed"]):
        ax.hist(proj[y_test == cls, 0], bins=40, alpha=0.6,
                color=col, label=lbl, edgecolor="white")
    ax.set_title("LDA Projection: Class Separation on LD1", fontweight="bold")
    ax.set_xlabel("LD1 Score"); ax.legend()
    plt.tight_layout()
    savefig(fig, f"{plot_dir}/lda_projection.png")


def plot_roc_comparison(models_dict: dict, y_test, plot_dir):
    """models_dict: {name: proba_array}"""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["steelblue", "coral", "green", "purple"]
    for (name, proba), col in zip(models_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, lw=2, color=col,
                label=f"{name}  (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison: LR vs LDA vs QDA vs NB",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    savefig(fig, f"{plot_dir}/model_comparison_roc.png")


def plot_model_comparison_bar(comp_df: pd.DataFrame, plot_dir):
    metrics = ["Accuracy", "F1 Score", "ROC AUC"]
    x  = np.arange(len(comp_df))
    w  = 0.22
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(metrics):
        ax.bar(x + i * w, comp_df[m], w, label=m, edgecolor="white")
    ax.set_xticks(x + w)
    ax.set_xticklabels(comp_df["Model"], fontsize=10)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Model Comparison: LR, LDA, QDA, Naive Bayes",
                 fontsize=12, fontweight="bold")
    ax.legend(); ax.set_ylabel("Score")
    plt.tight_layout()
    savefig(fig, f"{plot_dir}/model_comparison_bar.png")


def plot_regression_comparison(y_test, lin_pred, pois_pred, plot_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, pred, name in zip(axes, [lin_pred, pois_pred],
                               ["Linear Regression", "Poisson Regression"]):
        ax.scatter(y_test, pred, alpha=0.3, s=10, color="steelblue")
        mn, mx = y_test.min(), max(y_test.max(), pred.max())
        ax.plot([mn, mx], [mn, mx], "r--", lw=1.5)
        ax.set_xlabel("Actual (campaign contacts)")
        ax.set_ylabel("Predicted")
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2   = r2_score(y_test, pred)
        ax.set_title(f"{name}\nRMSE={rmse:.3f}  R²={r2:.4f}", fontweight="bold")
    plt.tight_layout()
    savefig(fig, f"{plot_dir}/regression_comparison.png")


def plot_regression_residuals(y_test, lin_pred, pois_pred, plot_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, pred, name in zip(axes, [lin_pred, pois_pred],
                               ["Linear", "Poisson"]):
        res = np.array(y_test) - pred
        ax.scatter(pred, res, alpha=0.3, s=10, color="steelblue")
        ax.axhline(0, color="red", linestyle="--", lw=1.5)
        ax.set_xlabel("Fitted Values"); ax.set_ylabel("Residuals")
        ax.set_title(f"{name} Residuals", fontweight="bold")
    plt.tight_layout()
    savefig(fig, f"{plot_dir}/regression_residuals.png")


def plot_regression_distribution(y_test, lin_pred, pois_pred, plot_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(y_test, bins=20, alpha=0.55, color="steelblue",
            label="Actual", edgecolor="white")
    ax.hist(np.clip(lin_pred, 0, None), bins=20, alpha=0.45,
            color="coral",  label="Linear Pred", edgecolor="white")
    ax.hist(pois_pred, bins=20, alpha=0.45, color="green",
            label="Poisson Pred", edgecolor="white")
    ax.set_title("Campaign Contacts: Distribution Comparison",
                 fontweight="bold")
    ax.set_xlabel("Number of Contacts"); ax.legend()
    plt.tight_layout()
    savefig(fig, f"{plot_dir}/regression_distribution.png")
