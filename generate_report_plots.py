import sys, os, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                             accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score,
                             mean_squared_error, r2_score)
import warnings
warnings.filterwarnings("ignore")

# ── local module ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
import analysis as an

PLOT_DIR = "plots"
DATA_DIR = "data"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA_DIR, "bank-full.csv")

# ─── Load ─────────────────────────────────────────────────────────────────────
print(f"Loading dataset from: {CSV_PATH}")
df_raw, df, all_features = an.load_and_preprocess(CSV_PATH)
print(f"  Rows: {len(df):,}  |  Features after encoding: {len(all_features)}")
print(f"  Subscription rate: {df['y'].mean()*100:.1f}%")

# Default to ALL features for standalone run
features = all_features

(X_tr, X_te, y_tr, y_te,
 Xs_tr, Xs_te, scaler) = an.split_and_scale(df, features, test_size=0.20)

# ─── EDA plots ────────────────────────────────────────────────────────────────
print("\n[EDA] Generating plots …")
an.plot_eda_distributions(df_raw, an.NUMERIC_COLS, PLOT_DIR)
an.plot_eda_categoricals(df_raw, an.CATEGORICAL_COLS, PLOT_DIR)
an.plot_eda_target(df_raw, PLOT_DIR)
an.plot_eda_correlation(df, an.NUMERIC_COLS, PLOT_DIR)
an.plot_eda_duration_balance(df_raw, PLOT_DIR)
print("  EDA plots saved.")

# ─── Logistic Regression ──────────────────────────────────────────────────────
print("\n[LR] Fitting logistic regression …")
lr_model, lr_stats = an.fit_logistic_with_stats(Xs_tr, y_tr, features)

print(lr_stats.head(12).to_string(index=False))
an.plot_lr_coefficients(lr_stats, PLOT_DIR, top_n=20)
an.plot_lr_pvalues(lr_stats, PLOT_DIR, top_n=20)

# Confounding: "duration" vs "campaign" (duration may suppress campaign's effect)
print("\n[LR] Confounding analysis: exposure=duration, confounder=previous …")
conf_result = an.confounding_analysis(Xs_tr, y_tr, features,
                                      exposure="duration",
                                      confounder="previous")
if conf_result:
    print(f"  duration coef with 'previous':    {conf_result['full']:.4f}")
    print(f"  duration coef without 'previous': {conf_result['reduced']:.4f}")
    print(f"  Change: {conf_result['change_pct']:.1f}%")
    an.plot_confounding(conf_result, "duration", "previous", PLOT_DIR)

# Multiclass: education level prediction (4 classes: unknown/primary/secondary/tertiary)
# We encode education directly for demonstration
print("\n[LR] Multiclass: predicting education level …")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_edu = le.fit_transform(df_raw["education"])
from sklearn.model_selection import train_test_split
Xe_tr, Xe_te, ye_tr, ye_te = train_test_split(Xs_tr, y_edu[:len(Xs_tr)],
                                                test_size=0.20, random_state=42,
                                                stratify=y_edu[:len(Xs_tr)])
lr_multi = LogisticRegression(max_iter=2000, random_state=42)
lr_multi.fit(Xe_tr, ye_tr)
ye_pred = lr_multi.predict(Xe_te)
an.plot_multiclass_cm(ye_te, ye_pred, le.classes_,
                      "Multiclass LR – Education Level Prediction",
                      "lr_multiclass_cm.png", PLOT_DIR)
print("  Multiclass CM saved.")

# ─── Discriminant Analysis ───────────────────────────────────────────────────
print("\n[DA] Fitting LDA and QDA …")
lda = LinearDiscriminantAnalysis()
lda.fit(Xs_tr, y_tr)
lda_proba = lda.predict_proba(Xs_te)[:, 1]
lda_pred  = lda.predict(Xs_te)

qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
qda.fit(Xs_tr, y_tr)
qda_proba = qda.predict_proba(Xs_te)[:, 1]
qda_pred  = qda.predict(Xs_te)

lda_auc, best_thresh = an.plot_lda_roc_threshold(y_te, lda_proba, PLOT_DIR)
an.plot_lda_projection(lda, Xs_te, y_te, PLOT_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc
fpr_q, tpr_q, _ = roc_curve(y_te, qda_proba)
fpr_l, tpr_l, _ = roc_curve(y_te, lda_proba)
auc_q = sk_auc(fpr_q, tpr_q)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr_l, tpr_l, lw=2, label=f"LDA  AUC={lda_auc:.3f}")
ax.plot(fpr_q, tpr_q, lw=2, label=f"QDA  AUC={auc_q:.3f}")
ax.plot([0,1],[0,1],"k--",lw=1)
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("LDA vs QDA ROC Curves", fontweight="bold"); ax.legend()
plt.tight_layout()
an.savefig(fig, f"{PLOT_DIR}/lda_qda_roc.png")
print(f"  LDA AUC={lda_auc:.4f}  QDA AUC={auc_q:.4f}  Best threshold={best_thresh:.4f}")

# ─── Naive Bayes + Comparison ─────────────────────────────────────────────────
print("\n[NB] Fitting Naive Bayes …")
nb = GaussianNB()
nb.fit(Xs_tr, y_tr)
nb_proba = nb.predict_proba(Xs_te)[:, 1]
nb_pred  = nb.predict(Xs_te)

lr_proba = lr_model.predict_proba(Xs_te)[:, 1]
lr_pred  = lr_model.predict(Xs_te)

an.plot_roc_comparison(
    {"Logistic Regression": lr_proba, "LDA": lda_proba,
     "QDA": qda_proba, "Naive Bayes": nb_proba},
    y_te, PLOT_DIR
)

rows = []
for name, (ypred, yprob) in {
    "Logistic Regression": (lr_pred, lr_proba),
    "LDA": (lda_pred, lda_proba),
    "QDA": (qda_pred, qda_proba),
    "Naive Bayes": (nb_pred, nb_proba),
}.items():
    rows.append({
        "Model": name,
        "Accuracy":  round(accuracy_score(y_te, ypred), 4),
        "Precision": round(precision_score(y_te, ypred, zero_division=0), 4),
        "Recall":    round(recall_score(y_te, ypred, zero_division=0), 4),
        "F1 Score":  round(f1_score(y_te, ypred, zero_division=0), 4),
        "ROC AUC":   round(roc_auc_score(y_te, yprob), 4),
    })
comp_df = pd.DataFrame(rows)
print("\n" + comp_df.to_string(index=False))
an.plot_model_comparison_bar(comp_df, PLOT_DIR)

# ─── Regression: Linear vs Poisson ───────────────────────────────────────────
print("\n[Regression] Linear vs Poisson on 'campaign' (contact count) …")
y_cnt_tr = y_tr.copy(); y_cnt_tr[:] = 0   # reset
y_cnt_full = df[an.POISSON_TARGET]
_, _, yc_tr, yc_te = train_test_split(df[features], y_cnt_full,
                                       test_size=0.20, random_state=42)
yc_tr_arr = yc_tr.values
yc_te_arr = yc_te.values

lin = LinearRegression()
lin.fit(Xs_tr, yc_tr_arr)
lin_pred = lin.predict(Xs_te)

pois = PoissonRegressor(max_iter=500, alpha=1e-4)
pois.fit(Xs_tr, yc_tr_arr)
pois_pred = pois.predict(Xs_te)

lin_rmse  = np.sqrt(mean_squared_error(yc_te_arr, lin_pred))
pois_rmse = np.sqrt(mean_squared_error(yc_te_arr, pois_pred))
lin_r2    = r2_score(yc_te_arr, lin_pred)
pois_r2   = r2_score(yc_te_arr, pois_pred)
print(f"  Linear  → RMSE={lin_rmse:.4f}  R²={lin_r2:.4f}")
print(f"  Poisson → RMSE={pois_rmse:.4f}  R²={pois_r2:.4f}")

an.plot_regression_comparison(yc_te_arr, lin_pred, pois_pred, PLOT_DIR)
an.plot_regression_residuals(yc_te_arr, lin_pred, pois_pred, PLOT_DIR)
an.plot_regression_distribution(yc_te_arr, lin_pred, pois_pred, PLOT_DIR)

# ─── Save summary JSON ───────────────────────────────────────────────────────
summary = {
    "n_samples": int(len(df)),
    "n_features_raw": 16,
    "n_features_encoded": int(len(features)),
    "subscription_rate_pct": round(float(df["y"].mean()) * 100, 2),
    "train_test_split": "80/20",
    "lr_auc":  float(rows[0]["ROC AUC"]),
    "lda_auc": float(rows[1]["ROC AUC"]),
    "qda_auc": float(rows[2]["ROC AUC"]),
    "nb_auc":  float(rows[3]["ROC AUC"]),
    "lda_best_threshold": round(float(best_thresh), 4),
    "confounding_exposure":  "duration",
    "confounding_confounder": "previous",
    "confounding_change_pct": round(float(conf_result["change_pct"]) if conf_result else 0, 2),
    "linear_rmse":  round(float(lin_rmse), 4),
    "poisson_rmse": round(float(pois_rmse), 4),
    "linear_r2":    round(float(lin_r2), 4),
    "poisson_r2":   round(float(pois_r2), 4),
    "model_comparison": rows,
    "lr_stats_top10": lr_stats.head(11).to_dict(orient="records"),
}
with open(f"{DATA_DIR}/summary_stats.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n✅  All done! Plots saved to ./{PLOT_DIR}/")
print(f"    Summary stats saved to ./{DATA_DIR}/summary_stats.json")
print("\nPlots generated:")
for p in sorted(os.listdir(PLOT_DIR)):
    print(f"  {p}")
