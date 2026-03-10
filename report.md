---
title: "Statistical Machine Learning Analysis Report"
subtitle: "Bank Marketing --- Subscription Prediction, Resampling \\& Model Selection"
author: "Tahar Masmaliyev \\& Abdulla Akhundzade"
date: "March 2026"
geometry: "top=1.5cm, bottom=1.5cm, left=2cm, right=2cm"
fontsize: 9pt
linestretch: 1.05
colorlinks: true
linkcolor: NavyBlue
urlcolor: NavyBlue
mainfont: "DejaVu Serif"
sansfont: "DejaVu Sans"
monofont: "DejaVu Sans Mono"
header-includes:
  - \usepackage{booktabs}
  - \usepackage{float}
  - \usepackage{fancyhdr}
  - \usepackage{graphicx}
  - \pagestyle{fancy}
  - \fancyhead[L]{\small Bank Marketing -- StatML Report}
  - \fancyhead[R]{\small Tahar Masmaliyev \& Abdulla Akhundzade}
  - \renewcommand{\headrulewidth}{0.4pt}
---

# 1. Dataset Overview

The **UCI Bank Marketing** dataset (Moro et al., 2011) contains 4,521 telephone marketing records from a Portuguese bank (2008--2010). After one-hot encoding, 43 features predict whether a client subscribes to a term deposit. The subscription rate is **11.52%** (severe 88:12 class imbalance). An 80/20 stratified train-test split is used. `duration` (call length) is the strongest numeric correlate with the target (r = 0.40).

# 2. Prior Classification \& Regression Results

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.892 | 0.554 | 0.298 | 0.388 | **0.891** |
| LDA | 0.881 | 0.477 | 0.394 | 0.432 | 0.889 |
| QDA | 0.845 | 0.350 | 0.404 | 0.375 | 0.820 |
| Naive Bayes | 0.825 | 0.318 | 0.452 | 0.373 | 0.786 |

**Logistic Regression** leads (AUC = 0.891). `duration` is the dominant predictor (coef = 1.121, Z = 18.69, p $\approx$ 0, OR = 3.07), followed by `poutcome_success` and `contact_unknown`. Also significant: `loan` (p = 0.002), `housing` (p = 0.032), `campaign` (p = 0.027). Non-significant: `age` (p = 0.390), `balance` (p = 0.504), `pdays` (p = 0.932).

**LDA** (0.889) nearly matches LR; **QDA** (0.820) underperforms, confirming shared covariance holds. **NB** (0.786) has lowest AUC but highest recall (0.452), with class priors 0.885/0.115. Confusion matrices: LDA (756 TN, 45 FP, 63 FN, 41 TP); QDA (723 TN, 78 FP, 62 FN, 42 TP); NB (700 TN, 101 FP, 57 FN, 47 TP).

\begin{figure}[H]
\centering
\includegraphics[width=0.48\textwidth]{plots/model_comparison_roc.png}
\caption{ROC curves. LR (0.891) and LDA (0.889) lead; QDA (0.820) and NB (0.786) trail.}
\end{figure}

**Optimal LDA threshold:** Youden's J = **0.027** (far below 0.5), yielding TPR $\approx$ 0.90, FPR $\approx$ 0.27. **Confounding:** `previous` changed `duration` coefficient by only **0.07%** --- not a confounder.

**Regression (Linear vs Poisson):** Both Linear (RMSE = 2.599, R$^2$ = --0.025) and Poisson (RMSE = 2.600, R$^2$ = --0.026) yield negative R$^2$: features do not explain `campaign` counts. Poisson is theoretically preferred (non-negativity) but neither is practically useful.

# 3. Task 1 --- Resampling Methods

## Task 1a: Cross-Validation (K=5, K=10, LOOCV)

Polynomial degrees 1--3 evaluated via K=5, K=10, and LOOCV (n=250). All methods agree: **degree 1** achieves minimum MSE. Degree 3 shows massive error increase, especially LOOCV (highest variance). K=10 provides the best bias-variance tradeoff.

## Task 1b: Bootstrap

The .632 estimator corroborates CV. For degree 1, mean OOB MSE = 12,631,525, 95% CI [7,036,284, 20,439,447]. Bootstrap and CV K=5 both select degree 1; degree 3 shows extreme variance inflation.

## Task 1c: MSE vs Polynomial Degree

\begin{figure}[H]
\centering
\includegraphics[width=0.48\textwidth]{plots/resample_cv_mse_degree.png}
\includegraphics[width=0.48\textwidth]{plots/resample_bootstrap_mse_degree.png}
\caption{Left: CV MSE vs degree (K=5, K=10, LOOCV). Right: Bootstrap vs CV K=5. Both confirm degree 1 as optimal.}
\end{figure}

# 4. Task 2 --- Model Selection

## Task 2a: Subset Selection

Five criteria select the optimal number of predictors: **Cp = 6, AIC = 6, BIC = 3, Adj-R$^2$ = 7, CV-MSE = 4**. BIC is most parsimonious ($\ln(n) \approx 8.2$ penalty per parameter vs AIC's 2). Most consistently retained features: `duration`, `poutcome_success`, `contact_unknown`.

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{plots/subset_selection_criteria.png}
\caption{Forward stepwise selection: Cp, AIC, BIC, Adj-R$^2$, and CV-MSE vs number of predictors.}
\end{figure}

## Task 2b: Shrinkage --- Ridge and Lasso

CV-optimal: **Ridge $\lambda$ = 322.46**, **Lasso $\lambda$ = 7.61**. Ridge shrinks all coefficients smoothly; Lasso drives many to zero (sparsity). Bias-variance decomposition: bias$^2$ dominates total MSE; variance stays small. Ridge min-MSE at $\lambda$ = 0.0045; Lasso at $\lambda$ = 1.3738 (bias-variance optimal).

\begin{figure}[H]
\centering
\includegraphics[width=0.48\textwidth]{plots/shrinkage_coef_paths.png}
\includegraphics[width=0.48\textwidth]{plots/shrinkage_cv_mse.png}
\caption{Left: Coefficient paths. Right: CV MSE vs $\lambda$ (optimal marked red).}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.75\textwidth]{plots/shrinkage_bias_variance.png}
\caption{Bias-variance tradeoff for Ridge (left) and Lasso (right). Bias$^2$ dominates; variance is negligible.}
\end{figure}

## Task 2c: PCR vs PLS

PC1 explains 24.3%, PC2 12.2% of variance. PCR minimum test MSE at **4 components**; PLS achieves comparable MSE with only **2 components** --- supervised reduction is more efficient. Full OLS baseline MSE = 7,735,224.

\begin{figure}[H]
\centering
\includegraphics[width=0.48\textwidth]{plots/pca_vs_pls.png}
\includegraphics[width=0.48\textwidth]{plots/model_selection_final_comparison.png}
\caption{Left: PCR vs PLS test MSE. Right: Final comparison across all methods.}
\end{figure}

All methods yield similar test MSE ($\sim$7.5 $\times$ 10$^6$) and R$^2$ ($\sim$0.01--0.014). PLS (2 components) is most efficient; Lasso provides best interpretability; Ridge retains all features with reduced magnitude.

# 5. Conclusions

1. **LR is the best classifier** (AUC = 0.891); `poutcome_success` is the most actionable predictor.
2. **Youden's J threshold** = 0.027 raises recall to $\sim$90% for the minority class.
3. **`previous` is not a confounder** (0.07% coefficient change).
4. **Task 1:** Degree 1 is optimal; all CV and bootstrap methods agree.
5. **Task 2a:** BIC selects 3; Cp/AIC select 6; CV selects 4 features.
6. **Task 2b:** Ridge ($\lambda$ = 322.46) and Lasso ($\lambda$ = 7.61) both improve over OLS.
7. **Task 2c:** PLS (2 components) matches PCR (4 components), confirming supervised reduction is most efficient.
8. QDA underperforms LDA; campaign regression fails (R$^2$ < 0); no overfitting advantage from regularisation.

## Team Contributions

| Task | Tahar Masmaliyev | Abdulla Akhundzade |
|:---|:---|:---|
| Data preprocessing, EDA, Classification | Lead | Support |
| Task 1: Resampling (CV \& Bootstrap) | Lead | Support |
| Task 2: Model Selection (Subset, Shrinkage, PCR/PLS) | Support | Lead |
| Streamlit Dashboard and Report | Joint | Joint |

\small
**References:**
Moro, S., Laureano, R., \& Cortez, P. (2011). *Using Data Mining for Bank Direct Marketing.* ESM'2011, EUROSIS.
James, G., Witten, D., Hastie, T., \& Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer.