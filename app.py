import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

warnings.filterwarnings("ignore")

from scipy import stats
from scipy.special import expit
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, PoissonRegressor,
    Ridge, Lasso, RidgeCV, LassoCV
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, mean_squared_error, r2_score,
)
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(__file__))
import analysis  as an
import analysis2 as an2

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH = "data/bank-full.csv"
PLOT_DIR  = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

st.set_page_config(
    page_title="Bank Marketing – StatML v2",
    page_icon="🏦",
    layout="wide",
)

# ── Colour helpers ────────────────────────────────────────────────────────────
PALETTE = ["#2196F3","#FF5722","#4CAF50","#9C27B0","#FF9800"]

def show(fig, fname=None):
    st.pyplot(fig)
    if fname:
        fig.savefig(os.path.join(PLOT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)

def mrow(d: dict):
    cols = st.columns(len(d))
    for c, (k, v) in zip(cols, d.items()):
        c.metric(k, v)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")
    uploaded  = st.file_uploader("Upload bank-full.csv", type="csv")
    test_size = st.slider("Test Set Size (%)", 10, 35, 20) / 100
    rand_seed = st.number_input("Random Seed", value=42, step=1)
    qda_reg   = st.slider("QDA reg_param", 0.0, 1.0, 0.1, 0.05)

    st.markdown("#### Confounding")
    conf_exp  = st.selectbox("Exposure",   an.NUMERIC_COLS, index=an.NUMERIC_COLS.index("duration"))
    conf_conf = st.selectbox("Confounder", an.NUMERIC_COLS, index=an.NUMERIC_COLS.index("previous"))

    st.markdown("#### Resampling Settings")
    max_deg    = st.slider("Max Polynomial Degree", 3, 12, 8)
    n_boot     = st.slider("Bootstrap Iterations",  50, 500, 200, 50)
    loocv_n    = st.slider("LOOCV Subsample Size",  500, 5000, 2000, 500)

    st.markdown("#### Model Selection Settings")
    max_sub    = st.slider("Max Features (Subset Selection)", 5, 30, 15)
    max_pca    = st.slider("Max PCA/PLS Components",          5, 40, 20)

    st.markdown("---")
    retrain    = st.button("🔄 Retrain / Refit All Models", type="primary")

    st.markdown("---")
    st.caption("UCI Bank Marketing – Moro et al. (2011)")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading dataset …")
def load_data(src, _seed):
    src = src if src is not None else (DATA_PATH if os.path.exists(DATA_PATH) else None)
    if src is None:
        st.error("No data file found. Upload bank-full.csv via the sidebar.")
        st.stop()
    return an.load_and_preprocess(src)

df_raw, df, all_feats = load_data(uploaded, int(rand_seed))

st.sidebar.markdown("#### Feature Selection")
sel_feats = st.sidebar.multiselect("Features", all_feats, default=all_feats)
if len(sel_feats) < 3:
    st.sidebar.error("Select ≥ 3 features.")
    st.stop()

# ── Splits ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Splitting data …")
def get_splits(fkey, seed, ts):
    return an.split_and_scale(df, sel_feats, test_size=ts, random_state=seed)

X_tr, X_te, y_tr, y_te, Xs_tr, Xs_te, scaler = get_splits(
    tuple(sel_feats), int(rand_seed), test_size
)

# Poisson / count target
y_cnt = df[an.POISSON_TARGET]
_, _, yc_tr, yc_te = train_test_split(
    df[sel_feats], y_cnt, test_size=test_size, random_state=int(rand_seed)
)
yc_tr, yc_te = yc_tr.values, yc_te.values

# Multiclass (education)
le_edu = LabelEncoder()
y_edu  = le_edu.fit_transform(df_raw["education"])
train_idx, test_idx = y_tr.index, y_te.index
ye_tr = y_edu[df.index.get_indexer(train_idx)]
ye_te = y_edu[df.index.get_indexer(test_idx)]

# ── Fit classification models ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Fitting models …")
def fit_models(fkey, seed, ts, qr):
    lr_m, lr_s   = an.fit_logistic_with_stats(Xs_tr, y_tr, sel_feats)
    lr_multi     = LogisticRegression(max_iter=2000, random_state=seed).fit(Xs_tr, ye_tr)
    lda          = LinearDiscriminantAnalysis().fit(Xs_tr, y_tr)
    qda          = QuadraticDiscriminantAnalysis(reg_param=qr).fit(Xs_tr, y_tr)
    nb            = GaussianNB().fit(Xs_tr, y_tr)
    lin           = LinearRegression().fit(Xs_tr, yc_tr)
    pois          = PoissonRegressor(max_iter=500, alpha=1e-4).fit(Xs_tr, yc_tr)
    return lr_m, lr_s, lr_multi, lda, qda, nb, lin, pois

lr_m, lr_s, lr_multi, lda, qda, nb_m, lin_reg, pois_reg = fit_models(
    tuple(sel_feats), int(rand_seed), test_size, qda_reg
)

# Probabilities
lr_prob  = lr_m.predict_proba(Xs_te)[:,1];   lr_pred  = lr_m.predict(Xs_te)
lda_prob = lda.predict_proba(Xs_te)[:,1];    lda_pred = lda.predict(Xs_te)
qda_prob = qda.predict_proba(Xs_te)[:,1];    qda_pred = qda.predict(Xs_te)
nb_prob  = nb_m.predict_proba(Xs_te)[:,1];   nb_pred  = nb_m.predict(Xs_te)
ye_pred  = lr_multi.predict(Xs_te)
lin_pred = lin_reg.predict(Xs_te)
pois_pred= pois_reg.predict(Xs_te)

fpr_l, tpr_l, thr_l = roc_curve(y_te, lda_prob)
j_l      = tpr_l - fpr_l
best_idx = np.argmax(j_l)
best_thr = thr_l[best_idx]

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_labels = ["📊 EDA","📈 Logistic Regression","🔵 Discriminant Analysis",
              "🟢 NB & Comparison","📉 Regression","🔁 Resampling",
              "🔬 Model Selection","🔮 Prediction"]
tabs = st.tabs(tab_labels)

# ══════════════════════════════ TAB 1 – EDA ══════════════════════════════════
with tabs[0]:
    st.header("Exploratory Data Analysis")
    mrow({"Clients": f"{len(df):,}", "Subscribed": f"{df['y'].mean()*100:.1f}%",
          "Features (raw)": "16", "Features (encoded)": str(len(all_feats))})

    st.subheader("Raw Data Preview")
    st.dataframe(df_raw.head(30), use_container_width=True)

    st.subheader("Descriptive Statistics")
    st.dataframe(df_raw[an.NUMERIC_COLS].describe().T.style.format("{:.2f}"),
                 use_container_width=True)

    st.subheader("Numeric Distributions")
    nc = 4; nr = (len(an.NUMERIC_COLS)+nc-1)//nc
    fig, axes = plt.subplots(nr, nc, figsize=(16, 4*nr))
    axes = np.array(axes).flatten()
    for i, col in enumerate(an.NUMERIC_COLS):
        axes[i].hist(df_raw[col].dropna(), bins=35, color="#2196F3",
                     edgecolor="white", alpha=0.85)
        axes[i].set_title(col, fontsize=11)
    for j in range(i+1, len(axes)): axes[j].set_visible(False)
    plt.suptitle("Numeric Feature Distributions", fontweight="bold"); plt.tight_layout()
    show(fig, "eda_distributions.png")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Categorical Features")
        fig, axes = plt.subplots(2,3,figsize=(15,9)); axes=axes.flatten()
        for i,col in enumerate(an.CATEGORICAL_COLS):
            vc=df_raw[col].value_counts()
            axes[i].bar(vc.index,vc.values,color="#2196F3",edgecolor="white")
            axes[i].set_title(col,fontsize=11); axes[i].tick_params(axis="x",rotation=30)
        plt.suptitle("Categorical Distributions",fontweight="bold"); plt.tight_layout()
        show(fig,"eda_categoricals.png")
    with c2:
        st.subheader("Target & Subscription by Job")
        vc=df_raw["y"].value_counts()
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        axes[0].bar(["No","Yes"],vc.values,color=["#2196F3","#FF5722"],edgecolor="white")
        axes[0].set_title("Subscribed?",fontweight="bold"); axes[0].set_ylabel("Count")
        rates=df_raw.groupby("job")["y"].apply(lambda s:(s=="yes").mean()).sort_values()
        axes[1].barh(rates.index,rates.values*100,color="#2196F3",edgecolor="white")
        axes[1].set_xlabel("Subscription Rate (%)"); axes[1].set_title("Rate by Job",fontweight="bold")
        plt.tight_layout(); show(fig,"eda_target.png")

    st.subheader("Correlation Heatmap")
    cols_c=[c for c in an.NUMERIC_COLS if c in df.columns]+["y"]
    fig,ax=plt.subplots(figsize=(9,7))
    mask=np.triu(np.ones_like(df[cols_c].corr(),dtype=bool))
    sns.heatmap(df[cols_c].corr(),mask=mask,annot=True,fmt=".2f",cmap="coolwarm",
                center=0,linewidths=0.5,ax=ax,square=True)
    ax.set_title("Correlation Heatmap",fontweight="bold"); plt.tight_layout()
    show(fig,"eda_correlation.png")

# ══════════════════════ TAB 2 – LOGISTIC REGRESSION ══════════════════════════
with tabs[1]:
    st.header("Logistic Regression")
    mrow({"Accuracy":  f"{accuracy_score(y_te,lr_pred):.4f}",
          "Precision": f"{precision_score(y_te,lr_pred,zero_division=0):.4f}",
          "Recall":    f"{recall_score(y_te,lr_pred,zero_division=0):.4f}",
          "F1":        f"{f1_score(y_te,lr_pred,zero_division=0):.4f}",
          "AUC":       f"{roc_auc_score(y_te,lr_prob):.4f}"})

    st.subheader("Coefficient Table")
    disp=lr_s.copy(); disp["Sig"]= disp["P_Value"].apply(lambda p:"✅" if p<0.05 else "❌")
    st.dataframe(disp.style.format({"Coefficient":"{:.4f}","Std_Error":"{:.4f}",
                                    "Z_Statistic":"{:.4f}","P_Value":"{:.3e}",
                                    "Odds_Ratio":"{:.4f}"}),
                 use_container_width=True, height=400)

    c1,c2=st.columns(2)
    with c1:
        top_n=st.slider("Top N features (coefficient plot)",5,30,20,key="lr_top")
        fd=lr_s[lr_s["Feature"]!="Intercept"].copy()
        fd["abs"]=fd["Coefficient"].abs(); fd=fd.nlargest(top_n,"abs").sort_values("Coefficient")
        cols_=[("#FF5722" if v>0 else "#2196F3") for v in fd["Coefficient"]]
        fig,ax=plt.subplots(figsize=(8,max(5,len(fd)*0.4)))
        ax.barh(fd["Feature"],fd["Coefficient"],color=cols_,edgecolor="white")
        ax.errorbar(fd["Coefficient"],fd["Feature"],xerr=1.96*fd["Std_Error"],
                    fmt="none",color="black",capsize=3,lw=1)
        ax.axvline(0,color="black",lw=0.8,linestyle="--")
        ax.set_title(f"Top {top_n} Coefficients ±95% CI",fontweight="bold"); plt.tight_layout()
        show(fig,"lr_coefficients.png")
    with c2:
        fd2=lr_s[lr_s["Feature"]!="Intercept"].copy()
        fd2["abs"]=fd2["Coefficient"].abs(); fd2=fd2.nlargest(top_n,"abs").sort_values("P_Value",ascending=False)
        cols2=["#4CAF50" if p<0.05 else "gray" for p in fd2["P_Value"]]
        fig,ax=plt.subplots(figsize=(8,max(5,len(fd2)*0.4)))
        ax.barh(fd2["Feature"],-np.log10(fd2["P_Value"].clip(1e-300)),color=cols2,edgecolor="white")
        ax.axvline(-np.log10(0.05),color="red",linestyle="--",label="p=0.05")
        ax.set_title("Feature Significance (−log₁₀ p)",fontweight="bold"); ax.legend(); plt.tight_layout()
        show(fig,"lr_pvalues.png")

    st.subheader("Confounding Analysis")
    if conf_exp in sel_feats and conf_conf in sel_feats:
        with st.spinner("Running …"):
            cr=an.confounding_analysis(Xs_tr,y_tr,sel_feats,conf_exp,conf_conf)
        if cr:
            mrow({f"'{conf_exp}' with '{conf_conf}'": f"{cr['full']:.4f}",
                  f"'{conf_exp}' without '{conf_conf}'": f"{cr['reduced']:.4f}",
                  "Change %": f"{cr['change_pct']:.1f}%",
                  "Confounding?": "✅ Yes" if cr['change_pct']>10 else "⚠️ Weak"})
            fig,ax=plt.subplots(figsize=(6,4))
            ax.bar([f"With\n'{conf_conf}'",f"Without\n'{conf_conf}'"],
                   [cr["full"],cr["reduced"]],color=["#2196F3","#FF5722"],edgecolor="white",width=0.45)
            ax.axhline(0,color="black",lw=0.8)
            ax.set_title(f"Confounding: '{conf_exp}'  ({cr['change_pct']:.1f}% change)",fontweight="bold")
            ax.set_ylabel("Coefficient (standardised)"); plt.tight_layout()
            show(fig,"lr_confounding.png")
    else:
        st.info("Select both exposure and confounder in the sidebar feature set.")

    st.subheader("Multiclass LR – Education Level")
    c1,c2=st.columns(2)
    with c1:
        rep=pd.DataFrame(an.classification_report(ye_te,ye_pred,target_names=le_edu.classes_,output_dict=True)).T
        st.dataframe(rep.style.format("{:.3f}"),use_container_width=True)
    with c2:
        fig,ax=plt.subplots(figsize=(6,5))
        sns.heatmap(confusion_matrix(ye_te,ye_pred),annot=True,fmt="d",cmap="Blues",ax=ax,
                    xticklabels=le_edu.classes_,yticklabels=le_edu.classes_)
        ax.set_title("Education Level – Confusion Matrix",fontweight="bold"); plt.tight_layout()
        show(fig,"lr_multiclass_cm.png")

# ══════════════════════ TAB 3 – DISCRIMINANT ANALYSIS ════════════════════════
with tabs[2]:
    st.header("Discriminant Analysis")
    lda_auc=roc_auc_score(y_te,lda_prob); qda_auc=roc_auc_score(y_te,qda_prob)
    mrow({"LDA Accuracy":f"{accuracy_score(y_te,lda_pred):.4f}",
          "LDA AUC":f"{lda_auc:.4f}","QDA AUC":f"{qda_auc:.4f}",
          "Optimal LDA Threshold":f"{best_thr:.4f}"})

    c1,c2=st.columns(2)
    with c1:
        fig,ax=plt.subplots(figsize=(7,5))
        ax.plot(fpr_l,tpr_l,color="#2196F3",lw=2,label=f"LDA (AUC={lda_auc:.3f})")
        ax.scatter(fpr_l[best_idx],tpr_l[best_idx],color="red",zorder=5,
                   label=f"Optimal={best_thr:.3f}")
        ax.plot([0,1],[0,1],"k--",lw=1); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("LDA ROC Curve",fontweight="bold"); ax.legend(); plt.tight_layout()
        show(fig,"lda_roc_threshold.png")
    with c2:
        thr_p=thr_l[thr_l<=1.0]
        fig,ax=plt.subplots(figsize=(7,5))
        ax.plot(thr_p,tpr_l[:len(thr_p)],label="TPR",color="#4CAF50")
        ax.plot(thr_p,fpr_l[:len(thr_p)],label="FPR",color="#FF5722")
        ax.plot(thr_p,j_l[:len(thr_p)],label="Youden's J",color="#2196F3",linestyle="--")
        ax.axvline(best_thr,color="black",linestyle=":",lw=1.5,label=f"Best={best_thr:.3f}")
        ax.set_xlabel("Threshold"); ax.set_title("Threshold Analysis",fontweight="bold"); ax.legend(); plt.tight_layout()
        show(fig,"lda_threshold_analysis.png")

    proj=lda.transform(Xs_te)
    fig,ax=plt.subplots(figsize=(10,4))
    for cls,col,lbl in zip([0,1],["#2196F3","#FF5722"],["Not Subscribed","Subscribed"]):
        ax.hist(proj[y_te==cls,0],bins=50,alpha=0.6,color=col,label=lbl,edgecolor="white",density=True)
    ax.set_title("LDA Projection – LD1",fontweight="bold"); ax.set_xlabel("LD1"); ax.legend(); plt.tight_layout()
    show(fig,"lda_projection.png")

    fpr_q,tpr_q,_=roc_curve(y_te,qda_prob)
    c1,c2=st.columns(2)
    with c1:
        fig,ax=plt.subplots(figsize=(7,5))
        ax.plot(fpr_l,tpr_l,lw=2,label=f"LDA {lda_auc:.3f}")
        ax.plot(fpr_q,tpr_q,lw=2,label=f"QDA {qda_auc:.3f}")
        ax.plot([0,1],[0,1],"k--",lw=1); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("LDA vs QDA ROC",fontweight="bold"); ax.legend(); plt.tight_layout()
        show(fig,"lda_qda_roc.png")
    with c2:
        fig,axes_=plt.subplots(1,2,figsize=(9,4))
        for ax_,pred_,title_ in zip(axes_,[lda_pred,qda_pred],["LDA","QDA"]):
            sns.heatmap(confusion_matrix(y_te,pred_),annot=True,fmt="d",cmap="Blues",ax=ax_,
                        xticklabels=["No","Yes"],yticklabels=["No","Yes"])
            ax_.set_title(f"{title_} CM",fontweight="bold")
        plt.tight_layout(); show(fig,"lda_qda_cm.png")

# ══════════════════════ TAB 4 – NB & COMPARISON ══════════════════════════════
with tabs[3]:
    st.header("Naïve Bayes & Model Comparison")
    nb_auc=roc_auc_score(y_te,nb_prob)
    mrow({"NB Acc":f"{accuracy_score(y_te,nb_pred):.4f}","NB F1":f"{f1_score(y_te,nb_pred,zero_division=0):.4f}","NB AUC":f"{nb_auc:.4f}"})

    c1,c2=st.columns(2)
    with c1:
        fig,ax=plt.subplots(figsize=(6,4))
        ax.bar(["Not Subscribed","Subscribed"],nb_m.class_prior_,color=["#2196F3","#FF5722"],edgecolor="white")
        ax.set_title("NB Class Priors",fontweight="bold"); plt.tight_layout(); show(fig,"nb_priors.png")
    with c2:
        fig,ax=plt.subplots(figsize=(6,4))
        sns.heatmap(confusion_matrix(y_te,nb_pred),annot=True,fmt="d",cmap="Greens",ax=ax,
                    xticklabels=["No","Yes"],yticklabels=["No","Yes"])
        ax.set_title("NB CM",fontweight="bold"); plt.tight_layout(); show(fig,"nb_cm.png")

    st.subheader("Full Model Comparison")
    rows=[]
    for name,(pred_,prob_) in {"LR":(lr_pred,lr_prob),"LDA":(lda_pred,lda_prob),
                                "QDA":(qda_pred,qda_prob),"NB":(nb_pred,nb_prob)}.items():
        rows.append({"Model":name,
                     "Accuracy": accuracy_score(y_te,pred_),
                     "Precision":precision_score(y_te,pred_,zero_division=0),
                     "Recall":   recall_score(y_te,pred_,zero_division=0),
                     "F1 Score": f1_score(y_te,pred_,zero_division=0),
                     "ROC AUC":  roc_auc_score(y_te,prob_)})
    comp_df=pd.DataFrame(rows)
    st.dataframe(comp_df.style.format({c:"{:.4f}" for c in comp_df.columns if c!="Model"})
                 .highlight_max(subset=["Accuracy","F1 Score","ROC AUC"],color="#c6efce"),
                 use_container_width=True)

    c1,c2=st.columns(2)
    with c1:
        fig,ax=plt.subplots(figsize=(8,6))
        for name,col in zip(["LR","LDA","QDA","NB"],["#2196F3","#FF5722","#4CAF50","#9C27B0"]):
            prob_={"LR":lr_prob,"LDA":lda_prob,"QDA":qda_prob,"NB":nb_prob}[name]
            f_,t_,_=roc_curve(y_te,prob_)
            ax.plot(f_,t_,lw=2,color=col,label=f"{name} ({auc(f_,t_):.3f})")
        ax.plot([0,1],[0,1],"k--",lw=1); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("ROC Comparison",fontweight="bold"); ax.legend(loc="lower right"); plt.tight_layout()
        show(fig,"model_comparison_roc.png")
    with c2:
        x_=np.arange(len(comp_df)); w=0.22
        fig,ax=plt.subplots(figsize=(8,5))
        for i,m in enumerate(["Accuracy","F1 Score","ROC AUC"]):
            ax.bar(x_+i*w,comp_df[m],w,label=m,edgecolor="white")
        ax.set_xticks(x_+w); ax.set_xticklabels(comp_df["Model"],fontsize=10)
        ax.set_ylim(0.5,1.0); ax.set_title("Performance Metrics",fontweight="bold")
        ax.legend(); plt.tight_layout(); show(fig,"model_comparison_bar.png")

# ══════════════════════ TAB 5 – REGRESSION ════════════════════════════════════
with tabs[4]:
    st.header("Linear vs Poisson Regression")
    lin_rmse=np.sqrt(mean_squared_error(yc_te,lin_pred)); pois_rmse=np.sqrt(mean_squared_error(yc_te,pois_pred))
    mrow({"Linear RMSE":f"{lin_rmse:.4f}","Poisson RMSE":f"{pois_rmse:.4f}",
          "Linear R²":f"{r2_score(yc_te,lin_pred):.4f}","Poisson R²":f"{r2_score(yc_te,pois_pred):.4f}",
          "Mean(campaign)":f"{yc_te.mean():.2f}","Var(campaign)":f"{yc_te.var():.2f}"})

    c1,c2=st.columns(2)
    with c1:
        fig,axes_=plt.subplots(1,2,figsize=(11,5))
        for ax_,pred_,name_ in zip(axes_,[lin_pred,pois_pred],["Linear","Poisson"]):
            ax_.scatter(yc_te,pred_,alpha=0.25,s=8,color="#2196F3")
            mn_,mx_=yc_te.min(),max(yc_te.max(),pred_.max())
            ax_.plot([mn_,mx_],[mn_,mx_],"r--",lw=1.5)
            ax_.set_xlabel("Actual"); ax_.set_ylabel("Predicted"); ax_.set_title(f"{name_}",fontweight="bold")
        plt.tight_layout(); show(fig,"regression_comparison.png")
    with c2:
        fig,axes_=plt.subplots(1,2,figsize=(11,5))
        for ax_,pred_,name_ in zip(axes_,[lin_pred,pois_pred],["Linear","Poisson"]):
            ax_.scatter(pred_,yc_te-pred_,alpha=0.25,s=8,color="#2196F3")
            ax_.axhline(0,color="red",linestyle="--",lw=1.5)
            ax_.set_xlabel("Fitted"); ax_.set_ylabel("Residuals"); ax_.set_title(f"{name_} Residuals",fontweight="bold")
        plt.tight_layout(); show(fig,"regression_residuals.png")

    if yc_te.var() > 2*yc_te.mean():
        st.warning(f"⚠️ Overdispersion detected (Var/Mean = {yc_te.var()/yc_te.mean():.2f}). Consider Negative Binomial regression.")
    else:
        st.success(f"✅ Equidispersion: Var/Mean = {yc_te.var()/yc_te.mean():.2f}. Poisson assumption holds.")

# ══════════════════════ TAB 6 – RESAMPLING (new module) ═════════════════════
from tab_resampling_modsel import render_resampling_tab, render_model_selection_tab

with tabs[5]:
    render_resampling_tab(df, all_feats, rand_seed)

# ══════════════════════ TAB 7 – MODEL SELECTION (new module) ═════════════════
with tabs[6]:
    render_model_selection_tab(df, all_feats, rand_seed)

# ══════════════════════ TAB 8 – PREDICTION ════════════════════════════════════
with tabs[7]:
    st.header("🔮 Live Prediction Interface")
    st.markdown("Enter client profile to predict subscription probability across all models.")

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("**Demographics**")
        inp_age  = st.slider("Age",18,95,41)
        inp_job  = st.selectbox("Job",sorted(df_raw["job"].unique()))
        inp_mar  = st.selectbox("Marital",df_raw["marital"].unique())
        inp_edu  = st.selectbox("Education",df_raw["education"].unique())
        inp_def  = st.selectbox("Credit Default",["no","yes"])
    with c2:
        st.markdown("**Financial**")
        inp_bal  = st.slider("Balance (€)",int(df_raw["balance"].min()),
                             min(int(df_raw["balance"].max()),20000),1362)
        inp_hou  = st.selectbox("Housing Loan",["no","yes"])
        inp_loan = st.selectbox("Personal Loan",["no","yes"])
    with c3:
        st.markdown("**Campaign**")
        inp_con  = st.selectbox("Contact Type",df_raw["contact"].unique())
        inp_day  = st.slider("Last Contact Day",1,31,15)
        inp_mon  = st.selectbox("Month",["jan","feb","mar","apr","may","jun",
                                         "jul","aug","sep","oct","nov","dec"])
        inp_dur  = st.slider("Duration (s)",0,3000,258)
        inp_camp = st.slider("Contacts This Campaign",1,30,2)
        inp_pday = st.slider("Days Since Last Contact (-1=never)",-1,400,-1)
        inp_prev = st.slider("Prior Contacts",0,20,0)
        inp_pout = st.selectbox("Previous Outcome",df_raw["poutcome"].unique())

    if st.button("🔮 Predict", type="primary"):
        raw_inp = pd.DataFrame([{"age":inp_age,"job":inp_job,"marital":inp_mar,
            "education":inp_edu,"default":inp_def,"balance":inp_bal,
            "housing":inp_hou,"loan":inp_loan,"contact":inp_con,"day":inp_day,
            "month":inp_mon,"duration":inp_dur,"campaign":inp_camp,
            "pdays":inp_pday,"previous":inp_prev,"poutcome":inp_pout,"y":"no"}])
        enc = raw_inp.copy()
        for col_ in an.BINARY_COLS+["y"]: enc[col_]=(enc[col_]=="yes").astype(int)
        enc = pd.get_dummies(enc,columns=an.CATEGORICAL_COLS,drop_first=True,dtype=int)
        enc["pdays_contacted"]=(enc["pdays"]>=0).astype(int)
        enc["pdays"]=enc["pdays"].replace(-1,0)
        enc=enc.drop(columns=["y"],errors="ignore")
        for c_ in sel_feats:
            if c_ not in enc.columns: enc[c_]=0
        enc=enc[sel_feats]; enc_s=scaler.transform(enc)

        lr_p_=lr_m.predict_proba(enc_s)[0,1]
        lda_p_=lda.predict_proba(enc_s)[0,1]
        qda_p_=qda.predict_proba(enc_s)[0,1]
        nb_p_=nb_m.predict_proba(enc_s)[0,1]

        st.markdown("---"); st.subheader("Results")
        mrow({"LR":f"{lr_p_:.1%}","LDA":f"{lda_p_:.1%}","QDA":f"{qda_p_:.1%}","NB":f"{nb_p_:.1%}"})

        fig,ax=plt.subplots(figsize=(8,3))
        probs_={"LR":lr_p_,"LDA":lda_p_,"QDA":qda_p_,"NB":nb_p_}
        bars_=ax.bar(probs_.keys(),probs_.values(),
                     color=["#4CAF50" if p<0.3 else "#FF9800" if p<0.6 else "#FF5722"
                            for p in probs_.values()],edgecolor="white")
        ax.axhline(0.5,color="black",linestyle="--",lw=1)
        ax.set_ylim(0,1); ax.set_ylabel("Subscription Probability")
        ax.set_title("Predicted Subscription Probability",fontweight="bold"); plt.tight_layout()
        show(fig,"prediction_result.png")

        v=("🟢 **LOW** — Unlikely to subscribe" if lr_p_<0.4
           else "🟡 **MEDIUM** — Uncertain" if lr_p_<0.65
           else "🔴 **HIGH** — Likely to subscribe")
        st.markdown(f"### Assessment: {v}")

st.markdown("---")
st.caption("📖 Citation: S. Moro, R. Laureano and P. Cortez. *Using Data Mining for Bank Direct Marketing.* ESM'2011. EUROSIS.")
