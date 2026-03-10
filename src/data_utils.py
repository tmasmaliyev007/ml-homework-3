import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

RANDOM_STATE = 42
N_SAMPLES = 690  # matches real dataset

# Continuous feature names
CONTINUOUS_COLS = ["A2", "A3", "A8", "A11", "A14", "A15"]
# Categorical feature names
CATEGORICAL_COLS = ["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"]
ALL_FEATURE_COLS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7",
                    "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15"]
TARGET_COL = "A16"


def load_credit_data(path: str = 'data/credit_approval.csv') -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[ALL_FEATURE_COLS + [TARGET_COL]]
    return df


def preprocess(df: pd.DataFrame):
    """
    Clean and encode the Credit Approval dataframe.
    - Impute missing values (median for numeric, mode for categorical).
    - Label‑encode all categorical columns.
    - Encode target as 0/1.

    Returns df_clean (all numeric, no NaN).
    """
    df = df.copy()

    # Impute continuous cols with median
    for col in CONTINUOUS_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # Impute & encode categorical cols
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "unknown"
            df[col] = df[col].fillna(mode_val)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Encode target: + → 1, - → 0
    df[TARGET_COL] = (df[TARGET_COL] == "+").astype(int)
    return df


def get_X_y(df: pd.DataFrame, continuous_only: bool = False):
    """Return feature matrix X, target vector y, and column names."""
    if continuous_only:
        cols = [c for c in CONTINUOUS_COLS if c in df.columns]
    else:
        cols = [c for c in df.columns if c != TARGET_COL]
    X = df[cols].values.astype(float)
    y = df[TARGET_COL].values
    return X, y, cols


def split_and_scale(X, y, test_size=0.2):
    """Train/test split + standard‑scaling (fit on train only)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test, scaler


# # ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     df_raw = load_credit_data()
#     print(f"Raw dataset : {df_raw.shape}")
#     print(f"Missing values:\n{df_raw.isnull().sum()}\n")
#     df = preprocess(df_raw)
#     print(f"Preprocessed: {df.shape}")
#     print(f"Target distribution:\n{df[TARGET_COL].value_counts()}\n")
#     X, y, cols = get_X_y(df)
#     print(f"X shape: {X.shape}, y shape: {y.shape}, Features: {cols}")