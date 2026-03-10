import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
CONTINUOUS_COLS = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]
N_SAMPLES = 2000  # manageable size for LOOCV yet statistically meaningful
SAMPLE_SIZE = 20000
N_CLASSES = 7


def load_covtype_data(csv_path: str = "data/covtype.csv") -> pd.DataFrame:
    """
    Returns
    -------
    df : pd.DataFrame  (N_SAMPLES rows, 55 columns including Cover_Type)
    """
    
    df = pd.read_csv(csv_path)
    return df


def get_X_y(df: pd.DataFrame, continuous_only: bool = False):
    """Return feature matrix X and target vector y."""
    target = "Cover_Type"
    if continuous_only:
        cols = [c for c in CONTINUOUS_COLS if c in df.columns]
    else:
        cols = [c for c in df.columns if c != target]
    X = df[cols].values.astype(float)[:SAMPLE_SIZE]
    y = df[target].values[:SAMPLE_SIZE]
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


# ---------------------------------------------------------------------------
# Quick self‑test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_covtype_data()
    print(f"Dataset shape : {df.shape}")
    print(f"Cover_Type distribution:\n{df['Cover_Type'].value_counts().sort_index()}")
    X, y, cols = get_X_y(df)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    X_tr, X_te, y_tr, y_te, sc = split_and_scale(X, y)
    print(f"Train: {X_tr.shape}, Test: {X_te.shape}")