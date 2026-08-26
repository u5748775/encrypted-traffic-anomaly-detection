import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# ── 1. Define feature columns ─────────────────────────────────────
# These features correspond to the feature vector xᵢ ∈ ℝᵈ defined in Chapter 2
# Selected based on their documented discriminative power in prior literature

FEATURE_COLUMNS = [
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',
    'Fwd Packet Length Mean',
    'Fwd Packet Length Std',
    'Bwd Packet Length Mean',
    'Bwd Packet Length Std',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Flow IAT Mean',
    'Flow IAT Std',
    'Fwd IAT Mean',
    'Fwd IAT Std',
    'Bwd IAT Mean',
    'Bwd IAT Std',
    'Fwd Header Length',
    'Bwd Header Length',
    'Fwd Packets/s',
    'Bwd Packets/s',
    'Packet Length Mean',
    'Packet Length Std',
    'Average Packet Size',
    'Avg Fwd Segment Size',
    'Avg Bwd Segment Size',
]

LABEL_COLUMN = 'Label'


# ── 2. Load data ──────────────────────────────────────────────────
def load_data(data_path):
    """
    Load CICIDS2017 CSV files from a file or directory.

    Parameters:
        data_path: path to a single CSV file or a directory of CSV files

    Returns:
        df: raw DataFrame
    """
    if os.path.isdir(data_path):
        all_files = [
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.endswith('.csv')
        ]
        print(f"Found {len(all_files)} CSV files")
        df = pd.concat(
            [pd.read_csv(f, encoding='utf-8', low_memory=False)
             for f in all_files],
            ignore_index=True
        )
    else:
        df = pd.read_csv(data_path, encoding='utf-8', low_memory=False)

    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


# ── 3. Clean data ─────────────────────────────────────────────────
def clean_data(df):
    """
    Clean the raw dataset by:
    - Stripping whitespace from column names (common issue in CICIDS2017)
    - Replacing infinite values with NaN
    - Dropping rows with missing values
    - Removing duplicate rows

    Parameters:
        df: raw DataFrame

    Returns:
        df: cleaned DataFrame
    """
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Report and drop missing values
    missing = df.isnull().sum().sum()
    print(f"Missing values found: {missing:,}")
    df.dropna(inplace=True)

    # Remove duplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    print(f"Removed {before - after:,} duplicate rows")
    print(f"Clean dataset size: {after:,} rows")

    return df


# ── 4. Extract features and labels ───────────────────────────────
def extract_features(df):
    """
    Extract the feature matrix X and label vector y from the DataFrame.

    Corresponds to the formalisation in Chapter 2:
        X = {x₁, x₂, ..., xₙ},  xᵢ ∈ ℝᵈ

    Label encoding:
        0 = normal (BENIGN)
        1 = anomalous (any attack type)

    Parameters:
        df: cleaned DataFrame

    Returns:
        X: feature matrix of shape (n, d)
        y: binary label vector of shape (n,)
        feature_names: list of feature column names used
    """
    # Keep only features that exist in this dataset
    available_features = [
        col for col in FEATURE_COLUMNS
        if col in df.columns
    ]
    print(f"Using {len(available_features)} features")

    # Extract feature matrix
    X = df[available_features].values

    # Encode labels: BENIGN = 0, all attacks = 1
    y = (df[LABEL_COLUMN].str.strip() != 'BENIGN').astype(int).values

    # Report class distribution
    n_normal = (y == 0).sum()
    n_anomaly = (y == 1).sum()
    print(f"Normal (BENIGN):  {n_normal:,} ({n_normal / len(y) * 100:.1f}%)")
    print(f"Anomalous (attack): {n_anomaly:,} ({n_anomaly / len(y) * 100:.1f}%)")

    return X, y, available_features


# ── 5. Split data into train and test sets ────────────────────────
def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and test sets.

    Strategy (consistent with the unsupervised learning setting in Chapter 2):
        - Training set: normal traffic only (BENIGN)
          The model learns the distribution of normal behaviour P(x)
        - Test set: mix of normal and anomalous traffic
          Used to evaluate anomaly score rankings

    Parameters:
        X: feature matrix
        y: binary label vector
        test_size: proportion of normal samples held out for testing
        random_state: random seed for reproducibility

    Returns:
        X_train: training set (normal traffic only)
        X_test: test set (normal + anomalous)
        y_test: test set labels
    """
    # Separate normal and anomalous samples
    X_normal = X[y == 0]
    X_anomaly = X[y == 1]

    # Split normal samples into train/test
    X_normal_train, X_normal_test = train_test_split(
        X_normal, test_size=test_size, random_state=random_state
    )

    # Test set = held-out normal samples + all anomalous samples
    X_test = np.vstack([X_normal_test, X_anomaly])
    y_test = np.concatenate([
        np.zeros(len(X_normal_test)),
        np.ones(len(X_anomaly))
    ])

    print(f"\nTraining set (normal only): {len(X_normal_train):,} samples")
    print(f"Test set total:             {len(X_test):,} samples")
    print(f"  - Normal:                 {len(X_normal_test):,}")
    print(f"  - Anomalous:              {len(X_anomaly):,}")

    return X_normal_train, X_test, y_test


# ── 6. Standardise features ───────────────────────────────────────
def scale_features(X_train, X_test=None):
    """
    Standardise features using StandardScaler (zero mean, unit variance).

    Important: the scaler is fitted on the training set only.
    Applying it to the test set avoids data leakage.

    This step is essential for LOF (distance-based) and the Autoencoder
    (gradient-based optimisation), as both are sensitive to feature scale.

    Parameters:
        X_train: training feature matrix
        X_test: test feature matrix (optional)

    Returns:
        X_train_scaled: standardised training set
        X_test_scaled: standardised test set (if provided)
        scaler: fitted StandardScaler object
    """
    scaler = StandardScaler()

    # Fit on training set only
    X_train_scaled = scaler.fit_transform(X_train)

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler

    return X_train_scaled, scaler


# ── 7. Full preprocessing pipeline ───────────────────────────────
def preprocess_pipeline(data_path, test_size=0.2):
    """
    End-to-end preprocessing pipeline combining all steps above.

    Parameters:
        data_path: path to CICIDS2017 CSV file(s)
        test_size: proportion of normal samples held out for testing

    Returns:
        X_train_scaled: standardised training set
        X_test_scaled: standardised test set
        y_test: test set labels
        scaler: fitted StandardScaler object
        feature_names: list of feature names used
    """
    print("=" * 50)
    print("Step 1: Loading data...")
    df = load_data(data_path)

    print("\nStep 2: Cleaning data...")
    df = clean_data(df)

    print("\nStep 3: Extracting features and labels...")
    X, y, feature_names = extract_features(df)

    print("\nStep 4: Splitting into train and test sets...")
    X_train, X_test, y_test = split_data(X, y, test_size)

    print("\nStep 5: Standardising features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    print("\n" + "=" * 50)
    print("Preprocessing complete!")
    print(f"  Training set shape: {X_train_scaled.shape}")
    print(f"  Test set shape:     {X_test_scaled.shape}")
    print("=" * 50)

    return X_train_scaled, X_test_scaled, y_test, scaler, feature_names