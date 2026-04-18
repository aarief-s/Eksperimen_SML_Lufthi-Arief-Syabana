"""
automate_Lufthi-Arief-Syabana.py
=================================
Script otomatisasi preprocessing dataset Smartphone Usage and Addiction Analysis.
Penulis : Lufthi Arief Syabana
Dataset : Smartphone Usage and Addiction Analysis (7500 Rows)
Task    : Binary Classification - Prediksi addicted_label (0/1)

Cara menjalankan:
    python automate_Lufthi-Arief-Syabana.py
    python automate_Lufthi-Arief-Syabana.py --input ../smartphone_usage_raw.csv --output smartphone_usage_preprocessing
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ─── Konfigurasi Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ─── Konstanta ────────────────────────────────────────────────────────────────
DEFAULT_INPUT  = '../smartphone_usage_raw.csv'
DEFAULT_OUTPUT = 'smartphone_usage_preprocessing'
TARGET_COL     = 'addicted_label'
RANDOM_STATE   = 42
TEST_SIZE      = 0.2

COLS_TO_DROP = ['transaction_id', 'user_id']

COLS_TO_SCALE = [
    'age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
    'work_study_hours', 'sleep_hours', 'notifications_per_day',
    'app_opens_per_day', 'weekend_screen_time'
]

STRESS_MAP    = {'Low': 0, 'Medium': 1, 'High': 2}
ADDICTION_MAP = {'Mild': 0, 'Moderate': 1, 'Severe': 2}
ACADEMIC_MAP  = {'No': 0, 'Yes': 1}


# ─── Fungsi-fungsi Preprocessing ─────────────────────────────────────────────

def load_data(input_path: str) -> pd.DataFrame:
    """
    Memuat dataset dari file CSV.

    Parameters
    ----------
    input_path : str
        Path ke file CSV raw.

    Returns
    -------
    pd.DataFrame
        DataFrame hasil load.
    """
    logger.info(f"Memuat dataset dari: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File tidak ditemukan: {input_path}")

    df = pd.read_csv(input_path)
    logger.info(f"Dataset berhasil dimuat — Shape: {df.shape}")
    return df


def drop_irrelevant_columns(df: pd.DataFrame, cols: list = COLS_TO_DROP) -> pd.DataFrame:
    """
    Menghapus kolom-kolom yang tidak relevan untuk pemodelan (ID columns).

    Parameters
    ----------
    df   : pd.DataFrame
    cols : list  Daftar nama kolom yang akan di-drop.

    Returns
    -------
    pd.DataFrame
    """
    existing = [c for c in cols if c in df.columns]
    df = df.drop(columns=existing)
    logger.info(f"Drop kolom tidak relevan: {existing} — Shape: {df.shape}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menangani missing values dengan strategi:
    - addiction_level : diimputasi dengan nilai modus (mode)

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    missing_before = df.isnull().sum().sum()
    logger.info(f"Total missing values sebelum penanganan: {missing_before}")

    if 'addiction_level' in df.columns and df['addiction_level'].isnull().sum() > 0:
        mode_val = df['addiction_level'].mode()[0]
        n_missing = df['addiction_level'].isnull().sum()
        df['addiction_level'] = df['addiction_level'].fillna(mode_val)
        logger.info(f"Imputasi addiction_level ({n_missing} nilai NaN) → modus: '{mode_val}'")

    missing_after = df.isnull().sum().sum()
    logger.info(f"Total missing values setelah penanganan: {missing_after}")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan encoding pada fitur-fitur kategorikal:
    - stress_level        : Label Encoding ordinal (Low=0, Medium=1, High=2)
    - addiction_level     : Label Encoding ordinal (Mild=0, Moderate=1, Severe=2)
    - academic_work_impact: Label Encoding biner (No=0, Yes=1)
    - gender              : One-Hot Encoding nominal (3 kategori)

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    # Label Encoding ordinal
    if 'stress_level' in df.columns:
        df['stress_level'] = df['stress_level'].map(STRESS_MAP)
        logger.info(f"Label Encoding stress_level: {STRESS_MAP}")

    if 'addiction_level' in df.columns:
        df['addiction_level'] = df['addiction_level'].map(ADDICTION_MAP)
        logger.info(f"Label Encoding addiction_level: {ADDICTION_MAP}")

    # Label Encoding biner
    if 'academic_work_impact' in df.columns:
        df['academic_work_impact'] = df['academic_work_impact'].map(ACADEMIC_MAP)
        logger.info(f"Label Encoding academic_work_impact: {ACADEMIC_MAP}")

    # One-Hot Encoding nominal
    if 'gender' in df.columns:
        df = pd.get_dummies(df, columns=['gender'], prefix='gender', drop_first=False)
        logger.info("One-Hot Encoding gender → gender_Female, gender_Male, gender_Other")

    # Pastikan kolom boolean OHE dikonversi ke integer
    ohe_cols = [c for c in df.columns if c.startswith('gender_')]
    df[ohe_cols] = df[ohe_cols].astype(int)

    logger.info(f"Encoding selesai — Shape: {df.shape}")
    return df


def normalize_features(df: pd.DataFrame, cols: list = COLS_TO_SCALE) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Melakukan normalisasi StandardScaler pada fitur numerik.

    Parameters
    ----------
    df   : pd.DataFrame
    cols : list  Daftar kolom numerik yang dinormalisasi.

    Returns
    -------
    tuple[pd.DataFrame, StandardScaler]
        DataFrame ternormalisasi dan objek scaler yang sudah fit.
    """
    existing_cols = [c for c in cols if c in df.columns]
    scaler = StandardScaler()
    df[existing_cols] = scaler.fit_transform(df[existing_cols])
    logger.info(f"StandardScaler diterapkan pada: {existing_cols}")
    return df, scaler


def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Membagi data menjadi train dan test set dengan stratified split.

    Parameters
    ----------
    df           : pd.DataFrame
    target_col   : str
    test_size    : float
    random_state : int

    Returns
    -------
    tuple: X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    logger.info(f"Train-Test Split 80:20 (stratified)")
    logger.info(f"  X_train: {X_train.shape} | X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def save_results(
    df_full: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str = DEFAULT_OUTPUT
) -> None:
    """
    Menyimpan hasil preprocessing ke folder output.

    Parameters
    ----------
    df_full    : pd.DataFrame  Dataset lengkap setelah preprocessing.
    X_train    : pd.DataFrame
    X_test     : pd.DataFrame
    y_train    : pd.Series
    y_test     : pd.Series
    output_dir : str
    """
    os.makedirs(output_dir, exist_ok=True)

    # Dataset lengkap
    full_path = os.path.join(output_dir, 'smartphone_usage_preprocessing.csv')
    df_full.to_csv(full_path, index=False)
    logger.info(f"Saved: {full_path} ({len(df_full):,} rows)")

    # Train set
    df_train = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    train_path = os.path.join(output_dir, 'train.csv')
    df_train.to_csv(train_path, index=False)
    logger.info(f"Saved: {train_path} ({len(df_train):,} rows)")

    # Test set
    df_test = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    test_path = os.path.join(output_dir, 'test.csv')
    df_test.to_csv(test_path, index=False)
    logger.info(f"Saved: {test_path} ({len(df_test):,} rows)")


def run_preprocessing(input_path: str = DEFAULT_INPUT, output_dir: str = DEFAULT_OUTPUT) -> pd.DataFrame:
    """
    Fungsi utama yang menjalankan seluruh pipeline preprocessing secara berurutan.

    Alur:
        load → drop_irrelevant → handle_missing → encode → normalize → split → save

    Parameters
    ----------
    input_path : str  Path ke file CSV raw.
    output_dir : str  Folder output hasil preprocessing.

    Returns
    -------
    pd.DataFrame
        Dataset lengkap setelah preprocessing (sebelum split).
    """
    logger.info("=" * 55)
    logger.info("  MEMULAI PIPELINE PREPROCESSING")
    logger.info("  Lufthi Arief Syabana — Smartphone Addiction")
    logger.info("=" * 55)

    # 1. Load data
    df = load_data(input_path)

    # 2. Drop kolom tidak relevan
    df = drop_irrelevant_columns(df)

    # 3. Tangani missing values
    df = handle_missing_values(df)

    # 4. Encoding kategorikal
    df = encode_categorical(df)

    # 5. Normalisasi fitur numerik
    df, scaler = normalize_features(df)

    # 6. Split data
    X_train, X_test, y_train, y_test = split_data(df)

    # 7. Simpan hasil
    save_results(df, X_train, X_test, y_train, y_test, output_dir)

    logger.info("=" * 55)
    logger.info("  PREPROCESSING SELESAI")
    logger.info(f"  Output disimpan di: {output_dir}/")
    logger.info("=" * 55)

    return df


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Automate Preprocessing — Smartphone Usage & Addiction (Lufthi Arief Syabana)'
    )
    parser.add_argument(
        '--input', type=str, default=DEFAULT_INPUT,
        help=f'Path ke file CSV raw (default: {DEFAULT_INPUT})'
    )
    parser.add_argument(
        '--output', type=str, default=DEFAULT_OUTPUT,
        help=f'Folder output hasil preprocessing (default: {DEFAULT_OUTPUT})'
    )
    args = parser.parse_args()

    run_preprocessing(input_path=args.input, output_dir=args.output)
