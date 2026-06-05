"""
HearAlert - Stratified Data Splitter & Normalizer
===================================================
Stratified train/val/test split + StandardScaler normalizasyonu.
Scaler sadece train set'e fit edilir (data leakage onleme).
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.logger import setup_logger
from src.config import config

logger = setup_logger(__name__, config.LOGS_DIR / "training.log")


def stratified_split(X: np.ndarray, y: np.ndarray,
                     test_size: float = 0.2,
                     val_size: float = 0.15,
                     random_state: int = config.RANDOM_SEED
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                np.ndarray, np.ndarray, np.ndarray]:
    """DIKKAT: Augmente verilerde kullanimi Veri Sizintisi (Data Leakage) yaratir.
    Rastgele bolme yaptigi icin orijinal dosyalarin kopya/augmente halleri
    hem Train hem Test'e dagilir. fold_based_split kullanin.
    """
    # Ilk bolme: train+val / test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y
    )

    # Ikinci bolme: train / val
    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=relative_val,
        random_state=random_state, stratify=y_trainval
    )

    logger.warning("stratified_split KULLANILDI (Veri Sizintisi riski yuksek!)")
    logger.info("  Train: %d, Val: %d, Test: %d", len(X_train), len(X_val), len(X_test))

    return X_train, X_val, X_test, y_train, y_val, y_test


def fold_based_split(X: np.ndarray, y: np.ndarray, folds: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                np.ndarray, np.ndarray, np.ndarray]:
    """ESC-50 Fold bilgisini kullanarak veriyi boler.
    Train: Folds 1, 2, 3
    Validation: Fold 4
    Test: Fold 5
    
    Bu yontem Veri Sizintisini (Data Leakage) onler.
    
    Args:
        X: Feature matrisi.
        y: Etiketler.
        folds: Her ornegin fold numarasi (1-5).
        
    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    train_mask = np.isin(folds, [1, 2, 3])
    val_mask = folds == 4
    test_mask = folds == 5
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    logger.info("Fold-based veri bolme (Leakage Free) tamamlandi:")
    logger.info("  Train (Folds 1-3): %d", len(X_train))
    logger.info("  Val (Fold 4): %d", len(X_val))
    logger.info("  Test (Fold 5): %d", len(X_test))
    logger.info("  Train sinif dagilimi: %s", dict(zip(*np.unique(y_train, return_counts=True))))
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize_features(X_train: np.ndarray, X_val: np.ndarray,
                       X_test: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Z-score normalizasyonu uygular. Scaler sadece train'e fit edilir.

    Args:
        X_train, X_val, X_test: Feature matrisleri.

    Returns:
        (X_train_norm, X_val_norm, X_test_norm, scaler)
    """
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)

    logger.info("Normalizasyon tamamlandi - mean range: [%.2f, %.2f], std range: [%.2f, %.2f]",
                scaler.mean_.min(), scaler.mean_.max(),
                scaler.scale_.min(), scaler.scale_.max())

    return X_train_norm, X_val_norm, X_test_norm, scaler


def save_scaler(scaler: StandardScaler, path: Optional[Path] = None) -> None:
    """Scaler'i kaydet."""
    if path is None:
        path = config.MODELS_MLP_DIR / "scaler.joblib"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    logger.info("Scaler kaydedildi: %s", path)
