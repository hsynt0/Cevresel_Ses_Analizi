"""
HearAlert - MFCC Feature Extractor
====================================
40 MFCC katsayisi cikarir, mean + std ile 80-boyutlu vektor olusturur.
MLP modeli icin birincil giris temsili.

Referans: Salamon & Bello (2016) - 80-dim MFCC feature vectors (40 means + 40 stds)
"""

import numpy as np
import librosa
from pathlib import Path
from typing import Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.logger import setup_logger
from src.config import config

logger = setup_logger(__name__, config.LOGS_DIR / "features.log")


def extract_mfcc(audio: np.ndarray, sr: int = config.SAMPLE_RATE,
                 n_mfcc: int = config.N_MFCC, n_fft: int = config.N_FFT,
                 hop_length: int = config.HOP_LENGTH) -> np.ndarray:
    """Tek bir ses sinyalinden 80-boyutlu MFCC vektor cikarir.

    Islem: audio -> STFT -> Mel filterbank -> log -> DCT -> 40 MFCC
    Sonra zaman ekseni uzerinde mean + std -> 80-dim vektor.

    Args:
        audio: Mono ses dizisi.
        sr: Ornekleme hizi.
        n_mfcc: MFCC katsayi sayisi.
        n_fft: FFT pencere boyutu.
        hop_length: Hop uzunlugu.

    Returns:
        80-boyutlu MFCC vektor (40 mean + 40 std).
    """
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc,
        n_fft=n_fft, hop_length=hop_length
    )
    # Zaman ekseni uzerinde istatistikler
    mfcc_mean = np.mean(mfcc, axis=1)  # (40,)
    mfcc_std = np.std(mfcc, axis=1)    # (40,)
    feature_vector = np.concatenate([mfcc_mean, mfcc_std])  # (80,)
    return feature_vector


def batch_extract_mfcc(file_paths: list, sr: int = config.SAMPLE_RATE
                       ) -> Tuple[np.ndarray, list]:
    """Birden fazla ses dosyasindan toplu MFCC cikarir.

    Args:
        file_paths: Ses dosyasi yollari listesi.
        sr: Ornekleme hizi.

    Returns:
        Tuple: (X: (N, 80) MFCC matrisi, basarisiz dosya indeksleri)
    """
    features = []
    failed_indices = []

    for i, fpath in enumerate(file_paths):
        try:
            audio, _ = librosa.load(str(fpath), sr=sr, mono=True,
                                     duration=config.DURATION)
            # Tam 3 saniyeye pad/trim
            target_len = int(sr * config.DURATION)
            if len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)))
            elif len(audio) > target_len:
                audio = audio[:target_len]

            feat = extract_mfcc(audio, sr)
            features.append(feat)
        except Exception as e:
            logger.error("MFCC cikarim hatasi - %s: %s", fpath, e)
            failed_indices.append(i)
            features.append(np.zeros(config.FEATURE_DIM))

    X = np.array(features, dtype=np.float32)
    logger.info("MFCC cikarimi tamamlandi - %d dosya, %d basarisiz",
                len(features), len(failed_indices))
    return X, failed_indices
