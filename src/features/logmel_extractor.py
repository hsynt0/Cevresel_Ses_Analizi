"""
HearAlert - Log-Mel Spectrogram Extractor
============================================
128 mel band, 128 frame -> (128, 128) log-mel spectrogram.
SB-CNN (Salamon & Bello 2016) icin birincil giris temsili.

STFT -> Mel Filterbank -> Power-to-dB -> (128, 128, 1) CNN girisi
"""

import numpy as np
import librosa
from pathlib import Path
from typing import Tuple, List
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.logger import setup_logger
from src.config import config

logger = setup_logger(__name__, config.LOGS_DIR / "features.log")


def extract_logmel(audio: np.ndarray, sr: int = config.SAMPLE_RATE,
                   n_mels: int = config.N_MELS, n_fft: int = config.N_FFT,
                   hop_length: int = config.HOP_LENGTH,
                   target_frames: int = 128) -> np.ndarray:
    """Tek bir ses sinyalinden log-mel spectrogram cikarir.

    Islem: audio -> STFT -> Mel filterbank (128 band) -> power_to_db
    Sonuc: (128, 128) matris — 128 mel band x 128 zaman frame'i

    Args:
        audio: Mono ses dizisi.
        sr: Ornekleme hizi.
        n_mels: Mel filtre sayisi.
        n_fft: FFT pencere boyutu.
        hop_length: Hop uzunlugu.
        target_frames: Hedef zaman frame sayisi.

    Returns:
        (n_mels, target_frames) boyutunda log-mel spectrogram.
    """
    # Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, power=2.0
    )

    # Log-scale (power to dB)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    # Frame sayisini hedef boyuta ayarla
    if log_mel.shape[1] > target_frames:
        # Ortadan kirp
        start = (log_mel.shape[1] - target_frames) // 2
        log_mel = log_mel[:, start:start + target_frames]
    elif log_mel.shape[1] < target_frames:
        # Sonda pad
        pad_width = target_frames - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)),
                         mode='constant', constant_values=log_mel.min())

    return log_mel  # (128, 128)


def batch_extract_logmel(file_paths: list, sr: int = config.SAMPLE_RATE
                         ) -> Tuple[np.ndarray, list]:
    """Birden fazla ses dosyasindan toplu log-mel spectrogram cikarir.

    Args:
        file_paths: Ses dosyasi yollari listesi.
        sr: Ornekleme hizi.

    Returns:
        Tuple: (X: (N, 128, 128, 1) log-mel tensor, basarisiz indeksler)
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

            logmel = extract_logmel(audio, sr)
            features.append(logmel)
        except Exception as e:
            logger.error("Log-mel cikarim hatasi - %s: %s", fpath, e)
            failed_indices.append(i)
            features.append(np.zeros((config.N_MELS, 128)))

    X = np.array(features, dtype=np.float32)
    # CNN icin kanal boyutu ekle: (N, 128, 128) -> (N, 128, 128, 1)
    X = X[..., np.newaxis]

    logger.info("Log-mel cikarimi tamamlandi - X shape: %s, %d basarisiz",
                X.shape, len(failed_indices))
    return X, failed_indices
