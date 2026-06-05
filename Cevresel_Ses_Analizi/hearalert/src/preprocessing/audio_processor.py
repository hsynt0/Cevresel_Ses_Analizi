"""
HearAlert — Ses Ön İşleme Modülü
====================================

Ham ses dosyalarını eğitim için hazırlar:
    1. Güvenli yükleme (hata yakalama ile)
    2. Mono dönüşüm (çevresel ses = kaynak tanımlama, uzamsal bilgi gereksiz)
    3. Yeniden örnekleme → 22050 Hz (Nyquist: 11025 Hz, tüm hedef frekansları kapsar)
    4. Kırpma/padding → tam 3 saniye (CNN sabit boyutlu giriş gerektirir)
    5. Peak normalizasyon (tutarlı genlik seviyeleri)

Referans:
    Salamon & Bello (2016): TF-patch boyutu 3 saniye, 128 frame,
    22050 Hz örnekleme hızı ile log-mel spectrogram.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.logger import setup_logger
from src.config import config

logger = setup_logger(__name__, config.LOGS_DIR / "preprocessing.log")


def load_audio_safe(file_path: Path, sr: int = config.SAMPLE_RATE
                    ) -> Optional[np.ndarray]:
    """Ses dosyasını güvenli şekilde yükler (hata yakalama ile).

    Args:
        file_path: Ses dosyası yolu (.wav, .ogg, .flac, vb.).
        sr: Hedef örnekleme hızı. None ise orijinal SR korunur.

    Returns:
        Mono ses dizisi veya hata durumunda None.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.error("Dosya bulunamadı: %s", file_path)
        return None

    try:
        audio, loaded_sr = librosa.load(str(file_path), sr=sr, mono=True)

        if len(audio) == 0:
            logger.warning("Boş ses dosyası: %s", file_path)
            return None

        if np.all(audio == 0):
            logger.warning("Sessiz dosya (tüm değerler sıfır): %s", file_path)
            # Yine de döndür — augmentation ile kullanılabilir

        logger.info("Yüklendi: %s — Süre: %.2fs, SR: %d",
                     file_path.name, len(audio) / sr, sr)
        return audio

    except Exception as e:
        logger.error("Ses yükleme hatası — %s: %s", file_path.name, e)
        return None


def convert_to_mono(audio: np.ndarray) -> np.ndarray:
    """Çok kanallı sesi mono'ya dönüştürür.

    Neden mono:
        Çevresel ses tanıma, kaynak kimliğini tespit eder (uzamsal konum değil).
        Mono ortalama, her iki kanaldan enerjiyi korur ve bellek
        gereksinimini yarıya indirir.

    Args:
        audio: Ses dizisi. Shape: (n_samples,) veya (n_channels, n_samples).

    Returns:
        Mono ses dizisi, shape: (n_samples,).
    """
    if audio.ndim == 1:
        return audio
    elif audio.ndim == 2:
        mono = np.mean(audio, axis=0)
        logger.info("Mono donusum: %d kanal -> 1 kanal", audio.shape[0])
        return mono
    else:
        raise ValueError(f"Beklenmeyen ses boyutu: {audio.ndim}D (1D veya 2D bekleniyor)")


def resample_audio(audio: np.ndarray, orig_sr: int,
                   target_sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Sesi hedef örnekleme hızına yeniden örnekler.

    Neden 22050 Hz:
        - FFT hesaplamayı yarıya indirir (44100 → 22050)
        - Nyquist limiti 11025 Hz: tüm hedef ses frekansları (<8000 Hz) korunur
        - Eğitim ve mikrofon çıkarımı arasında tutarlılık sağlar

    Args:
        audio: Giriş ses dizisi.
        orig_sr: Orijinal örnekleme hızı.
        target_sr: Hedef örnekleme hızı.

    Returns:
        Yeniden örneklenmiş ses dizisi.
    """
    if orig_sr == target_sr:
        return audio

    resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    logger.info("Yeniden ornekleme: %d Hz -> %d Hz", orig_sr, target_sr)
    return resampled


def trim_or_pad(audio: np.ndarray, sr: int = config.SAMPLE_RATE,
                duration: float = config.DURATION) -> np.ndarray:
    """Sesi tam olarak belirtilen süreye kırpar veya pad uygular.

    Neden sabit uzunluk (3 saniye):
        Sinir ağları sabit boyutlu girişler gerektirir.
        Salamon & Bello (2016) TF-patch boyutunu 3 saniye (128 frame) olarak sabitler.
        - Kısa klipler: Sonda sıfır-pad (onset bilgisini korur)
        - Uzun klipler: Ortadan kırpma (en bilgilendirici segmenti tutar)

    Args:
        audio: Giriş ses dizisi.
        sr: Örnekleme hızı.
        duration: Hedef süre (saniye).

    Returns:
        Tam olarak target_samples uzunluğunda ses dizisi.
    """
    target_samples = int(sr * duration)

    if len(audio) == target_samples:
        return audio
    elif len(audio) > target_samples:
        # Ortadan kırpma — en bilgilendirici segmenti tut
        start = (len(audio) - target_samples) // 2
        trimmed = audio[start:start + target_samples]
        logger.info("Kirpma: %d -> %d ornek (ortadan)", len(audio), target_samples)
        return trimmed
    else:
        # Sonda sıfır-padding — onset bilgisini koru
        padded = np.pad(audio, (0, target_samples - len(audio)), mode="constant")
        logger.info("Padding: %d -> %d ornek (sonda sifir)", len(audio), target_samples)
        return padded


def normalize_audio(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """Peak normalizasyon uygular.

    Args:
        audio: Giriş ses dizisi.
        peak: Hedef peak değeri (0-1 arası).

    Returns:
        Normalize edilmiş ses dizisi.
    """
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio * (peak / max_val)
    else:
        logger.warning("Sessiz sinyal - normalizasyon atlandi.")
    return audio


def process_single_file(file_path: Path, output_path: Path,
                        sr: int = config.SAMPLE_RATE,
                        duration: float = config.DURATION) -> bool:
    """Tek bir ses dosyasını ön işler ve kaydeder.

    Pipeline: yükle → mono → resample → trim/pad → normalize → kaydet

    Args:
        file_path: Giriş dosya yolu.
        output_path: Çıktı dosya yolu.
        sr: Hedef örnekleme hızı.
        duration: Hedef süre (saniye).

    Returns:
        Başarılıysa True, hata varsa False.
    """
    audio = load_audio_safe(file_path, sr=sr)
    if audio is None:
        return False

    try:
        audio = convert_to_mono(audio)
        audio = trim_or_pad(audio, sr=sr, duration=duration)
        audio = normalize_audio(audio)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, sr, subtype="PCM_16")

        return True

    except Exception as e:
        logger.error("İşleme hatası — %s: %s", file_path.name, e)
        return False


def process_dataset(df: pd.DataFrame, audio_dir: Path,
                    output_dir: Path, sr: int = config.SAMPLE_RATE,
                    duration: float = config.DURATION) -> pd.DataFrame:
    """Tüm veri setini ön işler.

    Args:
        df: Metadata DataFrame (en az 'filename' ve 'category' sütunları).
        audio_dir: Ham ses dosyalarının bulunduğu dizin.
        output_dir: İşlenmiş dosyaların kaydedileceği dizin.
        sr: Hedef örnekleme hızı.
        duration: Hedef süre.

    Returns:
        İşleme sonuçlarıyla güncellenmiş DataFrame
        ('processed_path' ve 'processed' sütunları eklenir).
    """
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["processed_path"] = ""
    df["processed"] = False

    success_count = 0
    fail_count = 0

    logger.info("Veri seti ön işleme başlıyor — %d dosya", len(df))

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Ön İşleme"):
        # Dosya yolunu belirle
        if "path" in row and row["path"] and Path(row["path"]).exists():
            input_path = Path(row["path"])
        else:
            # ESC-50 yapısı: audio_dir/filename
            input_path = audio_dir / row["filename"]
            # Veya sınıf alt dizini: audio_dir/category/filename
            if not input_path.exists():
                input_path = audio_dir / row["category"] / row["filename"]

        # Çıktı dosya adı
        output_filename = f"{row['category']}_{idx:04d}.wav"
        output_path = output_dir / output_filename

        success = process_single_file(input_path, output_path, sr, duration)

        if success:
            df.at[idx, "processed_path"] = str(output_path)
            df.at[idx, "processed"] = True
            success_count += 1
        else:
            fail_count += 1

    logger.info("Ön işleme tamamlandı — Başarılı: %d, Başarısız: %d",
                 success_count, fail_count)

    return df
