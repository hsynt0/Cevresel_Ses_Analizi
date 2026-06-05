"""
HearAlert - Data Augmentation (Salamon & Bello 2016, Section II-B)
====================================================================
Time Stretching, Pitch Shifting (PS1/PS2), Dynamic Range Compression,
Background Noise mixing. Class-conditional augmentation map uygulanir.
DRC rain/water_tap icin devre disi (paper Fig. 3).
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.logger import setup_logger
from src.config import config

logger = setup_logger(__name__, config.LOGS_DIR / "augmentation.log")


class AudioAugmentor:
    """Salamon & Bello (2016) Section II-B'ye gore ses augmentation.

    Desteklenen augmentasyonlar:
        - ts: Time Stretching (0.81, 0.93, 1.07, 1.23)
        - ps1: Pitch Shifting (-2, -1, +1, +2 semitone)
        - ps2: Pitch Shifting (-3.5, -2.5, +2.5, +3.5 semitone)
        - drc: Dynamic Range Compression (4 parametre seti)
        - bg: Background Noise mixing (w in [0.1, 0.5])
    """

    def __init__(self, sr: int = config.SAMPLE_RATE,
                 duration: float = config.DURATION,
                 seed: int = config.RANDOM_SEED):
        self.sr = sr
        self.duration = duration
        self.target_len = int(sr * duration)
        self.rng = np.random.default_rng(seed)
        logger.info("AudioAugmentor baslatildi - SR: %d, Sure: %.1fs", sr, duration)

    def _ensure_length(self, audio: np.ndarray) -> np.ndarray:
        """Augmente edilmis sesi tam 3 saniyeye trim/pad yapar."""
        if len(audio) > self.target_len:
            start = (len(audio) - self.target_len) // 2
            return audio[start:start + self.target_len]
        elif len(audio) < self.target_len:
            return np.pad(audio, (0, self.target_len - len(audio)))
        return audio

    def time_stretch(self, audio: np.ndarray, rate: float) -> np.ndarray:
        """Time stretching - tempo degistirme.

        Args:
            audio: Giris sinyali.
            rate: Stretch orani (< 1 = yavaslatma, > 1 = hizlandirma).
        """
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        return self._ensure_length(stretched)

    def pitch_shift(self, audio: np.ndarray, n_steps: float) -> np.ndarray:
        """Pitch shifting - perde kaydirma.

        Args:
            audio: Giris sinyali.
            n_steps: Semitone cinsinden kaydirma miktari.
        """
        shifted = librosa.effects.pitch_shift(
            audio, sr=self.sr, n_steps=n_steps
        )
        return self._ensure_length(shifted)

    def dynamic_range_compression(self, audio: np.ndarray,
                                   threshold: float = -20.0,
                                   ratio: float = 4.0) -> np.ndarray:
        """Dynamic Range Compression (DRC).

        Basit soft-knee compressor. Paper'da 4 farkli parametrizasyon kullanilir.
        DRC, surekli sesler (rain, water_tap) icin zararlıdir (paper Fig. 3).
        """
        # dB donusumu
        audio_db = 20 * np.log10(np.abs(audio) + 1e-10)
        # Threshold uzerindeki sinyalleri sikistir
        mask = audio_db > threshold
        compressed = audio.copy()
        if np.any(mask):
            gain_reduction = (audio_db[mask] - threshold) * (1 - 1/ratio)
            compressed[mask] = audio[mask] * 10 ** (-gain_reduction / 20)
        return self._ensure_length(compressed)

    def add_background_noise(self, audio: np.ndarray,
                              noise_pool: List[np.ndarray]) -> np.ndarray:
        """Background noise mixing: z = (1-w)*x + w*y, w in U(0.1, 0.5).

        Args:
            audio: Temiz sinyal.
            noise_pool: Gurultu sinyalleri havuzu.
        """
        if not noise_pool:
            return audio
        noise = self.rng.choice(noise_pool)
        noise = self._ensure_length(noise)
        w = self.rng.uniform(*config.BG_NOISE_WEIGHT_RANGE)
        mixed = (1 - w) * audio + w * noise
        return self._ensure_length(mixed)

    def augment_sample(self, audio: np.ndarray, class_name: str,
                       noise_pool: List[np.ndarray] = None
                       ) -> List[Tuple[np.ndarray, str]]:
        """Tek bir ornegi class-conditional augmentation map'e gore augmente eder.

        Args:
            audio: Giris sinyali (3sn, 22050Hz, mono).
            class_name: Sinif adi (augmentation map icin).
            noise_pool: BG noise icin diger siniflardan ornekler.

        Returns:
            List of (augmented_audio, augmentation_label) tuples.
        """
        enabled = config.CLASS_AUGMENTATION_MAP.get(class_name, [])
        augmented = []

        # Time Stretching
        if "ts" in enabled:
            for rate in config.TIME_STRETCH_FACTORS:
                aug = self.time_stretch(audio, rate)
                augmented.append((aug, f"ts_{rate}"))

        # Pitch Shift PS1
        if "ps1" in enabled:
            for steps in config.PITCH_SHIFT_PS1:
                aug = self.pitch_shift(audio, steps)
                augmented.append((aug, f"ps1_{steps}"))

        # Pitch Shift PS2
        if "ps2" in enabled:
            for steps in config.PITCH_SHIFT_PS2:
                aug = self.pitch_shift(audio, steps)
                augmented.append((aug, f"ps2_{steps}"))

        # Dynamic Range Compression (4 parametre seti)
        if "drc" in enabled:
            drc_params = [(-20, 4), (-25, 3), (-15, 5), (-30, 2)]
            for thresh, ratio in drc_params:
                aug = self.dynamic_range_compression(audio, thresh, ratio)
                augmented.append((aug, f"drc_{thresh}_{ratio}"))

        # Background Noise
        if "bg" in enabled and noise_pool:
            for i in range(2):  # 2 noise variant
                aug = self.add_background_noise(audio, noise_pool)
                augmented.append((aug, f"bg_{i}"))

        return augmented

    def augment_dataset(self, df: pd.DataFrame,
                        audio_dir: Path,
                        output_dir: Path) -> pd.DataFrame:
        """Tum veri setini augmente eder.

        Args:
            df: processed_meta.csv DataFrame'i.
            audio_dir: Islenmis ses dosyalari dizini.
            output_dir: Augmente dosyalarin kaydedilecegi dizin.

        Returns:
            Augmente edilmis metadata DataFrame (orijinal + augmente).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Noise pool olustur (tum dosyalardan rastgele ornekler)
        noise_pool = []
        for _, row in df[df["processed"] == True].sample(
                min(20, len(df)), random_state=config.RANDOM_SEED).iterrows():
            try:
                audio, _ = librosa.load(str(row["processed_path"]),
                                         sr=self.sr, mono=True)
                noise_pool.append(self._ensure_length(audio))
            except Exception:
                pass

        new_rows = []
        stats = {}

        logger.info("Augmentation basliyor - %d dosya", len(df))

        for idx, row in tqdm(df[df["processed"] == True].iterrows(),
                              total=df["processed"].sum(), desc="Augmentation"):
            class_name = row["category"]
            try:
                audio, _ = librosa.load(str(row["processed_path"]),
                                         sr=self.sr, mono=True)
                audio = self._ensure_length(audio)

                augments = self.augment_sample(audio, class_name, noise_pool)

                for aug_audio, aug_label in augments:
                    aug_filename = f"{class_name}_{idx:04d}_{aug_label}.wav"
                    aug_path = output_dir / aug_filename
                    sf.write(str(aug_path), aug_audio, self.sr, subtype="PCM_16")

                    new_row = row.copy()
                    new_row["filename"] = aug_filename
                    new_row["processed_path"] = str(aug_path)
                    new_row["augmentation"] = aug_label
                    new_rows.append(new_row)

                stats[class_name] = stats.get(class_name, 0) + len(augments)

            except Exception as e:
                logger.error("Augmentation hatasi - %s idx=%d: %s",
                             class_name, idx, e)

        # Orijinal veriye augmente veriyi ekle
        df_aug = pd.DataFrame(new_rows)
        df_combined = pd.concat([df, df_aug], ignore_index=True)

        logger.info("Augmentation tamamlandi:")
        for cls, count in sorted(stats.items()):
            logger.info("  - %s: +%d augmente ornek", cls, count)
        logger.info("  Toplam: %d orijinal + %d augmente = %d",
                     len(df), len(df_aug), len(df_combined))

        return df_combined
