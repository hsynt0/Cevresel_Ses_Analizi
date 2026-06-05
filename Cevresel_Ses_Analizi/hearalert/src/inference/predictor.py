"""
HearAlert - Sound Predictor
==============================
Egitilmis modeli yukler ve ses sinyalleri uzerinde tahmin yapar.
MLP (MFCC) veya CNN (LogMel) modeli destekler.
Confidence thresholding: >=0.75 normal, >=0.65 guvenlik-kritik.
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Tuple, Dict
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.logger import setup_logger
from src.config import config

logger = setup_logger(__name__, config.LOGS_DIR / "inference.log")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class SoundPredictor:
    """Ses sinifi tahmin motoru.

    Attributes:
        model_type: 'mlp' veya 'cnn'.
        model: Yuklu Keras modeli.
        scaler: MLP icin StandardScaler.
        label_encoder: LabelEncoder.
    """

    def __init__(self, model_type: str = "mlp"):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.class_names = []
        self.norm_mean = None
        self.norm_std = None
        self._loaded = False

    def load_model(self) -> bool:
        """Modeli ve iliskili dosyalari yukler."""
        from tensorflow import keras
        try:
            if self.model_type == "mlp":
                model_dir = config.MODELS_MLP_DIR
                self.model = keras.models.load_model(
                    str(model_dir / "best_model.keras"))
                self.scaler = joblib.load(model_dir / "scaler.joblib")
            else:
                model_dir = config.MODELS_CNN_DIR
                self.model = keras.models.load_model(
                    str(model_dir / "best_model.keras"))
                self.norm_mean = np.load(model_dir / "norm_mean.npy")
                self.norm_std = np.load(model_dir / "norm_std.npy")

            self.label_encoder = joblib.load(model_dir / "label_encoder.joblib")
            self.class_names = list(self.label_encoder.classes_)

            # Warm-up: TF predict fonksiyonunu ana thread'de derle
            # Bu olmadan, baska thread'den ilk predict() cagrisi sessizce basarisiz olur
            self._warmup_predict()

            self._loaded = True
            logger.info("%s modeli yuklendi - %d sinif",
                        self.model_type.upper(), len(self.class_names))
            return True
        except Exception as e:
            logger.error("Model yukleme hatasi: %s", e)
            return False

    def _warmup_predict(self):
        """Dummy tahmin yaparak TF predict fonksiyonunu thread-safe hale getirir."""
        try:
            if self.model_type == "mlp":
                dummy = np.zeros((1, config.FEATURE_DIM), dtype=np.float32)
            else:
                dummy = np.zeros((1, config.N_MELS, 128, 1), dtype=np.float32)
            _ = self.model.predict(dummy, verbose=0)
            logger.info("Warm-up predict basarili (thread-safe hazir).")
        except Exception as e:
            logger.warning("Warm-up predict hatasi: %s", e)

    def preprocess(self, audio: np.ndarray, sr: int = config.SAMPLE_RATE
                   ) -> np.ndarray:
        """Ses sinyalini tahmin icin on isler (egitim pipeline'iyla ayni)."""
        import librosa
        # Tam 3 saniyeye ayarla
        target_len = int(sr * config.DURATION)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        elif len(audio) > target_len:
            start = (len(audio) - target_len) // 2
            audio = audio[start:start + target_len]

        if self.model_type == "mlp":
            from src.features.mfcc_extractor import extract_mfcc
            feat = extract_mfcc(audio, sr)
            feat = feat.reshape(1, -1)
            if self.scaler is not None:
                feat = self.scaler.transform(feat)
            return feat
        else:
            from src.features.logmel_extractor import extract_logmel
            logmel = extract_logmel(audio, sr)
            logmel = logmel[np.newaxis, ..., np.newaxis]  # (1, 128, 128, 1)
            if self.norm_mean is not None:
                logmel = (logmel - self.norm_mean) / self.norm_std
            return logmel

    def predict(self, audio: np.ndarray,
                sr: int = config.SAMPLE_RATE) -> Tuple[str, float, Dict[str, float]]:
        """Tahmin yapar, en yuksek sinif + guven + tum olasiliklar dondurur."""
        if not self._loaded:
            raise RuntimeError("Model yuklenmedi! Once load_model() cagirin.")

        features = self.preprocess(audio, sr)
        probs = self.model.predict(features, verbose=0)[0]

        top_idx = np.argmax(probs)
        class_name = self.class_names[top_idx]
        confidence = float(probs[top_idx])

        all_probs = {name: float(probs[i])
                     for i, name in enumerate(self.class_names)}

        return class_name, confidence, all_probs

    def predict_with_threshold(self, audio: np.ndarray,
                                sr: int = config.SAMPLE_RATE
                                ) -> Optional[Tuple[str, float, bool, Dict[str, float]]]:
        """Esik degerli tahmin. Esik altindaysa None dondurur.

        Returns:
            (class_name, confidence, is_critical, all_probs) veya None.
        """
        class_name, confidence, all_probs = self.predict(audio, sr)

        is_critical = class_name in config.SAFETY_CRITICAL_CLASSES
        threshold = (config.CRITICAL_CONFIDENCE_THRESHOLD if is_critical
                     else config.DEFAULT_CONFIDENCE_THRESHOLD)

        if confidence >= threshold:
            return class_name, confidence, is_critical, all_probs
        return None
