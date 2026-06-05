"""
HearAlert - Real-Time Audio Listener
========================================
2 thread mimarisi:
    Thread 1 (Capture): sounddevice.InputStream -> queue
    Thread 2 (Inference): queue -> preprocess -> predict -> gui_queue
    Main thread (GUI): gui_queue -> UI guncelleme

Audio capture asla inference/GUI tarafindan ENGELLENMEZ.
"""

import numpy as np
import queue
import threading
import time
from typing import Optional, List, Dict
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.logger import setup_logger
from src.config import config

logger = setup_logger(__name__, config.LOGS_DIR / "inference.log")

try:
    import sounddevice as sd
except ImportError:
    sd = None
    logger.warning("sounddevice yuklu degil! pip install sounddevice")


class RealTimeListener:
    """Gercek zamanli ses dinleyici - 2 thread mimarisi.

    Attributes:
        predictor: SoundPredictor nesnesi.
        sr: Ornekleme hizi.
        duration: Yakalama suresi (saniye).
        gui_queue: GUI event queue.
    """

    def __init__(self, predictor, sr: int = config.SAMPLE_RATE,
                 duration: float = config.DURATION,
                 device_index: Optional[int] = config.MIC_DEVICE_INDEX,
                 gui_queue: Optional[queue.Queue] = None):
        self.predictor = predictor
        self.sr = sr
        self.duration = duration
        self.device_index = device_index
        self.gui_queue = gui_queue or queue.Queue(maxsize=50)
        self._audio_queue = queue.Queue(maxsize=10)
        self._running = False
        self._capture_thread = None
        self._inference_thread = None
        self._last_detections = {}  # Deduplication icin

    @staticmethod
    def list_audio_devices() -> List[Dict]:
        """Kullanilabilir ses giris cihazlarini listeler."""
        if sd is None:
            return []
        devices = sd.query_devices()
        input_devices = []
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                input_devices.append({
                    'index': i,
                    'name': dev['name'],
                    'channels': dev['max_input_channels'],
                    'sample_rate': dev['default_samplerate']
                })
        return input_devices

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice callback - ses verisini queue'ya ekler."""
        if status:
            logger.warning("Audio status: %s", status)
        try:
            self._audio_queue.put_nowait(indata.copy().flatten())
        except queue.Full:
            pass  # Queue doluysa atla, capture'i engelleme

    def _inference_worker(self):
        """Inference thread - queue'dan ses alip tahmin yapar."""
        buffer = np.array([], dtype=np.float32)
        target_len = int(self.sr * self.duration)
        logger.info("Inference worker baslatildi - hedef buffer: %d sample (%.1f sn)",
                     target_len, self.duration)

        while self._running:
            try:
                chunk = self._audio_queue.get(timeout=0.5)
                buffer = np.concatenate([buffer, chunk])

                if len(buffer) >= target_len:
                    audio_segment = buffer[:target_len]
                    buffer = buffer[target_len:]

                    # Ses seviyesi kontrolu
                    rms = np.sqrt(np.mean(audio_segment ** 2))
                    logger.debug("Audio RMS: %.6f, buffer len: %d", rms, len(audio_segment))

                    start_time = time.time()
                    try:
                        result = self.predictor.predict_with_threshold(
                            audio_segment, self.sr)
                    except Exception as e:
                        logger.error("predict_with_threshold hatasi: %s", e, exc_info=True)
                        continue

                    inference_ms = (time.time() - start_time) * 1000

                    # Tum olasiliklar (confidence panel icin)
                    try:
                        _, _, all_probs = self.predictor.predict(
                            audio_segment, self.sr)
                    except Exception as e:
                        logger.error("predict hatasi: %s", e, exc_info=True)
                        continue

                    logger.info("Inference OK (%.1fms) - probs: %s | threshold result: %s",
                                inference_ms,
                                {k: f"{v:.2f}" for k, v in all_probs.items()},
                                result[0] if result else "None (esik alti)")

                    timestamp = datetime.now().strftime("%H:%M:%S")

                    if result is not None:
                        class_name, confidence, is_critical, _ = result

                        # Deduplication kontrolu
                        now = time.time()
                        last = self._last_detections.get(class_name, 0)
                        if now - last < config.DEDUPLICATION_WINDOW_SEC:
                            continue
                        self._last_detections[class_name] = now

                        event = {
                            "class": class_name,
                            "confidence": confidence,
                            "is_critical": is_critical,
                            "all_probs": all_probs,
                            "timestamp": timestamp,
                            "inference_ms": inference_ms,
                            "detected": True
                        }
                    else:
                        event = {
                            "all_probs": all_probs,
                            "timestamp": timestamp,
                            "inference_ms": inference_ms,
                            "detected": False
                        }

                    try:
                        self.gui_queue.put_nowait(event)
                    except queue.Full:
                        logger.warning("GUI queue dolu, event atildi!")

            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Inference hatasi: %s", e, exc_info=True)

    def start(self):
        """Dinlemeyi baslatir (2 thread)."""
        if sd is None:
            logger.error("sounddevice yuklu degil!")
            return False
        if self._running:
            return True

        self._running = True

        # Inference thread
        self._inference_thread = threading.Thread(
            target=self._inference_worker, daemon=True)
        self._inference_thread.start()

        # Audio capture stream
        try:
            self._stream = sd.InputStream(
                samplerate=self.sr,
                channels=1,
                dtype='float32',
                blocksize=int(self.sr * 0.5),  # 500ms blok
                device=self.device_index,
                callback=self._audio_callback
            )
            self._stream.start()
            logger.info("Dinleme baslatildi - SR: %d, Device: %s",
                        self.sr, self.device_index or "varsayilan")
            return True
        except Exception as e:
            logger.error("Mikrofon baslatilamadi: %s", e)
            self._running = False
            return False

    def stop(self):
        """Dinlemeyi durdurur."""
        self._running = False
        if hasattr(self, '_stream') and self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        if self._inference_thread and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=2)
        logger.info("Dinleme durduruldu.")
