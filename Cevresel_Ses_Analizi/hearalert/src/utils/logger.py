"""
HearAlert — Merkezi Loglama Modülü
====================================

Tüm modüller için tutarlı loglama altyapısı sağlar.
Üretim kodunda asla print() kullanılmaz — her zaman logger kullanılır.

Kullanım:
    from src.utils.logger import setup_logger
    logger = setup_logger(__name__)
    logger.info("Model yüklendi.")
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Hem dosyaya hem konsola yazan bir logger oluşturur.

    Args:
        name: Logger adı (genellikle __name__ kullanılır).
        log_file: Log dosyası yolu. None ise sadece konsola yazar.
        level: Minimum log seviyesi (varsayılan: INFO).

    Returns:
        Yapılandırılmış logging.Logger nesnesi.

    Raises:
        OSError: Log dosyası dizini oluşturulamazsa.
    """
    logger = logging.getLogger(name)

    # Aynı logger'a tekrar handler eklenmesini önle
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Log formatı — zaman | modül | seviye | mesaj
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ── Konsol Handler ─────────────────────────────────────────────────
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # ── Dosya Handler ──────────────────────────────────────────────────
    if log_file is not None:
        try:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                log_file, mode="a", encoding="utf-8"
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError as e:
            logger.warning(
                "Log dosyası oluşturulamadı: %s — Sadece konsola yazılacak.", e
            )

    return logger
