"""
HearAlert - Veri Seti Yukleyici
=================================
ESC-50 metadata'sini yukler, hedef siniflari filtreler,
etiket kodlama ve sinif dagilim gorsellestirme saglar.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.logger import setup_logger
from src.config import config

logger = setup_logger(__name__, config.LOGS_DIR / "preprocessing.log")


def load_esc50_metadata(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """ESC-50 resmi metadata CSV dosyasini yukler.

    Args:
        csv_path: esc50.csv dosyasinin yolu. None ise config'den alinir.

    Returns:
        ESC-50 metadata DataFrame'i.
    """
    if csv_path is None:
        csv_path = config.DATASET_DIR / "meta" / "esc50.csv"
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"ESC-50 metadata bulunamadi: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
        logger.info("ESC-50 metadata yuklendi - %d satir, sutunlar: %s",
                     len(df), list(df.columns))
        return df
    except pd.errors.EmptyDataError:
        logger.error("CSV dosyasi bos: %s", csv_path)
        raise


def filter_target_classes(df: pd.DataFrame,
                          classes: Optional[List[str]] = None) -> pd.DataFrame:
    """DataFrame'i hedef ses siniflarina gore filtreler.

    ESC-50'deki sinif adlari bizim adlarimizdan farkli olabilir.
    ESC50_CLASS_MAP kullanarak eslestirme yapar.

    Args:
        df: Metadata DataFrame'i ('category' sutunu gerekli).
        classes: Hedef sinif listesi. None ise config.CLASSES kullanilir.

    Returns:
        Sadece hedef siniflari iceren filtrelenmis DataFrame.
    """
    if classes is None:
        classes = config.CLASSES

    if "category" not in df.columns:
        raise KeyError("'category' sutunu gerekli.")

    # ESC-50 sinif adlarini bul
    esc50_names = list(config.ESC50_CLASS_MAP.keys())
    filtered = df[df["category"].isin(esc50_names)].copy()

    # Sinif adlarini projemizin adlarina cevir
    filtered["original_category"] = filtered["category"]
    filtered["category"] = filtered["category"].map(config.ESC50_CLASS_MAP)

    filtered.reset_index(drop=True, inplace=True)

    found = filtered["category"].unique().tolist()
    missing = set(classes) - set(found)
    if missing:
        logger.warning("Su siniflar veri setinde bulunamadi: %s", missing)

    logger.info("Filtreleme tamamlandi - %d/%d satir, %d sinif",
                len(filtered), len(df), len(found))

    for cls in sorted(found):
        count = len(filtered[filtered["category"] == cls])
        logger.info("  - %s: %d ornek", cls, count)

    return filtered


def encode_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    """Kategori etiketlerini sayisal degerlere donusturur.

    Args:
        df: 'category' sutunlu DataFrame.

    Returns:
        Tuple: (Guncellenmis DataFrame, LabelEncoder)
    """
    le = LabelEncoder()
    df = df.copy()
    df["target"] = le.fit_transform(df["category"])

    logger.info("Etiket kodlama tamamlandi - %d sinif: %s",
                len(le.classes_), list(le.classes_))
    return df, le


def plot_class_distribution(df: pd.DataFrame,
                            save_path: Optional[Path] = None) -> None:
    """Sinif dagilim grafikini cizer ve kaydeder."""
    if save_path is None:
        save_path = config.PLOTS_DIR / "class_distribution.png"

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    class_counts = df["category"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e94560" if c in config.SAFETY_CRITICAL_CLASSES
              else "#00b4d8" for c in class_counts.index]

    bars = ax.bar(class_counts.index, class_counts.values, color=colors,
                  edgecolor="#1a1a2e", linewidth=1.5)

    for bar, val in zip(bars, class_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontweight="bold", fontsize=11,
                color="#eaeaea")

    ax.set_title("HearAlert - Sinif Dagilimi (6 Sinif)", fontsize=16,
                 fontweight="bold", pad=15, color="#eaeaea")
    ax.set_xlabel("Ses Sinifi", fontsize=13, color="#eaeaea")
    ax.set_ylabel("Ornek Sayisi", fontsize=13, color="#eaeaea")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#16213e")
    ax.tick_params(colors="#eaeaea", labelsize=10)
    plt.xticks(rotation=45, ha="right")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e94560", label="Guvenlik-Kritik"),
        Patch(facecolor="#00b4d8", label="Normal")
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              facecolor="#0f3460", edgecolor="#eaeaea", labelcolor="#eaeaea")

    plt.tight_layout()
    try:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        logger.info("Sinif dagilim grafigi kaydedildi: %s", save_path)
    except OSError as e:
        logger.error("Grafik kaydedilemedi: %s", e)
    finally:
        plt.close(fig)


def save_processed_metadata(df: pd.DataFrame,
                            save_path: Optional[Path] = None) -> None:
    """Filtrelenmis metadata'yi CSV olarak kaydeder."""
    if save_path is None:
        save_path = config.PROCESSED_METADATA_DIR / "filtered_meta.csv"

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(save_path, index=False, encoding="utf-8")
        logger.info("Metadata kaydedildi: %s (%d satir)", save_path, len(df))
    except OSError as e:
        logger.error("Metadata kaydedilemedi: %s", e)
        raise
