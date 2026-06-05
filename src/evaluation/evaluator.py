"""
HearAlert - Model Evaluator
==============================
Confusion matrix, classification report, safety-critical recall analizi.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_recall_fscore_support)
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.logger import setup_logger
from src.config import config

logger = setup_logger(__name__, config.LOGS_DIR / "evaluation.log")


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   class_names: list) -> Dict:
    """Modeli test seti uzerinde degerlendirir.

    Args:
        model: Egitilmis Keras modeli.
        X_test: Test feature matrisi.
        y_test: Test etiketleri.
        class_names: Sinif adlari listesi.

    Returns:
        Degerlendirme metrikleri sozlugu.
    """
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=range(len(class_names))
    )

    metrics = {
        "accuracy": acc,
        "per_class": {}
    }
    for i, name in enumerate(class_names):
        metrics["per_class"][name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i])
        }

    logger.info("Test accuracy: %.4f", acc)
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: list, normalize: bool = True,
                          save_path: Optional[Path] = None) -> None:
    """Confusion matrix cizer ve kaydeder."""
    if save_path is None:
        save_path = config.PLOTS_DIR / "confusion_matrix.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#16213e")

    sns.heatmap(cm, annot=True, fmt=fmt, cmap="YlOrRd",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5, linecolor="#1a1a2e")

    ax.set_title("Confusion Matrix (Normalized)", color="#eaeaea",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted", color="#eaeaea", fontsize=12)
    ax.set_ylabel("True", color="#eaeaea", fontsize=12)
    ax.tick_params(colors="#eaeaea")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Confusion matrix kaydedildi: %s", save_path)


def safety_critical_report(metrics: Dict) -> None:
    """Guvenlik-kritik siniflarin recall degerlerini raporlar."""
    logger.info("=== GUVENLIK-KRITIK SINIF RAPORU ===")

    for cls in config.SAFETY_CRITICAL_CLASSES:
        if cls in metrics["per_class"]:
            recall = metrics["per_class"][cls]["recall"]
            status = "OK" if recall >= 0.80 else "UYARI"
            logger.info("  [%s] %s: recall=%.4f", status, cls, recall)
            if recall < 0.80:
                logger.warning("  !!! %s recall < 0.80 - Guvenlik riski!", cls)
        else:
            logger.warning("  %s sinifi degerlendirme metriklerinde yok!", cls)

    logger.info("====================================")
