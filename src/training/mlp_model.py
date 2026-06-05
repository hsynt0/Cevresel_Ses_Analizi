"""
HearAlert - MLP Model
=======================
Input(80,) -> Dense(256,relu) -> BN -> Dropout(0.3)
           -> Dense(128,relu) -> BN -> Dropout(0.3)
           -> Dense(64,relu) -> Dropout(0.2)
           -> Dense(6, softmax)

~90K parametre, CPU'da anlik inference.
Referans: Salamon & Bello (2016) MLP baseline.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.logger import setup_logger
from src.config import config

logger = setup_logger(__name__, config.LOGS_DIR / "training.log")

# TensorFlow import
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_mlp_model(input_dim: int = config.FEATURE_DIM,
                    num_classes: int = config.NUM_CLASSES,
                    layer_sizes: list = None,
                    dropout_rates: list = None,
                    l2_reg: float = config.MLP_L2) -> keras.Model:
    """MLP modelini olusturur.

    Args:
        input_dim: Giris boyutu (varsayilan: 80).
        num_classes: Sinif sayisi (varsayilan: 6).
        layer_sizes: Katman boyutlari.
        dropout_rates: Dropout oranlari.
        l2_reg: L2 regularizasyon katsayisi.

    Returns:
        Derlenmemis Keras modeli.
    """
    if layer_sizes is None:
        layer_sizes = config.MLP_LAYERS
    if dropout_rates is None:
        dropout_rates = config.MLP_DROPOUT

    model = keras.Sequential(name="HearAlert_MLP")
    model.add(layers.Input(shape=(input_dim,)))

    for i, (units, drop) in enumerate(zip(layer_sizes, dropout_rates)):
        model.add(layers.Dense(
            units, activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"dense_{i}"
        ))
        if i < len(layer_sizes) - 1:
            model.add(layers.BatchNormalization(name=f"bn_{i}"))
        model.add(layers.Dropout(drop, name=f"dropout_{i}"))

    model.add(layers.Dense(num_classes, activation="softmax", name="output"))

    logger.info("MLP modeli olusturuldu - %d parametre", model.count_params())
    return model


def compile_model(model: keras.Model,
                  learning_rate: float = config.LEARNING_RATE) -> keras.Model:
    """Modeli derler."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    logger.info("Model derlendi - LR: %f", learning_rate)
    return model


def get_callbacks(model_save_path: Optional[Path] = None,
                  patience_es: int = config.EARLY_STOPPING_PATIENCE,
                  patience_lr: int = config.LR_REDUCE_PATIENCE
                  ) -> List[keras.callbacks.Callback]:
    """Egitim callback'lerini olusturur."""
    if model_save_path is None:
        model_save_path = config.MODELS_MLP_DIR / "best_model.keras"

    model_save_path = Path(model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience_es,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=config.LR_REDUCE_FACTOR,
            patience=patience_lr, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            str(model_save_path), monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
        keras.callbacks.CSVLogger(
            str(config.LOGS_DIR / "mlp_training_log.csv")
        )
    ]
    logger.info("Callback'ler hazir - EarlyStopping(%d), LRReduce(%d)",
                patience_es, patience_lr)
    return callbacks


def train_model(model: keras.Model,
                X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                epochs: int = config.EPOCHS,
                batch_size: int = config.BATCH_SIZE,
                callbacks: list = None) -> keras.callbacks.History:
    """Modeli egitir.

    Args:
        model: Derlenmis Keras modeli.
        X_train, y_train: Egitim verisi.
        X_val, y_val: Dogrulama verisi.
        epochs: Maksimum epoch sayisi.
        batch_size: Batch boyutu.
        callbacks: Callback listesi.

    Returns:
        Egitim gecmisi.
    """
    logger.info("MLP egitimi basliyor - %d epoch, batch=%d", epochs, batch_size)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    logger.info("Egitim tamamlandi - Son val_accuracy: %.4f",
                history.history["val_accuracy"][-1])
    return history


def plot_training_history(history: keras.callbacks.History,
                          save_path: Optional[Path] = None) -> None:
    """Egitim gecmisini cizip kaydeder."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if save_path is None:
        save_path = config.PLOTS_DIR / "mlp_training_history.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#16213e")

    # Accuracy
    ax1.plot(history.history["accuracy"], label="Train", color="#00b4d8", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Validation", color="#e94560", linewidth=2)
    ax1.set_title("Model Accuracy", color="#eaeaea", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch", color="#eaeaea")
    ax1.set_ylabel("Accuracy", color="#eaeaea")
    ax1.legend(facecolor="#0f3460", edgecolor="#eaeaea", labelcolor="#eaeaea")
    ax1.set_facecolor("#1a1a2e")
    ax1.tick_params(colors="#eaeaea")
    ax1.grid(True, alpha=0.2)

    # Loss
    ax2.plot(history.history["loss"], label="Train", color="#00b4d8", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Validation", color="#e94560", linewidth=2)
    ax2.set_title("Model Loss", color="#eaeaea", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch", color="#eaeaea")
    ax2.set_ylabel("Loss", color="#eaeaea")
    ax2.legend(facecolor="#0f3460", edgecolor="#eaeaea", labelcolor="#eaeaea")
    ax2.set_facecolor("#1a1a2e")
    ax2.tick_params(colors="#eaeaea")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Egitim grafigi kaydedildi: %s", save_path)
