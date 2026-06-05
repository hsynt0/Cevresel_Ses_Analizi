"""
HearAlert - SB-CNN Model (Salamon & Bello 2016, Section II-A)
================================================================
Exact reproduction:
    Input: (128, 128, 1)
    Conv2D(24, (5,5), valid, relu) -> MaxPool2D((4,2), stride (4,2))
    Conv2D(48, (5,5), valid, relu) -> MaxPool2D((4,2), stride (4,2))
    Conv2D(48, (5,5), valid, relu)
    Flatten()
    Dropout(0.5) -> Dense(64, relu, L2=0.001)
    Dropout(0.5) -> Dense(6, softmax, L2=0.001)
    Optimizer: SGD(lr=0.01, constant)
"""

import numpy as np
from pathlib import Path
from typing import List, Optional
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.logger import setup_logger
from src.config import config

logger = setup_logger(__name__, config.LOGS_DIR / "training.log")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_sbcnn_model(input_shape: tuple = (128, 128, 1),
                      num_classes: int = config.NUM_CLASSES,
                      l2_lambda: float = config.CNN_L2,
                      dropout_rate: float = config.CNN_DROPOUT
                      ) -> keras.Model:
    """SB-CNN modelini Salamon & Bello (2016) Section II-A'ya gore olusturur.

    Mimari:
        - 3 Conv2D katmani (24, 48, 48 filtre), 5x5 kernel, valid padding
        - 2 MaxPool2D katmani (4x2 stride)
        - 2 Dropout(0.5) + Dense(64) + Dense(num_classes)
        - L2 regularizasyon sadece dense katmanlarda

    Args:
        input_shape: Giris boyutu (128, 128, 1).
        num_classes: Sinif sayisi.
        l2_lambda: L2 regularizasyon katsayisi.
        dropout_rate: Dropout orani.

    Returns:
        Derlenmemis Keras modeli.
    """
    model = keras.Sequential(name="SB_CNN")

    # Conv Block 1: Conv2D(24, 5x5) + MaxPool(4x2)
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(
        24, (5, 5), padding="valid", activation="relu", name="conv1"
    ))
    model.add(layers.MaxPool2D(
        pool_size=(4, 2), strides=(4, 2), name="pool1"
    ))

    # Conv Block 2: Conv2D(48, 5x5) + MaxPool(4x2)
    model.add(layers.Conv2D(
        48, (5, 5), padding="valid", activation="relu", name="conv2"
    ))
    model.add(layers.MaxPool2D(
        pool_size=(4, 2), strides=(4, 2), name="pool2"
    ))

    # Conv Block 3: Conv2D(48, 5x5) - no pooling
    model.add(layers.Conv2D(
        48, (5, 5), padding="valid", activation="relu", name="conv3"
    ))

    # Flatten + Dense
    model.add(layers.Flatten())
    model.add(layers.Dropout(dropout_rate, name="dropout1"))
    model.add(layers.Dense(
        64, activation="relu",
        kernel_regularizer=regularizers.l2(l2_lambda),
        name="dense1"
    ))
    model.add(layers.Dropout(dropout_rate, name="dropout2"))
    model.add(layers.Dense(
        num_classes, activation="softmax",
        kernel_regularizer=regularizers.l2(l2_lambda),
        name="output"
    ))

    logger.info("SB-CNN modeli olusturuldu - %d parametre", model.count_params())
    return model


def compile_cnn(model: keras.Model,
                learning_rate: float = config.CNN_LR) -> keras.Model:
    """CNN'i SGD optimizer ile derler (paper'daki gibi constant lr)."""
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    logger.info("SB-CNN derlendi - SGD LR: %f", learning_rate)
    return model


def get_cnn_callbacks(model_save_path: Optional[Path] = None
                      ) -> List[keras.callbacks.Callback]:
    """CNN egitim callback'lerini olusturur."""
    if model_save_path is None:
        model_save_path = config.MODELS_CNN_DIR / "best_model.keras"
    model_save_path = Path(model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            str(model_save_path), monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
        keras.callbacks.CSVLogger(
            str(config.LOGS_DIR / "cnn_training_log.csv")
        )
    ]
    return callbacks


def train_cnn(model: keras.Model,
              X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = config.CNN_EPOCHS,
              batch_size: int = config.CNN_BATCH_SIZE,
              callbacks: list = None) -> keras.callbacks.History:
    """SB-CNN modelini egitir."""
    logger.info("SB-CNN egitimi basliyor - %d epoch, batch=%d", epochs, batch_size)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    logger.info("SB-CNN egitimi tamamlandi - Son val_accuracy: %.4f",
                history.history["val_accuracy"][-1])
    return history


def plot_cnn_training_history(history: keras.callbacks.History,
                              save_path: Optional[Path] = None) -> None:
    """CNN egitim gecmisini cizip kaydeder."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if save_path is None:
        save_path = config.PLOTS_DIR / "cnn_training_history.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#16213e")

    ax1.plot(history.history["accuracy"], label="Train", color="#00b4d8", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Validation", color="#e94560", linewidth=2)
    ax1.set_title("SB-CNN Accuracy", color="#eaeaea", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch", color="#eaeaea")
    ax1.set_ylabel("Accuracy", color="#eaeaea")
    ax1.legend(facecolor="#0f3460", edgecolor="#eaeaea", labelcolor="#eaeaea")
    ax1.set_facecolor("#1a1a2e")
    ax1.tick_params(colors="#eaeaea")
    ax1.grid(True, alpha=0.2)

    ax2.plot(history.history["loss"], label="Train", color="#00b4d8", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Validation", color="#e94560", linewidth=2)
    ax2.set_title("SB-CNN Loss", color="#eaeaea", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch", color="#eaeaea")
    ax2.set_ylabel("Loss", color="#eaeaea")
    ax2.legend(facecolor="#0f3460", edgecolor="#eaeaea", labelcolor="#eaeaea")
    ax2.set_facecolor("#1a1a2e")
    ax2.tick_params(colors="#eaeaea")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("CNN egitim grafigi kaydedildi: %s", save_path)
