"""
HearAlert - Ana Pipeline Scripti
===================================
Kullanim:
    python main.py prepare                    # ESC-50 filtrele ve on isle
    python main.py augment                    # Data augmentation
    python main.py extract --feature mfcc     # MFCC feature cikar
    python main.py extract --feature logmel   # Log-mel spectrogram cikar
    python main.py train --model mlp          # MLP egit
    python main.py train --model cnn          # SB-CNN egit
    python main.py evaluate --model mlp       # MLP degerlendir
    python main.py evaluate --model cnn       # CNN degerlendir
    python main.py all                        # Hepsini sirayla calistir
    python main.py run                        # GUI baslat
"""

import sys
import os
import argparse
import numpy as np
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.config import config
from src.utils.logger import setup_logger

logger = setup_logger("main", config.LOGS_DIR / "pipeline.log")


def step_prepare():
    """ADIM 1: ESC-50 veri setini filtrele ve on isle."""
    from src.preprocessing.dataset_loader import (
        load_esc50_metadata, filter_target_classes,
        encode_labels, plot_class_distribution, save_processed_metadata
    )
    from src.preprocessing.audio_processor import process_dataset

    logger.info("=" * 60)
    logger.info("[ADIM 1] ESC-50 veri seti hazirlaniyor...")
    logger.info("=" * 60)

    df = load_esc50_metadata()
    df = filter_target_classes(df)
    df, le = encode_labels(df)

    config.MODELS_MLP_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_CNN_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(le, config.MODELS_MLP_DIR / "label_encoder.joblib")
    joblib.dump(le, config.MODELS_CNN_DIR / "label_encoder.joblib")
    logger.info("Label encoder kaydedildi.")

    plot_class_distribution(df)
    save_processed_metadata(df)

    audio_dir = config.DATASET_DIR / "audio"
    df_processed = process_dataset(
        df=df, audio_dir=audio_dir,
        output_dir=config.PROCESSED_AUDIO_DIR,
        sr=config.SAMPLE_RATE, duration=config.DURATION
    )
    save_processed_metadata(df_processed,
                            config.PROCESSED_METADATA_DIR / "processed_meta.csv")

    success = df_processed["processed"].sum()
    logger.info("[OK] Hazirlama tamamlandi: %d/%d dosya", success, len(df_processed))
    return df_processed


def step_augment():
    """ADIM 2: Data augmentation (Salamon & Bello 2016)."""
    import pandas as pd
    from src.augmentation.augmentor import AudioAugmentor
    from src.preprocessing.dataset_loader import save_processed_metadata

    logger.info("=" * 60)
    logger.info("[ADIM 2] Data augmentation basliyor...")
    logger.info("=" * 60)

    meta_path = config.PROCESSED_METADATA_DIR / "processed_meta.csv"
    if not meta_path.exists():
        logger.error("processed_meta.csv bulunamadi! Once 'prepare' calistirin.")
        return None

    df = pd.read_csv(meta_path)
    augmentor = AudioAugmentor()

    aug_dir = config.PROCESSED_DIR / "augmented"
    df_combined = augmentor.augment_dataset(df, config.PROCESSED_AUDIO_DIR, aug_dir)

    save_processed_metadata(df_combined,
                            config.PROCESSED_METADATA_DIR / "augmented_meta.csv")
    logger.info("[OK] Augmentation tamamlandi: %d toplam ornek", len(df_combined))
    return df_combined


def step_extract(feature_type: str = "both"):
    """ADIM 3: Feature cikarimi."""
    import pandas as pd
    from src.features.mfcc_extractor import batch_extract_mfcc

    logger.info("=" * 60)
    logger.info("[ADIM 3] Feature cikarimi - %s...", feature_type)
    logger.info("=" * 60)

    # Augmente veri varsa onu kullan, yoksa orijinal
    aug_meta = config.PROCESSED_METADATA_DIR / "augmented_meta.csv"
    orig_meta = config.PROCESSED_METADATA_DIR / "processed_meta.csv"
    meta_path = aug_meta if aug_meta.exists() else orig_meta

    if not meta_path.exists():
        logger.error("Metadata bulunamadi! Once 'prepare' calistirin.")
        return

    df = pd.read_csv(meta_path)
    df_ok = df[df["processed"] == True].copy()
    file_paths = df_ok["processed_path"].tolist()
    y = df_ok["target"].values
    folds = df_ok["fold"].values

    if feature_type in ("mfcc", "both"):
        X_mfcc, failed = batch_extract_mfcc(file_paths)
        if failed:
            mask = np.ones(len(X_mfcc), dtype=bool)
            mask[failed] = False
            X_mfcc = X_mfcc[mask]
            y_mfcc = y[mask]
            folds_mfcc = folds[mask]
        else:
            y_mfcc = y
            folds_mfcc = folds
        config.PROCESSED_MFCC_DIR.mkdir(parents=True, exist_ok=True)
        np.save(config.PROCESSED_MFCC_DIR / "X_mfcc.npy", X_mfcc)
        np.save(config.PROCESSED_MFCC_DIR / "y_mfcc.npy", y_mfcc)
        np.save(config.PROCESSED_MFCC_DIR / "folds.npy", folds_mfcc)
        logger.info("[OK] MFCC: X=%s", X_mfcc.shape)

    if feature_type in ("logmel", "both"):
        from src.features.logmel_extractor import batch_extract_logmel
        X_logmel, failed = batch_extract_logmel(file_paths)
        if failed:
            mask = np.ones(len(X_logmel), dtype=bool)
            mask[failed] = False
            X_logmel = X_logmel[mask]
            y_logmel = y[mask]
            folds_logmel = folds[mask]
        else:
            y_logmel = y
            folds_logmel = folds
        config.PROCESSED_LOGMEL_DIR.mkdir(parents=True, exist_ok=True)
        np.save(config.PROCESSED_LOGMEL_DIR / "X_logmel.npy", X_logmel)
        np.save(config.PROCESSED_LOGMEL_DIR / "y_logmel.npy", y_logmel)
        np.save(config.PROCESSED_LOGMEL_DIR / "folds.npy", folds_logmel)
        logger.info("[OK] LogMel: X=%s", X_logmel.shape)


def step_train(model_type: str = "mlp"):
    """ADIM 4: Model egitimi."""
    from src.training.splitter import fold_based_split, normalize_features, save_scaler

    logger.info("=" * 60)
    logger.info("[ADIM 4] %s modeli egitimi...", model_type.upper())
    logger.info("=" * 60)

    if model_type == "mlp":
        from src.training.mlp_model import (
            build_mlp_model, compile_model, get_callbacks,
            train_model, plot_training_history
        )
        X = np.load(config.PROCESSED_MFCC_DIR / "X_mfcc.npy")
        y = np.load(config.PROCESSED_MFCC_DIR / "y_mfcc.npy")
        folds = np.load(config.PROCESSED_MFCC_DIR / "folds.npy")

        X_train, X_val, X_test, y_train, y_val, y_test = fold_based_split(X, y, folds)
        X_train, X_val, X_test, scaler = normalize_features(X_train, X_val, X_test)
        save_scaler(scaler)

        np.save(config.PROCESSED_MFCC_DIR / "X_test.npy", X_test)
        np.save(config.PROCESSED_MFCC_DIR / "y_test.npy", y_test)

        model = build_mlp_model()
        model = compile_model(model)
        model.summary()
        callbacks = get_callbacks()
        history = train_model(model, X_train, y_train, X_val, y_val, callbacks=callbacks)
        plot_training_history(history)

    elif model_type == "cnn":
        from src.training.cnn_model import (
            build_sbcnn_model, compile_cnn, get_cnn_callbacks,
            train_cnn, plot_cnn_training_history
        )
        X = np.load(config.PROCESSED_LOGMEL_DIR / "X_logmel.npy")
        y = np.load(config.PROCESSED_LOGMEL_DIR / "y_logmel.npy")
        folds = np.load(config.PROCESSED_LOGMEL_DIR / "folds.npy")

        X_train, X_val, X_test, y_train, y_val, y_test = fold_based_split(X, y, folds)

        # CNN icin per-feature normalizasyon (mean/std per mel band)
        mean = X_train.mean(axis=0, keepdims=True)
        std = X_train.std(axis=0, keepdims=True) + 1e-8
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        X_test = (X_test - mean) / std

        # Normalizasyon parametrelerini kaydet
        config.MODELS_CNN_DIR.mkdir(parents=True, exist_ok=True)
        np.save(config.MODELS_CNN_DIR / "norm_mean.npy", mean)
        np.save(config.MODELS_CNN_DIR / "norm_std.npy", std)

        np.save(config.PROCESSED_LOGMEL_DIR / "X_test.npy", X_test)
        np.save(config.PROCESSED_LOGMEL_DIR / "y_test.npy", y_test)

        model = build_sbcnn_model()
        model = compile_cnn(model)
        model.summary()
        callbacks = get_cnn_callbacks()
        history = train_cnn(model, X_train, y_train, X_val, y_val, callbacks=callbacks)
        plot_cnn_training_history(history)

    logger.info("[OK] %s egitimi tamamlandi!", model_type.upper())


def step_evaluate(model_type: str = "mlp"):
    """ADIM 5: Model degerlendirme."""
    from src.evaluation.evaluator import (
        evaluate_model, plot_confusion_matrix, safety_critical_report
    )
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from tensorflow import keras
    from sklearn.metrics import classification_report

    logger.info("=" * 60)
    logger.info("[ADIM 5] %s degerlendirme...", model_type.upper())
    logger.info("=" * 60)

    le = joblib.load(config.MODELS_MLP_DIR / "label_encoder.joblib")
    class_names = list(le.classes_)

    if model_type == "mlp":
        model_path = config.MODELS_MLP_DIR / "best_model.keras"
        X_test = np.load(config.PROCESSED_MFCC_DIR / "X_test.npy")
        y_test = np.load(config.PROCESSED_MFCC_DIR / "y_test.npy")
        suffix = "_mlp"
    elif model_type == "cnn":
        model_path = config.MODELS_CNN_DIR / "best_model.keras"
        X_test = np.load(config.PROCESSED_LOGMEL_DIR / "X_test.npy")
        y_test = np.load(config.PROCESSED_LOGMEL_DIR / "y_test.npy")
        suffix = "_cnn"

    if not model_path.exists():
        logger.error("Model bulunamadi: %s", model_path)
        return

    model = keras.models.load_model(str(model_path))
    metrics = evaluate_model(model, X_test, y_test, class_names)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    plot_confusion_matrix(y_test, y_pred, class_names,
                          save_path=config.PLOTS_DIR / f"confusion_matrix{suffix}.png")

    report = classification_report(y_test, y_pred, target_names=class_names)
    logger.info("\n%s Classification Report:\n%s", model_type.upper(), report)

    safety_critical_report(metrics)
    logger.info("[OK] %s degerlendirme tamamlandi!", model_type.upper())
    return metrics


def step_run():
    """ADIM 6: GUI baslatir."""
    logger.info("=" * 60)
    logger.info("[ADIM 6] HearAlert GUI baslatiliyor...")
    logger.info("=" * 60)
    from gui.app import launch_gui
    launch_gui()


def main():
    parser = argparse.ArgumentParser(description="HearAlert Pipeline")
    parser.add_argument("step", choices=["prepare", "augment", "extract",
                                          "train", "evaluate", "all", "run"],
                        help="Calistirilacak adim")
    parser.add_argument("--model", choices=["mlp", "cnn"], default="mlp",
                        help="Model tipi (train/evaluate icin)")
    parser.add_argument("--feature", choices=["mfcc", "logmel", "both"],
                        default="both", help="Feature tipi (extract icin)")

    args = parser.parse_args()

    if args.step == "prepare" or args.step == "all":
        step_prepare()
    if args.step == "augment" or args.step == "all":
        step_augment()
    if args.step == "extract" or args.step == "all":
        step_extract(args.feature if args.step != "all" else "both")
    if args.step == "train":
        step_train(args.model)
    if args.step == "evaluate":
        step_evaluate(args.model)
    if args.step == "all":
        step_train("mlp")
        step_train("cnn")
        step_evaluate("mlp")
        step_evaluate("cnn")
    if args.step == "run":
        step_run()


if __name__ == "__main__":
    main()

