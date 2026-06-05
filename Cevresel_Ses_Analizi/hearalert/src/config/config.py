"""
HearAlert - Merkezi Konfigurasyon Modulu
==========================================
Tum proje genelinde kullanilan sabitler, yollar, model parametreleri
ve sinif tanimlari burada merkezi olarak yonetilir.

Referanslar:
    - Salamon & Bello (2016): SB-CNN mimarisi ve augmentation parametreleri
    - IoT Elderly Care paper: Guvenlik-kritik sinif esik degerleri
"""

from pathlib import Path

# -- AUDIO -------------------------------------------------------------------
SAMPLE_RATE = 22050
DURATION = 3.0
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
N_MFCC = 40
FEATURE_DIM = 80  # N_MFCC * 2 (mean + std)

# -- MLP MODEL ---------------------------------------------------------------
MLP_LAYERS = [256, 128, 64]
MLP_DROPOUT = [0.3, 0.3, 0.2]
MLP_L2 = 0.0001
BATCH_SIZE = 64
EPOCHS = 150
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 20
LR_REDUCE_PATIENCE = 10
LR_REDUCE_FACTOR = 0.5

# -- CNN / SB-CNN (Salamon & Bello 2016) -------------------------------------
CNN_FILTERS = [24, 48, 48]
CNN_KERNEL_SIZE = (5, 5)
CNN_POOLING = [(4, 2), (4, 2)]
CNN_DENSE_UNITS = 64
CNN_DROPOUT = 0.5
CNN_L2 = 0.001
CNN_LR = 0.01
CNN_BATCH_SIZE = 100
CNN_EPOCHS = 50

# -- INFERENCE ----------------------------------------------------------------
DEFAULT_CONFIDENCE_THRESHOLD = 0.75
CRITICAL_CONFIDENCE_THRESHOLD = 0.65
DEDUPLICATION_WINDOW_SEC = 5.0
MIC_DEVICE_INDEX = None

# -- SOUND CLASSES (5 sinif) -------------------------------------------------
CLASSES = [
    "alarm_clock", "crying_baby",
    "glass_breaking", "dog_bark", "siren"
]
NUM_CLASSES = 5
SAFETY_CRITICAL_CLASSES = ["alarm_clock", "crying_baby", "glass_breaking", "siren"]

# -- ESC-50 sinif adlari (ESC-50 CSV'deki karsiliklar) -----------------------
ESC50_CLASS_MAP = {
    "clock_alarm": "alarm_clock",
    "crying_baby": "crying_baby",
    "glass_breaking": "glass_breaking",
    "dog": "dog_bark",
    "siren": "siren",
}

# -- CLASS-CONDITIONAL AUGMENTATION MAP (paper Fig. 3) ------------------------
CLASS_AUGMENTATION_MAP = {
    "alarm_clock":    ["ts", "ps1", "ps2", "drc", "bg"],
    "crying_baby":    ["ts", "ps1", "ps2", "drc", "bg"],
    "glass_breaking": ["ts", "ps1", "ps2",        "bg"],
    "dog_bark":       ["ts", "ps1", "ps2", "drc", "bg"],
    "siren":          ["ts", "ps1", "ps2",        "bg"],
}

# -- AUGMENTATION PARAMS (Salamon & Bello 2016, Section II-B) -----------------
TIME_STRETCH_FACTORS = [0.81, 0.93, 1.07, 1.23]
PITCH_SHIFT_PS1 = [-2, -1, 1, 2]
PITCH_SHIFT_PS2 = [-3.5, -2.5, 2.5, 3.5]
BG_NOISE_WEIGHT_RANGE = (0.1, 0.5)

# -- GUI ----------------------------------------------------------------------
FLASH_COUNT = 5
FLASH_DURATION_MS = 120
CARD_DISPLAY_DURATION_MS = 6000
MAX_LOG_ENTRIES = 200
GUI_POLL_INTERVAL_MS = 100

# -- PATHS --------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset" / "ESC-50"
PROCESSED_DIR = PROJECT_ROOT / "processed"
PROCESSED_AUDIO_DIR = PROCESSED_DIR / "audio_clips"
PROCESSED_FEATURES_DIR = PROCESSED_DIR / "features"
PROCESSED_MFCC_DIR = PROCESSED_FEATURES_DIR / "mfcc"
PROCESSED_LOGMEL_DIR = PROCESSED_FEATURES_DIR / "logmel"
PROCESSED_METADATA_DIR = PROCESSED_DIR / "metadata"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_MLP_DIR = MODELS_DIR / "mlp"
MODELS_CNN_DIR = MODELS_DIR / "cnn"
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
REPORTS_DIR = OUTPUTS_DIR / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

# -- REPRODUCIBILITY ----------------------------------------------------------
RANDOM_SEED = 42
