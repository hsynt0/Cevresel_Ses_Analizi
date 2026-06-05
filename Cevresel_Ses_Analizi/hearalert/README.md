# HearAlert - Real-Time Sound Event Recognition System

**Isitme engelli bireyler icin gercek zamanli cevresel ses olay tanima sistemi.**

## Arastirma Referanslari

1. **Salamon & Bello (2017)** - *"Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification"* (IEEE Signal Processing Letters)
2. **IoT for Elderly Care** - *"Design and Development of Sound Event Recognition System for Hearing Impaired People"*

## Hedef Ses Siniflari (5 Sinif)

| Sinif | ESC-50 Karsiligi | Guvenlik-Kritik |
|-------|------------------|-----------------|
| alarm_clock | clock_alarm | **Evet** |
| crying_baby | crying_baby | **Evet** |
| glass_breaking | glass_breaking | **Evet** |
| dog_bark | dog | Hayir |
| siren | siren | **Evet** |

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanim

```bash
cd hearalert

# Tum pipeline'i calistir (prepare -> augment -> extract -> train -> evaluate)
python main.py all

# GUI'yi baslat (gercek zamanli dinleme)
python main.py run

# Tek tek adimlar:
python main.py prepare                    # ESC-50 filtrele ve on isle
python main.py augment                    # Data augmentation (Salamon & Bello 2017)
python main.py extract --feature mfcc     # MFCC feature cikar
python main.py extract --feature logmel   # Log-mel spectrogram cikar
python main.py train --model mlp          # MLP egit
python main.py train --model cnn          # SB-CNN egit
python main.py evaluate --model mlp       # MLP degerlendir
python main.py evaluate --model cnn       # CNN degerlendir
```

## Mimari

### MLP Model (~64K parametre)
```
Input(80) -> Dense(256,relu) -> BN -> Dropout(0.3)
          -> Dense(128,relu) -> BN -> Dropout(0.3)
          -> Dense(64,relu) -> Dropout(0.2)
          -> Dense(5, softmax)
```

### SB-CNN Model (~241K parametre, Salamon & Bello 2017)
```
Input(128, 128, 1)
-> Conv2D(24, 5x5, valid, relu) -> MaxPool2D(4x2)
-> Conv2D(48, 5x5, valid, relu) -> MaxPool2D(4x2)
-> Conv2D(48, 5x5, valid, relu)
-> Flatten -> Dropout(0.5) -> Dense(64, relu, L2=0.001)
-> Dropout(0.5) -> Dense(5, softmax, L2=0.001)
Optimizer: SGD(lr=0.01)
```

## Data Augmentation (Salamon & Bello 2017, Section II-B)

| Yontem | Parametreler |
|--------|-------------|
| Time Stretching | 0.81, 0.93, 1.07, 1.23 |
| Pitch Shift PS1 | -2, -1, +1, +2 semitone |
| Pitch Shift PS2 | -3.5, -2.5, +2.5, +3.5 semitone |
| DRC | 4 parametre seti (surekli sesler haric) |
| Background Noise | w in U(0.1, 0.5) |

## GUI

- **CustomTkinter** dark mode arayuzu (1100x700)
- Gercek zamanli mikrofon dinleme (2 thread: capture + inference)
- Guvenlik-kritik tespitlerde **kirmizi flas uyari**
- Canli confidence barlari (5 sinif)
- Detection log tablosu (maks 200 girdi)
- Model secimi (MLP / CNN) ve mikrofon secimi

## Proje Yapisi

```
hearalert/
├── dataset/ESC-50/              # ESC-50 veri seti
├── processed/
│   ├── audio_clips/             # 3sn mono 22050Hz WAV
│   ├── augmented/               # Augmente edilmis dosyalar
│   ├── features/
│   │   ├── mfcc/                # X_mfcc.npy, y_mfcc.npy
│   │   └── logmel/              # X_logmel.npy, y_logmel.npy
│   └── metadata/                # CSV metadata dosyalari
├── models/
│   ├── mlp/                     # MLP model + scaler + label_encoder
│   └── cnn/                     # SB-CNN model + norm params
├── outputs/plots/               # Grafikler
├── src/
│   ├── config/config.py         # Merkezi konfigurasyon
│   ├── utils/logger.py          # Loglama altyapisi
│   ├── preprocessing/           # dataset_loader, audio_processor
│   ├── augmentation/            # augmentor (Salamon & Bello 2017)
│   ├── features/                # mfcc_extractor, logmel_extractor
│   ├── training/                # splitter, mlp_model, cnn_model
│   ├── evaluation/              # evaluator
│   └── inference/               # predictor, realtime_listener
├── gui/
│   ├── styles/theme.py          # Renk paleti, fontlar
│   ├── components/
│   │   ├── flash_overlay.py     # Gorsel flas uyari
│   │   ├── alert_card.py        # Bildirim kartlari
│   │   ├── confidence_panel.py  # Canli guven cubugu
│   │   └── log_table.py         # Tespit gecmisi
│   └── app.py                   # Ana uygulama penceresi
├── main.py                      # CLI pipeline + GUI baslat
├── requirements.txt
└── README.md
```

## Model Sonuclari (Data Leakage Giderilmis Gercek Degerler)

| Model | Test Accuracy | Veri Bölme (Fold Tabanlı) |
|-------|--------------|------|
| MLP | **%79.74** | Sızıntısız: Folds 1-3 Train, Fold 4 Val, Fold 5 Test |
| SB-CNN | **%77.16** | Sızıntısız: Folds 1-3 Train, Fold 4 Val, Fold 5 Test |

## Guvenlik-Kritik Sinif Recall

| Sinif | MLP | CNN | Hedef |
|-------|-----|-----|-------|
| alarm_clock | 0.9868 | 0.9934 | >= 0.80 |
| crying_baby | 0.6250 | 0.8816 | >= 0.80 |
| glass_breaking | 0.8250 | 0.4000 | >= 0.80 |
| siren | 0.7583 | 0.7083 | >= 0.80 |

## Sinif Bazli Detayli Performans (Classification Report)

### MLP
| Sinif | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| alarm_clock | 0.82 | 0.99 | 0.89 | 152 |
| crying_baby | 0.88 | 0.62 | 0.73 | 152 |
| dog_bark | 0.78 | 0.79 | 0.78 | 152 |
| glass_breaking | 0.95 | 0.82 | 0.88 | 120 |
| siren | 0.62 | 0.76 | 0.68 | 120 |

### SB-CNN
| Sinif | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| alarm_clock | 0.94 | 0.99 | 0.96 | 152 |
| crying_baby | 0.86 | 0.88 | 0.87 | 152 |
| dog_bark | 0.57 | 0.78 | 0.66 | 152 |
| glass_breaking | 0.72 | 0.40 | 0.51 | 120 |
| siren | 0.82 | 0.71 | 0.76 | 120 |

## Future Improvements

1. **Attention Mechanism**: CNN katmanlarinin uzerine sinif-spesifik attention eklenebilir (spectrogram'in hangi bolgesinin onemli oldugunu ogrenme).
2. **Transfer Learning**: VGGish veya YAMNet gibi onceden egitilmis modellerden feature extraction ve fine-tuning.
3. **Multi-Scale CNN**: Farkli kernel boyutlari (3x3, 5x5, 7x7) ile paralel convolution (Inception-benzeri).
4. **Online Learning**: Kullanici geri bildirimiyle modelin zamanla iyilesmesi.
5. **Edge Deployment**: TFLite/ONNX donusumu ile Raspberry Pi veya mobil cihazda calisma.
6. **AudioSet Entegrasyonu**: Google AudioSet'ten ek veri indirerek sinif basina 1000+ ornek.
7. **Noise Robustness**: Gercek ortam gurultusune karsi direnc testleri ve gurultuye ozel augmentation.
8. **Multi-label Classification**: Birden fazla sesin ayni anda algilanmasi.

## Lisans

Bu proje akademik amaclarla gelistirilmistir.
