# 🏥 Intelligent Clinical Early Warning System

### *Can AI Save Lives? Building a Deep Learning Pipeline for Sepsis Detection*


> **Author:** Muhammad Sohaib · Roll No: 22F-BSAI-40
>
> **Course:** Deep Learning — Assignment 01
>
> **Submitted to:** Engr. Hamza Farooqui
>
> **Platform:** Google Colab (GPU)

## Overview

This project builds a **multi-generation deep learning pipeline** to predict patient deterioration risk in ICU settings — specifically **sepsis onset** — using two complementary data modalities:

- **Time-series vital signs** (hourly ICU readings)
- **Free-text clinical notes** (discharge summaries, nursing notes)

The pipeline is structured across three progressively advanced generations, each building on the last, and culminates in a full NLP-based clinical note classifier using **ClinicalBERT**.

> In healthcare AI, **Recall (Sensitivity) is the primary metric**. A missed sepsis case (False Negative) can cost a patient's life. A false alarm (False Positive) causes unnecessary but clinically safe review.

---

## 📂 Project Structure

```
Intelligent-Clinical-Early-Warning-System/
│
├── Dataset.csv                        # PhysioNet Sepsis Challenge 2019 (raw)
├── NOTEEVENTS.csv                     # MIMIC-III clinical notes (50k rows)
├── cleaned_sepsis_data.csv            # After preprocessing (post-imputation)
├── balanced_sepsis_data.csv           # After class balancing (1:1 ratio)
│
├── sepsis-early-warning-system.ipynb  # Main Colab notebook
│
├── models/
│   ├── gen1_dnn_adam.keras            # DNN (Adam optimizer)
│   ├── gen1_dnn_sgd.keras             # DNN (SGD optimizer)
│   ├── gen2_lstm.pt                   # LSTM model weights
│   ├── gen2_gru.pt                    # GRU model weights
│   ├── gen2_bilstm.pt                 # Bidirectional LSTM weights
│   ├── gen3_clinicalbert_frozen.pt    # ClinicalBERT (frozen base)
│   └── gen3_clinicalbert_full_finetune.pt  # ClinicalBERT (full fine-tune)
│
└── results/
    ├── fig_1_0_overview.png
    ├── optimizer_comparison.png
    ├── fig_2_0_dataset_overview.png
    ├── fig_2_1_sample_sequences.png
    ├── fig_2_7_loss_curves.png
    ├── fig_2_7_comparison_overlay.png
    ├── fig_2_8_confusion_matrices.png
    ├── fig_2_8_roc_curves.png
    ├── fig_2_8_benchmarks.png
    ├── fig_2_9_radar_comparison.png
    ├── gen3_training_comparison.png
    ├── gen3_attention_heatmap.png
    ├── gen3_confusion_matrices.png
    ├── gen3_perclass_breakdown.png
    └── confusion_metrics_of_all_models.png
```

---

## 🗃️ Dataset

### PhysioNet Sepsis Challenge 2019

Used in **Generations 1 & 2** for time-series vital sign modelling.

| Property                | Value                              |
| ----------------------- | ---------------------------------- |
| Total rows              | 1,552,210                          |
| Unique patients         | 40,336                             |
| Features                | 40 clinical features + SepsisLabel |
| Sepsis prevalence (raw) | ~1.80%                             |
| After balancing         | 50% / 50% (27,916 each class)      |
| Window length (Gen 2)   | 12 hours sliding window            |

**Key preprocessing steps:**

1. Dropped columns with >95% missing values (13 columns removed)
2. Patient-level forward-fill → backward-fill → global median imputation
3. Downsampled majority class to 1:1 ratio
4. StandardScaler normalization (fit on train only)
5. Patient-wise stratified 70/15/15 train/val/test split (prevents data leakage)

### MIMIC-III NOTEEVENTS

Used in **Generation 3** for clinical NLP.

| Property                  | Value                                             |
| ------------------------- | ------------------------------------------------- |
| Notes loaded              | 50,000                                            |
| Categories used           | Discharge summary, Nursing, Physician             |
| Sample size (fine-tuning) | 3,000                                             |
| Label method              | Keyword heuristic (sepsis, ICU, intubation, etc.) |
| Class balance             | ~47% high-risk / 52% stable                       |

---

## 🧬 Pipeline Architecture

### Generation 1 — Baseline DNN

A shallow feedforward neural network to establish baseline performance.

**Architecture:** `Input(28) → Dense(256, ReLU) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(1, Sigmoid)`

**Optimizer Comparison:**

| Optimizer         | Accuracy | Precision | Recall | F1-Score | Train Time |
| ----------------- | -------- | --------- | ------ | -------- | ---------- |
| **Adam** ✅ | 0.7037   | 0.7275    | 0.6513 | 0.6873   | 10.65s     |
| SGD               | 0.6653   | 0.6916    | 0.5965 | 0.6405   | 4.07s      |

**Winner:** Adam — superior across all metrics including the clinically critical Recall.

---

### Generation 2 — Temporal Sequence Models (LSTM / GRU / BiLSTM)

Treats ICU records as patient timelines using 12-hour sliding windows over 14 vital + lab features.

**Architectures:**

| Model            | Key Property                            | Parameters | Real-Time Safe? |
| ---------------- | --------------------------------------- | ---------- | --------------- |
| **LSTM**   | Causal (left→right), long-range memory | 214,401    | ✅ Yes          |
| **GRU**    | Lighter gated unit, faster training     | 162,945    | ✅ Yes          |
| **BiLSTM** | Reads sequence in both directions       | 559,745    | ❌ No           |

All models use:

- `BCEWithLogitsLoss` with `pos_weight` for class imbalance
- Orthogonal weight initialization for recurrent layers
- AdamW optimizer + ReduceLROnPlateau scheduler
- Early stopping (patience = 7)
- Decision threshold = **0.35** (lowered to maximize recall)

**Test Set Results (threshold = 0.35):**

| Model            | Accuracy | Precision | Recall ⬆        | F1-Score | AUC              | Train Time |
| ---------------- | -------- | --------- | ---------------- | -------- | ---------------- | ---------- |
| LSTM             | 0.6022   | 0.9723    | 0.6906           | 0.7348   | 0.7050           | 2.5 min    |
| **GRU** ✅ | 0.3276   | 0.9764    | **0.9092** | 0.4720   | **0.7330** | 2.4 min    |
| BiLSTM           | 0.3346   | 0.9757    | 0.8952           | 0.4803   | 0.7256           | 4.2 min    |

**Clinical Deployment Recommendation:**

- **Real-time EWS:** Use GRU (highest recall, fastest) or LSTM (better balanced performance)
- **Retrospective audit:** BiLSTM (uses full sequence context for best AUC)

---

### Generation 3 — Clinical NLP with ClinicalBERT

Fine-tunes [`emilyalsentzer/Bio_ClinicalBERT`](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) on MIMIC-III clinical notes to classify patient risk from unstructured text.

**Model:** `BERT(768) → Dropout(0.3) → Linear(768→256) → GELU → Dropout(0.1) → Linear(256→2)`

**Two Fine-Tuning Strategies:**

| Strategy                                  | Trainable Params | Train Time | Val Accuracy     | Val Loss         |
| ----------------------------------------- | ---------------- | ---------- | ---------------- | ---------------- |
| Strategy 1 — Frozen Base                 | 197,378          | 52.1s      | 0.6266           | 0.6563           |
| **Strategy 2 — Full Fine-Tune** ✅ | 108,507,650      | 97.6s      | **0.6449** | **0.6457** |

**Test Set Results:**

| Strategy                    | Accuracy         | Precision        | Recall           | F1-Score         |
| --------------------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| Frozen Base                 | 0.6044           | 0.6034           | 0.6044           | 0.5942           |
| **Full Fine-Tune** ✅ | **0.6178** | **0.6167** | **0.6178** | **0.6169** |

**Explainability:** Attention heatmaps visualize which clinical terms (e.g., "CARDIOTHORACIC", "trach", "COPD") drive the model's prediction — critical for physician trust.

---

## 📊 Unified Model Comparison

| Model                  | Accuracy | Precision | Recall | F1-Score | Train Time |
| ---------------------- | -------- | --------- | ------ | -------- | ---------- |
| DNN (Baseline)         | 0.7037   | 0.7059    | 0.7037 | 0.7029   | 10.65s     |
| LSTM                   | 0.6022   | 0.9723    | 0.6022 | 0.7348   | 148.8s     |
| GRU                    | 0.3276   | 0.9764    | 0.3276 | 0.4720   | 141.5s     |
| Bi-LSTM                | 0.3346   | 0.9757    | 0.3346 | 0.4803   | 250.1s     |
| ClinicalBERT (Frozen)  | 0.6044   | 0.6034    | 0.6044 | 0.5942   | 52.1s      |
| ClinicalBERT (Full FT) | 0.6178   | 0.6167    | 0.6178 | 0.6169   | 97.6s      |

---

## ⚙️ Setup & Usage

### Requirements

```bash
pip install torch torchvision torchaudio
pip install tensorflow
pip install transformers datasets
pip install scikit-learn pandas numpy matplotlib seaborn tqdm
```

### Run on Google Colab

1. Mount your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Place `Dataset.csv` and `NOTEEVENTS.csv` in:

```
/content/drive/MyDrive/Intelligent Clinical Early Warning System/
```

3. Run all cells sequentially. The notebook is self-contained — it handles preprocessing, training, evaluation, and saving.

### GPU Requirement

A CUDA-capable GPU is strongly recommended. The notebook auto-detects:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## 🔑 Key Design Decisions

**Why lower the decision threshold to 0.35?**
In a life-critical system, a False Negative (missed sepsis) is far more dangerous than a False Positive (unnecessary review). Lowering the threshold from 0.5 to 0.35 maximizes Recall at the cost of some Precision — a deliberate and clinically justified trade-off.

**Why patient-wise data splits?**
Random row-level splits allow data from the same patient to appear in both train and test sets, artificially inflating performance. Patient-wise splits prevent this leakage and give honest generalization estimates.

**Why ClinicalBERT instead of general BERT?**
ClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`) is pre-trained on MIMIC-III clinical notes, giving it native understanding of medical abbreviations, clinical shorthand, and ICU-specific language.

---

## ⚠️ Clinical Disclaimer

> **This model is a decision support tool, not a decision-maker.**
>
> In all deployments, a qualified physician must review every alert generated by the system. The model provides a probabilistic second opinion based on patterns in historical data. It does not replace clinical judgment, and its predictions should always be interpreted in the full clinical context of the patient.

---

## 📚 References

- Reyna, M. et al. (2019). *Early Prediction of Sepsis from Clinical Data — PhysioNet/CinC Challenge 2019.* PhysioNet.
- Johnson, A. et al. (2016). *MIMIC-III, a freely accessible critical care database.* Scientific Data.
- Alsentzer, E. et al. (2019). *Publicly Available Clinical BERT Embeddings.* NAACL Clinical NLP Workshop.
- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation.
- Cho, K. et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder.* EMNLP.

---

## 👤 Author

**Muhammad Sohaib**
BS Artificial Intelligence — Roll No: 22F-BSAI-40
Deep Learning Course, Assignment 01
