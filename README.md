# 🏦 Smart Mailroom: Customer Support Intent Classification (NLP & Deep Learning)

An end-to-end NLP and Deep Learning system built using **TensorFlow and Keras** that automatically classifies incoming banking customer support inquiries across **77 fine-grained intent categories**.

Powered by a lightweight, bidirectional recurrent architecture (**BiLSTM**) with custom learned embeddings trained strictly from scratch, this system delivers **85.26% Macro-F1 accuracy** with sub-400ms CPU inference latency and a compact **4.5 MB** disk footprint.

---

## 📊 Measured Benchmark Results (Official Test Set: 3,080 Queries)

Evaluated under reproducible CPU conditions on the official untouched **PolyAI Banking77 test set**:

| Evaluation Metric | Measured Result |
| :--- | :---: |
| **Test Accuracy** | **85.26%** (2,626 / 3,080 correct) |
| **Macro-F1 (Primary Metric)** | **0.8526** |
| **Weighted-F1** | **0.8526** |
| **Macro Precision** | **0.8584** |
| **Macro Recall** | **0.8526** |
| **CPU Inference Latency (Median)** | **~395 ms / query** |
| **Total Trainable Parameters** | **391,757** |
| **Model Weight / Disk Footprint** | **4.53 MB** |
| **Target Classes** | **77 distinct intent categories** |

---

## 🏗️ Architecture Design

```text
Incoming Customer Query (Raw Text)
                 │
                 ▼
Keras TextVectorization Layer (Vocabulary: 2,211 tokens, fit strictly on train split)
                 │
                 ▼
Padded Token Sequence (Shape: batch_size x 45)
                 │
                 ▼
Trainable Embedding Layer (Input dim: 2,211 -> Output dim: 128, mask_zero=True)
                 │
                 ▼
Bidirectional LSTM Layer (Forward: 64 units + Backward: 64 units = 128 dimensions)
                 │
                 ▼
Dropout Regularization (Rate = 0.30)
                 │
                 ▼
Dense Softmax Classifier (77 output units)
                 │
                 ▼
Calibrated Probabilities & Top Intent Class (e.g., 'card_arrival', 'pending_transfer')
```

---

## 📁 Repository Structure

```text
Customer_ticket/
├── app.py                      # Interactive Streamlit Web Application
├── requirements.txt            # Pure TensorFlow/Keras requirements
├── README.md                   # Complete documentation and benchmark report
├── data/
│   ├── train.csv               # 8,002 stratified training queries (80%)
│   ├── val.csv                 # 2,001 stratified validation queries (20%)
│   ├── test.csv                # 3,080 untouched test queries
│   ├── categories.json         # 77 intent category strings
│   ├── label2id.json           # Label to integer ID mapping
│   ├── id2label.json           # Integer ID to label mapping
│   └── eda_length_distribution.png
├── models/
│   ├── bilstm_model.keras      # Saved trained BiLSTM model (4.53 MB)
│   ├── bilstm_vocab.json       # Train-set vocabulary (2,211 tokens)
│   ├── bilstm_val_metrics.json # Validation split metrics
│   └── evaluation_results.json # Comprehensive test evaluation metrics & error analysis
├── notebooks/
│   └── 01_eda.ipynb            # Exploratory Data Analysis notebook with executed outputs
└── src/
    ├── data_utils.py           # Ingestion, validation, and stratified splitting
    ├── train_bilstm.py         # BiLSTM training pipeline
    └── evaluate.py             # Rigorous test set evaluation and error breakdown
```

---

## 🚀 Quickstart Guide

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Download Data & Prepare Splits
```bash
python src/data_utils.py
```

### 3. Train the Model from Scratch
```bash
python src/train_bilstm.py
```

### 4. Run Rigorous Test Set Evaluation
```bash
python src/evaluate.py
```

### 5. Launch the Interactive Web Application
```bash
streamlit run app.py
```

---

## 📜 Dataset & Licensing

Dataset: **Banking77** released by PolyAI under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.
