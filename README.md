# 🇳🇴 NorBERT Multitask Classifier

This project fine-tunes the Norwegian BERT model [`NbAiLab/nb-bert-base`](https://huggingface.co/NbAiLab/nb-bert-base) to classify **Norwegian customer service messages** across three key dimensions — all in one model pass:

* 🎭 **Sentiment**: `negativ`, `nøytral`, `positiv`
* 🔥 **Priority**: `lav`, `normal`, `høy`
* 📌 **Category**:
  `leveringsproblem`, `feil vare`, `fakturaspørsmål`, `generell henvendelse`, `reklamasjon`

---

## 🧠 Highlights

* ✅ Multitask learning (3 outputs from 1 model)
* ✅ Supports Norwegian customer support use cases
* ✅ Fully synthetic training data generated using structured LLM prompts
* ✅ Lightweight interface (CLI, JSONL format)
* ✅ Fast to train, accurate in production

---

## 📦 Project Structure

```
norbert-multitask-classifier/
├── data/
│   └── norbert_synthetic_sample.jsonl        # Sample synthetic dataset
├── examples/
│   └── sample_predictions.txt                # Example output from prediction script
├── src/
│   ├── model.py                              # NorBERT with 3 classification heads
│   ├── prepare_data.py                       # Loads + tokenizes JSONL dataset
│   ├── train_all.py                          # Multitask fine-tuning pipeline
│   └── predict_all.py                        # CLI-based prediction on sample inputs
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### 🧪 Step 1: Prepare the Dataset

```bash
python src/prepare_data.py
```

This tokenizes the `data/norbert_synthetic_sample.jsonl` dataset and stores a Hugging Face-compatible version in `processed_dataset/`.

---

### 🧠 Step 2: Train the Model

```bash
python src/train_all.py
```

This will:

* Load the processed dataset
* Train a multitask BERT model
* Save weights as `norbert-multitask.pt`

---

### 🔍 Step 3: Run Predictions

```bash
python src/predict_all.py
```

Example output:

```
🟢 Tekst: Jeg har blitt trukket dobbelt på kortet. Kan dere rydde opp?
  📌 Kategori : fakturaspørsmål
  🔥 Prioritet: høy
  🎭 Sentiment: negativ
```

---

## ❗ Model File

Trained model weights (`norbert-multitask.pt`, \~600MB) are **not included in this repo**.
You can regenerate them by running `train_all.py`, or save your own copy with:

```python
model.bert.save_pretrained("norbert-multitask/")
tokenizer.save_pretrained("norbert-multitask/")
```

---

## 📌 Label Definitions

### Category

| Code | Label                |
| ---- | -------------------- |
| 0    | leveringsproblem     |
| 1    | feil vare            |
| 2    | fakturaspørsmål      |
| 3    | generell henvendelse |
| 4    | reklamasjon          |

### Priority

| Code | Label  |
| ---- | ------ |
| 0    | lav    |
| 1    | normal |
| 2    | høy    |

### Sentiment

| Code | Label   |
| ---- | ------- |
| 0    | negativ |
| 1    | nøytral |
| 2    | positiv |

---

## 📜 License

MIT – Free to use, modify, and share. Credit appreciated.
