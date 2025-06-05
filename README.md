# NorBERT Multitask Classifier 🇳🇴

This project fine-tunes the Norwegian BERT model `NbAiLab/nb-bert-base` to classify **customer service messages** in Norwegian.

It predicts **three labels at once**:
- 🎭 Sentiment (negativ, nøytral, positiv)
- 🔥 Priority (lav, normal, høy)
- 📌 Category (leveringsproblem, feil vare, fakturaspørsmål, generell henvendelse, reklamasjon)

---

## 🛠️ How it works

- Trained on a synthetic dataset generated via LLM prompts (Grok API)
- Multitask architecture with 3 classifier heads
- Input: plain text message  
- Output: structured labels

---

## 📁 Project Structure

- `src/model.py` – BERT with 3 parallel output heads  
- `src/train_all.py` – multitask fine-tuning loop  
- `src/prepare_data.py` – loads and tokenizes `JSONL` format data  
- `src/predict_all.py` – demo script with pre-written inputs  
- `data/` – includes `norbert_synthetic_sample.jsonl` (subset)

---

📦 Model

Weights (norbert-multitask.pt) not included due to size (600MB).
You can train your own using the provided dataset and code.

---

## 🚀 Getting Started

```bash
pip install -r requirements.txt

# Prepare data
python src/prepare_data.py

# Train model
python src/train_all.py

# Run prediction demo
python src/predict_all.py
