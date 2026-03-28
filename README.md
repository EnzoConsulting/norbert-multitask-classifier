# NorBERT Multitask Classifier

A multitask fine-tune of `NbAiLab/nb-bert-base` that classifies Norwegian 
customer service messages across three dimensions in a single model pass.

## Background

Customer service teams handling Norwegian-language messages need to triage 
incoming requests quickly — routing by urgency, understanding tone, and 
categorizing the issue type. Most approaches train separate models for each 
task. This project handles all three with one fine-tuned BERT model, reducing 
latency and infrastructure overhead.

## What it classifies

| Task      | Labels                                                                 |
|-----------|------------------------------------------------------------------------|
| Sentiment | `negativ`, `nøytral`, `positiv`                                        |
| Priority  | `lav`, `normal`, `høy`                                                 |
| Category  | `leveringsproblem`, `feil vare`, `fakturaspørsmål`, `generell henvendelse`, `reklamasjon` |

## Approach

Training data was generated synthetically using structured LLM prompts. This 
was a deliberate choice — real customer service data carries privacy 
constraints, and synthetic data allowed rapid iteration on label balance and 
edge cases. The trade-off is reduced real-world noise; a production deployment 
would benefit from fine-tuning on actual labelled messages.

## Example output
```
Input:  Jeg har blitt trukket dobbelt på kortet. Kan dere rydde opp?
Output: kategori=fakturaspørsmål  prioritet=høy  sentiment=negativ
```

## Project structure
```
norbert-multitask-classifier/
├── data/
│   └── norbert_synthetic_sample.jsonl
├── examples/
│   └── sample_predictions.txt
├── src/
│   ├── model.py          # BERT with 3 classification heads
│   ├── prepare_data.py   # Tokenization pipeline
│   ├── train_all.py      # Multitask fine-tuning
│   └── predict_all.py    # CLI prediction
├── requirements.txt
└── README.md
```

## Usage
```bash
pip install -r requirements.txt
python src/prepare_data.py   # tokenize dataset
python src/train_all.py      # fine-tune model (~600MB output)
python src/predict_all.py    # run predictions
```

Note: trained weights are not included. Run `train_all.py` to generate, 
or save your own with `model.bert.save_pretrained()`.

## License

MIT
