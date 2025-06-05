import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

MODEL_NAME = "NbAiLab/nb-bert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

# Load CSV
from datasets import load_dataset

dataset = load_dataset("json", data_files="norbert_synthetic.jsonl", split="train")


# Tokenize + map
dataset = dataset.map(tokenize)

# Save processed dataset
dataset.save_to_disk("processed_dataset")
