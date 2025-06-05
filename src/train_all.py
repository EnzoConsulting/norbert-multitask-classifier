import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from transformers import AutoTokenizer
from datasets import load_from_disk
from model import MultiHeadNorBERT
from tqdm import tqdm

# === Config ===
MODEL_NAME = "NbAiLab/nb-bert-base"
NUM_EPOCHS = 4
BATCH_SIZE = 16
LEARNING_RATE = 5e-5

# === Load dataset ===
dataset = load_from_disk("processed_dataset")
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# === DataLoader wrapper ===
def collate_fn(batch):
    input_ids = torch.tensor([item["input_ids"] for item in batch])
    attention_mask = torch.tensor([item["attention_mask"] for item in batch])
    sentiment = torch.tensor([item["sentiment"] for item in batch])
    priority = torch.tensor([item["priority"] for item in batch])
    category = torch.tensor([item["category"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": {
            "sentiment": sentiment,
            "priority": priority,
            "category": category
        }
    }

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# === Model ===
num_labels = {
    "sentiment": 3,
    "priority": 3,
    "category": 5
}
model = MultiHeadNorBERT(MODEL_NAME, num_labels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Optimizer & Scheduler ===
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
lr_scheduler = get_scheduler("linear", optimizer=optimizer,
                             num_warmup_steps=0,
                             num_training_steps=len(train_loader) * NUM_EPOCHS)

# === Loss Function ===
loss_fn = torch.nn.CrossEntropyLoss()

# === Training Loop ===
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = {k: v.to(device) for k, v in batch["labels"].items()}

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = (
            loss_fn(outputs["sentiment"], labels["sentiment"]) +
            loss_fn(outputs["priority"], labels["priority"]) +
            loss_fn(outputs["category"], labels["category"])
        )

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} average loss: {total_loss / len(train_loader):.4f}")

# === Save model ===
torch.save(model.state_dict(), "norbert-multitask.pt")
print("✅ Model saved as norbert-multitask.pt")
