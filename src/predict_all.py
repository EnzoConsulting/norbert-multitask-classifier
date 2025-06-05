import torch
from transformers import AutoTokenizer
from model import MultiHeadNorBERT

# === CONFIG ===
MODEL_NAME = "NbAiLab/nb-bert-base"
MODEL_PATH = "norbert-multitask.pt"

LABELS = {
    "sentiment": {0: "negativ", 1: "nøytral", 2: "positiv"},
    "priority": {0: "lav", 1: "normal", 2: "høy"},
    "category": {
        0: "leveringsproblem",
        1: "feil vare",
        2: "fakturaspørsmål",
        3: "generell henvendelse",
        4: "reklamasjon"
    }
}

# === LOAD MODEL ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = MultiHeadNorBERT(MODEL_NAME, {
    "sentiment": 3,
    "priority": 3,
    "category": 5
})
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# === PREDICTION FUNCTION ===
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

    sentiment = torch.argmax(outputs["sentiment"], dim=1).item()
    priority = torch.argmax(outputs["priority"], dim=1).item()
    category = torch.argmax(outputs["category"], dim=1).item()

    return {
        "text": text,
        "sentiment": LABELS["sentiment"][sentiment],
        "priority": LABELS["priority"][priority],
        "category": LABELS["category"][category]
    }

# === SAMPLE MESSAGES ===
if __name__ == "__main__":
    messages = [
        "Hei, jeg har fortsatt ikke mottatt pakken min. Hva skjer?",
        "Produktet jeg fikk var feil størrelse. Hvordan bytter jeg det?",
        "Fakturaen jeg fikk stemmer ikke med bestillingen.",
        "Hei! Når åpner butikken deres i morgen?",
        "Skuffet over at varen sluttet å fungere etter bare én uke.",
        "Takk for god hjelp i chatten i går, veldig fornøyd!",
        "Hei, jeg vurderer å bestille men lurer på om dere har varen på lager.",
        "Jeg har blitt trukket dobbelt på kortet. Kan dere rydde opp?",
        "Tusen takk for rask levering! Alt var i orden.",
        "Det er gått tre uker og jeg har ikke fått pakken. Dette er ikke greit!"
    ]

    for msg in messages:
        result = predict(msg)
        print(f"\n🟢 Tekst: {result['text']}")
        print(f"  📌 Kategori : {result['category']}")
        print(f"  🔥 Prioritet: {result['priority']}")
        print(f"  🎭 Sentiment: {result['sentiment']}")
