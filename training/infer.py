from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "saved_models" / "saved_model"

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
model.eval()

def predict(text1, text2, max_length=128):
    text1 = str(text1).strip()
    text2 = str(text2).strip()

    inputs = tokenizer(
        text1,
        text2,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)[0]
        pred = int(torch.argmax(probs).item())

    return {
        "label": pred,
        "prediction": "plagiarism" if pred == 1 else "not_plagiarism",
        "confidence": float(probs[pred]),
        "probabilities": {
            "not_plagiarism": float(probs[0]),
            "plagiarism": float(probs[1])
        }
    }