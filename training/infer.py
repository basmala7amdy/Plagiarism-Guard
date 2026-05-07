import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

Model_Path = "saved_models/saved_model"

tokenizer = AutoTokenizer.from_pretrained(Model_Path)
model = AutoModelForSequenceClassification.from_pretrained(Model_Path)
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

if __name__ == "__main__":
    result1 = predict(
        "The company launched a new AI model.",
        "A new artificial intelligence model was released by the company."
    )

    result2 = predict(
        "The cat is sleeping on the sofa.",
        "Quantum computing uses qubits to perform calculations."
    )
    print(result1)
    print(result2)