
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

model_path = (
    PROJECT_ROOT / "models" / "intent_classifier.joblib"
)

model = joblib.load(model_path)

query = "What is the daily ATM withdrawal limit?"

prediction = model.predict([query])[0]
scores = model.decision_function([query])[0]

print("\nINTENT CLASSIFICATION ANALYSIS")
print("-" * 55)

print("Customer Query:", query)
print("Predicted Intent:", prediction)

print("\nCATEGORY DECISION SCORES")

ranked_scores = sorted(
    zip(model.classes_, scores),
    key=lambda item: item[1],
    reverse=True
)

for category, score in ranked_scores:
    print(f"{category:30s} {score:.4f}")
