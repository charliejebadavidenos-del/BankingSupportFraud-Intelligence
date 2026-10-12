
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

model_path = (
    PROJECT_ROOT / "models" / "intent_classifier.joblib"
)

model = joblib.load(model_path)

test_queries = [
    "Someone transferred money from my account without permission",
    "I cannot log into my mobile banking app",
    "My home loan application is still pending",
    "My KYC verification was rejected",
    "My UPI payment failed but money was deducted",
    "I want to update my registered mobile number",
    "My fixed deposit maturity amount is incorrect"
]

print("\nSAVED ML MODEL PREDICTIONS")
print("-" * 60)

for query in test_queries:
    prediction = model.predict([query])[0]

    print("\nCustomer Query:", query)
    print("Predicted Intent:", prediction)
