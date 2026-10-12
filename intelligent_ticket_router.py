
import joblib
from pathlib import Path

from src.fraud_escalation import evaluate_escalation

PROJECT_ROOT = Path(__file__).resolve().parent.parent

model_path = (
    PROJECT_ROOT / "models" / "intent_classifier.joblib"
)

model = joblib.load(model_path)


def route_ticket(query):

    predicted_intent = model.predict([query])[0]

    escalation = evaluate_escalation(query)

    decision = escalation["decision"]

    # ML fraud predictions must also trigger review.
    if predicted_intent == "Fraud/Unauthorized":
        if decision == "NORMAL_PROCESSING":
            decision = "MANUAL_REVIEW"

    return {
        "query": query,
        "predicted_intent": predicted_intent,
        "escalation_decision": decision
    }


if __name__ == "__main__":

    test_queries = [
        "Someone transferred money from my account without permission",
        "I found a duplicate charge on my statement",
        "My UPI payment failed but money was deducted",
        "I cannot access my online banking account"
    ]

    print("\nINTELLIGENT BANKING TICKET ROUTER")
    print("-" * 60)

    for query in test_queries:
        result = route_ticket(query)

        print("\nCustomer Query:", result["query"])
        print("Predicted Intent:", result["predicted_intent"])
        print("Escalation Decision:", result["escalation_decision"])
