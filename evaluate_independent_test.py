
import pandas as pd
from pathlib import Path
from src.intelligent_ticket_router import route_ticket

project_root = Path(__file__).resolve().parent.parent
test_file = project_root / "data" / "tickets" / "03_independent_test.csv"

df = pd.read_csv(test_file)

intent_correct = 0
escalation_correct = 0

print("\nINDEPENDENT TEST EVALUATION")
print("-" * 65)

for _, row in df.iterrows():
    result = route_ticket(row["query_text"])

    predicted_intent = result["predicted_intent"]
    predicted_escalation = result["escalation_decision"]

    intent_match = predicted_intent == row["expected_intent"]
    escalation_match = predicted_escalation == row["expected_escalation"]

    intent_correct += int(intent_match)
    escalation_correct += int(escalation_match)

    print("\nQuery:", row["query_text"])
    print("Expected Intent:", row["expected_intent"])
    print("Predicted Intent:", predicted_intent)
    print("Intent Match:", intent_match)
    print("Expected Escalation:", row["expected_escalation"])
    print("Predicted Escalation:", predicted_escalation)
    print("Escalation Match:", escalation_match)

print("\nFINAL RESULTS")
print("-" * 65)
print("Total Queries:", len(df))
print(f"Intent Accuracy: {intent_correct / len(df):.2%}")
print(f"Escalation Agreement: {escalation_correct / len(df):.2%}")



results = []

for _, row in df.iterrows():
    result = route_ticket(row["query_text"])

    results.append({
        "query_text": row["query_text"],
        "expected_intent": row["expected_intent"],
        "predicted_intent": result["predicted_intent"],
        "expected_escalation": row["expected_escalation"],
        "predicted_escalation": result["escalation_decision"]
    })

output_path = (
    project_root / "data" / "tickets"
    / "04_independent_test_results.csv"
)

pd.DataFrame(results).to_csv(output_path, index=False)

print("\nEvaluation results saved to:", output_path)
