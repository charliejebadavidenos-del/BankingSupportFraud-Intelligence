
import pandas as pd
from pathlib import Path

from src.intelligent_ticket_router import route_ticket

PROJECT_ROOT = Path(__file__).resolve().parent.parent

dataset_path = (
    PROJECT_ROOT / "data" / "tickets"
    / "02_improved_support_tickets.csv"
)

df = pd.read_csv(dataset_path)

total_fraud = 0
fraud_escalated = 0
missed_fraud = []
non_fraud_escalated = 0

for _, row in df.iterrows():

    query = row["query_text"]
    actual_category = row["category"]

    result = route_ticket(query)
    decision = result["escalation_decision"]

    escalated = decision in [
        "HIGH_PRIORITY_FRAUD_REVIEW",
        "MANUAL_REVIEW"
    ]

    if actual_category == "Fraud/Unauthorized":
        total_fraud += 1

        if escalated:
            fraud_escalated += 1
        else:
            missed_fraud.append(query)

    elif escalated:
        non_fraud_escalated += 1

print("\nCOMBINED ROUTER EVALUATION")
print("-" * 55)

print("Total Records:", len(df))
print("Total Fraud Queries:", total_fraud)
print("Fraud Queries Escalated:", fraud_escalated)
print("Fraud Queries Missed:", len(missed_fraud))
print("Non-Fraud Queries Escalated:", non_fraud_escalated)

print(
    "Fraud Escalation Recall:",
    f"{fraud_escalated / total_fraud:.2%}"
)

print("\nMISSED FRAUD QUERIES")

for query in missed_fraud:
    print("-", query)
