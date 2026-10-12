
from src.fraud_signals import (
    detect_fraud_signals,
    detect_review_signals
)


def evaluate_escalation(query):

    fraud_result = detect_fraud_signals(query)
    review_result = detect_review_signals(query)

    if fraud_result["fraud_signal_detected"]:
        decision = "HIGH_PRIORITY_FRAUD_REVIEW"

    elif review_result["review_required"]:
        decision = "MANUAL_REVIEW"

    else:
        decision = "NORMAL_PROCESSING"

    return {
        "query": query,
        "decision": decision,
        "fraud_signal": fraud_result["fraud_signal_detected"],
        "review_signal": review_result["review_required"]
    }


if __name__ == "__main__":

    test_queries = [
        "Someone used my credit card online without my knowledge",
        "I found a duplicate charge of ₹999 on my statement",
        "My UPI payment failed but money was debited"
    ]

    for query in test_queries:
        result = evaluate_escalation(query)

        print("\nCustomer Query:", result["query"])
        print("Decision:", result["decision"])
        print("Fraud Signal:", result["fraud_signal"])
        print("Review Signal:", result["review_signal"])
