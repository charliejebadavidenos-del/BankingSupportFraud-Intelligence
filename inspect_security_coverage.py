
from src.fraud_signals import (
    detect_fraud_signals,
    detect_review_signals
)

queries = [
    "An online purchase appeared on my card although I was asleep",
    "A caller pretending to be bank staff wants my internet banking password",
    "I forgot my internet banking password",
    "My debit card payment was declined",
    "Someone is asking me to disclose my OTP",
    "I noticed a payment that I cannot explain",
    "My card was stolen during my commute",
    "How can I change my ATM withdrawal limit?"
]

print("\nSECURITY RULE COVERAGE ANALYSIS")
print("-" * 65)

for query in queries:
    fraud = detect_fraud_signals(query)
    review = detect_review_signals(query)

    print("\nQuery:", query)
    print("Fraud signal:", fraud)
    print("Review signal:", review)
    print("Matched fraud patterns:", fraud["matched_patterns"])
    print("Matched review patterns:", review["matched_patterns"])
