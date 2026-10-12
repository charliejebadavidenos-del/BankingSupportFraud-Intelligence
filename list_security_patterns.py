
from src.fraud_signals import FRAUD_PATTERNS, REVIEW_PATTERNS

print("\nEXISTING FRAUD PATTERNS")
print("-" * 60)

for number, pattern in enumerate(FRAUD_PATTERNS, start=1):
    print(f"{number}. {pattern}")

print("\nEXISTING REVIEW PATTERNS")
print("-" * 60)

for number, pattern in enumerate(REVIEW_PATTERNS, start=1):
    print(f"{number}. {pattern}")
