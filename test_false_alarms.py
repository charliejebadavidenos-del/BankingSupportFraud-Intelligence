
from src.intelligent_ticket_router import route_ticket

test_queries = [
    "How do I replace my expired debit card?",
    "What is the daily ATM withdrawal limit?",
    "My debit card payment was declined",
    "Can I authorize my spouse to operate my account?",
    "I want to change my credit card PIN",
    "How can I enable international card transactions?",
    "My friend transferred money to my account",
    "I need to reset my mobile banking password",
    "What should I do if I lose my debit card?",
    "How do I report a suspicious transaction?"
]

print("\nFALSE ALARM SCENARIO TEST")
print("-" * 65)

for query in test_queries:
    result = route_ticket(query)

    print("\nQuery:", query)
    print("Predicted Intent:", result["predicted_intent"])
    print("Escalation:", result["escalation_decision"])
