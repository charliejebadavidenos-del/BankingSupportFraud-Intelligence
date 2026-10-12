
from src.intelligent_ticket_router import route_ticket

test_queries = [
    "I woke up to several card payments I never approved",
    "My wallet was stolen and my bank cards are missing",
    "A person claiming to be from my bank asked for my OTP",
    "My account balance dropped overnight for no apparent reason",
    "I accidentally transferred money to the wrong UPI ID",
    "Can I increase my debit card spending limit?",
    "How do I temporarily block my credit card?",
    "I received a refund but the amount is incorrect",
    "Someone is threatening me to reveal my banking password",
    "Can I withdraw cash from another bank's ATM?"
]

print("\nNEW BANKING QUERY TEST")
print("-" * 65)

for query in test_queries:
    result = route_ticket(query)

    print("\nQuery:", query)
    print("Predicted Intent:", result["predicted_intent"])
    print("Escalation:", result["escalation_decision"])
