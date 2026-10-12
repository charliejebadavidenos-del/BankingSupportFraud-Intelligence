
from src.intelligent_ticket_router import route_ticket

queries = [
    "The bank asked me to reset my password",
    "I requested a new ATM PIN",
    "My bank wants me to update my contact details",
    "I asked the bank to send a new OTP",
    "The customer service agent asked for my account number",
    "Can I share my account number with my employer?",
    "Why does the bank ask for OTP during login?",
    "I want to change my internet banking password",
    "How do I report someone asking for my PIN?",
    "Is it safe to disclose my OTP to a bank employee?"
]

print("\nSECURITY FALSE-POSITIVE TEST")
print("-" * 65)

for query in queries:
    result = route_ticket(query)

    print("\nQuery:", query)
    print("Predicted Intent:", result["predicted_intent"])
    print("Escalation:", result["escalation_decision"])
