
from src.intelligent_ticket_router import route_ticket

test_cases = [
    ("My debit card was stolen yesterday", "Fraud/Unauthorized"),
    ("I shared my OTP with a stranger", "Fraud/Unauthorized"),
    ("A stranger withdrew cash from my account", "Fraud/Unauthorized"),
    ("My salary has not been credited", "Transaction/Payment Issue"),
    ("I need to change my account nominee", "Account Services"),
    ("My personal loan EMI is incorrect", "Loan"),
    ("My Aadhaar verification keeps failing", "KYC"),
    ("My fixed deposit interest was not credited", "Deposit Issue"),
    ("My banking account is locked", "Account Access"),
    ("My NEFT transfer is still pending", "Transaction/Payment Issue")
]

print("\nBANKING ROUTER SCENARIO TEST")
print("-" * 65)

correct = 0

for query, expected_intent in test_cases:
    result = route_ticket(query)

    predicted = result["predicted_intent"]
    decision = result["escalation_decision"]

    if predicted == expected_intent:
        correct += 1
        status = "PASS"
    else:
        status = "CHECK"

    print("\nQuery:", query)
    print("Expected Intent:", expected_intent)
    print("Predicted Intent:", predicted)
    print("Escalation:", decision)
    print("Status:", status)

print("\nIntent Matches:", correct, "/", len(test_cases))
