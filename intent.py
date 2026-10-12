
def classify_intent(query: str) -> str:
    """Classify a banking customer query using keyword rules."""

    text = query.lower().replace("’", "'")

    if (
        "atm withdrawal" in text
        and "but i'm in" in text
    ):
        return "Fraud/Unauthorized"


    if (
        "otp" in text
        and "never requested" in text
    ):
        return "Fraud/Unauthorized"

    if (
        "card was charged" in text
        and "unknown merchant" in text
    ):
        return "Fraud/Unauthorized"

    if (
        "was debited" in text
        and "didn't shop" in text
    ):
        return "Fraud/Unauthorized"

    if (
        "transactions" in text
        and "don't recognize" in text
    ):
        return "Fraud/Unauthorized"

    

    # 1. Fraud or unauthorized activity
    fraud_terms = [
        "fraud",
        "unauthorized",
        "unauthorised",
        "someone appears to have used my debit card",
        "someone used my card",
        "someone used my debit card",
        "someone used my credit card",
        "used my card without permission",
        "transaction i didn't make",
        "transaction i did not make",
        "i didn't transact",
        "i didnt transact",
        "unknown transaction",
        "unrecognized transaction",
        "unrecognised transaction",
        "suspicious transaction",
        "money stolen",
        "account hacked",
        "i think someone has hacked my net banking account",
        "someone hacked my bank account",
        "my account has been compromised",
        "net banking account compromised",
        "someone accessed my bank account",
        "unauthorized access to my account",
        "unauthorised access to my account",
        "i didn't make",
        "i did not make",
        "without my consent",
        "never signed up",
    ]

    # 2. Transaction or payment issues
    transaction_terms = [
        "transaction failed",
        "transaction not showing",
        "transaction not reflected",
        "money deducted",
        "money was deducted",
        "amount deducted",
        "amount debited",
        "money was debited",
        "merchant did not receive",
        "merchant has not received",
        "payment not received",
        "payment failed",
        "payment pending",
        "cash not dispensed",
    ]

    # 3. Account access
    access_terms = [
        "login",
        "log in",
        "password",
        "forgot password",
        "account locked",
        "cannot access",
        "can't access",
        "unable to access",
        "blocked account",
        "atm pin",
    ]

    # 4. KYC
    kyc_terms = [
        "kyc",
        "pan card",
        "identity proof",
        "update kyc",
        "kyc verification",
        "aadhaar",
        "pan verification",
    ]

    # 5. Loans
    loan_terms = [
        "loan",
        "emi",
        "interest rate",
        "loan application",
        "loan repayment",
    ]
    
    # 6. Refund
    refund_terms = [
        "refund",
        "refund not received",
        "my refund has not been received",
        "refund not credited",
        "money not refunded",
        "refund pending",
    ]
    if any(term in text for term in fraud_terms):
        return "Fraud/Unauthorized"

    if any(term in text for term in transaction_terms):
        return "Transaction/Payment Issue"

    if any(term in text for term in access_terms):
        return "Account Access"

    if any(term in text for term in kyc_terms):
        return "KYC"

    if any(term in text for term in loan_terms):
        return "Loan"

    if any(term in text for term in refund_terms):
        return "Refund"
    
    return "General"
