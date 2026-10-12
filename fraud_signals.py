
import re

FRAUD_PATTERNS = [
    r"\bwithout my consent\b",
    r"\bwithout my knowledge\b",
    r"\bnever requested\b",
    r"\bdon't recognize\b",
    r"\bdo not recognize\b",
    r"\bunknown merchant\b",
    r"\bunauthorized\b",
    r"\bunauthorised\b",
    r"\bnot made by me\b",
    r"\bsomeone used my\b",
    r"\bnot initiated by me\b",
    r"\bi didn't make\b",
    r"\bi did not make\b",
    r"\bi didn't shop\b",
    r"\bi never signed up\b",
    r"\bwithout permission\b",
    r"\bwithout my permission\b",
    r"\bi did not authorize\b",
    r"\b(?:debit|credit|atm)\s+card\b.*\b(?:stolen|lost)\b",
    r"\b(?:stolen|lost)\b.*\b(?:debit|credit|atm)\s+card\b",
    r"\b(?:stranger|unknown person)\b.*\b(?:withdrew|withdrawn)\b",
    r"\b(?:shared|gave)\b.*\botp\b.*\b(?:stranger|unknown person)\b",

    r"\bwallet\b.*\bstolen\b",
    r"\bstolen\b.*\bwallet\b",
    r"\b(?:bank|debit|credit)\s+cards?\b.*\bmissing\b",
    r"\baccount balance\b.*\b(?:dropped|disappeared)\b",
    r"\b(?:threatening|forcing|coercing)\b.*\b(?:password|pin|otp)\b",
    r"\b(?:password|pin|otp)\b.*\b(?:threatening|forcing|coercing)\b",

    # Stolen cards, including generic references to "my card"
    r"\b(?:my|the)\s+card\b.*\b(?:stolen|lost)\b",
    r"\b(?:stolen|lost)\b.*\b(?:my|the)\s+card\b",

    # Requests to disclose sensitive credentials
    
    r"\b(?:asking|asked|wants|requested)\b.*\b(?:share|disclose|reveal|provide|give)\b.*\b(?:otp|password|pin)\b",

    r"\b(?:otp|password|pin)\b.*\b(?:share|disclose|reveal)\b",

    # Bank impersonation
    r"\b(?:pretending|posing|impersonating)\b.*\bbank\b",


]

def detect_fraud_signals(query):
    text = str(query).lower()

    matched_patterns = [
        pattern
        for pattern in FRAUD_PATTERNS
        if re.search(pattern, text)
    ]

    return {
        "fraud_signal_detected": bool(matched_patterns),
        "matched_patterns": matched_patterns
    }


# Step 31 - Suspicious activity review signals

REVIEW_PATTERNS = [
    r"\bduplicate charge\b",
    r"\batm withdrawal\b.*\bbut i'm in\b",
    r"\batm withdrawal\b.*\bbut i am in\b",
    # Unexpected purchases or payments
    r"\b(?:purchase|payment|transaction|charge)\b.*\b(?:unexpected|unrecognized|unrecognised)\b",

    # Customer cannot explain a payment or transaction
    r"\b(?:payment|transaction|charge)\b.*\b(?:cannot explain|can't explain)\b",

    # Purchases appearing while the customer was asleep
    r"\b(?:purchase|payment|charge)\b.*\b(?:while|although)\b.*\b(?:asleep|sleeping)\b",
]

def detect_review_signals(query):
    text = str(query).lower()

    matched_patterns = [
        pattern
        for pattern in REVIEW_PATTERNS
        if re.search(pattern, text)
    ]

    return {
        "review_required": bool(matched_patterns),
        "matched_patterns": matched_patterns
    }



if __name__ == "__main__":
    test_queries = [
        "My card was charged ₹999 by an unknown merchant",
        "My UPI ID was used to transfer money without my consent",
        "I received an OTP but never requested it",
        "My UPI payment failed but money was debited"
    ]

    for query in test_queries:
        result = detect_fraud_signals(query)

        print("\nQuery:", query)
        print("Fraud Signal:", result["fraud_signal_detected"])
        print("Matched:", result["matched_patterns"])



if __name__ == "__main__":
    # Step 29 - Evaluate fraud signals against the dataset

    import pandas as pd
    from pathlib import Path
    from sklearn.metrics import confusion_matrix, classification_report

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    dataset_path = (
        PROJECT_ROOT
        / "data"
        / "tickets"
        / "02_improved_support_tickets.csv"
    )

    df = pd.read_csv(dataset_path)

    actual_fraud = df["category"].eq("Fraud/Unauthorized")

    predicted_fraud = df["query_text"].apply(
        lambda query: detect_fraud_signals(query)["fraud_signal_detected"]
    )

    print("\nFRAUD SIGNAL DETECTOR EVALUATION")
    print("-" * 55)

    print(
        confusion_matrix(
            actual_fraud,
            predicted_fraud,
            labels=[False, True]
        )
    )

    print(
        classification_report(
            actual_fraud,
            predicted_fraud,
            target_names=["Non-Fraud", "Fraud"],
            zero_division=0
        )
    )

    missed_fraud = df[
        actual_fraud & ~predicted_fraud
    ]

    false_alarms = df[
        ~actual_fraud & predicted_fraud
    ]

    print("\nMISSED FRAUD QUERIES:")
    for query in missed_fraud["query_text"]:
        print("-", query)

    print("\nFALSE ALARMS:")
    for query in false_alarms["query_text"]:
        print("-", query)



    # Test review signals

    print("\nSUSPICIOUS ACTIVITY REVIEW TEST")
    print("-" * 55)

    test_review_queries = [
        "There's an ATM withdrawal of ₹5,000 from Bangalore but I'm in Ahmedabad",
        "I found a duplicate charge of ₹999 on my statement",
        "My ATM withdrawal was successful",
        "My UPI payment failed"
    ]

    for query in test_review_queries:
        result = detect_review_signals(query)

        print("\nQuery:", query)
        print("Review Required:", result["review_required"])