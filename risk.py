
def classify_risk(intent, urgency, escalation_decision="NORMAL_PROCESSING"):
    """
    Classify banking risk using ML intent, urgency,
    and independent fraud escalation.
    """

    # Security escalation takes precedence over ML intent
    if escalation_decision == "HIGH_PRIORITY_FRAUD_REVIEW":
        return "High"

    if escalation_decision == "MANUAL_REVIEW":
        return "Medium"

    # Existing intent-based risk logic
    if intent == "Fraud/Unauthorized":
        return "High"

    if urgency == "High":
        return "Medium"

    if intent in ["Account Access", "KYC", "Loan"]:
        return "Medium"

    return "Low"
