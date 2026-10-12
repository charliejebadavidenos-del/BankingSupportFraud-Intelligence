def recommend_action(intent, risk, escalation_decision="NORMAL_PROCESSING"):
    """
    Recommend next steps based on intent, risk and security escalation.

    These are recommendations for the customer/support agent. The function
    does not itself block accounts, file complaints, or perform bank actions.
    """

    # Escalation must override the ML intent when a security incident is detected.
    if escalation_decision == "HIGH_PRIORITY_FRAUD_REVIEW":
        return [
            "Immediately contact the bank through its official app, website, or verified helpline to secure the affected card or account",
            "If a card is missing or stolen, request that the bank block it and arrange a replacement",
            "Review recent account and card activity; report any transactions you do not recognize",
            "Ask the bank to investigate the security incident and provide a complaint reference number",
            "Preserve relevant messages, transaction details, and other evidence",
            "Do not share your OTP, PIN, CVV, or password with anyone",
        ]

    if escalation_decision == "MANUAL_REVIEW":
        return [
            "Review the transaction or account activity that appears unusual",
            "Contact the bank through an official channel to report the concern and request a review",
            "Preserve transaction details, messages, and any relevant screenshots",
            "Keep the complaint or case reference number for follow-up",
            "Do not share your OTP, PIN, CVV, or password",
        ]

    # Fraud / unauthorized transactions
    if intent == "Fraud/Unauthorized":
        if risk == "High":
            return [
                "Contact the bank immediately through its official app, website, or the number on your card",
                "Request that the card/account and affected payment instruments be blocked",
                "Report the unauthorized transaction and keep the complaint reference number",
                "Preserve transaction details and supporting evidence",
                "Never share your OTP, PIN, CVV, or password",
                "Escalate the case for fraud investigation",
            ]
        return [
            "Review the transaction details and confirm whether you authorized it",
            "Report the suspicious transaction through an official bank channel",
            "Keep the complaint reference number and relevant evidence",
            "Never share your OTP, PIN, CVV, or password",
        ]

    # Refund requests
    if intent == "Refund":
        return [
            "Check the transaction status and account statement for a refund or reversal",
            "Identify whether this is a failed transaction, merchant refund, or chargeback",
            "If the refund remains missing, contact the bank through an official channel and ask how to raise or track a dispute",
            "Keep the transaction ID, date, amount, merchant details, and any receipts",
            "Confirm the applicable timeline with the bank; do not assume all refund types have the same timeline",
        ]

    # Failed / pending payments and amount debited without receipt
    if intent == "Transaction/Payment Issue":
        return [
            "Check the transaction status in the banking app or online banking",
            "Confirm with the merchant whether the payment was received",
            "Keep the transaction ID, date, amount, merchant details, and receipt or screenshot",
            "If the amount was debited but the transaction failed or the merchant did not receive it, contact the bank through an official channel to request a status check or dispute",
            "Confirm the applicable reversal timeline with the bank because it depends on the payment method and transaction type",
        ]

    # Account access
    if intent == "Account Access":
        if risk == "Medium":
            return [
                "Use the bank's official password-reset or account-recovery process",
                "Verify identity only through the bank's official channel",
                "Contact the bank if access remains blocked",
                "Never share your OTP, PIN, or password with anyone",
            ]
        return [
            "Use the bank's official account-recovery guidance",
            "Contact the bank through an official channel if you still cannot access the account",
        ]

    # KYC
    if intent == "KYC":
        return [
            "Check the KYC status through the bank's official app, website, or branch",
            "Confirm which documents are required before submitting them",
            "Do not send identity documents or account credentials to unverified contacts",
        ]

    # Loan
    if intent == "Loan":
        return [
            "Check the loan application status through the bank's official channel",
            "Ask whether additional documents or information are required",
            "Keep the application or reference number for follow-up",
        ]

    # General
    return [
        "Review the issue and relevant transaction or application details",
        "Contact the bank through its official app, website, or branch for case-specific guidance",
    ]
