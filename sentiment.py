def analyze_sentiment(text):
    """
    Rule-based customer sentiment and urgency analysis.

    Sentiment measures expressed emotion.
    Urgency measures how quickly attention may be needed.
    """

    text = str(text).lower().replace("’", "'")

    # Emotional language: sentiment indicators
    negative_words = [
        "angry",
        "frustrated",
        "frustrating",
        "terrible",
        "worst",
        "upset",
        "scared",
        "worried",
        "afraid",
        "anxious",
        "disappointed",
        "unhappy",
        "horrible",
        "stressful",
    ]

    positive_words = [
        "thank you",
        "thanks",
        "excellent",
        "great service",
        "happy",
        "satisfied",
        "wonderful",
    ]

    # Explicit urgency indicators
    urgent_words = [
        "urgent",
        "immediately",
        "asap",
        "emergency",
        "right now",
        "today",
        "stolen",
        "fraud",
        "hacked",
        "unauthorized",
        "unauthorised",
        "without my permission",
        "someone used my card",
        "someone used my debit card",
        "someone used my credit card",
        "account takeover",
        "multiple fraudulent transactions",
        "card compromised",
        "lost my card",
        "can't access my account",
        "cannot access my account",
        "someone appears to have used my debit card",
    ]

    negative_found = any(
        word in text for word in negative_words
    )

    positive_found = any(
        word in text for word in positive_words
    )

    # Avoid treating a security incident as low urgency
    # merely because emotional words are absent.
    urgency_score = sum(
        1 for word in urgent_words if word in text
    )

    if negative_found and positive_found:
        sentiment = "Mixed"
    elif negative_found:
        sentiment = "Negative"
    elif positive_found:
        sentiment = "Positive"
    else:
        sentiment = "Neutral"

    if urgency_score >= 2:
        urgency = "High"
    elif urgency_score == 1:
        urgency = "Medium"
    else:
        urgency = "Low"

    return {
        "sentiment": sentiment,
        "urgency": urgency,
        "urgency_score": urgency_score,
    }
