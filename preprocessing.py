import re


def clean_text(text):
    """
    Basic preprocessing for customer support queries.
    """

    # Convert to string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Replace currency amounts such as ₹10,000 or ₹25000
    text = re.sub(r"₹\s?[\d,]+(?:\.\d+)?", " <amount> ", text)

    # Replace numbers
    text = re.sub(r"\b\d+\b", " <number> ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text