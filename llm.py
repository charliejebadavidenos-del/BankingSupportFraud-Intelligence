import os
from openai import OpenAI


def generate_response(prompt):
    """Generate a grounded customer response using the configured Groq model."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    response = client.responses.create(
        model="openai/gpt-oss-20b",
        input=prompt,
    )
    return response.output_text
