from pathlib import Path
import re

from src.embedding import embed_text
from src.retriever import search_index


def load_document(file_path):
    """Read a text document and return its contents."""
    path = Path(file_path)
    return path.read_text(encoding="utf-8")


def chunk_text(text, chunk_size=500, overlap=100):
    """Chunk policy text at line/sentence boundaries where possible."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size.")

    # Policy documents often use one numbered clause per line. Keep these
    # clauses intact where possible instead of splitting at arbitrary offsets.
    units = []
    for line in text.splitlines():
        line = " ".join(line.split())
        if not line:
            continue
        if len(line) <= chunk_size:
            units.append(line)
            continue

        # Split unusually long lines at sentence boundaries. The whitespace
        # requirement avoids splitting numbered clauses such as "3.1".
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", line)
        for sentence in sentences:
            if len(sentence) <= chunk_size:
                units.append(sentence)
                continue
            # A single sentence may exceed the limit: split on word boundaries.
            words = sentence.split()
            piece = ""
            for word in words:
                candidate = f"{piece} {word}".strip()
                if piece and len(candidate) > chunk_size:
                    units.append(piece)
                    piece = word
                else:
                    piece = candidate
            if piece:
                units.append(piece)

    chunks = []
    current_units = []
    for unit in units:
        candidate = "\n".join(current_units + [unit])
        if current_units and len(candidate) > chunk_size:
            chunks.append("\n".join(current_units))
            # Overlap only complete clauses, not fragments of a word/sentence.
            retained = []
            for old_unit in reversed(current_units):
                candidate_tail = [old_unit] + retained
                if len("\n".join(candidate_tail)) > overlap:
                    break
                retained = candidate_tail
            current_units = retained
            # Never exceed chunk_size just to preserve overlap.
            if len("\n".join(current_units + [unit])) > chunk_size:
                current_units = []
        current_units.append(unit)

    if current_units:
        chunks.append("\n".join(current_units))
    return chunks


# Map intents to likely relevant policy documents.
# Only use these names if your knowledge-base filenames match.
INTENT_SOURCE_HINTS = {
    "Fraud/Unauthorized": ("fraud_handling_policy.txt","lost_stolen_card_guidance.txt",),
    "Refund": ("refund_dispute_policy.txt",),
    "Transaction/Payment Issue": ("refund_dispute_policy.txt",),
    "KYC": ("kyc_policy.txt",),
    "Loan": ("loan_processing_policy.txt",),
    # No dedicated account-access policy is present in the currently loaded
    # knowledge-base file list, so don't force unrelated policy chunks.
    "Account Access": ("account_access_policy.txt",),
}


def retrieve_context(
    query,
    index,
    chunks,
    top_k=3,
    intent=None,
    chunk_sources=None,
    min_score=0.30,
):
    """
    Retrieve relevant chunks, optionally restricted to documents for the intent.

    `chunk_sources` must contain one source filename per chunk, in the same
    order as `chunks`. `min_score` is a conservative default; tune it using
    examples from your own knowledge base before relying on it in production.
    """
    if index is None or not chunks:
        return []

    query_embedding = embed_text(query)

    # Search all indexed chunks before applying intent filters.
    search_k = len(chunks)

    scores, indices = search_index(
        index,
        query_embedding,
        top_k=search_k
    )

    allowed_names = INTENT_SOURCE_HINTS.get(intent) if intent else None
    results = []

    for score, index_number in zip(scores, indices):
        if index_number == -1:
            continue

        idx = int(index_number)
        if idx < 0 or idx >= len(chunks):
            continue

        source_name = None
        if chunk_sources and idx < len(chunk_sources):
            source_name = Path(chunk_sources[idx]).name

        if allowed_names and source_name is not None:
            if source_name not in allowed_names:
                continue

        score = float(score)
        if score < min_score:
            continue

        results.append({
            "chunk": chunks[idx],
            "score": score,
            "index": idx,
            "source": source_name,
        })

        if len(results) >= top_k:
            break
        
    return results
