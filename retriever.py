import faiss
import numpy as np


def build_index(embeddings):
    """Build a FAISS index using cosine similarity."""
    embeddings = np.asarray(embeddings).astype("float32").copy()

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array.")
    if embeddings.shape[0] == 0:
        raise ValueError("Cannot build a FAISS index from empty embeddings.")
    if embeddings.shape[1] == 0:
        raise ValueError("Embedding vectors must have at least one dimension.")

    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def search_index(index, query_embedding, top_k=3):
    """Return cosine-similarity scores and indices for the nearest chunks."""
    if index is None or index.ntotal == 0:
        return np.array([], dtype="float32"), np.array([], dtype="int64")

    query_embedding = np.asarray(query_embedding).astype("float32").reshape(1, -1)

    if query_embedding.shape[1] != index.d:
        raise ValueError(
            f"Query embedding dimension ({query_embedding.shape[1]}) "
            f"does not match index dimension ({index.d})."
        )

    faiss.normalize_L2(query_embedding)
    k = min(max(int(top_k), 1), index.ntotal)
    scores, indices = index.search(query_embedding, k)
    return scores[0], indices[0]
