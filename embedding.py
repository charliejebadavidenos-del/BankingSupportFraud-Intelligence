from sentence_transformers import SentenceTransformer


# Load the embedding model once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text):
    """
    Convert a single text string into a 384-dimensional embedding.
    """
    return embedding_model.encode(text)


def embed_texts(texts):
    """
    Convert multiple text strings into embeddings.

    Parameters:
        texts: list of strings

    Returns:
        numpy array containing one embedding per text
    """
    return embedding_model.encode(texts)


def get_embedding_dimension():
    """
    Return the dimension of the embedding vectors.
    """
    return embedding_model.get_sentence_embedding_dimension()