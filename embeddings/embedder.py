from sentence_transformers import SentenceTransformer

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if not _embedding_model:
        _embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')

    return _embedding_model

def embed_documents(documents):
    model = get_embedding_model()
    passage_embeddings = model.encode(
        ["passage: " + doc for doc in documents],
        normalize_embeddings=True
    )
    return passage_embeddings.tolist()

def embed_query(query):
    model = get_embedding_model()
    embeded_query = model.encode(
        ["query: " + query], 
        normalize_embeddings=True
    )
    return embeded_query.tolist()