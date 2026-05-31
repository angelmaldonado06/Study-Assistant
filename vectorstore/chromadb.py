import chromadb

_client = None
_collection = None
COLLECTION_NAME = 'study_assistant'

def get_collection():
    global _collection, _client
    if not _collection:
        _client = chromadb.PersistentClient(path="./chroma_db")
        _collection = _client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection

def store_chunks(chunks, embeddings, metadata):
    collection = get_collection()
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    if metadata is None:
        metadata = [{"source": "unknown"} for _ in chunks]
    
    collection.upsert(
        embeddings = embeddings,
        documents=chunks,
        metadatas = metadata,
        ids=ids
    )


def query_store(query_embedding, n_results):
    collection = get_collection()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    return results["documents"][0]

