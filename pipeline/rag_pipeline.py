from embeddings.embedder import embed_query
from vectorstore.chromadb import query_store
from pipeline.judge import generate_with_judge


def run_rag(user_query, n_results=3):
    """
    Full RAG pipeline:
    query -> embed -> retrieve -> generate -> evaluate -> answer
    """
    query_embedding = embed_query(user_query)

    retrieved_docs = query_store(query_embedding, n_results=n_results)

    if not retrieved_docs:
        return "No relevant content found in your notes. Try uploading some documents first."
    
    retrieved_docs_str = "\n".join(retrieved_docs)

    answer = generate_with_judge(retrieved_docs_str,user_query)

    return answer