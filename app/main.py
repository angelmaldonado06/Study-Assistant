from pipeline.rag_pipeline import run_rag

from embeddings.embedder import embed_documents
from vectorstore.chromadb import store_chunks

def seed_test_data():
    documents = [
        "Photosynthesis is the process by which plants convert sunlight into glucose using carbon dioxide and water.",
        "Mitosis is a type of cell division that results in two identical daughter cells from a single parent cell.",
        "The capital of France is Paris, which is also the largest city in the country.",
        "Water boils at 100 degrees Celsius at standard atmospheric pressure."
    ]
    metadata = [{"source": "test_data"} for _ in documents]
    embeddings = embed_documents(documents)
    store_chunks(documents, embeddings, metadata)
    print("Test data seeded.\n")

if __name__ == "__main__":
    # seed_test_data()
    query = "What is photosynthesis?"
    response = run_rag(query)
    print("\nAnswer:", response)



# from user_pipeline import run_user_pipeline

# query = "What is photosynthesis?"
# answer = run_user_pipeline(query)

# print(answer)


# from langchain_ollama import OllamaLLM
# from langchain_classic.chains import LLMChain
# from langchain_core.prompts import PromptTemplate
# from sentence_transformers import SentenceTransformer
# import chromadb

# #CREATING MODELS / VECTOR DATA BASE
# def create_llm():
#     return OllamaLLM(model='llama3', temperature=0.3)

# def create_vlm():
#     return OllamaLLM(model="granite3.2-vision")

# def create_llm_judge():
#     return OllamaLLM(model="mistral-nemo")

# def create_embedding_model():
#     return SentenceTransformer('intfloat/multilingual-e5-large')

# def create_db():
#     client = chromadb.Client()
#     return client.create_collection(name="my_collection")


# # DOCUMENT EMBEDDINGS / STORAGE
# documents = [
#     "Photosynthesis is the process by which plants convert sunlight into energy.",
#     "Mitosis is a type of cell division that results in two identical daughter cells.",
#     "The capital of France is Paris.",
#     "Water boils at 100 degrees Celsius at standard atmospheric pressure."
# ]

# ids = ["doc1", "doc2", "doc3", "doc4"]

# embedding_model = create_embedding_model()
# collection = create_db()

# #PASSAGE FOR EMBEDDING MODEL (RELEVANT CONTEXT)
# passage_embeddings = embedding_model.encode(
#     ["passage: " + doc for doc in documents],
#     normalize_embeddings=True
# )
# #ADD TO STORAGE
# collection.upsert(
#     embeddings = passage_embeddings.tolist(),
#     documents=documents,
#     ids=ids
# )

# #USER QUERY -> EMBED QUERY -> SIMILARITY SEARCH
# user_query = "What is photosyntesis?"
# embeded_query = embedding_model.encode(
#     ["query: " + user_query], 
#     normalize_embeddings=True
# )

# res = collection.query(
#     query_embeddings= embeded_query.tolist(),
#     n_results = 3
# )
# retrieved_docs = res["documents"][0]

# #RESULTS
# print("User Question:", user_query)
# print("\nTop Results:")

# for i, doc in enumerate(retrieved_docs):
#     print(f"{i+1}. {doc}")

# prompt = PromptTemplate(

#     template = '''
#     You are an AI assitant tasked with giving correct answers based on relevant context. Provide a concise, and detailed answer.

#     Instructions:
#     1. Generate an answer to the user's query with the given context
#     2. Answer in no more than 2 sentences. Output only the answer.

#     Relevant Context:
#     {retrieved_docs}
#     Based on the relevant context, please answer the following question:
#     {user_query}
    

#     Answer:
#     ''',
#     input_variables=["retrieved_docs", "user_query"]
# )

# #LLM AS A JUDGE
# judge_prompt = PromptTemplate(

#     template = '''
#     You are an evaluator judging the quality of an AI response.

#     Evaluate based on :
#     1. Correctned (is it factually correct based on context?)
#     2. Relevance ( does it answer the question?)
#     3. Completeness (is it sufficient?)

#     Context: {retrieved_docs}
#     Question: {user_query}
#     Answer: {answer}

#     Instructions:
#     - Give it a score from 1 to 10 
#     - If score >= 7 -> Pass
#     - If score < 7 -> Fail
#     - Provide brief feedback

#     Output format:
#     Score: <number>
#     Decision: <PASS or FAIL>
#     Feedback: <short explanation>
#     ''',

#     input_variables=["retrieved_docs", "user_query", "answer"]
# )

# llm = create_llm()
# chain = prompt | llm 
# retrieved_docs_str = "\n".join(retrieved_docs)

# judge_llm = create_llm_judge()
# judge_chain = judge_prompt | judge_llm



# def generate_with_judge(llm_chain, judge_chain, retrieved_docs_str, user_query, attempts=3):

#     for attempt in range(attempts):

#         answer = llm_chain.invoke({
#             "retrieved_docs" : retrieved_docs_str,
#             "user_query" : user_query

#         })

#         evaluation = judge_chain.invoke({
#             "retrieved_docs" : retrieved_docs_str,
#             "user_query" : user_query,
#             "answer": answer
#         })

#         if "pass" in evaluation.lower():
#             print(f"\nAPPROVED BY JUDGE\nFeedback: {evaluation}\n")
#             return answer

#         print("\nRETRYING...\n")

#     return "Failed to generate a good answer after {attempts} attempts."

# final_answer = generate_with_judge(chain, judge_chain, retrieved_docs_str, user_query)

# print(final_answer)



# #RAG PIPLINE: CHUNK -> EMBED -> STORE -> RETRIEVE TOP-K

# #PDF UPLOAD/EXTRACTION -> RAG PIPLINE -> LLM GENERATION

# #IMAGE UPLOAD/EXTRATION

