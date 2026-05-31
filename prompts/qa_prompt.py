from langchain_core.prompts import PromptTemplate


def get_qa_prompt():
    template = '''
    You are an AI assitant tasked with giving correct answers based on relevant context. Provide a concise, and detailed answer.

    Instructions:
    1. Generate an answer to the user's query with the given context
    2. Answer in no more than 2 sentences. Output only the answer.

    Relevant Context:
    {retrieved_docs}
    Based on the relevant context, please answer the following question:
    {user_query}


    Answer:
    '''

    return PromptTemplate(
        template = template,
        input_variables=["retrieved_docs", "user_query"]
)