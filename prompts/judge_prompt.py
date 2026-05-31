from langchain_core.prompts import PromptTemplate


def get_judge_prompt():
    template = '''
    You are an evaluator judging the quality of an AI response.

    Evaluate based on :
    1. Correctned (is it factually correct based on context?)
    2. Relevance ( does it answer the question?)
    3. Completeness (is it sufficient?)

    Context: {retrieved_docs}
    Question: {user_query}
    Answer: {answer}

    Instructions:
    - Give it a score from 1 to 10 
    - If score >= 7 -> Pass
    - If score < 7 -> Fail
    - Provide brief feedback

    Output format:
    Score: <number>
    Decision: <PASS or FAIL>
    Feedback: <short explanation>
    '''

    return PromptTemplate(
        template = template,
        input_variables=["retrieved_docs", "user_query", "answer"]
)