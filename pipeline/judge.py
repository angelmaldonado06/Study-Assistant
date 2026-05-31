from pipeline.chain import run_judge_chain
from pipeline.chain import run_qa_chain


def evaluate(retrieved_docs_str, user_query, answer):
    """
    Evaluate the answer using the llm as a judge
    Returns passed:bool, evaluation:str
    """
    evaluation = run_judge_chain(retrieved_docs_str, user_query, answer)
    passed = "pass" in evaluation.lower()

    return passed, evaluation


def generate_with_judge(retrieved_docs_str, user_query, attempts=3):
    
    for attempt in range(attempts):

        answer = run_qa_chain(retrieved_docs_str, user_query)
        passed, feedback = evaluate(retrieved_docs_str, user_query, answer)

        if passed:
            print(f"\nAPPROVED BY JUDGE\nFeedback: {feedback}\n")
            return answer

        print("\nRETRYING...\n")
    return "I wasn't able to generate a reliable answer from your notes. Try rephrasing your question."

