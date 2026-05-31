from models.llms import get_llm, get_judge_llm
from prompts.qa_prompt import get_qa_prompt
from prompts.judge_prompt import get_judge_prompt

# Built once, reused on every call
_qa_chain = None
_judge_chain = None


def build_qa_chain():
    llm = get_llm()
    prompt = get_qa_prompt()
    return prompt | llm


def get_qa_chain():
    global _qa_chain
    if _qa_chain is None:
        _qa_chain = build_qa_chain()
    return _qa_chain


def run_qa_chain(retrieved_docs_str, user_query):
    return get_qa_chain().invoke({
        "retrieved_docs": retrieved_docs_str,
        "user_query": user_query
    })


def build_judge_chain():
    judge_llm = get_judge_llm()
    judge_prompt = get_judge_prompt()
    return judge_prompt | judge_llm


def get_judge_chain():
    global _judge_chain
    if _judge_chain is None:
        _judge_chain = build_judge_chain()
    return _judge_chain


def run_judge_chain(retrieved_docs_str, user_query, answer):
    return get_judge_chain().invoke({
        "retrieved_docs": retrieved_docs_str,
        "user_query": user_query,
        "answer": answer
    })