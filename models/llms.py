from langchain_ollama import OllamaLLM

def get_vlm():
    """Vision LLM — Granite Vision 3.2 via Ollama for dev."""
    return OllamaLLM(model="granite3.2-vision")

def get_llm(model_name="llama3", temperature=0.3):
    """Main generation LLM — Llama 3 via Ollama"""
    return OllamaLLM(
        model=model_name,
        temperature=temperature
    )

def get_judge_llm():
    """Evaluation LLM — Mistral Nemo as judge."""
    return OllamaLLM(
        model="mistral-nemo",
        temperature=0
    )