from langchain_community.llms import LlamaCpp

# MODEL_PATH = "models/Phi-3-mini-4k-instruct-q4.gguf"
MODEL_PATH = "models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"

def load_llm():
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=8192,
        n_batch=512,
        max_tokens=512,
        temperature=0.1,
        repeat_penalty=1.3,
        verbose=False,
        n_gpu_layers=-1,
    )
    return llm