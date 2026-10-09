import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"

AVAILABLE_MODELS = {
    "GPT-4o mini": "openai/gpt-4o-mini",
    "Claude Haiku 4.5": "anthropic/claude-haiku-4.5",
    "Gemini 2.5 Flash Lite": "google/gemini-2.5-flash-lite",
    "Llama 3.3 70B": "meta-llama/llama-3.3-70b-instruct",
}


def get_llm(model: str = DEFAULT_MODEL, temperature: float = 0.3) -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found. Copy .env.example to .env and add your key.")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
    )
