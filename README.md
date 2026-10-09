# Hopscotch Support Chatbot (LangChain)

Same Hopscotch return-policy assistant as the OpenAI SDK version, rebuilt with LangChain:

- `llm.py` — model / OpenRouter client
- `prompts.py` — prompt templates
- `chains.py` — chat history + ticket triage
- `app.py` — Streamlit UI

## Run locally

```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env
# put OPENROUTER_API_KEY in .env
streamlit run app.py
```

## Streamlit Cloud

Set `OPENROUTER_API_KEY` in app secrets. Do not commit `.env`.
