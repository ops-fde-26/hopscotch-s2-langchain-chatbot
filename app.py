import json
import os
import re
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

import chains
from chains import SAMPLE_TICKETS, build_chatbot, run_triage
from llm import AVAILABLE_MODELS, DEFAULT_MODEL, get_llm

load_dotenv()
st.set_page_config(page_title="Hopscotch Support", page_icon="🛍️", layout="wide")

if not os.getenv("OPENROUTER_API_KEY"):
    st.error("Missing OPENROUTER_API_KEY. Add it to .env and restart.")
    st.stop()


@st.cache_resource
def get_store():
    return chains.store


store = get_store()

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:8]


def new_session():
    st.session_state.session_id = uuid.uuid4().hex[:8]


with st.sidebar:
    st.header("Hopscotch Support")
    model_label = st.selectbox("Model", list(AVAILABLE_MODELS), index=list(AVAILABLE_MODELS.values()).index(DEFAULT_MODEL))
    model = AVAILABLE_MODELS[model_label]
    sid = st.session_state.session_id
    st.write(f"Session: `{sid}`")
    st.button("New session", on_click=new_session)

try:
    llm = get_llm(model)
except Exception as e:
    st.error(str(e))
    st.stop()

tab_chat, tab_triage = st.tabs(["Chat", "Ticket triage"])

with tab_chat:
    user_msg = st.chat_input("Ask about a return, refund, or defect")
    if user_msg:
        try:
            build_chatbot(llm).invoke(
                {"input": user_msg},
                config={"configurable": {"session_id": sid}},
            )
        except Exception as e:
            st.error(str(e))

    msgs = store[sid].messages if sid in store else []
    if not msgs:
        st.info("No messages yet. Example: My son's sneaker sole peeled off after 3 wears.")
    for m in msgs:
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(m.content)
            if isinstance(m, AIMessage):
                st.caption(f"model: {model_label}")

with tab_triage:
    choice = st.selectbox("Sample ticket", list(SAMPLE_TICKETS))
    ticket = st.text_area("Ticket", SAMPLE_TICKETS[choice], height=140)
    if st.button("Run triage"):
        try:
            st.session_state.triage = run_triage(ticket, llm)
        except Exception as e:
            st.error(str(e))

    if "triage" in st.session_state:
        res = st.session_state.triage
        c1, c2, c3 = st.columns(3)
        c1.metric("Category", res["category"])
        c2.metric("Priority", res["priority"])
        cleaned = re.sub(r"^```(?:json)?|```$", "", res["details"].strip(), flags=re.M).strip()
        with c3:
            st.write("Details")
            try:
                st.json(json.loads(cleaned))
            except ValueError:
                st.text(res["details"])
        st.write("Draft reply")
        st.write(res["draft_reply"])
