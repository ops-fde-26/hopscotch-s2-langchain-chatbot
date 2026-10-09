from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

from prompts import (
    CLASSIFICATION_PROMPT,
    EXTRACTION_PROMPT,
    PRIORITY_PROMPT,
    RESPONSE_PROMPT,
    SUPPORT_PROMPT,
)

store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def build_chatbot(llm):
    chain = SUPPORT_PROMPT | llm | StrOutputParser()
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )


def run_triage(ticket_text: str, llm) -> dict:
    parser = StrOutputParser()
    classification = CLASSIFICATION_PROMPT | llm | parser
    extraction = EXTRACTION_PROMPT | llm | parser
    priority = PRIORITY_PROMPT | llm | parser
    response = RESPONSE_PROMPT | llm | parser

    category = classification.invoke({"ticket_text": ticket_text}).strip()
    details = extraction.invoke({"ticket_text": ticket_text}).strip()
    pri = priority.invoke({"category": category, "details": details}).strip()
    draft_reply = response.invoke(
        {"category": category, "details": details, "ticket_text": ticket_text}
    ).strip()

    return {
        "category": category,
        "priority": pri,
        "details": details,
        "draft_reply": draft_reply,
    }


SAMPLE_TICKETS = {
    "Peeling sneaker sole": (
        "Hi, I'm Ananya. Order #HS88213 Dino-Print Sneakers for my son, two weeks ago. "
        "The sole started peeling after three wears at school. INR 2,499. Can I get a replacement?"
    ),
    "Worn jacket, wrong size": (
        "Wrong size Winter Jacket, already worn outdoors. Delivered 40 days ago. "
        "Order #HS77102, INR 3,200. Can I exchange it?"
    ),
    "Loose button": (
        "Rohan here. Button on my 2-year-old's romper (Order #HS90455) came off after one wash "
        "and she put it in her mouth. Delivered 10 days ago. INR 1,899."
    ),
    "High value zip": (
        "Meera. Zip on Party Dress set Order #HS91777, INR 7,200, broke the second wear. "
        "Delivered 25 days ago. I want a refund."
    ),
}
