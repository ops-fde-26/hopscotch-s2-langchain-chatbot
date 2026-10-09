from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

POLICY = (Path(__file__).parent / "hopscotch_policy.md").read_text(encoding="utf-8")

SUPPORT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a customer support assistant for Hopscotch, a kids fashion store in Mumbai. "
        "Be warm and short.\n\n"
        "Use only this policy:\n\n"
        "=== RETURN POLICY ===\n{policy}\n=== END POLICY ===\n\n"
        "Rules:\n"
        "1. Answer from the policy. Do not invent rules.\n"
        "2. If delivery date, item condition, or order value is missing, ask for it.\n"
        "3. For Section 4 cases, do not decide. Say a human agent will review it.\n"
        "4. For a suspected defect, ask for photos."
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
]).partial(policy=POLICY)

CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Classify the ticket as exactly one of: policy_violation, genuine_defect, other.\n"
        "policy_violation: standard return that policy does not allow, or misuse / wear and tear.\n"
        "genuine_defect: manufacturing defect within 90 days.\n"
        "other: anything else.\n"
        "Reply with only the label."
    )),
    ("human", "{ticket_text}"),
])

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Extract JSON with keys: customer_name, order_id, product, issue, "
        "days_since_delivery, order_value_inr. Use null if unknown. JSON only."
    )),
    ("human", "{ticket_text}"),
])

PRIORITY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Assign High, Medium, or Low.\n"
        "High: child safety, allergy, or genuine_defect on an order above INR 5,000.\n"
        "Medium: routine defect or other follow-up.\n"
        "Low: policy_violation with no safety issue.\n"
        "Reply with only the priority."
    )),
    ("human", "Category: {category}\nDetails: {details}"),
])

RESPONSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Draft a 3-4 sentence Hopscotch support reply.\n"
        "genuine_defect: ask for photos, then replacement/refund.\n"
        "policy_violation: explain the policy politely.\n"
        "other: route to the right team.\n"
        "Section 4: do not promise an outcome; a human will review.\n\n"
        "=== RETURN POLICY ===\n{policy}\n=== END POLICY ==="
    )),
    ("human", "Category: {category}\nDetails: {details}\nOriginal message: {ticket_text}"),
]).partial(policy=POLICY)
