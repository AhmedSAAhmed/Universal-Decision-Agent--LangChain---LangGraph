import os
from typing import Literal, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()


# v1 fixed for rubric
class CustomerPreference(BaseModel):
    category: Literal["contact_channel", "language", "reply_style", "accessibility", "experiences"]
    quote: str = Field(description="Exact words in the latest message expressing a lasting preference.")


# getting user input intent & routing

class TriageDecision(BaseModel):
    primary_intent: Literal[
        "ticketing",
        "billing",
        "reservation",
        "refund_cancel",
        "outage",
        "account_access",
        "general_qa",
    ] = Field(description="Primary support domain for this ticket")
    sentiment: Literal[
        "positive",
        "neutral",
        "frustrated",
        "angry",
    ] = Field(description="Customer emotional tone")
    urgency: Literal[
        "low",
        "medium",
        "high",
        "critical",
    ] = Field(description="Urgency level of this ticket")
    reasoning: str = Field(description="Brief reason for the chosen intent, sentiment, and urgency")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score in the classification")
    rewritten_query: str = Field(description="Concise, context-enriched query for the specialist agent")
    # v1 fixed for rubric
    knowledge_only: bool = Field(
        default=False,
        description="Whether a general policy, FAQ, or how-to answer can resolve the ticket without an account lookup, live status check, or action.",
    )
    customer_preferences: list[CustomerPreference] = Field(default_factory=list)


# Creating system prompt for traigre agent
TRIAGE_SYSTEM_PROMPT = """You are the Triage Agent for UDA-Hub, the customer support brain for CultPass.
Analyze the incoming ticket, channel metadata, and any prior user context, then return a structured TriageDecision.

1. primary_intent (choose ONE best-fit domain):
- "ticketing": Checking the status or history of an existing support ticket or follow-up.
- "billing": Subscription tiers (Basic/Premium), invoices, charges, prorated upgrades/downgrades, or monthly experience quotas.
- "reservation": Browsing CultPass experiences, slot availability, viewing bookings, or venue QR code/check-in help.
- "refund_cancel": Cancelling a reservation or subscription, or requesting a refund/credit, duplicate charges, refund requests, and cancellations.
- "outage": App crashes, widespread payment/booking errors, 500 errors, or suspected service downtime.
- "account_access": Login failures, password resets, updating registered email, or blocked/suspended account inquiries.
- "general_qa": General CultPass how-to questions, policies, or FAQs that do not require account changes.

2. sentiment & urgency:
- sentiment: "positive", "neutral", "frustrated" (impatient/annoyed), or "angry" (hostile, threatening churn/chargeback, or all-caps/profanity).
- urgency:
  - "low": General questions or non-time-sensitive inquiries.
  - "medium": Standard billing, reservation, or account requests.
  - "high": Event happening soon (today/tomorrow), blocked account, or failed/duplicate payment.
  - "critical": Suspected outage, security/fraud issue, or severe customer anger.
reasonig: 
- Choose high urgency if ANY of these are true: 
  - "critical" (e.g., active outage, security/fraud concern, or severe customer anger)
  - User explicitly says “urgent”, “now”, “ASAP”, or uses ALL CAPS frequently.
  - Event is today/tomorrow, user is locked out, or payment failed/duplicated.
  - User mentions “can’t book”, “can’t check in”, or “error”.
- Choose low urgency ONLY IF all of these are true:
  - No time pressure in the text.
  - Not about login, blocked access, or payment issues.
  - Request is purely informational (“how do I…”, “what is…”).
- Everything else is medium.

3. confidence (0.0 to 1.0):
- 0.80 - 1.00: Clear single intent with unambiguous details.
- 0.50 - 0.79: Likely intent, but missing some context or mildly ambiguous.
- 0.00 - 0.49: Vague, conflicting, or unclassifiable request.

4. rewritten_query:
- Rewrite the customer's request into a clear, concise instruction for the downstream specialist agent.
- Preserve all critical entities (user email, user_id, reservation_id, experience name, dates, dollar amounts) and include any relevant prior context.
- Keep to 1–2 sentences using only verified facts from the ticket; do not invent step-by-step playbooks or database fields

5. knowledge_only:
- Set true for questions fully answerable from general documentation, including plan rules, refund policy, and how-to instructions. Retain the primary domain intent.
- Set false when resolving the request requires a customer-specific lookup, live status, or an action. For example, "What are the refund rules?" is knowledge-only and "Refund my last charge" is an operation.

6. customer_preferences:
- Record only lasting preferences the customer explicitly states in the latest message. Use their exact words and return an empty list for one-time instructions or inferred preferences.
- Treat previous turns and saved preferences as context. Use them to interpret references in the latest request. Never treat earlier requests as authorization for a new action.
"""
# creating triage agent funtion
def triage_agent(user_input: str, channel: str = "chat", memory_context: Optional[str] = None, model: str = "gpt-5-mini") -> TriageDecision:
    """Analyze the incoming ticket, channel metadata, and any prior user context, then return a structured TriageDecision."""
    # set up correct openai endpoints connection to vocaruem
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing OPENAI_API_KEY in environment.")
    base_url = os.getenv("base_url") or os.getenv("OPENAI_BASE_URL")


    llm = ChatOpenAI(model=model, temperature=0, api_key=api_key, base_url=base_url)
    structured_llm = llm.with_structured_output(TriageDecision)

    user_content = (
    f"Current Date: {datetime.now().strftime('%Y-%m-%d')}\n"
    f"Channel: {channel}\n"
    )
    if memory_context:
        user_content += f"Prior User Context: {memory_context}\n"
    user_content += f"Ticket: {user_input}"

    return structured_llm.invoke([
        SystemMessage(content=TRIAGE_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ])
    
