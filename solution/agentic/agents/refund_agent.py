import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import Field
from agentic.agents.tool_history import ToolTrackedResult
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from agentic.tools.refund_tool import (
    cancel_reservation,
    cancel_subscription,
    process_refund,
)
from agentic.tools.reservation_tool import get_user_reservations

load_dotenv()


# Structured output returned by the Refund & Cancellation Agent so the workflow router
# knows whether the request was auto-resolved or requires human escalation.
class RefundResult(ToolTrackedResult):
    resolved: bool = Field(
        description="True if the cancellation or refund was completed or clearly answered."
    )
    needs_escalation: bool = Field(
        description="True if the request exceeds policy thresholds or requires human approval."
    )
    response: str = Field(
        description="Clear, empathetic response to send to the customer."
    )
    escalation_reason: Optional[str] = Field(
        default=None,
        description="Short explanation if needs_escalation is True.",
    )


# v1 fixed for rubric
REFUND_SYSTEM_PROMPT = """You are the Refund & Cancellation Specialist Agent for UDA-Hub (CultPass customer support).
Your responsibility is to handle reservation cancellations, subscription cancellations, and refund requests while enforcing company thresholds.
Report the customer's case outcome confirmed by your tools. General policy guidance comes from the knowledge base. Request escalation if a policy answer is needed to resolve the case.

1. Workflow:
- If the customer wants to cancel a reservation and you don't have the `reservation_id`, call `get_user_reservations(user_id_or_email)` first to find it.
- Call `cancel_reservation(reservation_id)` for reservation cancellations.
- Call `cancel_subscription(user_id_or_email)` ONLY when the customer explicitly asks to cancel their CultPass subscription.
- Call `process_refund(user_id_or_email, amount, reason)` when a specific dollar refund is requested (e.g., duplicate charge, cancelled event).

2. Threshold & Policy Guardrails:
- If `process_refund` returns `requires_escalation=True`, set `resolved=False`, `needs_escalation=True`, and tell the customer that a billing specialist will review the request.
- If no dollar amount is stated for a refund request, do NOT invent an amount—ask for the charge amount or escalate to Billing.
- Never fabricate transaction IDs, card digits, or dates not returned by your tools.

3. Resolution vs. Escalation Rules (`RefundResult`):
- Set `resolved=True` and `needs_escalation=False` when:
  - A reservation or subscription cancellation succeeded, OR
  - `process_refund` returned `approved=True`.
- Set `resolved=False` and `needs_escalation=True` (with a specific `escalation_reason`) when:
  - `process_refund` returns `requires_escalation=True`,
  - A user/reservation cannot be found, or
  - The user disputes a policy/forfeited quota or requests a refund without an eligible tool action."""


def refund_agent(
    rewritten_query: str,
    user_id_or_email: Optional[str] = None,
    model: str = "gpt-5-mini",
) -> RefundResult:
    """Run the refund & cancellation ReAct agent and return a structured RefundResult."""
    # Step 1: Load and validate OpenAI credentials.
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing OPENAI_API_KEY in environment.")
    base_url = os.getenv("base_url") or os.getenv("OPENAI_BASE_URL")

    # Step 2: Initialize the LLM and bind refund, cancellation, and lookup tools.
    llm = ChatOpenAI(
        model=model, temperature=0, api_key=api_key, base_url=base_url
    )
    agent = create_react_agent(
        model=llm,
        tools=[
            get_user_reservations,
            cancel_reservation,
            cancel_subscription,
            process_refund,
        ],
        prompt=REFUND_SYSTEM_PROMPT,
        response_format=RefundResult,
    )

    # Step 3: Pass available user identifier + rewritten query to the agent.
    context_parts = []
    if user_id_or_email:
        context_parts.append(f"user_id_or_email: {user_id_or_email}")
    context_parts.append(f"Request: {rewritten_query}")

    # Step 4: Invoke the agent and return the parsed RefundResult.
    result = agent.invoke({"messages": [HumanMessage(content="\n".join(context_parts))]})
    # v1 fixed for rubric
    return result["structured_response"].record_tool_messages(result["messages"])
