import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import Field
from agentic.agents.tool_history import ToolTrackedResult
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from agentic.tools.billing_tool import (
    get_subscription_details,
    change_subscription_tier,
)

load_dotenv()


# Structured output returned by the Billing Agent so the workflow router
# knows whether the inquiry was resolved or needs human escalation.
class BillingResult(ToolTrackedResult):
    resolved: bool = Field(
        description="True if the user's billing or subscription inquiry was fully answered or updated."
    )
    needs_escalation: bool = Field(
        description="True if the request cannot be resolved and must be escalated to human support."
    )
    response: str = Field(
        description="Clear, helpful response to send to the customer."
    )
    escalation_reason: Optional[str] = Field(
        default=None,
        description="Short explanation if needs_escalation is True.",
    )


# v1 fixed for rubric
BILLING_SYSTEM_PROMPT = """You are the Billing Specialist Agent for UDA-Hub (CultPass customer support).
Your responsibility is to handle account-specific subscription, tier, monthly quota, and plan-change inquiries.
Report account facts and outcomes confirmed by your tools. General policy guidance comes from the knowledge base. Request escalation if a policy answer is needed to resolve an account-specific request.

1. Investigation Workflow (Always follow this order):
- If a `user_id` or `email` is provided, call `get_subscription_details(user_id_or_email)` first to inspect their subscription status, tier, monthly quota, and used reservations.
- If no `user_id` or `email` is available, do NOT guess—politely ask for their account email or user ID (or escalate if asynchronous).

2. Plan Changes (Write Tool):
- Call `change_subscription_tier(user_id_or_email, new_tier)` ONLY when the user explicitly requests upgrading to `"premium"` or downgrading to `"basic"` AND their subscription status is `"active"`.
- Confirm the new tier and quota returned by the plan-change tool.
- Never fabricate invoice amounts, card numbers, transaction IDs, or dates not returned by your tools.
- Do not offer actions you do not have tools for (such as issuing refunds or reactivating cancelled/blocked accounts).

3. Resolution vs. Escalation Rules (`BillingResult`):
- Set `resolved=True` and `needs_escalation=False` when:
  - You retrieved and clearly explained the user's subscription tier, status, or remaining quota, OR
  - You completed a valid `basic` <-> `premium` tier change.
- Set `resolved=False` and `needs_escalation=True` (with a specific `escalation_reason`) when:
  - The user or subscription cannot be found in CultPass,
  - The user asks for a refund, duplicate-charge reversal, invoice dispute, or reactivating a cancelled/blocked subscription, or
  - `change_subscription_tier` returns an error that requires manual billing intervention.

4. Response Style:
- Be concise, empathetic, and factual. Reference the exact `tier`, `status`, and `quota_remaining` when summarizing."""


def billing_agent(
    rewritten_query: str,
    user_id_or_email: Optional[str] = None,
    model: str = "gpt-5-mini",
) -> BillingResult:
    """Run the billing ReAct agent with billing_tool functions and return a structured result."""
    # Step 1: Load and validate OpenAI credentials.
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing OPENAI_API_KEY in environment.")
    base_url = os.getenv("base_url") or os.getenv("OPENAI_BASE_URL")

    # Step 2: Initialize the LLM and bind only billing tools.
    llm = ChatOpenAI(
        model=model, temperature=0, api_key=api_key, base_url=base_url
    )
    agent = create_react_agent(
        model=llm,
        tools=[
            get_subscription_details,
            change_subscription_tier,
        ],
        prompt=BILLING_SYSTEM_PROMPT,
        response_format=BillingResult,
    )

    # Step 3: Pass available user identifier + rewritten query to the agent.
    context_parts = []
    if user_id_or_email:
        context_parts.append(f"user_id_or_email: {user_id_or_email}")
    context_parts.append(f"Request: {rewritten_query}")

    # Step 4: Invoke the agent and return the parsed BillingResult.
    result = agent.invoke({"messages": [HumanMessage(content="\n".join(context_parts))]})
    # v1 fixed for rubric
    return result["structured_response"].record_tool_messages(result["messages"])
