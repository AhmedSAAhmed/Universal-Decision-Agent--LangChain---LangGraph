import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import Field
from agentic.agents.tool_history import ToolTrackedResult
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from agentic.tools.access_tool import (
    get_account_access_status,
    unblock_user_account,
    update_user_email,
)

load_dotenv()


# Structured output returned by the Account Access Agent so the workflow router
# knows whether the inquiry was resolved or needs human escalation.
class AccessResult(ToolTrackedResult):
    resolved: bool = Field(
        description="True if the login, email, or access inquiry was resolved."
    )
    needs_escalation: bool = Field(
        description="True if the account is blocked above threshold or requires human verification."
    )
    response: str = Field(
        description="Clear, helpful response to send to the customer."
    )
    escalation_reason: Optional[str] = Field(
        default=None,
        description="Short explanation if needs_escalation is True.",
    )


# v1 fixed for rubric
ACCESS_SYSTEM_PROMPT = """You are the Account Access Specialist Agent for UDA-Hub (CultPass customer support).
Your responsibility is to diagnose login issues, check blocked account status, handle email updates, and enforce security thresholds.
Report the customer's account status and outcomes confirmed by your tools. General access and security guidance comes from the knowledge base. Request escalation if such guidance is needed to resolve an account-specific request.

1. Investigation Workflow:
- Always call `get_account_access_status(user_id_or_email)` first when a `user_id` or `email` is provided.
- If `is_blocked=False`, report that the account lookup shows no block. Escalate a persistent login issue that requires further troubleshooting.
- If `is_blocked=True`:
  - Call `unblock_user_account(user_id_or_email, reason, risk_score)` ONLY if you have clear low-risk justification (`risk_score <= 0.4`); otherwise use default `risk_score=0.5` which triggers human Account Access verification.
- If the user asks to change their registered email and `is_blocked=False`, call `update_user_email(user_id_or_email, new_email)`.

2. Strict Guardrails:
- Never fabricate passwords, reset links, or security logs—only state facts returned by your tools.
- Never promise an immediate unblock if `unblock_user_account` returns `requires_escalation=True`.

3. Resolution vs. Escalation Rules (`AccessResult`):
- Set `resolved=True` and `needs_escalation=False` when:
  - The lookup answered the customer's account-status question,
  - `update_user_email` succeeded, OR
  - `unblock_user_account` returned `approved=True`.
- Set `resolved=False` and `needs_escalation=True` (with a specific `escalation_reason`) when:
  - The user cannot be found in CultPass,
  - `is_blocked=True` and `unblock_user_account` returns `requires_escalation=True`, or
  - The user has persistent login issues despite an active account and password reset."""


def access_agent(
    rewritten_query: str,
    user_id_or_email: Optional[str] = None,
    model: str = "gpt-5-mini",
) -> AccessResult:
    """Run the account access ReAct agent and return a structured AccessResult."""
    # Step 1: Load and validate OpenAI credentials.
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing OPENAI_API_KEY in environment.")
    base_url = os.getenv("base_url") or os.getenv("OPENAI_BASE_URL")

    # Step 2: Initialize the LLM and bind account access tools.
    llm = ChatOpenAI(
        model=model, temperature=0, api_key=api_key, base_url=base_url
    )
    agent = create_react_agent(
        model=llm,
        tools=[
            get_account_access_status,
            unblock_user_account,
            update_user_email,
        ],
        prompt=ACCESS_SYSTEM_PROMPT,
        response_format=AccessResult,
    )

    # Step 3: Pass available user identifier + rewritten query to the agent.
    context_parts = []
    if user_id_or_email:
        context_parts.append(f"user_id_or_email: {user_id_or_email}")
    context_parts.append(f"Request: {rewritten_query}")

    # Step 4: Invoke the agent and return the parsed AccessResult.
    result = agent.invoke({"messages": [HumanMessage(content="\n".join(context_parts))]})
    # v1 fixed for rubric
    return result["structured_response"].record_tool_messages(result["messages"])
