import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import Field
from agentic.agents.tool_history import ToolTrackedResult
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from agentic.tools.reservation_tool import (
    list_upcoming_experiences,
    get_user_reservations,
    book_experience,
)

load_dotenv()


# Structured output returned by the Reservation Agent so the workflow router
# knows whether the inquiry was resolved or needs human escalation.
class ReservationResult(ToolTrackedResult):
    resolved: bool = Field(
        description="True if the user's experience or reservation inquiry was fully answered or booked."
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
RESERVATION_SYSTEM_PROMPT = """You are the Reservation Specialist Agent for UDA-Hub (CultPass customer support).
Report availability, reservations, and booking outcomes confirmed by your tools. General policy or venue instructions come from the knowledge base. Request escalation if such guidance is needed to resolve an account-specific request.
Your responsibility is to help customers browse upcoming experiences, check their reservations, book available spots, and resolve venue check-in questions.

1. Investigation & Booking Workflow:
- To browse or check slot availability, call `list_upcoming_experiences(location, include_premium)`.
- To inspect a customer's bookings, call `get_user_reservations(user_id_or_email)`.
- To book an experience, call `book_experience(user_id_or_email, experience_id)` ONLY when the user has confirmed which experience they want and their `user_id` or `email` is known.
- For a customer-specific check-in request, look up the reservation via `get_user_reservations` and share the reservation ID if found.

2. Strict Guardrails:
- Never fabricate `experience_id`, `reservation_id`, dates, locations, or slot counts—only state facts returned by your tools.
- Do not offer actions you do not have tools for (such as generating new QR code links or issuing refunds).

3. Resolution vs. Escalation Rules (`ReservationResult`):
- Set `resolved=True` and `needs_escalation=False` when:
  - You listed matching experiences, retrieved their reservation details/ID, or completed a booking via `book_experience`.
- Set `resolved=False` and `needs_escalation=True` (with a specific `escalation_reason`) when:
  - The user cannot be found, their account is blocked (`is_blocked`), or their subscription is inactive and they dispute it,
  - They ask to cancel a reservation or request a refund (outside this agent's tools), or
  - No matching experiences or slots are available and they request manual accommodation.

4. Response Style:
- Be concise, helpful, and factual. Include exact experience titles, dates, locations, and `reservation_id` when relevant."""


def reservation_agent(
    rewritten_query: str,
    user_id_or_email: Optional[str] = None,
    model: str = "gpt-5-mini",
) -> ReservationResult:
    """Run the reservation ReAct agent with reservation_tool functions and return a structured result."""
    # Step 1: Load and validate OpenAI credentials.
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing OPENAI_API_KEY in environment.")
    base_url = os.getenv("base_url") or os.getenv("OPENAI_BASE_URL")

    # Step 2: Initialize the LLM and bind only reservation tools.
    llm = ChatOpenAI(
        model=model, temperature=0, api_key=api_key, base_url=base_url
    )
    agent = create_react_agent(
        model=llm,
        tools=[
            list_upcoming_experiences,
            get_user_reservations,
            book_experience,
        ],
        prompt=RESERVATION_SYSTEM_PROMPT,
        response_format=ReservationResult,
    )

    # Step 3: Pass available user identifier + rewritten query to the agent.
    context_parts = []
    if user_id_or_email:
        context_parts.append(f"user_id_or_email: {user_id_or_email}")
    context_parts.append(f"Request: {rewritten_query}")

    # Step 4: Invoke the agent and return the parsed ReservationResult.
    result = agent.invoke({"messages": [HumanMessage(content="\n".join(context_parts))]})
    # v1 fixed for rubric
    return result["structured_response"].record_tool_messages(result["messages"])
