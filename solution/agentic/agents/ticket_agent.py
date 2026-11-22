import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import Field
from agentic.agents.tool_history import ToolTrackedResult
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from agentic.tools.ticket_tool import (
    get_ticket_details,
    get_user_tickets,
    update_ticket_metadata,
    add_ticket_message,
)

load_dotenv()


# creating the ticket agent
class TicketResult(ToolTrackedResult):
    resolved: bool = Field(
        description="True if the user's ticket inquiry was fully answered or updated"
    )
    needs_escalation: bool = Field(
        description="True if the ticket cannot be resolved and must be escalated to human support"
    )
    response: str = Field(
        description="Clear, helpful response to send to the customer"
    )
    escalation_reason: Optional[str] = Field(
        default=None,
        description="Short explanation if needs_escalation is True"
    )


TICKET_SYSTEM_PROMPT = """You are the Ticket Specialist Agent for UDA-Hub (CultPass customer support).
Your responsibility is to investigate and resolve inquiries about existing support tickets, ticket statuses, message histories, and ticket follow-ups.

1. Investigation Workflow (Always follow this order):
- If a `ticket_id` is provided, call `get_ticket_details(ticket_id)` first to inspect its status, channel, tags, issue type, and full message history.
- If no `ticket_id` is provided (or the user asks about "my other/previous tickets"), call `get_user_tickets(user_id_or_external)` using their `user_id` or `external_user_id`.
- If neither `ticket_id` nor `user_id` is available, do NOT guess—politely ask for their ticket ID or escalate if this is an asynchronous channel.

2. Updating Tickets (Write Tools):
- Call `update_ticket_metadata` ONLY when a status, `main_issue_type`, or `tags` change is explicitly warranted (e.g., reopening a closed ticket on user request, or marking a resolved inquiry as `"resolved"`).
- Call `add_ticket_message` ONLY to log a meaningful `ai` response or `system` note after verifying the ticket exists.
- Never fabricate `ticket_id`, `status`, timestamps, or past message content—only state facts returned by your tools.

3. Resolution vs. Escalation Rules (`TicketResult`):
- Set `resolved=True` and `needs_escalation=False` when:
  - You successfully retrieved and clearly explained the ticket's current status, history, or next steps, OR
  - You completed a valid status/note update requested by the customer.
- Set `resolved=False` and `needs_escalation=True` (with a specific `escalation_reason`) when:
  - The requested ticket or user cannot be found in the database,
  - The ticket requires domain actions outside ticket lookup/updates (e.g., issuing a refund, unblocking an account, investigating a live outage), or
  - The ticket is already marked `"escalated"` and the customer is demanding urgent human follow-up.

4. Response Style:
- Be concise, empathetic, and factual. Reference the exact `ticket_id` and `status` when summarizing."""


def ticket_agent(
    rewritten_query: str,
    ticket_id: Optional[str] = None,
    user_id: Optional[str] = None,
    model: str = "gpt-5-mini",
) -> TicketResult:
    """Run the ticket ReAct agent with ticket_tool functions and return a structured result."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing OPENAI_API_KEY in environment")
    base_url = os.getenv("base_url") or os.getenv("OPENAI_BASE_URL")

    llm = ChatOpenAI(
        model=model, temperature=0, api_key=api_key, base_url=base_url
    )

    agent = create_react_agent(
        model=llm,
        tools=[
            get_ticket_details,
            get_user_tickets,
            update_ticket_metadata,
            add_ticket_message,
        ],
        prompt=TICKET_SYSTEM_PROMPT,
        response_format=TicketResult,
    )

    context_parts = []
    if ticket_id:
        context_parts.append(f"ticket_id: {ticket_id}")
    if user_id:
        context_parts.append(f"user_id: {user_id}")
    context_parts.append(f"Request: {rewritten_query}")

    result = agent.invoke({"messages": [HumanMessage(content="\n".join(context_parts))]})
    # v1 fixed for rubric
    return result["structured_response"].record_tool_messages(result["messages"])
