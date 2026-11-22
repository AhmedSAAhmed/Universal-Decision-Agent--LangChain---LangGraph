from typing import Optional

from sqlalchemy import Engine

from agentic.agents.tool_history import ToolTrackedResult
from agentic.tools.escalation_tool import escalate_ticket, record_escalation


_TEAMS = {
    "billing": "billing_team",
    "refund_cancel": "billing_team",
    "account_access": "account_access_team",
    "reservation": "reservations_team",
    "outage": "engineering_oncall",
}


class EscalationResult(ToolTrackedResult):
    escalated: bool
    customer_response: str
    handoff: dict


# v1 fixed for rubric
def escalation_agent(
    ticket_text: str,
    escalation_reason: str,
    sentiment: str = "neutral",
    urgency: str = "medium",
    primary_intent: Optional[str] = None,
    ticket_id: Optional[str] = None,
    core_engine: Engine | None = None,
) -> EscalationResult:
    """Record the handoff before confirming it in the customer reply."""
    team = _TEAMS.get(primary_intent, "general_support_lead")
    payload = {
        "ticket_id": ticket_id,
        "target_team": team,
        "priority": urgency,
        "reason": escalation_reason,
        "summary": f"Customer request: {ticket_text[:500]} (sentiment: {sentiment})",
    }
    handoff = (
        record_escalation(core_engine, **payload)
        if core_engine is not None
        else escalate_ticket.invoke(payload)
    )
    confirmed = handoff.get("db_updated") is True
    response = (
        f"I've sent your request to our support team for review. "
        f"Your handoff reference is {handoff['escalation_id']}."
        if confirmed
        else "I couldn't record the handoff. Please try again."
    )
    return EscalationResult(
        escalated=confirmed, customer_response=response, handoff=handoff,
    ).record_tool_call("escalate_ticket", "success" if confirmed else "error")
