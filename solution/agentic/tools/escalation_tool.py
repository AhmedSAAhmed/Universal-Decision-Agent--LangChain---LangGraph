import uuid
from pathlib import Path
from typing import Literal, Optional
from sqlalchemy import Engine, create_engine
from langchain_core.tools import tool

from utils import get_session
from data.models import udahub

# Locate udahub.db relative to solution/ so this works from any working directory.
SOLUTION_DIR = Path(__file__).resolve().parents[2]
UDAHUB_DB_PATH = SOLUTION_DIR / "data" / "core" / "udahub.db"
engine = create_engine(f"sqlite:///{UDAHUB_DB_PATH}", echo=False)

EscalationTeam = Literal[
    "billing_team", "account_access_team", "reservations_team",
    "engineering_oncall", "general_support_lead",
]
EscalationPriority = Literal["low", "medium", "high", "critical"]

@tool
def escalate_ticket(
    target_team: EscalationTeam,
    priority: EscalationPriority,
    reason: str,
    summary: str,
    ticket_id: Optional[str] = None,
) -> dict:
    """Record a human escalation handoff, update TicketMetadata status to 'escalated', and log an internal system note."""
    return record_escalation(engine, target_team, priority, reason, summary, ticket_id)


# v1 fixed for rubric
def record_escalation(
    db: Engine, target_team: EscalationTeam, priority: EscalationPriority,
    reason: str, summary: str, ticket_id: Optional[str],
) -> dict:
    escalation_id = f"ESC-{str(uuid.uuid4())[:6].upper()}"

    # Step 1: If no ticket_id was supplied, return the structured handoff payload directly.
    if not ticket_id:
        return {
            "escalation_id": escalation_id,
            "ticket_id": None,
            "target_team": target_team,
            "priority": priority,
            "status": "failed",
            "db_updated": False,
            "error": "A ticket ID is required to save a handoff.",
        }

    with get_session(db) as session:
        # Step 2: Check whether the ticket exists in udahub.db.
        ticket = (
            session.query(udahub.Ticket)
            .filter(udahub.Ticket.ticket_id == ticket_id)
            .first()
        )
        if not ticket:
            return {
                "escalation_id": escalation_id,
                "ticket_id": ticket_id,
                "target_team": target_team,
                "priority": priority,
                "status": "failed",
                "db_updated": False,
                "error": "The ticket record does not exist.",
            }

        # Step 3: Update TicketMetadata status and tags.
        meta = ticket.ticket_metadata
        if meta is None:
            return {"ticket_id": ticket_id, "status": "failed", "db_updated": False,
                    "error": "The ticket metadata does not exist."}
        meta.status = "escalated"
        meta.main_issue_type = target_team
        existing_tags = (tag.strip() for tag in (meta.tags or "").split(","))
        tags = [tag for tag in existing_tags if tag and not tag.startswith(("escalated:", "priority:"))]
        meta.tags = ",".join([*tags, f"escalated:{target_team}", f"priority:{priority}"])

        # Step 4: Log a system handoff note in TicketMessage for the human agent.
        handoff_note = (
            f"[{escalation_id}] Escalated to {target_team} (priority={priority}). "
            f"Reason: {reason} | Summary: {summary}"
        )
        session.add(
            udahub.TicketMessage(
                message_id=str(uuid.uuid4()),
                ticket_id=ticket_id,
                role=udahub.RoleEnum.system,
                content=handoff_note,
            )
        )

        return {
            "escalation_id": escalation_id,
            "ticket_id": ticket_id,
            "target_team": target_team,
            "priority": priority,
            "status": "escalated",
            "db_updated": True,
            "message": f"Ticket '{ticket_id}' escalated to {target_team} ({priority}).",
        }
