# import packages
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Optional
from sqlalchemy import Engine, create_engine
from langchain_core.tools import tool
from utils import get_session
from data.models import udahub

SOLUTION_DIR = Path(__file__).resolve().parents[2]
UDAHUB_DB_PATH = SOLUTION_DIR / "data" / "core" / "udahub.db"
engine = create_engine(f"sqlite:///{UDAHUB_DB_PATH}", echo=False)

# v1 fixed for rubric
_active_engine: ContextVar[Engine | None] = ContextVar("ticket_tool_engine", default=None)


@contextmanager
def use_ticket_engine(db: Engine):
    token = _active_engine.set(db)
    try:
        yield
    finally:
        _active_engine.reset(token)


def _ticket_engine() -> Engine:
    return _active_engine.get() or engine

# Getting ticket details form the database
@tool
def get_ticket_details(ticket_id: str) -> dict:
    """Retrieve a ticket's metadata, channel, user info, and full message history."""
    with get_session(_ticket_engine()) as session:
        ticket = (
            session.query(udahub.Ticket).filter(udahub.Ticket.ticket_id == ticket_id)
            .first()
        )
        if not ticket:
            return {
                "error": f"Ticket '{ticket_id}' not found."
            }

        meta = ticket.ticket_metadata
        messages = (
            session.query(udahub.TicketMessage)
            .filter(udahub.TicketMessage.ticket_id == ticket_id)
            .order_by(udahub.TicketMessage.created_at.asc())
            .all()
        )

        return {
            "ticket_id": ticket.ticket_id,
            "account_id": ticket.account_id,
            "user_id": ticket.user_id,
            "user_name": ticket.user.user_name if ticket.user else None,
            "channel": ticket.channel,
            "created_at": str(ticket.created_at),
            "status": meta.status if meta else "unknown",
            "main_issue_type": meta.main_issue_type if meta else None,
            "tags": meta.tags if meta else None,
            "messages": [{
                "role": msg.role.value,
                "content": msg.content,
                "created_at": str(msg.created_at)
            }
            for msg in messages
            ],
        }

# Getting user details form the database
@tool
def get_user_tickets(user_id_or_external: str)->list[dict]:
    """list all support tickets belongs to a user_id or external_user_id"""
    with get_session(_ticket_engine()) as session:
        user = (
            session.query(udahub.User)
            .filter(
            (udahub.User.user_id == user_id_or_external)
            | (udahub.User.external_user_id == user_id_or_external)
            )
            .first()
        )
        if not user:
            return [{
                "error": f"No user found matching'{user_id_or_external}'."
            }]
        return [{
            "ticket_id": t.ticket_id,
            "channel": t.channel,
            "status": t.ticket_metadata.status if t.ticket_metadata else "unknown",
            "main_issue_type": t.ticket_metadata.main_issue_type if t.ticket_metadata else None,
            "created_at": str(t.created_at),
        } for t in user.tickets
        ]

# Getting update ticket status details into the database
@tool
def update_ticket_metadata(
    ticket_id: str,
    status: str,
    main_issue_type: Optional[str] = None,
    tags: Optional[str] = None,
) -> dict:
    """ Update a ticket status (e.g. open, resolved, escalated), issue type, or tags """
    with get_session(_ticket_engine()) as session:
        meta = (
            session.query(udahub.TicketMetadata)
            .filter(udahub.TicketMetadata.ticket_id == ticket_id)
            .first()
        )
        if not meta:
            return {"error": f"Metadata for ticket '{ticket_id}' not found"}

        meta.status = status
        if main_issue_type is not None:
            meta.main_issue_type = main_issue_type
        if tags is not None:
            meta.tags = tags

        return {
            "ticket_id": ticket_id,
            "status": meta.status,
            "main_issue_type": meta.main_issue_type,
            "tags": meta.tags
        }

# adding ticket message to the after the new updated status
@tool
def add_ticket_message(ticket_id: str, role: str, content: str) -> dict:
    """Append a new message (role: 'user', 'ai', 'agent', or 'system') to a ticket."""
    with get_session(_ticket_engine()) as session:
        ticket = (
            session.query(udahub.Ticket)
            .filter(udahub.Ticket.ticket_id == ticket_id)
            .first()
        )
        if not ticket:
            return {"error": f"Ticket '{ticket_id}' not found."}
        try:
            role_enum = udahub.RoleEnum(role)
        except ValueError:
            return {"error": f"Invalid role '{role}'. Must be user, ai, agent, or system."}
        msg = udahub.TicketMessage(
            message_id=str(uuid.uuid4()),
            ticket_id=ticket_id,
            role=role_enum,
            content=content,
        )
        session.add(msg)
        return {"message_id": msg.message_id, "ticket_id": ticket_id, "role": role}
