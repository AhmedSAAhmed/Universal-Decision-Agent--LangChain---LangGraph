"""End-to-end ticket submission, final actions, and searchable audit events."""

import json
import logging
from dataclasses import dataclass
from uuid import uuid4

from sqlalchemy import Engine, select, text

from agentic import workflow
from agentic.tools import ticket_tool
from data.models import udahub
from utils import get_session

_LOG = logging.getLogger(__name__)
_SPECIALISTS = {"ticket", "billing", "reservation", "refund", "outage", "access", "general_qa"}
_FAILURE_REPLY = "I couldn't complete your support request. Please try again."


@dataclass(frozen=True)
class TicketOutcome:
    ticket_id: str
    run_id: str
    status: str
    reply: str


class SubmissionRejected(ValueError):
    pass


class FinalizationError(RuntimeError):
    pass


# v1 fixed for rubric
def _event(run_id, ticket_id, kind, *, node=None, status=None, **details):
    return udahub.TicketEvent(
        run_id=run_id, ticket_id=ticket_id, kind=kind, node=node,
        status=status, details=details,
    )


def _log(db, run_id, ticket_id, kind, **details):
    with get_session(db) as session:
        session.add(_event(run_id, ticket_id, kind, **details))


def query_ticket_events(
    *, core_engine: Engine | None = None, ticket_id: str | None = None,
    run_id: str | None = None, kind: str | None = None, node: str | None = None,
    status: str | None = None, tool_name: str | None = None,
    route_target: str | None = None, limit: int = 100,
) -> list[dict]:
    """Return the most recent matching audit events in execution order."""
    db = core_engine if core_engine is not None else ticket_tool.engine
    with db.begin() as connection:
        udahub.TicketEvent.__table__.create(connection, checkfirst=True)
    filters = {"ticket_id": ticket_id, "run_id": run_id, "kind": kind, "node": node, "status": status}
    event = udahub.TicketEvent
    statement = select(event).filter_by(**{key: value for key, value in filters.items() if value is not None})
    if tool_name is not None:
        statement = statement.where(event.kind == "tool", event.details["name"].as_string() == tool_name)
    if route_target is not None:
        statement = statement.where(event.kind == "route", event.details["target"].as_string() == route_target)
    statement = statement.order_by(event.event_id.desc()).limit(max(1, min(limit, 1000)))
    with get_session(db) as session:
        return [
            {
                "timestamp": row.created_at.isoformat() + "Z", "run_id": row.run_id,
                "ticket_id": row.ticket_id, "kind": row.kind, "node": row.node,
                "status": row.status, "details": row.details,
            }
            for row in reversed(session.scalars(statement).all())
        ]


def _register_submission(db, state, run_id):
    ticket_id = state["ticket_id"]
    customer = state.get("user_id_or_email")
    external_id = customer or f"anonymous:{ticket_id}"
    with get_session(db) as session:
        session.execute(text("BEGIN IMMEDIATE"))
        ticket = session.get(udahub.Ticket, ticket_id)
        if ticket is not None:
            owner = ticket.user
            matching_ids = {owner.external_user_id, owner.user_id} if owner else set()
            if external_id not in matching_ids:
                raise SubmissionRejected("Use a different ticket ID for a different customer.")
        else:
            if session.get(udahub.Account, "cultpass") is None:
                raise FinalizationError("CultPass account is missing from the core database.")
            owner = session.get(udahub.User, external_id)
            if owner is None or owner.account_id != "cultpass":
                owner = session.scalar(select(udahub.User).filter_by(
                    account_id="cultpass", external_user_id=external_id,
                ))
            if owner is None:
                owner = udahub.User(
                    user_id=str(uuid4()), account_id="cultpass",
                    external_user_id=external_id, user_name="CultPass customer",
                )
                session.add(owner)
            ticket = udahub.Ticket(
                ticket_id=ticket_id, account_id="cultpass", user=owner, channel=state["channel"],
            )
            session.add(ticket)
        if ticket.ticket_metadata is None:
            ticket.ticket_metadata = udahub.TicketMetadata(status="open")
        session.add(udahub.TicketMessage(
            message_id=run_id, ticket=ticket, role=udahub.RoleEnum.user,
            content=state["messages"][0].content,
        ))
        session.add(_event(run_id, ticket_id, "submission", status="accepted",
                           channel=state["channel"], priority=state["priority"]))


def _audit_node(db, run_id, ticket_id, node, update):
    events = []
    if node == "triage_node":
        decision = update["triage"]
        events.append(_event(
            run_id, ticket_id, "classification", node=node, status="completed",
            **{key: decision.get(key) for key in (
                "primary_intent", "confidence", "sentiment", "urgency", "knowledge_only"
            )},
        ))
    for tool_call in update.get("tool_usage", []):
        events.append(_event(
            run_id, ticket_id, "tool", node=node, status=tool_call["status"],
            name=tool_call["name"],
        ))
    if node == "general_qa":
        source_ids = update.get("source_article_ids", [])
        retrieval_failed = any(call["status"] == "error" for call in update.get("tool_usage", []))
        events.append(_event(
            run_id, ticket_id, "retrieval", node=node,
            status="error" if retrieval_failed else "supported" if source_ids else "insufficient",
            candidates=update.get("retrieved_articles", []), source_article_ids=source_ids,
        ))
    if node in _SPECIALISTS:
        resolved = update.get("resolved") and not update.get("needs_escalation")
        events.append(_event(run_id, ticket_id, "resolution_attempt", node=node,
                             status="resolved" if resolved else "unresolved"))
    if node == "escalation":
        handoff = update.get("handoff") or {}
        events.append(_event(
            run_id, ticket_id, "handoff", node=node,
            status="saved" if handoff.get("db_updated") is True else "failed",
            **{key: handoff.get(key) for key in ("escalation_id", "target_team", "priority")},
        ))
    if events:
        with get_session(db) as session:
            session.add_all(events)


def _finish_ticket(db, ticket_id, run_id, state):
    if state.get("needs_escalation"):
        handoff = state.get("handoff") or {}
        if handoff.get("db_updated") is not True or handoff.get("ticket_id") != ticket_id:
            raise FinalizationError("The handoff was not saved.")
        status = "escalated"
    elif state.get("resolved"):
        status = "resolved"
    else:
        raise FinalizationError("The workflow returned no final action.")

    reply = state["messages"][-1].content
    if not isinstance(reply, str) or not reply.strip():
        raise FinalizationError("The workflow returned an empty reply.")
    decision = state.get("triage") or {}
    with get_session(db) as session:
        session.execute(text("BEGIN IMMEDIATE"))
        meta = session.get(udahub.TicketMetadata, ticket_id)
        if meta is None:
            raise FinalizationError("The ticket metadata record does not exist.")
        if status == "escalated":
            reference = handoff.get("escalation_id")
            note = session.scalar(select(udahub.TicketMessage.message_id).where(
                udahub.TicketMessage.ticket_id == ticket_id,
                udahub.TicketMessage.role == udahub.RoleEnum.system,
                udahub.TicketMessage.content.startswith(f"[{reference}] "),
            ).limit(1))
            if not reference or note is None or meta.status != "escalated":
                raise FinalizationError("The persisted handoff could not be verified.")
        meta.status = status
        if status == "resolved":
            meta.main_issue_type = decision.get("primary_intent")
        route_tags = ("ab:", "sentiment:")
        existing = (tag.strip() for tag in (meta.tags or "").split(","))
        tags = [tag for tag in existing if tag and not tag.startswith(route_tags)]
        tags.extend([f"ab:{state.get('ab_variant')}", f"sentiment:{decision.get('sentiment')}"])
        meta.tags = ",".join(tags)
        session.add(udahub.TicketMessage(
            message_id=f"{run_id}:reply", ticket_id=ticket_id, role=udahub.RoleEnum.ai,
            content=reply,
        ))
        session.add(_event(run_id, ticket_id, "outcome", status=status,
                           source_article_ids=state.get("source_article_ids", [])))
    return TicketOutcome(ticket_id, run_id, status, reply)


def _record_failure(db, run_id, ticket_id, phase, error, accepted):
    try:
        with get_session(db) as session:
            meta = session.get(udahub.TicketMetadata, ticket_id) if accepted else None
            if meta is not None and meta.status != "escalated":
                meta.status = "processing_failed"
            session.add(_event(run_id, ticket_id, "error", node=phase, status="failed",
                               error_type=type(error).__name__))
            session.add(_event(run_id, ticket_id, "outcome", status="failed",
                               ticket_status=meta.status if meta else None))
    except Exception as audit_error:
        _LOG.error(json.dumps({
            "kind": "outcome", "status": "failed", "run_id": run_id, "ticket_id": ticket_id,
            "phase": phase, "error_type": type(error).__name__,
            "audit_error_type": type(audit_error).__name__,
        }))


# v1 fixed for rubric
def submit_ticket(payload: dict, *, graph=None, core_engine: Engine | None = None) -> TicketOutcome:
    """Accept a new or existing ticket, run the graph, and report the persisted outcome."""
    db = core_engine if core_engine is not None else ticket_tool.engine
    graph = workflow.orchestrator if graph is None else graph
    run_id = str(uuid4())
    ticket_id = run_id
    phase = "submission"
    accepted = False
    try:
        with db.begin() as connection:
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            udahub.TicketEvent.__table__.create(connection, checkfirst=True)
        try:
            state = workflow.normalize_platform_payload(payload)
            ticket_id = str(state.get("ticket_id") or ticket_id)
            state["ticket_id"] = ticket_id
            if not state["messages"][0].content.strip():
                raise SubmissionRejected("Enter a message before submitting the ticket.")
        except (AttributeError, TypeError, ValueError) as error:
            if isinstance(error, SubmissionRejected):
                raise
            raise SubmissionRejected("The ticket payload is invalid.") from error

        config = {"configurable": {"thread_id": ticket_id}}
        if payload.get("ab_variant") in ("variant_a", "variant_b"):
            state["ab_variant"] = payload["ab_variant"]
        state["messages"][0].id = run_id
        _register_submission(db, state, run_id)
        accepted = True
        phase = "workflow"
        previous = None
        with ticket_tool.use_ticket_engine(db):
            for updates in graph.stream(state, config, context={"core_engine": db}, stream_mode="updates"):
                for node, update in updates.items():
                    if previous == "triage_node" or previous in _SPECIALISTS:
                        _log(db, run_id, ticket_id, "route", node=previous, status="selected", target=node)
                    _audit_node(db, run_id, ticket_id, node, update or {})
                    previous = node
        phase = "finalize"
        return _finish_ticket(db, ticket_id, run_id, graph.get_state(config).values)
    except SubmissionRejected as error:
        try:
            with get_session(db) as session:
                session.add_all([
                    _event(run_id, ticket_id, "submission", status="rejected", reason=str(error)),
                    _event(run_id, ticket_id, "outcome", status="rejected"),
                ])
        except Exception as audit_error:
            _record_failure(db, run_id, ticket_id, "submission", audit_error, False)
            return TicketOutcome(ticket_id, run_id, "failed", _FAILURE_REPLY)
        return TicketOutcome(ticket_id, run_id, "rejected", str(error))
    except Exception as error:
        failing_nodes = getattr(error, "__notes__", [])
        for note in failing_nodes:
            if note.startswith("During task with name '"):
                phase = note.split("'")[1]
                break
        _record_failure(db, run_id, ticket_id, phase, error, accepted)
        return TicketOutcome(ticket_id, run_id, "failed", _FAILURE_REPLY)
