from pathlib import Path
from typing import Optional
from sqlalchemy import create_engine
from fastmcp import FastMCP

from utils import get_session
from data.models import cultpass

# Initialize the FastMCP server for all external CultPass operations.
mcp = FastMCP("CultPassMCP")

# Locate cultpass.db relative to solution/ so the MCP server works from any CWD.
SOLUTION_DIR = Path(__file__).resolve().parents[2]
CULTPASS_DB_PATH = SOLUTION_DIR / "data" / "external" / "cultpass.db"
engine = create_engine(f"sqlite:///{CULTPASS_DB_PATH}", echo=False)


def _find_user(session, user_id_or_email: str):
    """Look up a CultPass user by 6-char user_id or email."""
    return (
        session.query(cultpass.User)
        .filter(
            (cultpass.User.user_id == user_id_or_email)
            | (cultpass.User.email == user_id_or_email)
        )
        .first()
    )


# ---------------------------------------------------------------------------
# 1. Account Access Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def get_user_profile(user_id_or_email: str) -> dict:
    """Fetch a CultPass user's profile and whether their account is blocked."""
    with get_session(engine) as session:
        user = _find_user(session, user_id_or_email)
        if not user:
            return {"error": f"User '{user_id_or_email}' not found in CultPass."}

        return {
            "user_id": user.user_id,
            "full_name": user.full_name,
            "email": user.email,
            "is_blocked": user.is_blocked,
            "created_at": str(user.created_at),
        }


@mcp.tool()
def unblock_user_account(
    user_id_or_email: str,
    reason: str,
    risk_score: float = 0.5,
) -> dict:
    """Unblock a CultPass user if risk_score <= 0.4; otherwise require human escalation."""
    # Threshold gate: high-risk unblocks must go through human verification.
    if risk_score > 0.4:
        return {
            "approved": False,
            "requires_escalation": True,
            "reason": f"Risk score {risk_score} exceeds auto-unblock threshold (0.4). Escalate to Account Access team.",
        }

    with get_session(engine) as session:
        user = _find_user(session, user_id_or_email)
        if not user:
            return {"error": f"User '{user_id_or_email}' not found in CultPass."}

        if not user.is_blocked:
            return {
                "approved": True,
                "user_id": user.user_id,
                "is_blocked": False,
                "message": "Account is already active (not blocked).",
            }

        user.is_blocked = False
        return {
            "approved": True,
            "requires_escalation": False,
            "user_id": user.user_id,
            "is_blocked": False,
            "reason": reason,
        }


# ---------------------------------------------------------------------------
# 2. Experience & Reservation Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def list_upcoming_experiences(
    location: Optional[str] = None,
    include_premium: bool = True,
) -> list[dict]:
    """List available CultPass experiences with open slots, optionally filtered by location."""
    with get_session(engine) as session:
        query = session.query(cultpass.Experience).filter(
            cultpass.Experience.slots_available > 0
        )
        if location:
            query = query.filter(cultpass.Experience.location.ilike(f"%{location}%"))
        if not include_premium:
            query = query.filter(cultpass.Experience.is_premium.is_(False))

        experiences = query.order_by(cultpass.Experience.when.asc()).all()
        return [
            {
                "experience_id": exp.experience_id,
                "title": exp.title,
                "description": exp.description,
                "location": exp.location,
                "when": str(exp.when),
                "slots_available": exp.slots_available,
                "is_premium": exp.is_premium,
            }
            for exp in experiences
        ]


@mcp.tool()
def get_user_reservations(user_id_or_email: str) -> list[dict]:
    """Retrieve all reservations (and linked experience details) for a CultPass user."""
    with get_session(engine) as session:
        user = _find_user(session, user_id_or_email)
        if not user:
            return [{"error": f"User '{user_id_or_email}' not found in CultPass."}]

        return [
            {
                "reservation_id": res.reservation_id,
                "status": res.status,
                "experience_id": res.experience_id,
                "title": res.experience.title if res.experience else None,
                "location": res.experience.location if res.experience else None,
                "when": str(res.experience.when) if res.experience else None,
                "is_premium": res.experience.is_premium if res.experience else None,
            }
            for res in user.reservations
        ]


# ---------------------------------------------------------------------------
# 3. Cancellation & Threshold-Gated Refund Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def cancel_reservation(reservation_id: str) -> dict:
    """Cancel an active ('reserved') CultPass reservation and restore its experience slot."""
    with get_session(engine) as session:
        res = (
            session.query(cultpass.Reservation)
            .filter(cultpass.Reservation.reservation_id == reservation_id)
            .first()
        )
        if not res:
            return {"error": f"Reservation '{reservation_id}' not found."}

        if res.status != "reserved":
            return {
                "error": f"Reservation '{reservation_id}' cannot be cancelled (current status: '{res.status}')."
            }

        res.status = "cancelled"
        if res.experience:
            res.experience.slots_available += 1

        return {
            "reservation_id": res.reservation_id,
            "status": res.status,
            "experience_title": res.experience.title if res.experience else None,
            "message": "Reservation cancelled and slot restored.",
        }


@mcp.tool()
def process_refund(
    user_id_or_email: str,
    amount: float,
    reason: str,
    auto_approve_limit: float = 50.0,
) -> dict:
    """Process a refund if amount <= auto_approve_limit ($50); otherwise require escalation."""
    if amount <= 0:
        return {"error": "Refund amount must be greater than 0."}

    with get_session(engine) as session:
        user = _find_user(session, user_id_or_email)
        if not user:
            return {"error": f"User '{user_id_or_email}' not found in CultPass."}

        # Threshold gate: refunds above the policy limit require human approval.
        if amount > auto_approve_limit:
            return {
                "approved": False,
                "requires_escalation": True,
                "user_id": user.user_id,
                "amount": amount,
                "reason": (
                    f"Refund of ${amount:.2f} exceeds auto-approval limit "
                    f"(${auto_approve_limit:.2f}). Escalate to Billing/Refund lead."
                ),
            }

        return {
            "approved": True,
            "requires_escalation": False,
            "user_id": user.user_id,
            "email": user.email,
            "amount": amount,
            "reason": reason,
            "message": f"Refund of ${amount:.2f} auto-approved.",
        }


if __name__ == "__main__":
    # Run via stdio transport so LangChain's MultiServerMCPClient can launch it.
    mcp.run()
