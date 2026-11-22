from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine
from langchain_core.tools import tool

from utils import get_session
from data.models import cultpass

# Locate the external CultPass SQLite database relative to solution/
# so this module works regardless of the current working directory.
SOLUTION_DIR = Path(__file__).resolve().parents[2]
CULTPASS_DB_PATH = SOLUTION_DIR / "data" / "external" / "cultpass.db"
engine = create_engine(f"sqlite:///{CULTPASS_DB_PATH}", echo=False)


def _find_user(session, user_id_or_email: str):
    """Find a CultPass user by either their 6-character user_id or email address."""
    return (
        session.query(cultpass.User)
        .filter(
            (cultpass.User.user_id == user_id_or_email)
            | (cultpass.User.email == user_id_or_email)
        )
        .first()
    )


@tool
def cancel_reservation(reservation_id: str) -> dict:
    """Cancel an active ('reserved') CultPass reservation and restore its experience slot."""
    with get_session(engine) as session:
        # Step 1: Look up the reservation by ID.
        res = (
            session.query(cultpass.Reservation)
            .filter(cultpass.Reservation.reservation_id == reservation_id)
            .first()
        )
        if not res:
            return {"error": f"Reservation '{reservation_id}' not found."}

        # Step 2: Ensure only active ('reserved') bookings can be cancelled.
        if res.status != "reserved":
            return {
                "error": f"Reservation '{reservation_id}' cannot be cancelled (current status: '{res.status}')."
            }

        # Step 3: Cancel the reservation and restore 1 slot on the linked experience.
        res.status = "cancelled"
        if res.experience:
            res.experience.slots_available += 1

        return {
            "reservation_id": res.reservation_id,
            "status": res.status,
            "experience_title": res.experience.title if res.experience else None,
            "slots_available": res.experience.slots_available if res.experience else None,
            "message": "Reservation cancelled and experience slot restored.",
        }


@tool
def cancel_subscription(user_id_or_email: str) -> dict:
    """Cancel an active CultPass subscription (effective at the end of the billing cycle)."""
    with get_session(engine) as session:
        # Step 1: Locate the user and their subscription.
        user = _find_user(session, user_id_or_email)
        if not user:
            return {"error": f"User '{user_id_or_email}' not found in CultPass."}

        sub = user.subscription
        if not sub:
            return {"error": f"No subscription found for '{user.email}'."}

        # Step 2: Check whether the subscription is already cancelled.
        if sub.status == "cancelled":
            return {
                "subscription_id": sub.subscription_id,
                "status": sub.status,
                "message": "Subscription is already cancelled.",
            }

        # Step 3: Mark the subscription as cancelled and record ended_at timestamp.
        sub.status = "cancelled"
        sub.ended_at = datetime.now()

        return {
            "subscription_id": sub.subscription_id,
            "user_id": user.user_id,
            "status": sub.status,
            "ended_at": str(sub.ended_at),
            "message": "Subscription cancelled. Access continues until the end of the current billing cycle.",
        }


@tool
def process_refund(
    user_id_or_email: str,
    amount: float,
    reason: str,
    auto_approve_limit: float = 50.0,
) -> dict:
    """Process a refund if amount <= $50.00; otherwise return an escalation signal."""
    if amount <= 0:
        return {"error": "Refund amount must be greater than 0."}

    with get_session(engine) as session:
        # Step 1: Verify the customer exists in CultPass.
        user = _find_user(session, user_id_or_email)
        if not user:
            return {"error": f"User '{user_id_or_email}' not found in CultPass."}

        # Step 2: Threshold gate — refunds above $50 require human billing approval.
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

        # Step 3: Auto-approve refunds at or below the policy threshold.
        return {
            "approved": True,
            "requires_escalation": False,
            "user_id": user.user_id,
            "email": user.email,
            "amount": amount,
            "reason": reason,
            "message": f"Refund of ${amount:.2f} auto-approved.",
        }
