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
def get_account_access_status(user_id_or_email: str) -> dict:
    """Check a CultPass user's account profile, whether is_blocked is True, and subscription status."""
    with get_session(engine) as session:
        # Step 1: Look up the user in CultPass.
        user = _find_user(session, user_id_or_email)
        if not user:
            return {"error": f"User '{user_id_or_email}' not found in CultPass."}

        # Step 2: Return access/block state alongside subscription status.
        return {
            "user_id": user.user_id,
            "full_name": user.full_name,
            "email": user.email,
            "is_blocked": user.is_blocked,
            "subscription_status": user.subscription.status if user.subscription else None,
            "created_at": str(user.created_at),
        }


@tool
def unblock_user_account(
    user_id_or_email: str,
    reason: str,
    risk_score: float = 0.5,
) -> dict:
    """Unblock a CultPass user if risk_score <= 0.4; otherwise require human escalation."""
    # Step 1: Enforce the risk threshold gate before modifying account access.
    if risk_score > 0.4:
        return {
            "approved": False,
            "requires_escalation": True,
            "reason": (
                f"Risk score {risk_score:.2f} exceeds auto-unblock threshold (0.40). "
                "Escalate to the Account Access team for identity verification."
            ),
        }

    with get_session(engine) as session:
        # Step 2: Locate the user and verify they are currently blocked.
        user = _find_user(session, user_id_or_email)
        if not user:
            return {"error": f"User '{user_id_or_email}' not found in CultPass."}

        if not user.is_blocked:
            return {
                "approved": True,
                "requires_escalation": False,
                "user_id": user.user_id,
                "is_blocked": False,
                "message": "Account is already active (not blocked).",
            }

        # Step 3: Unblock the account; get_session() commits on exit.
        user.is_blocked = False
        return {
            "approved": True,
            "requires_escalation": False,
            "user_id": user.user_id,
            "email": user.email,
            "is_blocked": False,
            "reason": reason,
        }


@tool
def update_user_email(user_id_or_email: str, new_email: str) -> dict:
    """Update the registered email address for an unblocked CultPass user."""
    with get_session(engine) as session:
        # Step 1: Verify the user exists and is not blocked.
        user = _find_user(session, user_id_or_email)
        if not user:
            return {"error": f"User '{user_id_or_email}' not found in CultPass."}

        if user.is_blocked:
            return {
                "approved": False,
                "requires_escalation": True,
                "error": f"Cannot update email while account '{user.email}' is blocked.",
            }

        # Step 2: Ensure the new email is not already in use by another account.
        existing = (
            session.query(cultpass.User)
            .filter(cultpass.User.email == new_email)
            .first()
        )
        if existing and existing.user_id != user.user_id:
            return {"error": f"Email '{new_email}' is already registered to another account."}

        # Step 3: Update the email; get_session() commits on exit.
        old_email = user.email
        user.email = new_email
        return {
            "user_id": user.user_id,
            "old_email": old_email,
            "new_email": user.email,
            "message": "Registered email updated successfully.",
        }
