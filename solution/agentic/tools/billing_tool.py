from pathlib import Path
from typing import Literal
from sqlalchemy import create_engine
from langchain_core.tools import tool

from utils import get_session
from data.models import cultpass

# Locate the external CultPass SQLite database relative to the solution/ folder
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
def get_subscription_details(user_id_or_email: str) -> dict:
    """Retrieve a CultPass user's subscription tier, status, monthly quota, and used reservations."""
    with get_session(engine) as session:
        # Step 1: Look up the customer in CultPass using their ID or email.
        user = _find_user(session, user_id_or_email)
        if not user:
            return {"error": f"User '{user_id_or_email}' not found in CultPass."}

        # Step 2: Check whether the user has an associated subscription record.
        sub = user.subscription
        if not sub:
            return {
                "user_id": user.user_id,
                "email": user.email,
                "error": "No subscription found for this user.",
            }

        # Step 3: Count active/completed reservations to calculate remaining monthly quota.
        # Cancelled reservations do not count against the user's quota.
        used_count = (
            session.query(cultpass.Reservation)
            .filter(
                (cultpass.Reservation.user_id == user.user_id)
                & (cultpass.Reservation.status.in_(["reserved", "completed"]))
            )
            .count()
        )

        # Step 4: Return a structured billing summary for the Billing Agent.
        return {
            "user_id": user.user_id,
            "full_name": user.full_name,
            "email": user.email,
            "is_blocked": user.is_blocked,
            "subscription_id": sub.subscription_id,
            "status": sub.status,
            "tier": sub.tier,
            "monthly_quota": sub.monthly_quota,
            "reservations_used": used_count,
            "quota_remaining": max(0, sub.monthly_quota - used_count),
            "started_at": str(sub.started_at),
            "ended_at": str(sub.ended_at) if sub.ended_at else None,
        }


@tool
def change_subscription_tier(
    user_id_or_email: str,
    new_tier: Literal["basic", "premium"],
) -> dict:
    """Upgrade or downgrade an active CultPass subscription between 'basic' and 'premium'."""
    # Standard monthly experience allowances per CultPass plan tier.
    tier_quotas = {"basic": 4, "premium": 10}

    with get_session(engine) as session:
        # Step 1: Locate the customer and verify they have a subscription.
        user = _find_user(session, user_id_or_email)
        if not user:
            return {"error": f"User '{user_id_or_email}' not found in CultPass."}

        sub = user.subscription
        if not sub:
            return {"error": f"No subscription found for '{user_id_or_email}'."}

        # Step 2: Only active subscriptions can be upgraded or downgraded.
        if sub.status != "active":
            return {
                "error": f"Cannot change tier because subscription '{sub.subscription_id}' is '{sub.status}'."
            }

        # Step 3: Avoid unnecessary writes if the user is already on the requested tier.
        if sub.tier == new_tier:
            return {
                "subscription_id": sub.subscription_id,
                "tier": sub.tier,
                "message": f"Subscription is already on the '{new_tier}' tier.",
            }

        # Step 4: Update the tier and adjust monthly quota; get_session() commits on exit.
        old_tier = sub.tier
        sub.tier = new_tier
        sub.monthly_quota = tier_quotas[new_tier]

        return {
            "subscription_id": sub.subscription_id,
            "user_id": user.user_id,
            "old_tier": old_tier,
            "new_tier": sub.tier,
            "new_monthly_quota": sub.monthly_quota,
            "proration_note": "Prorated difference will be applied on the current billing cycle.",
        }
