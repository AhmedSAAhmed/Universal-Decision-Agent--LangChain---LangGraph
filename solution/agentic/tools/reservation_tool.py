import uuid
from pathlib import Path
from typing import Optional
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
def list_upcoming_experiences(
    location: Optional[str] = None,
    include_premium: bool = True,
) -> list[dict]:
    """List CultPass experiences that still have open slots, optionally filtered by location."""
    with get_session(engine) as session:
        # Step 1: Start with experiences that have at least 1 slot left.
        query = session.query(cultpass.Experience).filter(
            cultpass.Experience.slots_available > 0
        )

        # Step 2: Apply optional location and premium filters.
        if location:
            query = query.filter(cultpass.Experience.location.ilike(f"%{location}%"))
        if not include_premium:
            query = query.filter(cultpass.Experience.is_premium.is_(False))

        # Step 3: Return results sorted by soonest date first.
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


@tool
def get_user_reservations(user_id_or_email: str) -> list[dict]:
    """Retrieve all reservations (and linked experience details) for a CultPass user."""
    with get_session(engine) as session:
        # Step 1: Locate the customer in CultPass.
        user = _find_user(session, user_id_or_email)
        if not user:
            return [{"error": f"User '{user_id_or_email}' not found in CultPass."}]

        # Step 2: Return each reservation joined with its experience metadata.
        if not user.reservations:
            return [{"message": f"No reservations found for '{user.email}'."}]

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


@tool
def book_experience(user_id_or_email: str, experience_id: str) -> dict:
    """Book an open CultPass experience for an active, unblocked user with remaining quota."""
    with get_session(engine) as session:
        # Step 1: Verify the user exists and is not blocked.
        user = _find_user(session, user_id_or_email)
        if not user:
            return {"error": f"User '{user_id_or_email}' not found in CultPass."}
        if user.is_blocked:
            return {"error": f"User '{user.email}' is currently blocked and cannot book."}

        # Step 2: Verify the user has an active subscription and remaining monthly quota.
        sub = user.subscription
        if not sub or sub.status != "active":
            return {"error": "An active CultPass subscription is required to book."}

        used_count = (
            session.query(cultpass.Reservation)
            .filter(
                (cultpass.Reservation.user_id == user.user_id)
                & (cultpass.Reservation.status.in_(["reserved", "completed"]))
            )
            .count()
        )
        if used_count >= sub.monthly_quota:
            return {
                "error": f"Monthly quota reached ({used_count}/{sub.monthly_quota} used)."
            }

        # Step 3: Verify the experience exists and still has slots available.
        exp = (
            session.query(cultpass.Experience)
            .filter(cultpass.Experience.experience_id == experience_id)
            .first()
        )
        if not exp:
            return {"error": f"Experience '{experience_id}' not found."}
        if exp.slots_available <= 0:
            return {"error": f"Experience '{exp.title}' has no slots available."}

        # Step 4: Create the reservation and decrement available slots; get_session() commits.
        new_res = cultpass.Reservation(
            reservation_id=str(uuid.uuid4())[:6],
            user_id=user.user_id,
            experience_id=exp.experience_id,
            status="reserved",
        )
        exp.slots_available -= 1
        session.add(new_res)

        return {
            "reservation_id": new_res.reservation_id,
            "user_id": user.user_id,
            "experience_id": exp.experience_id,
            "title": exp.title,
            "when": str(exp.when),
            "status": new_res.status,
            "slots_remaining": exp.slots_available,
        }
