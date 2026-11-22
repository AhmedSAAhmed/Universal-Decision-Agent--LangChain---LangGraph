import json
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool

# Locate the synthetic incident feed relative to solution/ (replaces live incident.co API).
SOLUTION_DIR = Path(__file__).resolve().parents[2]
OUTAGES_FEED_PATH = SOLUTION_DIR / "data" / "external" / "outages.json"


def _load_feed() -> dict:
    """Load the synthetic incident.co JSON feed from disk."""
    if not OUTAGES_FEED_PATH.exists():
        return {"provider": "incident.co (synthetic)", "incidents": []}
    with open(OUTAGES_FEED_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@tool
def get_outage_status(service: Optional[str] = None) -> dict:
    """Check active CultPass service outages and ETAs (optionally filtered by service name like 'booking-api', 'payments', or 'auth')."""
    feed = _load_feed()
    incidents = feed.get("incidents", [])

    # Step 1: Filter to active (non-resolved) incidents.
    active = [inc for inc in incidents if inc.get("status") != "resolved"]

    # Step 2: Narrow by service name if provided.
    if service:
        needle = service.strip().lower()
        active = [
            inc for inc in active
            if needle in inc.get("service", "").lower()
            or needle in inc.get("title", "").lower()
        ]

    # Step 3: Return active incidents or an all-clear operational summary.
    if not active:
        return {
            "provider": feed.get("provider"),
            "overall_status": "operational",
            "active_incidents": [],
            "message": f"No active outages found{' for ' + service if service else ''}.",
        }

    return {
        "provider": feed.get("provider"),
        "overall_status": "degraded",
        "updated_at": feed.get("updated_at"),
        "active_incidents": active,
    }


@tool
def list_all_incidents(include_resolved: bool = False) -> list[dict]:
    """List incidents from the synthetic status feed, optionally including recently resolved ones."""
    feed = _load_feed()
    incidents = feed.get("incidents", [])

    if include_resolved:
        return incidents
    return [inc for inc in incidents if inc.get("status") != "resolved"]
