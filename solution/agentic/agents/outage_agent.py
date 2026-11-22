import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import Field
from agentic.agents.tool_history import ToolTrackedResult
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from agentic.tools.outage_tool import (
    get_outage_status,
    list_all_incidents,
)

load_dotenv()


# Structured output returned by the Outage Agent so the workflow router
# knows whether the user's outage inquiry was answered or needs escalation.
class OutageResult(ToolTrackedResult):
    resolved: bool = Field(
        description="True if the outage status/ETA was clearly communicated to the user."
    )
    needs_escalation: bool = Field(
        description="True if this appears to be an unreported outage or requires engineering/on-call escalation."
    )
    response: str = Field(
        description="Clear, empathetic status update to send to the customer."
    )
    escalation_reason: Optional[str] = Field(
        default=None,
        description="Short explanation if needs_escalation is True.",
    )


OUTAGE_SYSTEM_PROMPT = """You are the Outage Specialist Agent for UDA-Hub (CultPass customer support).
Your responsibility is to check the synthetic incident feed for active service disruptions, share accurate status updates and ETAs, and escalate potential unreported outages.

1. Investigation Workflow:
- Always call `get_outage_status(service)` (or `get_outage_status()` with no filter) first to check for active incidents.
- If the user asks about a past or recently fixed issue, call `list_all_incidents(include_resolved=True)`.

2. Strict Guardrails:
- Never invent incident IDs, ETAs, affected regions, or root causes—only report what is returned by the status feed.
- Do not promise refunds or compensation (escalate if the user demands compensation for an outage).

3. Resolution vs. Escalation Rules (`OutageResult`):
- Set `resolved=True` and `needs_escalation=False` when:
  - An active or recently resolved incident matches the user's issue and you clearly share the status, latest update, and ETA.
- Set `resolved=False` and `needs_escalation=True` (with a specific `escalation_reason`) when:
  - The user reports widespread 500 errors / app downtime, but the status feed shows NO active incident (possible unreported outage requiring on-call engineering), OR
  - The user demands outage compensation or urgent manual intervention.

4. Response Style:
- Be empathetic, transparent, and concise. Include the `incident_id`, `status`, `eta_resolution`, and `latest_update` when an active incident exists."""


def outage_agent(
    rewritten_query: str,
    model: str = "gpt-5-mini",
) -> OutageResult:
    """Run the outage ReAct agent with outage_tool functions and return a structured OutageResult."""
    # Step 1: Load and validate OpenAI credentials.
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing OPENAI_API_KEY in environment.")
    base_url = os.getenv("base_url") or os.getenv("OPENAI_BASE_URL")

    # Step 2: Initialize the LLM and bind outage status tools.
    llm = ChatOpenAI(
        model=model, temperature=0, api_key=api_key, base_url=base_url
    )
    agent = create_react_agent(
        model=llm,
        tools=[
            get_outage_status,
            list_all_incidents,
        ],
        prompt=OUTAGE_SYSTEM_PROMPT,
        response_format=OutageResult,
    )

    # Step 3: Invoke the agent with the rewritten query and return OutageResult.
    result = agent.invoke({"messages": [HumanMessage(content=f"Request: {rewritten_query}")]})
    # v1 fixed for rubric
    return result["structured_response"].record_tool_messages(result["messages"])
