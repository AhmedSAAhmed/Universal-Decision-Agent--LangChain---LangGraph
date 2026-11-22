# UDA-Hub Tools (`agentic/tools`)

This folder contains all reusable tools for the UDA-Hub multi-agent system. Tools are intentionally decoupled from agent prompts and LangGraph routing logic so they can be tested independently and shared across agents.

## Planned Modules

| Module | Backing Source | Responsibilities | Used By |
| --- | --- | --- | --- |
| `cultpass_mcp.py` | `data/external/cultpass.db` (`data/models/cultpass.py`) | **FastMCP** server exposing CultPass `User`, `Subscription`, `Experience`, and `Reservation` lookups, plus threshold-gated cancellations, refunds, and access checks. | Billing, Reservation, Refund, Access Agents |
| `outage_tools.py` | Synthetic incident feed (`data/external/outages.json` or MCP) | Simulates `incident.co` / status-page lookups (`get_outage_status`) for active incidents, severity, and latest updates. | Outage Agent |
| `kb_tools.py` | `udahub.Knowledge` + **ChromaDB** | Embeds `Knowledge` articles (`title + content + tags`) into ChromaDB and exposes `semantic_kb_search` for RAG retrieval. | General QA, Billing, Refund Agents |
| `ticket_tools.py` | `data/core/udahub.db` (`data/models/udahub.py`) | Queries `Ticket` / `TicketMessage` history, updates `TicketMetadata` (`status`, `main_issue_type`, `tags`), and logs A/B variant outcomes. | Ticketing, Escalation, Metrics Logger |

## Design Guidelines

- **Typed & Self-Documenting:** Every tool uses clear type annotations and concise docstrings so LangChain/FastMCP schemas are accurate for LLM tool-calling.
- **No Routing Logic:** Tools perform data retrieval or validated mutations only; decisions and escalation routing belong in `agentic/agents` and `agentic/workflow.py`.
- **Threshold Enforcement:** Mutating tools (e.g., refunds, cancellations, account unblocking) validate policy thresholds and return structured approval/escalation signals rather than bypassing guardrails.
