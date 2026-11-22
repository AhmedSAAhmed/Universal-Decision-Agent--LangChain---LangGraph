# Universal Decision Agent

A LangGraph customer support workflow for the sample CultPass service. It classifies tickets, routes them to specialist agents, answers policy questions from a knowledge base, and records resolutions and human handoffs. SQLite stores ticket records, audit events, and customer memory.

See the [architecture and routing rules](solution/agentic/design/README.md).

For grading, open the [review evidence and source excerpts](SUBMISSION.md). The setup and app notebooks also retain their verification outputs.

## Setup

1. Create a Python 3.12 virtual environment and install the notebook dependencies.

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   Copy-Item .env.example .env
   ```

2. Set `OPENAI_API_KEY` in `.env` for live model calls. Set `OPENAI_BASE_URL` if you use an OpenAI-compatible endpoint. The routing, end-to-end, and memory demos use controlled model responses and run without a key.
3. Select the virtual environment as the notebook kernel. Run the notebooks with `solution/` as the working directory.

   1. [External database setup](solution/01_external_db_setup.ipynb) creates the sample CultPass database.
   2. [Core database setup](solution/02_core_db_setup.ipynb) creates the ticket database and seeds the support articles.
   3. [Agentic app](solution/03_agentic_app.ipynb) runs the routing, integration, knowledge retrieval, and memory demos. Run the interactive chat cell to submit a ticket.

The setup notebooks recreate their databases. On a fresh checkout, the live knowledge search builds the Chroma index on first use. The repository contains seed inputs and saved demo outputs. Generated databases, indexes, and local credentials are ignored.
