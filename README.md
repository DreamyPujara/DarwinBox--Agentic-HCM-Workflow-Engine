# Darwinbox Agentic HCM Workflow Engine

Multi-agent HR workflow prototype for the Darwinbox AI Engineering assignment. It includes an orchestrator, a grounded policy RAG agent, an action agent with structured mock HRIS tools, session memory, trace logging, token/cost accounting, and a React trace viewer.

## Stack Recommendation

This submission uses Python 3.10+ for the backend and a browser-loaded React UI. I would normally choose a TypeScript monorepo with Express plus Vite React, but this machine does not have Node installed. The Python stdlib implementation keeps the project runnable with zero dependency install while still delivering the requested React UI and Groq integration.

## Architecture

```text
React UI
  |
  | POST /api/chat
  v
HTTP Server
  |
  v
Orchestrator Agent
  |-- classifies route with compact session memory
  |
  |-- Policy Agent
  |     |-- chunks data/hr_policy.md
  |     |-- retrieves top-k grounded policy chunks
  |
  |-- Action Agent
        |-- parses leave intent
        |-- calls structured tools with retry
            |-- leave_balance_check(JSON)
            |-- leave_apply(JSON)

Response Agent
  |-- uses Groq openai/gpt-oss-20b when GROQ_API_KEY is set
  |-- deterministic fallback when the API is unavailable

Trace + Cost Summary
  |-- agent, action, input, output, latency
  |-- token estimate and naive all-LLM comparison
```

## Setup

1. Copy `.env.example` to `.env`.
2. Set:

```bash
GROQ_API_KEY=your_key
GROQ_MODEL=openai/gpt-oss-20b
```

3. Run:

```bash
python server.py
```

4. Open:

```text
http://127.0.0.1:8000
```

No package installation is required.

## Demo Prompts

- `What is our maternity leave policy?`
- `Apply for 3 days annual leave starting June 15`
- `What is my leave balance?`
- `Fetch the payroll policy for payslip corrections`

## Tool Schemas

```json
{
  "name": "leave_balance_check",
  "input": {
    "employee_id": "string",
    "leave_type": "annual | sick"
  },
  "output": {
    "ok": "boolean",
    "employee_id": "string",
    "leave_type": "string",
    "balance_days": "number",
    "error": "string?"
  }
}
```

```json
{
  "name": "leave_apply",
  "input": {
    "employee_id": "string",
    "leave_type": "annual | sick",
    "days": "number",
    "start": "date"
  },
  "output": {
    "ok": "boolean",
    "request_id": "string",
    "status": "PENDING_MANAGER_APPROVAL",
    "warning": "string?",
    "error": "string?"
  }
}
```

## Design Decisions

- Grounding first: policy answers are composed from retrieved chunks only. If no chunk matches, the system refuses to invent policy details.
- Cost control: the orchestrator routes requests before LLM synthesis, sends only top-k policy chunks, and handles tool execution deterministically. The UI compares this against a naive baseline that sends the full policy and full history to the LLM each turn.
- Resilience: mock HRIS actions use retry for transient failures and return structured fallback errors.
- State: sessions persist in memory for the server process, allowing follow-up leave requests to reuse pending fields.

## Production Scale Notes

- Replace lexical retrieval with a managed vector store and audited embedding model.
- Move sessions to Redis or the Darwinbox conversation store.
- Add auth, employee scoping, policy versioning, and permission checks before tool execution.
- Stream trace events over SSE/WebSockets for true live step-by-step rendering.
- Store traces in OpenTelemetry-compatible spans for search and replay.
