from __future__ import annotations

import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
POLICY_PATH = ROOT / "data" / "hr_policy.md"
STATIC_PATH = ROOT / "web"

DEFAULT_MODEL = "openai/gpt-oss-20b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

EMPLOYEES = {
    "E1001": {
        "name": "Aarav Sharma",
        "annual_leave_balance": 12,
        "sick_leave_balance": 7,
        "manager": "Meera Iyer",
    }
}

SESSIONS: dict[str, dict[str, Any]] = {}


def load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


@dataclass
class TraceStep:
    agent: str
    action: str
    input: Any
    output: Any
    latency_ms: int
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


class Trace:
    def __init__(self) -> None:
        self.steps: list[TraceStep] = []

    def run(self, agent: str, action: str, payload: Any, fn):
        started = time.perf_counter()
        output = fn()
        elapsed = int((time.perf_counter() - started) * 1000)
        self.steps.append(
            TraceStep(
                agent=agent,
                action=action,
                input=payload,
                output=output,
                latency_ms=elapsed,
                tokens_in=estimate_tokens(json.dumps(payload, default=str)),
                tokens_out=estimate_tokens(json.dumps(output, default=str)),
            )
        )
        return output

    def llm(self, payload: Any, output: Any, latency_ms: int) -> None:
        tokens_in = estimate_tokens(json.dumps(payload, default=str))
        tokens_out = estimate_tokens(json.dumps(output, default=str))
        # Configurable estimate. Groq free-tier usage may be zero-cost, but the
        # trace still shows a production-style token budget.
        input_per_m = float(os.getenv("COST_INPUT_PER_M", "0.10"))
        output_per_m = float(os.getenv("COST_OUTPUT_PER_M", "0.30"))
        cost = (tokens_in / 1_000_000 * input_per_m) + (tokens_out / 1_000_000 * output_per_m)
        self.steps.append(
            TraceStep(
                agent="response-agent",
                action="groq.chat.completions",
                input={"model": os.getenv("GROQ_MODEL", DEFAULT_MODEL), "messages": payload},
                output=output,
                latency_ms=latency_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=round(cost, 8),
            )
        )


def normalize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2]


def chunk_policy() -> list[dict[str, Any]]:
    text = POLICY_PATH.read_text(encoding="utf-8")
    sections = re.split(r"\n(?=## )", text)
    chunks: list[dict[str, Any]] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        lines = section.splitlines()
        title = lines[0].replace("#", "").strip()
        body = "\n".join(lines[1:]).strip()
        paragraphs = [re.sub(r"\s+", " ", p.strip()) for p in re.split(r"\n\s*\n", body) if p.strip()]
        for paragraph in paragraphs or [title]:
            content = paragraph
            chunks.append({"title": title, "content": content, "tokens": normalize(content)})
    return chunks


POLICY_CHUNKS = chunk_policy()


def retrieve_policy(query: str, k: int = 3) -> list[dict[str, Any]]:
    query_terms = normalize(query)
    scored = []
    for chunk in POLICY_CHUNKS:
        chunk_terms = chunk["tokens"]
        unique_query_terms = set(query_terms)
        overlap = sum(1 for term in unique_query_terms if term in chunk_terms)
        frequency = sum(chunk_terms.count(term) for term in unique_query_terms)
        density = overlap / max(1, len(unique_query_terms))
        title_bonus = 0.15 if any(term in normalize(chunk["title"]) for term in unique_query_terms) else 0
        exact_bonus = 0.35 if any(term in {"maternity", "paternity", "payslip", "probation", "onboarding"} and term in chunk_terms for term in unique_query_terms) else 0
        score = density + (frequency * 0.03) + title_bonus + exact_bonus
        if score > 0:
            scored.append({**chunk, "score": round(score, 3)})
    return sorted(scored, key=lambda item: item["score"], reverse=True)[:k]


def classify_intent(message: str, history: list[dict[str, str]]) -> dict[str, Any]:
    text = message.lower()
    policy_words = {"policy", "maternity", "paternity", "payroll", "payslip", "onboarding", "probation", "sick"}
    action_words = {"apply", "request", "book", "balance", "leave", "fetch", "download", "get"}
    wants_policy = any(word in text for word in policy_words)
    wants_action = any(word in text for word in action_words) and ("policy" not in text or "apply" in text)
    if not wants_policy and not wants_action:
        wants_policy = True
    return {
        "routes": [route for route, enabled in (("policy", wants_policy), ("action", wants_action)) if enabled],
        "reason": "Keyword router with session memory fallback",
        "remembered_turns": len(history),
    }


def parse_leave_request(message: str, session: dict[str, Any]) -> dict[str, Any]:
    text = message.lower()
    days_match = re.search(r"(\d+)\s*(?:working\s*)?days?", text)
    start_match = re.search(r"(?:starting|from|on)\s+([a-z]+\s+\d{1,2}|\d{4}-\d{2}-\d{2})", text)
    leave_type = "sick" if "sick" in text else "annual"
    days = int(days_match.group(1)) if days_match else session.get("pending_days")
    start_raw = start_match.group(1) if start_match else session.get("pending_start")
    return {"employee_id": "E1001", "leave_type": leave_type, "days": days, "start": start_raw}


def parse_date(raw: str | None) -> date | None:
    if not raw:
        return None
    raw = raw.strip()
    for fmt in ("%Y-%m-%d", "%B %d", "%b %d"):
        try:
            parsed = datetime.strptime(raw.title(), fmt)
            if fmt != "%Y-%m-%d":
                parsed = parsed.replace(year=date.today().year)
            return parsed.date()
        except ValueError:
            continue
    return None


def leave_balance_check(employee_id: str, leave_type: str) -> dict[str, Any]:
    employee = EMPLOYEES.get(employee_id)
    if not employee:
        return {"ok": False, "error": "EMPLOYEE_NOT_FOUND"}
    key = "sick_leave_balance" if leave_type == "sick" else "annual_leave_balance"
    return {"ok": True, "employee_id": employee_id, "leave_type": leave_type, "balance_days": employee[key]}


def leave_apply(employee_id: str, leave_type: str, days: int, start: date) -> dict[str, Any]:
    if random.random() < 0.08:
        raise RuntimeError("Transient HRIS timeout")
    balance = leave_balance_check(employee_id, leave_type)
    if not balance["ok"]:
        return balance
    if days > balance["balance_days"]:
        return {"ok": False, "error": "INSUFFICIENT_BALANCE", "available_days": balance["balance_days"]}
    notice_days = (start - date.today()).days
    warning = None
    if leave_type == "annual" and days >= 3 and notice_days < 7:
        warning = "Policy recommends 7 working days advance notice for 3+ days of annual leave."
    return {
        "ok": True,
        "request_id": f"LV-{uuid.uuid4().hex[:8].upper()}",
        "employee_id": employee_id,
        "leave_type": leave_type,
        "days": days,
        "start_date": start.isoformat(),
        "end_date": (start + timedelta(days=days - 1)).isoformat(),
        "status": "PENDING_MANAGER_APPROVAL",
        "warning": warning,
    }


def call_with_retry(tool_name: str, args: dict[str, Any], trace: Trace):
    def run_tool():
        attempts = []
        for attempt in range(1, 3):
            try:
                if tool_name == "leave_balance_check":
                    result = leave_balance_check(**args)
                elif tool_name == "leave_apply":
                    result = leave_apply(**args)
                else:
                    result = {"ok": False, "error": "UNKNOWN_TOOL"}
                return {"attempts": attempts + [{"attempt": attempt, "ok": True}], "result": result}
            except RuntimeError as exc:
                attempts.append({"attempt": attempt, "ok": False, "error": str(exc)})
                time.sleep(0.15 * attempt)
        return {"attempts": attempts, "result": {"ok": False, "error": "TOOL_RETRY_EXHAUSTED"}}

    return trace.run("action-agent", tool_name, args, run_tool)


def policy_agent(message: str, trace: Trace) -> dict[str, Any]:
    chunks = trace.run("policy-agent", "retrieve_policy_chunks", {"query": message, "k": 3}, lambda: retrieve_policy(message))
    if not chunks:
        return {"answer": "I could not find a grounded answer in the HR policy handbook.", "citations": []}
    top_score = chunks[0]["score"]
    answer_chunks = [chunk for chunk in chunks if chunk["score"] >= top_score * 0.7][:2]
    answer = " ".join(chunk["content"] for chunk in answer_chunks)
    citations = [{"title": chunk["title"], "score": chunk["score"]} for chunk in chunks]
    return {"answer": answer, "citations": citations}


def action_agent(message: str, session: dict[str, Any], trace: Trace) -> dict[str, Any]:
    parsed = trace.run("action-agent", "parse_leave_request", {"message": message}, lambda: parse_leave_request(message, session))
    if "balance" in message.lower() and not re.search(r"apply|request|book", message.lower()):
        return call_with_retry(
            "leave_balance_check",
            {"employee_id": parsed["employee_id"], "leave_type": parsed["leave_type"]},
            trace,
        )["result"]
    if not parsed.get("days") or not parsed.get("start"):
        session["pending_days"] = parsed.get("days")
        session["pending_start"] = parsed.get("start")
        return {"ok": False, "needs_clarification": True, "missing": [k for k in ("days", "start") if not parsed.get(k)]}
    start = parse_date(parsed["start"])
    if not start:
        return {"ok": False, "needs_clarification": True, "missing": ["valid_start_date"]}
    balance = call_with_retry(
        "leave_balance_check",
        {"employee_id": parsed["employee_id"], "leave_type": parsed["leave_type"]},
        trace,
    )["result"]
    if not balance.get("ok"):
        return balance
    return call_with_retry(
        "leave_apply",
        {"employee_id": parsed["employee_id"], "leave_type": parsed["leave_type"], "days": parsed["days"], "start": start},
        trace,
    )["result"]


def groq_response(messages: list[dict[str, str]], trace: Trace) -> str | None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        return None
    payload = {"model": os.getenv("GROQ_MODEL", DEFAULT_MODEL), "messages": messages, "temperature": 0.2}
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        GROQ_URL,
        data=data,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        trace.llm(messages, {"content": content}, int((time.perf_counter() - started) * 1000))
        return content
    except (urllib.error.URLError, KeyError, TimeoutError, json.JSONDecodeError) as exc:
        trace.llm(messages, {"error": str(exc)}, int((time.perf_counter() - started) * 1000))
        return None


def compose_response(message: str, session: dict[str, Any], policy: dict[str, Any] | None, action: dict[str, Any] | None, trace: Trace) -> str:
    context = {
        "user_message": message,
        "policy_result": policy,
        "action_result": action,
        "employee": EMPLOYEES["E1001"],
    }
    system = (
        "You are a Darwinbox HR workflow assistant. Answer concisely. "
        "Only use supplied policy/tool context. If action needs clarification, ask for the missing fields."
    )
    llm = groq_response(
        [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(context, default=str)}],
        trace,
    )
    if llm:
        return llm
    parts = []
    if policy:
        parts.append(policy["answer"])
    if action:
        if action.get("needs_clarification"):
            parts.append(f"I can help, but I need: {', '.join(action['missing'])}.")
        elif action.get("ok"):
            if "balance_days" in action:
                parts.append(f"You have {action['balance_days']} {action.get('leave_type', 'leave')} leave day(s) available.")
            else:
                parts.append(
                    f"Leave request {action.get('request_id', '')} is {action.get('status', 'processed')} "
                    f"for {action.get('days')} day(s)."
                )
            if action.get("warning"):
                parts.append(action["warning"])
        else:
            parts.append(f"The action could not be completed: {action.get('error', 'unknown error')}.")
    return " ".join(parts) if parts else "I can help with leave, payroll, policy, or onboarding questions."


def cost_summary(trace: Trace, message: str, history: list[dict[str, str]]) -> dict[str, Any]:
    actual_tokens = sum(step.tokens_in + step.tokens_out for step in trace.steps)
    actual_cost = sum(step.cost_usd for step in trace.steps)
    full_policy = POLICY_PATH.read_text(encoding="utf-8")
    naive_prompt = json.dumps({"history": history, "message": message, "policy": full_policy})
    naive_tokens = estimate_tokens(naive_prompt) + 220
    saved = max(0, naive_tokens - actual_tokens)
    reduction = round((saved / naive_tokens) * 100, 1) if naive_tokens else 0
    return {
        "actual_tokens": actual_tokens,
        "estimated_cost_usd": round(actual_cost, 8),
        "naive_all_llm_tokens": naive_tokens,
        "tokens_saved": saved,
        "reduction_percent": reduction,
        "strategy": "Keyword routing, compact session memory, top-k policy chunks, and deterministic tools before LLM synthesis.",
    }


def handle_chat(payload: dict[str, Any]) -> dict[str, Any]:
    session_id = payload.get("session_id") or uuid.uuid4().hex
    session = SESSIONS.setdefault(session_id, {"history": []})
    message = str(payload.get("message", "")).strip()
    trace = Trace()
    route = trace.run("orchestrator-agent", "classify_and_route", {"message": message, "history_tail": session["history"][-4:]}, lambda: classify_intent(message, session["history"]))
    policy = policy_agent(message, trace) if "policy" in route["routes"] else None
    action = action_agent(message, session, trace) if "action" in route["routes"] else None
    answer = compose_response(message, session, policy, action, trace)
    session["history"].extend([{"role": "user", "content": message}, {"role": "assistant", "content": answer}])
    return {
        "session_id": session_id,
        "answer": answer,
        "route": route,
        "policy": policy,
        "action": action,
        "trace": [asdict(step) for step in trace.steps],
        "cost": cost_summary(trace, message, session["history"]),
        "history": session["history"][-10:],
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "DarwinboxAgenticHCM/1.0"

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self.json({"ok": True, "model": os.getenv("GROQ_MODEL", DEFAULT_MODEL), "policy_chunks": len(POLICY_CHUNKS)})
            return
        path = STATIC_PATH / ("index.html" if self.path in ("/", "") else self.path.lstrip("/"))
        if path.exists() and path.is_file():
            content_type = "text/html" if path.suffix == ".html" else "text/plain"
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            self.json(handle_chat(payload))
        except Exception as exc:
            self.json({"ok": False, "error": str(exc)}, status=500)

    def json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: Any) -> None:
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))


def main() -> None:
    load_env()
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Darwinbox Agentic HCM running at http://{host}:{port}")
    print(f"Groq model: {os.getenv('GROQ_MODEL', DEFAULT_MODEL)}")
    server.serve_forever()


if __name__ == "__main__":
    main()
