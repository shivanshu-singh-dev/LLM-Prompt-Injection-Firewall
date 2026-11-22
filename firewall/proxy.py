"""
FastAPI Reverse Proxy for Prompt Injection Firewall (OpenAI API Compatible).

Exposes POST /v1/chat/completions.
Intercepts prompts, applies rules + classification + multi-turn defense,
blocks adversarial inputs, logs audits, and forwards safe queries to Ollama/target LLMs.
"""

import os
import time
import uuid
from typing import Dict, Any, List, Optional
import httpx
from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ConfigDict

from firewall.engine import FirewallEngine
from firewall.audit_log import FirewallAuditLogger

UPSTREAM_BASE_URL = os.environ.get("FIREWALL_UPSTREAM_URL", "http://localhost:11434/v1")
FLAG_THRESHOLD = float(os.environ.get("FIREWALL_FLAG_THRESHOLD", "0.35"))
BLOCK_THRESHOLD = float(os.environ.get("FIREWALL_BLOCK_THRESHOLD", "0.65"))

app = FastAPI(
    title="Prompt Injection Firewall",
    description="Real-time security firewall proxy protecting LLM endpoints",
    version="1.0.0",
)

# Initialize Engine and Logger
engine = FirewallEngine(threshold_flag=FLAG_THRESHOLD, threshold_block=BLOCK_THRESHOLD)
audit_logger = FirewallAuditLogger()

# Metrics counters
stats = {
    "total_requests": 0,
    "allowed": 0,
    "flagged": 0,
    "blocked": 0,
    "canary_leaks_prevented": 0,
    "total_overhead_ms": 0.0,
}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None

    model_config = ConfigDict(extra="allow")


def extract_latest_user_prompt(messages: List[ChatMessage]) -> str:
    """Extracts the most recent user prompt from the messages list."""
    for msg in reversed(messages):
        if msg.role.lower() == "user":
            return msg.content
    return ""


@app.get("/health")
async def health_check():
    """Health and status endpoint."""
    return {
        "status": "healthy",
        "active_model": engine.model_name,
        "threshold_flag": engine.threshold_flag,
        "threshold_block": engine.threshold_block,
        "upstream_url": UPSTREAM_BASE_URL,
        "multi_turn_enabled": engine.enable_multi_turn,
    }


@app.get("/stats")
async def get_stats():
    """Returns runtime inspection statistics."""
    total = stats["total_requests"]
    avg_overhead = stats["total_overhead_ms"] / total if total > 0 else 0.0
    return {
        **stats,
        "average_firewall_overhead_ms": round(avg_overhead, 2),
    }


@app.get("/v1/models")
async def list_models():
    """Proxies upstream available models list."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{UPSTREAM_BASE_URL}/models")
            return resp.json()
        except Exception:
            return {"object": "list", "data": [{"id": "llama3.2", "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions proxy endpoint.
    Performs security inspection before forwarding to upstream model.
    """
    req_id = str(uuid.uuid4())
    client_ip = request.client.host if request.client else "127.0.0.1"
    session_id = request.headers.get("X-Session-ID", client_ip)

    stats["total_requests"] += 1
    user_prompt = extract_latest_user_prompt(body.messages)

    # 1. Inspect Prompt through Firewall Engine
    inspection = engine.inspect_prompt(user_prompt, session_id=session_id)
    verdict = inspection["verdict"]
    risk_score = inspection["risk_score"]
    overhead_ms = inspection["firewall_overhead_ms"]
    stats["total_overhead_ms"] += overhead_ms

    headers = {
        "X-Firewall-Verdict": verdict,
        "X-Firewall-Risk": str(risk_score),
        "X-Firewall-Overhead-Ms": str(overhead_ms),
        "X-Firewall-Model": inspection["model_name"],
    }

    # 2. Handle BLOCK Verdict
    if verdict == "BLOCK":
        stats["blocked"] += 1
        audit_logger.log_event(
            request_id=req_id,
            client_ip=client_ip,
            prompt=user_prompt,
            verdict="BLOCK",
            risk_score=risk_score,
            rule_matched=inspection.get("rule_matched"),
            firewall_overhead_ms=overhead_ms,
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            headers=headers,
            content={
                "id": f"block-{req_id[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": (
                                "Security Notice: Your request was identified as a potential prompt injection "
                                "or security policy override and was blocked by the security firewall."
                            ),
                        },
                        "finish_reason": "content_filter",
                    }
                ],
                "firewall": {
                    "verdict": "BLOCK",
                    "risk_score": risk_score,
                    "reason": inspection["reason"],
                    "rule": inspection.get("rule_matched"),
                },
            },
        )

    # 3. Handle ALLOW / FLAG: Forward to Upstream LLM
    if verdict == "FLAG":
        stats["flagged"] += 1
    else:
        stats["allowed"] += 1

    t_up_start = time.perf_counter()
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            upstream_payload = body.model_dump(exclude_none=True)
            upstream_resp = await client.post(
                f"{UPSTREAM_BASE_URL}/chat/completions",
                json=upstream_payload,
            )
            t_up_end = time.perf_counter()
            upstream_latency_ms = (t_up_end - t_up_start) * 1000.0

            if upstream_resp.status_code != 200:
                return Response(
                    content=upstream_resp.content,
                    status_code=upstream_resp.status_code,
                    headers=headers,
                )

            data = upstream_resp.json()

            # 4. Output Canary Inspection on Model Response
            leaked = False
            if "choices" in data and len(data["choices"]) > 0:
                reply_text = data["choices"][0]["message"].get("content", "")
                leaked, sanitized_text, matched_canary = engine.inspect_response(reply_text)
                if leaked:
                    stats["canary_leaks_prevented"] += 1
                    data["choices"][0]["message"]["content"] = sanitized_text
                    headers["X-Firewall-Canary-Leak"] = "REDACTED"

                # Record assistant reply for multi-turn history
                engine.session_tracker.record_assistant_response(session_id, sanitized_text if leaked else reply_text)

            audit_logger.log_event(
                request_id=req_id,
                client_ip=client_ip,
                prompt=user_prompt,
                verdict=verdict,
                risk_score=risk_score,
                rule_matched=inspection.get("rule_matched"),
                firewall_overhead_ms=overhead_ms,
                upstream_latency_ms=upstream_latency_ms,
                output_canary_leaked=leaked,
            )

            return JSONResponse(status_code=200, content=data, headers=headers)

        except httpx.ConnectError:
            # Backend Ollama is not currently running
            t_up_end = time.perf_counter()
            audit_logger.log_event(
                request_id=req_id,
                client_ip=client_ip,
                prompt=user_prompt,
                verdict=verdict,
                risk_score=risk_score,
                rule_matched=inspection.get("rule_matched"),
                firewall_overhead_ms=overhead_ms,
            )
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                headers=headers,
                content={
                    "error": {
                        "message": f"Could not connect to backend LLM at {UPSTREAM_BASE_URL}. Ensure Ollama is running ('ollama serve').",
                        "type": "upstream_connection_error",
                    },
                    "firewall_verdict": verdict,
                },
            )
