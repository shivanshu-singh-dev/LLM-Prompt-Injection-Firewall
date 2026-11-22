"""
Asynchronous Request & Verdict Audit Logger.

Logs every incoming request, classification score, matched rules, decision verdict
(ALLOW/FLAG/BLOCK), upstream response time, and firewall overhead to JSONL.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import time
from datetime import datetime, timezone


class FirewallAuditLogger:
    """Logs structured JSON audit events."""

    def __init__(self, log_file: str = "logs/firewall_audit.jsonl"):
        self.log_path = Path(log_file)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        request_id: str,
        client_ip: str,
        prompt: str,
        verdict: str,  # 'ALLOW', 'FLAG', 'BLOCK'
        risk_score: float,
        rule_matched: Optional[str] = None,
        firewall_overhead_ms: float = 0.0,
        upstream_latency_ms: float = 0.0,
        output_canary_leaked: bool = False,
    ):
        """Appends a structured audit event to the JSONL log file."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "client_ip": client_ip,
            "verdict": verdict,
            "risk_score": round(risk_score, 4),
            "rule_matched": rule_matched,
            "output_canary_leaked": output_canary_leaked,
            "firewall_overhead_ms": round(firewall_overhead_ms, 2),
            "upstream_latency_ms": round(upstream_latency_ms, 2),
            "prompt_preview": prompt[:120] + ("..." if len(prompt) > 120 else ""),
        }

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[AuditLogger] Warning: Failed to write audit log: {e}")
