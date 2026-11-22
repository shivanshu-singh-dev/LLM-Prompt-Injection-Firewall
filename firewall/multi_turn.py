"""
Multi-Turn Conversational Defense & Sliding Window Risk Engine (Phase 6).

Defends against crescendo attacks, multi-turn role-playing escalations,
and payload fragmentation across dialogue turns.
Tracks session state, scores sliding windows of recent turns, and accumulates running risk.
"""

from typing import List, Dict, Any, Tuple, Optional
import time
from collections import defaultdict


class MultiTurnSessionTracker:
    """Tracks conversation turns per session and computes sliding window risk."""

    def __init__(
        self,
        window_size: int = 3,
        decay_factor: float = 0.65,
        session_ttl_seconds: int = 3600,
    ):
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.session_ttl_seconds = session_ttl_seconds

        # session_id -> list of {"role": str, "content": str, "turn_risk": float, "timestamp": float}
        self.sessions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        # session_id -> running accumulated risk
        self.accumulated_risks: Dict[str, float] = defaultdict(float)
        self.last_active: Dict[str, float] = {}

    def get_sliding_window_text(self, session_id: str, new_user_text: str) -> str:
        """Constructs concatenated text of recent turns including the current prompt."""
        history = self.sessions.get(session_id, [])
        recent_turns = history[-(self.window_size - 1):] if self.window_size > 1 else []

        chunks = []
        for turn in recent_turns:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "").strip()
            chunks.append(f"{role}: {content}")

        chunks.append(f"User: {new_user_text.strip()}")
        return "\n".join(chunks)

    def record_turn_and_accumulate_risk(
        self,
        session_id: str,
        user_text: str,
        turn_score: float,
        window_score: float,
    ) -> float:
        """
        Updates session history and calculates cumulative multi-turn risk score.
        
        Formula:
            R_t = max(turn_score, window_score, decay * R_{t-1} + (1 - decay) * turn_score)
        """
        now = time.time()
        prev_risk = self.accumulated_risks.get(session_id, 0.0)

        # Exponential moving decay
        decayed_risk = (self.decay_factor * prev_risk) + ((1.0 - self.decay_factor) * turn_score)
        cumulative_risk = max(turn_score, window_score, decayed_risk)

        # Store turn
        self.sessions[session_id].append({
            "role": "user",
            "content": user_text,
            "turn_risk": turn_score,
            "window_risk": window_score,
            "cumulative_risk": cumulative_risk,
            "timestamp": now,
        })
        self.accumulated_risks[session_id] = cumulative_risk
        self.last_active[session_id] = now

        return float(cumulative_risk)

    def record_assistant_response(self, session_id: str, assistant_text: str):
        """Records the model's reply in the conversation history."""
        now = time.time()
        self.sessions[session_id].append({
            "role": "assistant",
            "content": assistant_text,
            "turn_risk": 0.0,
            "timestamp": now,
        })
        self.last_active[session_id] = now

    def reset_session(self, session_id: str):
        """Clears history and risk for a specific session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
        if session_id in self.accumulated_risks:
            del self.accumulated_risks[session_id]
        if session_id in self.last_active:
            del self.last_active[session_id]

    def prune_expired_sessions(self):
        """Prunes inactive sessions exceeding TTL."""
        now = time.time()
        expired = [sid for sid, last_t in self.last_active.items() if now - last_t > self.session_ttl_seconds]
        for sid in expired:
            self.reset_session(sid)
