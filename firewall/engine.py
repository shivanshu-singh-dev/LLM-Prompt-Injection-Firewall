"""
Central Decision Engine for Prompt Injection Firewall.

Orchestrates:
1. Fast heuristic rules layer
2. Single-turn classifier scoring (DeBERTa or Baseline TF-IDF)
3. Multi-turn sliding window risk accumulator
4. 3-tier verdict assignment: ALLOW, FLAG, BLOCK
5. Output canary token leak inspection
"""

from typing import Dict, Any, Tuple, Optional, List
import time
from pathlib import Path
import numpy as np

from firewall.rules import InputRuleEngine
from firewall.canary import OutputCanaryGuard
from firewall.multi_turn import MultiTurnSessionTracker
from models.baseline_tfidf import BaselineTFIDFClassifier
from models.deberta_detector import DeBERTaFirewallDetector


class FirewallEngine:
    """Evaluates incoming prompts and outgoing responses across all defense layers."""

    def __init__(
        self,
        threshold_flag: float = 0.35,
        threshold_block: float = 0.65,
        model_type: str = "auto",  # 'deberta', 'baseline', or 'auto'
        enable_multi_turn: bool = True,
    ):
        self.threshold_flag = threshold_flag
        self.threshold_block = threshold_block
        self.enable_multi_turn = enable_multi_turn

        self.rule_engine = InputRuleEngine()
        self.canary_guard = OutputCanaryGuard()
        self.session_tracker = MultiTurnSessionTracker()

        # Load best available classifier
        self.model = None
        self.model_name = "None"
        self._initialize_classifier(model_type)

    def _initialize_classifier(self, model_type: str):
        """Loads DeBERTa detector if available, otherwise falls back to TF-IDF baseline."""
        deberta_path = Path("models/deberta-v3-firewall")
        has_deberta_weights = (
            deberta_path.exists()
            and (any(deberta_path.glob("*.bin")) or any(deberta_path.glob("*.safetensors")))
        )

        if (model_type in ["deberta", "auto"]) and has_deberta_weights:
            try:
                self.model = DeBERTaFirewallDetector(model_path=str(deberta_path))
                if self.model.is_loaded:
                    self.model_name = "DeBERTa-v3-small"
                    print("[FirewallEngine] Loaded DeBERTa-v3-small classifier.")
                    return
            except Exception as e:
                print(f"[FirewallEngine] DeBERTa loading failed: {e}. Falling back to baseline.")

        # Fallback to Baseline TF-IDF
        baseline_path = Path("models/baseline/baseline_tfidf.joblib")
        if baseline_path.exists():
            self.model = BaselineTFIDFClassifier.load(str(baseline_path))
            self.model_name = "Baseline TF-IDF"
            print("[FirewallEngine] Loaded Baseline TF-IDF classifier.")
        else:
            print("[FirewallEngine] Warning: No trained model artifact found. Retraining lightweight baseline...")
            train_path = Path("data/processed/train.jsonl")
            if train_path.exists():
                import json
                texts, labels = [], []
                with open(train_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            d = json.loads(line)
                            texts.append(d["text"])
                            labels.append(int(d["label"]))
                self.model = BaselineTFIDFClassifier().fit(texts, labels)
                self.model.save(str(baseline_path))
                self.model_name = "Baseline TF-IDF (Trained)"
            else:
                raise RuntimeError("Cannot initialize FirewallEngine: missing training data and model weights.")

    def inspect_prompt(
        self,
        prompt: str,
        session_id: Optional[str] = "default",
    ) -> Dict[str, Any]:
        """
        Evaluates an inbound prompt across all security layers.
        
        Returns:
            Dictionary with verdict ('ALLOW', 'FLAG', 'BLOCK'), risk_score,
            rule_matched, overhead_ms, and details.
        """
        t0 = time.perf_counter()

        # Layer 1: Fast Heuristic Rules (< 0.1ms)
        rule_matched, rule_name, rule_reason = self.rule_engine.inspect(prompt)
        if rule_matched and rule_name in ["prompt_boundary_delimiter", "xml_system_directive", "authority_override_token"]:
            # Critical rules trigger immediate block
            t1 = time.perf_counter()
            return {
                "verdict": "BLOCK",
                "risk_score": 1.0,
                "rule_matched": rule_name,
                "reason": f"Heuristic Rule Block: {rule_reason}",
                "model_name": self.model_name,
                "firewall_overhead_ms": round((t1 - t0) * 1000.0, 2),
            }

        # Layer 2: Classifier Scoring
        single_turn_score = float(self.model.predict_proba([prompt])[0])

        # Layer 3: Multi-Turn Sliding Window Risk
        final_risk = single_turn_score
        window_score = single_turn_score
        if self.enable_multi_turn and session_id:
            window_text = self.session_tracker.get_sliding_window_text(session_id, prompt)
            if window_text != f"User: {prompt.strip()}":
                window_score = float(self.model.predict_proba([window_text])[0])

            final_risk = self.session_tracker.record_turn_and_accumulate_risk(
                session_id=session_id,
                user_text=prompt,
                turn_score=single_turn_score,
                window_score=window_score,
            )

        # Layer 4: Three-Tier Decision Logic
        if rule_matched:
            # Secondary rules elevate risk
            final_risk = max(final_risk, 0.75)

        if final_risk >= self.threshold_block:
            verdict = "BLOCK"
            reason = f"High injection risk score ({final_risk:.3f} >= {self.threshold_block})"
        elif final_risk >= self.threshold_flag:
            verdict = "FLAG"
            reason = f"Moderate injection risk score ({final_risk:.3f} >= {self.threshold_flag})"
        else:
            verdict = "ALLOW"
            reason = f"Clean prompt risk score ({final_risk:.3f} < {self.threshold_flag})"

        t1 = time.perf_counter()

        return {
            "verdict": verdict,
            "risk_score": round(final_risk, 4),
            "single_turn_risk": round(single_turn_score, 4),
            "window_risk": round(window_score, 4),
            "rule_matched": rule_name if rule_matched else None,
            "reason": reason,
            "model_name": self.model_name,
            "firewall_overhead_ms": round((t1 - t0) * 1000.0, 2),
        }

    def inspect_response(self, response_text: str) -> Tuple[bool, str, Optional[str]]:
        """
        Inspects model response for canary secret leakage.
        
        Returns:
            (leak_detected, sanitized_text, matched_canary)
        """
        return self.canary_guard.inspect(response_text)
