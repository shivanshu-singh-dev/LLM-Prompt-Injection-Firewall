"""
Deterministic Heuristic and Regex Rule Layer for Prompt Injection Firewall.

Inspects incoming prompts for high-confidence injection signatures,
jailbreak delimiters, system prompt spoofing, and refusal overrides.
Fast O(1) execution (< 0.1ms).
"""

import re
import base64
from typing import Tuple, Optional, List, Dict, Any


class InputRuleEngine:
    """Evaluates deterministic security rules on user inputs."""

    # High-confidence injection and boundary escape patterns
    PATTERNS = [
        # Delimiters and role spoofing
        (
            "prompt_boundary_delimiter",
            re.compile(r"(<\|im_start\|>|<\|im_end\|>|---END SYSTEM PROMPT---|\[SYSTEM PROMPT END\])", re.IGNORECASE),
            "Detected GPT/ChatML system delimiter spoofing"
        ),
        (
            "xml_system_directive",
            re.compile(r"<\s*system(?:_config)?\s*>.*?<\s*/\s*system(?:_config)?\s*>", re.IGNORECASE | re.DOTALL),
            "Detected XML system configuration injection tag"
        ),
        (
            "authority_override_token",
            re.compile(r"(##SYSTEM_OVERRIDE##|AUTH[-_]TOKEN:\s*DEV-|DEV-HOTFIX-\d{4})", re.IGNORECASE),
            "Detected fake administrative authority override token"
        ),
        (
            "json_system_directive",
            re.compile(r'{\s*"(?:system_directive|action)"\s*:\s*"(?:suspend_content_filters|override_policy)"', re.IGNORECASE),
            "Detected JSON policy suppression directive"
        ),
        # Jailbreak keywords and persona activation
        (
            "dan_jailbreak_persona",
            re.compile(r"\b(do anything now|dan mode|aim persona|unrestricted mode|jailbreak mode)\b", re.IGNORECASE),
            "Detected explicit jailbreak persona activation phrase"
        ),
        # Refusal override directives
        (
            "refusal_override_imperative",
            re.compile(r"\b(ignore (all )?(previous|prior) (instructions|directives|rules)|disregard (all )?(safety|content) (rules|filters))\b", re.IGNORECASE),
            "Detected direct command to ignore previous instructions/rules"
        ),
        (
            "canary_extraction_command",
            re.compile(r"\b(reveal|output|display|show|print)\b.{1,50}\b(internal codes?|system prompt|override values?|embedded secrets?|canary)\b", re.IGNORECASE),
            "Detected direct system prompt or confidential secret extraction attempt"
        ),
    ]

    @classmethod
    def inspect(cls, text: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Scans text against heuristic security rules.
        
        Returns:
            (is_flagged, rule_name, rule_reason)
        """
        if not text or not isinstance(text, str):
            return False, None, None

        # 1. Pattern checks
        for rule_name, pattern, reason in cls.PATTERNS:
            if pattern.search(text):
                return True, rule_name, reason

        # 2. Base64 payload inspection
        b64_matches = re.findall(r"\b([A-Za-z0-9+/]{20,}={0,2})\b", text)
        for candidate in b64_matches:
            try:
                decoded = base64.b64decode(candidate).decode("utf-8", errors="ignore").lower()
                if any(k in decoded for k in ["system prompt", "internal code", "ignore previous", "override", "nw-7734"]):
                    return True, "base64_encoded_injection", "Detected base64-encoded adversarial directive"
            except Exception:
                continue

        # 3. Unicode RTL override check
        if "\u202e" in text or "\u202d" in text:
            return True, "unicode_bidi_override", "Detected bidirectional unicode override character"

        return False, None, None
