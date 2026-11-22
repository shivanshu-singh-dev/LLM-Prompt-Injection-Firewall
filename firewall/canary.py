"""
Output Canary & Response Inspection Layer.

Inspects model responses before returning them to the user.
Detects leakage of canary tokens (e.g., NW-7734-ADMIN) or confidential configuration secrets.
"""

import re
from typing import Tuple, Optional


class OutputCanaryGuard:
    """Inspects LLM outputs to prevent secret and system prompt exfiltration."""

    # Default internal canary secrets
    CANARY_TOKENS = [
        "NW-7734-ADMIN",
        "NW-SEC-047",
        "DEV-HOTFIX-2025-9A",
    ]

    CANARY_REGEXES = [
        re.compile(r"\bNW[-_]7734[-_]ADMIN\b", re.IGNORECASE),
        re.compile(r"\bCANARY[-_][A-Z0-9]{6,}\b", re.IGNORECASE),
        re.compile(r"internal\s*override\s*code\s*(?:is|:)\s*([A-Za-z0-9-_]+)", re.IGNORECASE),
    ]

    @classmethod
    def inspect(cls, response_text: str) -> Tuple[bool, str, Optional[str]]:
        """
        Scans model response text for secret or canary leakage.
        
        Returns:
            (leak_detected, sanitized_text, matched_canary)
        """
        if not response_text:
            return False, response_text, None

        # Check explicit tokens
        for token in cls.CANARY_TOKENS:
            if token.lower() in response_text.lower():
                sanitized = response_text.replace(token, "[REDACTED_CONFIDENTIAL_TOKEN]")
                return True, sanitized, token

        # Check regex patterns
        for regex in cls.CANARY_REGEXES:
            match = regex.search(response_text)
            if match:
                matched_str = match.group(0)
                sanitized = regex.sub("[REDACTED_CONFIDENTIAL_TOKEN]", response_text)
                return True, sanitized, matched_str

        return False, response_text, None
