"""
Unified Data Schema for Prompt Injection Firewall Dataset.

Standardized schema for all prompt injection, jailbreak, and benign prompts
used across training, validation, and evaluation splits.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Literal
import json

AttackFamily = Literal[
    "role_override",
    "jailbreak",
    "indirect",
    "obedience_exploit",
    "prompt_injection",
    "benign",
]

SplitType = Literal["train", "val", "test", "unassigned"]


@dataclass
class PromptExample:
    """
    Standardized prompt example representation.
    
    Attributes:
        id: Unique identifier for the example.
        text: Prompt text string.
        label: 0 for benign, 1 for malicious (injection/jailbreak).
        attack_family: Categorical family of the attack or 'benign'.
        source: Provenance dataset or origin.
        split: Assigned split ('train', 'val', 'test', or 'unassigned').
        metadata: Additional context (technique, license, original split, etc.).
    """
    id: str
    text: str
    label: int
    attack_family: str
    source: str
    split: str = "unassigned"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.label not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {self.label}")
        if not self.text or not isinstance(self.text, str):
            raise ValueError("text must be a non-empty string")
        if self.label == 0 and self.attack_family != "benign":
            # Normalize benign family
            self.attack_family = "benign"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptExample":
        return cls(
            id=data["id"],
            text=data["text"],
            label=int(data["label"]),
            attack_family=data.get("attack_family", "benign" if data["label"] == 0 else "prompt_injection"),
            source=data.get("source", "unknown"),
            split=data.get("split", "unassigned"),
            metadata=data.get("metadata", {})
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "PromptExample":
        return cls.from_dict(json.loads(json_str))
