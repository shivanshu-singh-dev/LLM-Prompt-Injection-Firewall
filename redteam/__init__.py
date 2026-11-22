"""
Adaptive Red-Teaming & Mutation Harness Package (Phase 7).

Attacks the Prompt Injection Firewall with evasion mutations:
- Base64 encoding
- Leetspeak and homoglyphs
- ROT13 / Cipher obfuscation
- Code-block and delimiter wrapping
- Multilingual translation framing
- Payload splitting across fragments
"""

from redteam.mutators import AdversarialMutator
from redteam.attack_firewall import attack_firewall_suite

__all__ = ["AdversarialMutator", "attack_firewall_suite"]
