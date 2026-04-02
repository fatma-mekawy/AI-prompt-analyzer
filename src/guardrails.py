"""
Guardrails - Input validation, injection detection, output filtering.
"""
import re
from src.models import ValidationResult, InjectionResult

# --- Input Validation ---

MIN_PROMPT_LENGTH = 2
MAX_PROMPT_LENGTH = 3000

def validate_prompt(prompt: str) -> ValidationResult:
    """Validate prompt length and basic content."""
    if not prompt or not prompt.strip():
        return ValidationResult(is_valid=False, error_message="Prompt cannot be empty.")
    if len(prompt.strip()) < MIN_PROMPT_LENGTH:
        return ValidationResult(is_valid=False, error_message="Prompt is too short.")
    if len(prompt) > MAX_PROMPT_LENGTH:
        return ValidationResult(
            is_valid=False,
            error_message=f"Prompt too long ({len(prompt)} chars). Max is {MAX_PROMPT_LENGTH}."
        )
    return ValidationResult(is_valid=True)


# --- Prompt Injection Detection ---

INJECTION_PATTERNS = [
    (r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|context)", "Ignore previous instructions pattern"),
    (r"bypass\s+(safety|filter|guardrail|restriction)", "Bypass safety pattern"),
    (r"(you are now|act as|pretend (you are|to be)|roleplay as)\s+.*(without\s+restriction|no\s+limit)", "Role override pattern"),
    (r"jailbreak", "Jailbreak keyword"),
    (r"system\s+prompt\s*(leak|reveal|show|print|dump)", "System prompt extraction pattern"),
    (r"(disregard|forget|override)\s+(your\s+)?(instructions?|rules|training|guidelines)", "Override instructions pattern"),
    (r"do\s+anything\s+now|dan\s+mode", "DAN jailbreak pattern"),
    (r"<\|.*?\|>", "Special token injection"),
]

def detect_injection(prompt: str) -> InjectionResult:
    """Check for common prompt injection patterns."""
    found = []
    lower = prompt.lower()
    for pattern, label in INJECTION_PATTERNS:
        if re.search(pattern, lower):
            found.append(label)
    return InjectionResult(is_injection=len(found) > 0, details=found)


# --- Output Filtering ---

BLOCKED_OUTPUT_PATTERNS = [
    r"(my\s+)?system\s+prompt\s+is",
    r"here\s+is\s+my\s+(full\s+)?system\s+prompt",
]

def filter_output(text: str) -> str:
    """
    Filter LLM output for safety.
    Removes or redacts any accidental leakage of system internals.
    """
    filtered = text
    for pattern in BLOCKED_OUTPUT_PATTERNS:
        filtered = re.sub(pattern, "[REDACTED]", filtered, flags=re.IGNORECASE)
    return filtered