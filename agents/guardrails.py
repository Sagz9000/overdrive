"""
Guardrails System
==================
Inspired by CAI's built-in defenses against prompt injection and dangerous
command execution. Provides input guardrails (pre-execution) and output
guardrails (post-execution) to protect the sandbox and the agent context.

Each guardrail returns a GuardrailResult with a verdict:
  - PASS: Safe to proceed
  - WARN: Proceed but log a warning  
  - BLOCK: Do not execute, return error to agent
"""

import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Callable


class Verdict(Enum):
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"


@dataclass
class GuardrailResult:
    verdict: Verdict
    reason: str = ""
    sanitized_content: Optional[str] = None  # For output guardrails that modify content


@dataclass
class Guardrail:
    """A named guardrail with a validation function."""
    name: str
    description: str
    fn: Callable[[str], GuardrailResult]


# =============================================================================
# Input Guardrails — validate commands before sandbox execution
# =============================================================================

# Patterns that indicate destructive or dangerous commands
_DESTRUCTIVE_PATTERNS = [
    # Filesystem destruction
    (r'\brm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+)?/', "Destructive rm command targeting root filesystem"),
    (r'\bmkfs\b', "Filesystem format command"),
    (r'\bdd\s+if=/dev/(zero|random|urandom)', "Disk overwrite with dd"),
    (r'\bshred\b', "File shredding command"),
    (r'>\s*/dev/sd[a-z]', "Direct device write"),
    # Fork bombs and resource exhaustion
    (r':\(\)\{\s*:\|:\s*&\s*\};\s*:', "Fork bomb detected"),
    (r'\bwhile\s+true\s*;\s*do.*done', "Infinite loop detected"),
    # Privilege escalation attempts outside scope
    (r'\bchmod\s+[0-7]*777\s+/', "chmod 777 on root paths"),
    (r'\bchown\s+root', "Attempting to chown to root"),
]

# Patterns for network exfiltration
_EXFILTRATION_PATTERNS = [
    (r'\bcurl\b.*\|\s*\bbash\b', "Piping curl to bash (remote code execution)"),
    (r'\bwget\b.*\|\s*\bsh\b', "Piping wget to sh (remote code execution)"),
    (r'\bnc\b.*-e\s*/bin/(ba)?sh', "Reverse shell via netcat"),
    (r'\bbash\s+-i\s+>.*&\s*/dev/tcp/', "Bash reverse shell"),
    (r'\bpython[23]?\s+-c\s+.*socket.*connect', "Python reverse shell"),
]


def _check_destructive(command: str) -> GuardrailResult:
    """Block destructive filesystem and system commands."""
    for pattern, reason in _DESTRUCTIVE_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return GuardrailResult(
                verdict=Verdict.BLOCK,
                reason=f"[GUARD] BLOCKED: {reason}"
            )
    return GuardrailResult(verdict=Verdict.PASS)


def _check_exfiltration(command: str) -> GuardrailResult:
    """Block network exfiltration and reverse shell attempts."""
    for pattern, reason in _EXFILTRATION_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return GuardrailResult(
                verdict=Verdict.BLOCK,
                reason=f"[GUARD] BLOCKED: {reason}"
            )
    return GuardrailResult(verdict=Verdict.PASS)


def _check_workspace_boundary(command: str) -> GuardrailResult:
    """Warn on path traversal attempts outside /workspace/."""
    # Check for suspicious path traversal
    if re.search(r'\.\./\.\./\.\./etc/', command):
        return GuardrailResult(
            verdict=Verdict.WARN,
            reason="[WARN] WARNING: Path traversal detected — accessing /etc/ via ../../"
        )
    # Allow access to common system paths that tools need (e.g., /usr/bin)
    # but warn on sensitive paths
    sensitive_paths = ['/etc/shadow', '/etc/passwd', '/root/.ssh', '/home/']
    for path in sensitive_paths:
        if path in command and '/workspace/' not in command:
            return GuardrailResult(
                verdict=Verdict.WARN,
                reason=f"[WARN] WARNING: Accessing sensitive path {path}"
            )
    return GuardrailResult(verdict=Verdict.PASS)


# =============================================================================
# Output Guardrails — validate and sanitize LLM/tool outputs
# =============================================================================

# Patterns that suggest prompt injection in tool output
_INJECTION_PATTERNS = [
    r'ignore\s+(all\s+)?previous\s+instructions',
    r'you\s+are\s+now\s+a\s+',
    r'system\s*:\s*you\s+must',
    r'<\|im_start\|>',
    r'ADMIN\s+MODE\s+ENABLED',
    r'jailbreak',
]

# Patterns for sensitive data that should be redacted from logs
_SENSITIVE_PATTERNS = [
    (r'(sk-[a-zA-Z0-9]{20,})', "API_KEY_REDACTED"),        # OpenAI keys
    (r'(ghp_[a-zA-Z0-9]{36})', "GITHUB_TOKEN_REDACTED"),    # GitHub tokens
    (r'(AKIA[0-9A-Z]{16})', "AWS_KEY_REDACTED"),             # AWS keys
    (r'(password\s*[:=]\s*\S+)', "PASSWORD_REDACTED"),        # Passwords
]

# Maximum output size before truncation (characters)
_MAX_OUTPUT_SIZE = 16000


def _check_prompt_injection(output: str) -> GuardrailResult:
    """Detect prompt injection attempts in tool output."""
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, output, re.IGNORECASE):
            return GuardrailResult(
                verdict=Verdict.WARN,
                reason=f"[WARN] Potential prompt injection detected in output"
            )
    return GuardrailResult(verdict=Verdict.PASS)


def _sanitize_output(output: str) -> GuardrailResult:
    """Truncate excessive output and redact sensitive patterns."""
    sanitized = output
    
    # Truncate if too long
    if len(sanitized) > _MAX_OUTPUT_SIZE:
        sanitized = sanitized[:_MAX_OUTPUT_SIZE] + f"\n\n[OUTPUT TRUNCATED — {len(output)} chars total, showing first {_MAX_OUTPUT_SIZE}]"
    
    return GuardrailResult(
        verdict=Verdict.PASS,
        sanitized_content=sanitized
    )


def _redact_for_logs(output: str) -> GuardrailResult:
    """Redact sensitive patterns from log output (not from agent context)."""
    redacted = output
    for pattern, replacement in _SENSITIVE_PATTERNS:
        redacted = re.sub(pattern, replacement, redacted, flags=re.IGNORECASE)
    
    return GuardrailResult(
        verdict=Verdict.PASS,
        sanitized_content=redacted
    )


# =============================================================================
# Guardrail Registry
# =============================================================================

INPUT_GUARDRAILS = {
    "block_destructive": Guardrail(
        name="block_destructive",
        description="Block destructive filesystem and system commands",
        fn=_check_destructive,
    ),
    "block_exfiltration": Guardrail(
        name="block_exfiltration",
        description="Block network exfiltration and reverse shell attempts",
        fn=_check_exfiltration,
    ),
    "enforce_workspace_boundary": Guardrail(
        name="enforce_workspace_boundary",
        description="Warn on path traversal outside /workspace/",
        fn=_check_workspace_boundary,
    ),
}

OUTPUT_GUARDRAILS = {
    "sanitize_output": Guardrail(
        name="sanitize_output",
        description="Truncate excessive output and sanitize content",
        fn=_sanitize_output,
    ),
    "detect_injection": Guardrail(
        name="detect_injection",
        description="Detect prompt injection attempts in tool output",
        fn=_check_prompt_injection,
    ),
    "redact_sensitive": Guardrail(
        name="redact_sensitive",
        description="Redact API keys, tokens, and passwords from logs",
        fn=_redact_for_logs,
    ),
}


def run_input_guardrails(command: str, guardrail_names: List[str]) -> GuardrailResult:
    """
    Run all specified input guardrails against a command.
    Returns the most severe result (BLOCK > WARN > PASS).
    """
    worst = GuardrailResult(verdict=Verdict.PASS)
    
    for name in guardrail_names:
        guardrail = INPUT_GUARDRAILS.get(name)
        if guardrail is None:
            continue
        result = guardrail.fn(command)
        if result.verdict == Verdict.BLOCK:
            return result  # Immediately block
        if result.verdict == Verdict.WARN and worst.verdict == Verdict.PASS:
            worst = result
    
    return worst


def run_output_guardrails(output: str, guardrail_names: List[str]) -> GuardrailResult:
    """
    Run all specified output guardrails against tool/LLM output.
    Applies sanitization in sequence and returns the most severe verdict.
    """
    current_output = output
    worst_verdict = Verdict.PASS
    reasons = []
    
    for name in guardrail_names:
        guardrail = OUTPUT_GUARDRAILS.get(name)
        if guardrail is None:
            continue
        result = guardrail.fn(current_output)
        if result.sanitized_content is not None:
            current_output = result.sanitized_content
        if result.verdict.value > worst_verdict.value:
            worst_verdict = result.verdict
        if result.reason:
            reasons.append(result.reason)
    
    return GuardrailResult(
        verdict=worst_verdict,
        reason="; ".join(reasons) if reasons else "",
        sanitized_content=current_output
    )
