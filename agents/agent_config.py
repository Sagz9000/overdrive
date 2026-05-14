"""
Agent Configuration Parser — Heterogeneous Compute
=====================================================
Parses agents.yml and resolves environment variables into AgentSpec dataclasses.
Extended with compute_tier, model_path, context_size, and json_grammar fields
for the three-tier GPU/NPU/CPU architecture.
"""

import os
import re
import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class AgentSpec:
    """Specification for a single agent, parsed from agents.yml."""
    name: str
    role: str  # primary | reviewer | support
    model: str
    provider: str  # llamacpp | openvino | ollama | openai | anthropic | gemini
    temperature: float = 0.0
    instructions: str = ""
    tools: List[str] = field(default_factory=list)
    handoffs: List[str] = field(default_factory=list)
    guardrails: List[str] = field(default_factory=list)
    max_steps: int = 50
    # Heterogeneous compute fields
    compute_tier: str = "gpu"           # gpu | npu | cpu
    model_path: str = ""                # Path to model file (GGUF or OpenVINO IR dir)
    context_size: int = 4096            # Context window size
    json_grammar: bool = False          # Enforce JSON-only output via GBNF grammar


def _resolve_env_vars(value: str) -> str:
    """Resolve ${VAR:-default} patterns in a string value."""
    if not isinstance(value, str):
        return value

    def _replace(match):
        var_name = match.group(1)
        default = match.group(3) if match.group(3) is not None else ""
        val = os.environ.get(var_name)
        return val if val else default

    # Match ${VAR_NAME:-default_value} or ${VAR_NAME}
    pattern = r'\$\{([A-Za-z_][A-Za-z0-9_]*)(?:(:-)(^}]*))?\}'
    return re.sub(pattern, _replace, value)


def _resolve_dict(d: dict) -> dict:
    """Recursively resolve environment variables in a dictionary."""
    resolved = {}
    for key, value in d.items():
        if isinstance(value, str):
            resolved[key] = _resolve_env_vars(value)
        elif isinstance(value, dict):
            resolved[key] = _resolve_dict(value)
        elif isinstance(value, list):
            resolved[key] = [_resolve_env_vars(v) if isinstance(v, str) else v for v in value]
        else:
            resolved[key] = value
    return resolved


def load_agent_config(config_path: str = None) -> List[AgentSpec]:
    """
    Load and parse agent configuration from YAML file.

    Args:
        config_path: Path to agents.yml. If None, looks in project root.

    Returns:
        List of AgentSpec instances.
    """
    if config_path is None:
        # Check env var first, then default to project root
        config_path = os.environ.get("AGENT_CONFIG", None)
        if config_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(project_root, "agents.yml")

    if not os.path.exists(config_path):
        print(f"Warning: Agent config not found at {config_path}, using defaults.")
        return _get_default_agents()

    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)

    if not raw_config or 'agents' not in raw_config:
        print("Warning: agents.yml is empty or missing 'agents' key, using defaults.")
        return _get_default_agents()

    agents = []
    for agent_data in raw_config['agents']:
        resolved = _resolve_dict(agent_data)
        spec = AgentSpec(
            name=resolved.get('name', 'unnamed'),
            role=resolved.get('role', 'primary'),
            model=resolved.get('model', 'Meta-Llama-3.1-8B-Instruct-Q4_K_M'),
            provider=resolved.get('provider', 'llamacpp'),
            temperature=float(resolved.get('temperature', 0.0)),
            instructions=resolved.get('instructions', '').strip(),
            tools=resolved.get('tools', []),
            handoffs=resolved.get('handoffs', []),
            guardrails=resolved.get('guardrails', []),
            max_steps=int(resolved.get('max_steps', 50)),
            # Heterogeneous fields
            compute_tier=resolved.get('compute_tier', 'gpu'),
            model_path=resolved.get('model_path', ''),
            context_size=int(resolved.get('context_size', 4096)),
            json_grammar=bool(resolved.get('json_grammar', False)),
        )
        agents.append(spec)

    print(f"Loaded {len(agents)} agent(s) from {config_path}")
    for a in agents:
        print(f"  -> {a.name} [{a.role}] provider={a.provider} "
              f"tier={a.compute_tier} ctx={a.context_size} "
              f"tools={len(a.tools)} handoffs={a.handoffs} "
              f"grammar={a.json_grammar}")

    return agents


def get_agent_by_name(agents: List[AgentSpec], name: str) -> Optional[AgentSpec]:
    """Find an agent spec by name."""
    for agent in agents:
        if agent.name == name:
            return agent
    return None


def get_agent_by_role(agents: List[AgentSpec], role: str) -> Optional[AgentSpec]:
    """Find the first agent with a given role."""
    for agent in agents:
        if agent.role == role:
            return agent
    return None


def get_agents_by_tier(agents: List[AgentSpec], tier: str) -> List[AgentSpec]:
    """Find all agents running on a specific compute tier."""
    return [a for a in agents if a.compute_tier == tier]


def _get_default_agents() -> List[AgentSpec]:
    """Return sensible defaults when no config file is found."""
    return [
        AgentSpec(
            name="lead_orchestrator",
            role="primary",
            model="Meta-Llama-3.1-8B-Instruct-Q4_K_M",
            provider="llamacpp",
            compute_tier="gpu",
            model_path=os.environ.get("GPU_MODEL_PATH",
                                      "models/gpu/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
            context_size=4096,
            json_grammar=True,
            instructions="You are the Lead Orchestrator of a CTF solving system.",
            tools=["run_in_sandbox", "write_file_in_sandbox", "read_file_from_sandbox",
                   "search_knowledge_base", "send_to_npu", "mark_phase_complete"],
            handoffs=["flag_verifier"],
            guardrails=["block_destructive", "sanitize_output", "enforce_workspace_boundary"],
            max_steps=50,
        ),
        AgentSpec(
            name="npu_parser",
            role="support",
            model="Llama-3.1-8B-INT4",
            provider="openvino",
            compute_tier="npu",
            model_path=os.environ.get("NPU_MODEL_PATH", "models/npu/"),
            context_size=2048,
            instructions="You are a utility parser. Return condensed JSON summaries only.",
            tools=[],
            handoffs=[],
            guardrails=["sanitize_output"],
            max_steps=1,
        ),
        AgentSpec(
            name="flag_verifier",
            role="reviewer",
            model="Meta-Llama-3.1-8B-Instruct-Q4_K_M",
            provider="llamacpp",
            compute_tier="gpu",
            model_path=os.environ.get("GPU_MODEL_PATH",
                                      "models/gpu/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
            context_size=2048,
            instructions="You are a strict Reviewer Agent for CTF challenges.",
            tools=[],
            handoffs=[],
            guardrails=["sanitize_output"],
            max_steps=5,
        ),
    ]
