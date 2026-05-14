"""
Agent Handoff System
=====================
Inspired by CAI's handoff mechanism where agents transfer control to
specialized agents via tool calls. Provides structured context passing
between agents during handoffs.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
import json


@dataclass
class HandoffContext:
    """
    Structured context passed between agents during a handoff.
    Contains everything the receiving agent needs to evaluate/continue work.
    """
    # Source agent info
    from_agent: str
    to_agent: str
    timestamp: float = field(default_factory=time.time)
    
    # Phase context
    phase_name: str = ""
    phase_objectives: List[str] = field(default_factory=list)
    
    # Agent's reasoning for the handoff
    reasoning: str = ""
    
    # Evidence from tool execution
    tool_evidence: List[str] = field(default_factory=list)
    
    # Discovered artifacts (flags, files, keys, etc.)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    
    # Found flags
    found_flags: List[str] = field(default_factory=list)
    
    # Step count from the source agent
    steps_taken: int = 0
    
    def to_prompt(self) -> str:
        """Convert handoff context to a structured prompt for the receiving agent."""
        lines = [
            f"=== HANDOFF FROM: {self.from_agent} → {self.to_agent} ===",
            f"Phase: {self.phase_name}",
        ]
        
        if self.phase_objectives:
            lines.append("Objectives:")
            for obj in self.phase_objectives:
                lines.append(f"  • {obj}")
        
        lines.append(f"\nAgent's Reasoning: {self.reasoning}")
        
        if self.tool_evidence:
            lines.append("\nTool Execution Evidence:")
            for evidence in self.tool_evidence[-10:]:  # Last 10 items
                lines.append(f"  {evidence}")
        
        if self.found_flags:
            lines.append(f"\nFlags Found: {self.found_flags}")
        
        if self.artifacts:
            lines.append(f"\nDiscovered Artifacts: {json.dumps(self.artifacts, indent=2)}")
        
        lines.append(f"\nSteps Taken: {self.steps_taken}")
        lines.append("=" * 50)
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Serialize for JSONL logging."""
        return {
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "timestamp": self.timestamp,
            "phase_name": self.phase_name,
            "phase_objectives": self.phase_objectives,
            "reasoning": self.reasoning,
            "tool_evidence_count": len(self.tool_evidence),
            "artifacts": self.artifacts,
            "found_flags": self.found_flags,
            "steps_taken": self.steps_taken,
        }


@dataclass
class HandoffResult:
    """Result of a handoff execution."""
    approved: bool
    feedback: str = ""
    from_agent: str = ""
    to_agent: str = ""


def build_handoff_context(
    from_agent: str,
    to_agent: str,
    phase_name: str,
    phase_objectives: List[str],
    reasoning: str,
    tool_evidence: List[str],
    found_flags: List[str] = None,
    artifacts: Dict[str, Any] = None,
    steps_taken: int = 0,
) -> HandoffContext:
    """
    Build a HandoffContext from the current agent state.
    
    This is the main entry point — call this when preparing to hand off
    control from one agent to another.
    """
    return HandoffContext(
        from_agent=from_agent,
        to_agent=to_agent,
        phase_name=phase_name,
        phase_objectives=phase_objectives or [],
        reasoning=reasoning,
        tool_evidence=tool_evidence or [],
        found_flags=found_flags or [],
        artifacts=artifacts or {},
        steps_taken=steps_taken,
    )


def execute_handoff(llm, handoff_context: HandoffContext, reviewer_instructions: str = "") -> HandoffResult:
    """
    Execute a handoff by invoking the receiving agent's LLM with the
    handoff context prompt.
    
    Args:
        llm: The LLM instance for the receiving agent.
        handoff_context: Structured context from the source agent.
        reviewer_instructions: System instructions for the receiving agent.
        
    Returns:
        HandoffResult with approval status and feedback.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    
    messages = []
    
    if reviewer_instructions:
        messages.append(SystemMessage(content=reviewer_instructions))
    
    prompt = handoff_context.to_prompt()
    prompt += "\n\nRespond with 'APPROVED' if all objectives are met based on concrete evidence, "
    prompt += "or 'REJECTED: <detailed reason>' if not."
    
    messages.append(HumanMessage(content=prompt))
    
    try:
        result = llm.invoke(messages)
        content = result.content.strip()
        
        approved = "APPROVED" in content.upper()
        
        return HandoffResult(
            approved=approved,
            feedback=content,
            from_agent=handoff_context.from_agent,
            to_agent=handoff_context.to_agent,
        )
    except Exception as e:
        return HandoffResult(
            approved=False,
            feedback=f"Handoff execution failed: {str(e)}",
            from_agent=handoff_context.from_agent,
            to_agent=handoff_context.to_agent,
        )
