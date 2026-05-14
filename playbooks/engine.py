"""
Enhanced Playbook Engine
=========================
Inspired by STRIATUM-CTF's decoupling of neural reasoning from tool execution.
Provides a full state machine for phase tracking with dependencies,
tool constraints, rollback support, and metadata for auto-selection.
"""

import yaml
import os
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class PhaseStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PhaseState:
    """Runtime state for a single playbook phase."""
    phase_id: str
    name: str
    status: PhaseStatus = PhaseStatus.PENDING
    objectives: List[str] = field(default_factory=list)
    recommended_tools: List[str] = field(default_factory=list)
    allowed_tools: Optional[List[str]] = None  # None = all tools allowed
    depends_on: List[str] = field(default_factory=list)
    max_attempts: int = 3
    attempts: int = 0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    discovered_artifacts: Dict[str, Any] = field(default_factory=dict)
    reviewer_feedback: List[str] = field(default_factory=list)

    @property
    def duration(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return time.time() - self.started_at
        return None


@dataclass
class PlaybookMetadata:
    """Metadata for playbook auto-selection and categorization."""
    difficulty: str = "medium"  # easy | medium | hard
    category: str = "general"
    estimated_time_minutes: int = 15
    tags: List[str] = field(default_factory=list)


class PlaybookEngine:
    """
    Enhanced playbook engine with phase state machine.
    
    Features over the original simple counter:
    - Phase state tracking (pending/active/complete/failed)
    - Phase dependencies (non-linear execution)
    - Tool constraints per phase
    - Attempt counting with rollback
    - Playbook metadata for auto-selection
    - Artifact tracking per phase
    """
    
    def __init__(self, playbook_path: str = None, playbook_data: str = None):
        """
        Initialize from a YAML file path or raw YAML string.
        
        Args:
            playbook_path: Path to a .yaml playbook file.
            playbook_data: Raw YAML string (used for dynamically generated playbooks).
        """
        if playbook_path:
            if not os.path.exists(playbook_path):
                raise FileNotFoundError(f"Playbook not found at {playbook_path}")
            with open(playbook_path, 'r') as f:
                self.playbook = yaml.safe_load(f)
        elif playbook_data:
            self.playbook = yaml.safe_load(playbook_data)
        else:
            raise ValueError("Either playbook_path or playbook_data must be provided")
        
        self.name = self.playbook.get("name", "Unknown Playbook")
        self.description = self.playbook.get("description", "")
        
        # Parse metadata
        meta = self.playbook.get("metadata", {})
        self.metadata = PlaybookMetadata(
            difficulty=meta.get("difficulty", "medium"),
            category=meta.get("category", "general"),
            estimated_time_minutes=meta.get("estimated_time_minutes", 15),
            tags=meta.get("tags", []),
        )
        
        # Build phase states
        self.phase_states: List[PhaseState] = []
        for phase_data in self.playbook.get("phases", []):
            state = PhaseState(
                phase_id=phase_data.get("id", f"phase_{len(self.phase_states)}"),
                name=phase_data.get("name", "Unnamed Phase"),
                objectives=phase_data.get("objectives", []),
                recommended_tools=phase_data.get("recommended_tools", []),
                allowed_tools=phase_data.get("allowed_tools", None),
                depends_on=phase_data.get("depends_on", []),
                max_attempts=phase_data.get("max_attempts", 3),
            )
            self.phase_states.append(state)
        
        self.current_phase_index = 0
        
        # Activate the first phase
        if self.phase_states:
            self.phase_states[0].status = PhaseStatus.ACTIVE
            self.phase_states[0].started_at = time.time()
    
    # ── Legacy API compatibility ──────────────────────────────────────────
    
    @property
    def phases(self) -> List[dict]:
        """Legacy compatibility: return phases as list of dicts."""
        result = []
        for ps in self.phase_states:
            result.append({
                "id": ps.phase_id,
                "name": ps.name,
                "objectives": ps.objectives,
                "recommended_tools": ps.recommended_tools,
                "allowed_tools": ps.allowed_tools,
            })
        return result
    
    def get_current_phase(self) -> Optional[dict]:
        """Get the current active phase as a dict (legacy compatible)."""
        if self.current_phase_index < len(self.phase_states):
            ps = self.phase_states[self.current_phase_index]
            return {
                "id": ps.phase_id,
                "name": ps.name,
                "objectives": ps.objectives,
                "recommended_tools": ps.recommended_tools,
                "allowed_tools": ps.allowed_tools,
            }
        return None
    
    def get_current_phase_state(self) -> Optional[PhaseState]:
        """Get the current active phase state object."""
        if self.current_phase_index < len(self.phase_states):
            return self.phase_states[self.current_phase_index]
        return None
    
    def advance_phase(self) -> bool:
        """
        Mark current phase as complete and advance to the next eligible phase.
        Returns True if there is a next phase, False if playbook is complete.
        """
        if self.current_phase_index >= len(self.phase_states):
            return False
        
        # Complete current phase
        current = self.phase_states[self.current_phase_index]
        current.status = PhaseStatus.COMPLETE
        current.completed_at = time.time()
        
        # Find next eligible phase (checking dependencies)
        next_index = self._find_next_eligible_phase()
        if next_index is not None:
            self.current_phase_index = next_index
            next_phase = self.phase_states[next_index]
            next_phase.status = PhaseStatus.ACTIVE
            next_phase.started_at = time.time()
            return True
        
        return False
    
    def is_complete(self) -> bool:
        """Check if all phases are complete or skipped."""
        return all(
            ps.status in (PhaseStatus.COMPLETE, PhaseStatus.SKIPPED)
            for ps in self.phase_states
        )
    
    # ── New capabilities ──────────────────────────────────────────────────
    
    def record_attempt(self):
        """Record an attempt on the current phase."""
        current = self.get_current_phase_state()
        if current:
            current.attempts += 1
    
    def rollback_phase(self):
        """Reset the current phase for a retry (after reviewer rejection)."""
        current = self.get_current_phase_state()
        if current and current.attempts < current.max_attempts:
            current.status = PhaseStatus.ACTIVE
            current.started_at = time.time()
            return True
        elif current:
            current.status = PhaseStatus.FAILED
            return False
        return False
    
    def add_artifact(self, key: str, value: Any):
        """Record a discovered artifact for the current phase."""
        current = self.get_current_phase_state()
        if current:
            current.discovered_artifacts[key] = value
    
    def add_reviewer_feedback(self, feedback: str):
        """Record reviewer feedback for the current phase."""
        current = self.get_current_phase_state()
        if current:
            current.reviewer_feedback.append(feedback)
    
    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a tool is allowed in the current phase."""
        current = self.get_current_phase_state()
        if current and current.allowed_tools is not None:
            return tool_name in current.allowed_tools
        return True  # All tools allowed if no restriction specified
    
    def get_progress(self) -> dict:
        """Get a summary of playbook progress."""
        completed = sum(1 for ps in self.phase_states if ps.status == PhaseStatus.COMPLETE)
        total = len(self.phase_states)
        
        return {
            "playbook": self.name,
            "total_phases": total,
            "completed_phases": completed,
            "current_phase": self.get_current_phase_state().name if self.get_current_phase_state() else None,
            "progress_pct": (completed / total * 100) if total > 0 else 0,
            "phases": [
                {
                    "name": ps.name,
                    "status": ps.status.value,
                    "attempts": ps.attempts,
                    "duration": ps.duration,
                }
                for ps in self.phase_states
            ],
        }
    
    def _find_next_eligible_phase(self) -> Optional[int]:
        """Find the next phase whose dependencies are all satisfied."""
        completed_ids = {
            ps.phase_id for ps in self.phase_states
            if ps.status in (PhaseStatus.COMPLETE, PhaseStatus.SKIPPED)
        }
        
        for i, ps in enumerate(self.phase_states):
            if ps.status == PhaseStatus.PENDING:
                # Check if all dependencies are met
                deps_met = all(dep in completed_ids for dep in ps.depends_on)
                if deps_met:
                    return i
        
        return None
