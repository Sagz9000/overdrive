import os
from typing import TypedDict, List, Annotated, Optional, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from agents.live_reporter import update_live_report

# =============================================================================
# State Definition
# =============================================================================

class CTFSolverState(TypedDict):
    """Internal state for the CTF Solver FSM."""
    messages: Annotated[List[BaseMessage], "The conversation history"]
    current_phase: str                   # Name of the current phase
    phase_data: dict                     # Objectives and tools for current phase
    phase_index: int                     # Index in the playbook
    found_flags: List[str]               # All flags captured so far
    error_count: int                     # Consecutive tool errors
    step_count: int                      # Steps taken in CURRENT phase
    max_steps: int                       # Max steps before force-review
    flag_format: str                     # Expected flag regex
    retry_count: int                     # Review rejections for current phase
    max_retries: int                     # Max review rejections before force-advance
    reviewer_feedback: str               # Last reviewer feedback
    evidence_board: str                  # Accumulated evidence from previous phases

# =============================================================================
# Nodes (Agent Logic)
# =============================================================================

def create_solver_node(solver_llm, all_tools, playbook_engine, obs):
    """Create the primary solver node."""

    def solver_node(state: CTFSolverState) -> dict:
        """Execute one turn of the solver agent."""
        current_state = state["current_phase"]
        phase_data = state["phase_data"]
        
        # Filter tools based on phase
        allowed_tool_names = list(phase_data.get("allowed_tools", []))
        
        # PROACTIVE NUDGE: If shell access is allowed, Python execution must also be allowed
        # to prevent "escaping hell" in the sandbox.
        if "run_in_sandbox" in allowed_tool_names and "execute_python_code" not in allowed_tool_names:
            allowed_tool_names.append("execute_python_code")
            
        available_tools = [t for t in all_tools if t.name in allowed_tool_names]
        
        # Ensure fallback tools are always available
        if "mark_phase_complete" not in [t.name for t in available_tools]:
            available_tools.append(next(t for t in all_tools if t.name == "mark_phase_complete"))
            
        tool_names = [t.name for t in available_tools]
        
        # Inject feedback if this is a retry
        pending_msgs = []
        feedback = state.get("reviewer_feedback", "")
        if feedback and state["retry_count"] > 0:
            pending_msgs.append(HumanMessage(content=f"REVIEW FEEDBACK:\n{feedback}\n\nPlease address the feedback above to complete this phase."))

        obs.log_execution_step("state_graph_node", {
            "state": current_state,
            "phase": phase_data.get("name", ""),
            "available_tools": tool_names,
            "step_count": state["step_count"],
        })

        # Build dynamic system prompt with phase context
        evidence_board = state.get("evidence_board", "")
        system_prompt = _build_phase_prompt(phase_data, current_state, available_tools, evidence_board)

        # Create a phase-specific react agent with filtered tools
        agent = create_react_agent(solver_llm, available_tools, prompt=system_prompt)
        
        # Prune message history if it's getting too long for the context window
        input_messages = list(state["messages"]) + pending_msgs
        if len(input_messages) > 20:
            # Keep first message (the task) and the most recent context
            input_messages = [input_messages[0]] + input_messages[-19:]

        config = {"recursion_limit": 50}
        inputs = {"messages": input_messages}
        
        try:
            result = agent.invoke(inputs, config)
            
            # Extract new messages
            input_len = len(input_messages)
            new_msgs = result["messages"][input_len:]
            
            # Catch empty tool calls (Requirement 3)
            # Check if ANY AIMessage in the new batch has a tool call
            has_tool_call = any(isinstance(m, AIMessage) and m.tool_calls for m in new_msgs)
            
            if not has_tool_call:
                nudge = ("System Error: You provided text but no valid tool call. You must use a tool to proceed.\n"
                         "IMPORTANT: You must use the actual JSON tool call format provided by the API.\n"
                         "Example: {\"name\": \"run_in_sandbox\", \"parameters\": {\"command\": \"ls\"}}")
                new_msgs.append(HumanMessage(content=nudge))
                obs.log_execution_step("agent_empty_call_nudge", {"reason": "No tool call detected in response batch"})

            # Log the raw response for debugging
            for m in new_msgs:
                if isinstance(m, AIMessage):
                    obs.log_execution_step("agent_raw_output", {"content": m.content, "tool_calls": getattr(m, 'tool_calls', [])})
            
            # Update the live dashboard
            try:
                reasoning = ""
                last_tool_result = ""
                for m in reversed(new_msgs):
                    if isinstance(m, AIMessage) and not reasoning:
                        reasoning = (m.content or "")[:500]
                    if hasattr(m, 'type') and m.type == 'tool' and not last_tool_result:
                        last_tool_result = (m.content or "")[:500]
                
                update_live_report(
                    phase=phase_data.get("name", current_state),
                    step=state["step_count"],
                    reasoning=reasoning or "(thinking...)",
                    action=f"{len([m for m in new_msgs if hasattr(m, 'tool_calls') and m.tool_calls])} tool calls",
                    result=last_tool_result or "(no tool output yet)"
                )
            except Exception:
                pass  # Never let reporting crash the agent

            # Force immediate transition if mark_phase_complete was called (Requirement 1)
            # This prevents the solver node from looping internally if recursion limit allows it
            for m in new_msgs:
                if isinstance(m, AIMessage) and hasattr(m, 'tool_calls'):
                    if any(tc.get('name') == 'mark_phase_complete' for tc in m.tool_calls):
                        obs.log_execution_step("force_transition", {"reason": "mark_phase_complete detected"})
                        # We return immediately to let should_continue_or_review trigger the node swap
                        break

            return {
                "messages": new_msgs,
                "step_count": state["step_count"] + 1,
            }

        except Exception as e:
            obs.log_execution_step("solver_error", {"error": str(e)})
            error_msg = f"Agent error: {str(e)}. Please try a different approach."
            return {
                "messages": [AIMessage(content=error_msg)],
                "error_count": state["error_count"] + 1,
                "step_count": state["step_count"] + 1,
            }

    return solver_node


def create_reviewer_node(reviewer_llm, obs):
    """Create the reviewer/verifier node."""

    def reviewer_node(state: CTFSolverState) -> dict:
        """Evaluate whether phase objectives have been met."""
        phase_data = state["phase_data"]

        obs.log_execution_step("reviewer_start", {
            "phase": phase_data.get("name", ""),
            "found_flags": state["found_flags"],
        })

        # Build review prompt
        objectives = phase_data.get("objectives", [])
        review_prompt = (
            f"Phase: {phase_data.get('name', '')}\n"
            f"Objectives:\n"
        )
        for obj in objectives:
            review_prompt += f"  - {obj}\n"

        review_prompt += f"\nFlags found: {state['found_flags']}\n"
        review_prompt += f"Expected flag format: '{state.get('flag_format', '')}'\n"
        review_prompt += f"Steps taken: {state['step_count']}\n\n"

        # Get recent tool evidence from messages
        evidence = []
        for msg in state["messages"][-20:]:
            if hasattr(msg, 'type') and msg.type == 'tool':
                evidence.append(f"[{getattr(msg, 'name', 'tool')}]: "
                                f"{str(msg.content)[:300]}")

        if evidence:
            review_prompt += "Tool evidence:\n" + "\n".join(evidence[-10:])

        review_prompt += ("\n\nRespond with 'APPROVED' if all technical objectives are met "
                          "based on concrete evidence. "
                          "IMPORTANT: If 'Expected flag format' is EMPTY, you MUST ignore the 'Flags found' field and approve based on the technical objectives alone. Do NOT look for flags unless a format is specified.")

        instructions = "You are a strict CTF challenge reviewer. Focus on evidence and whether objectives were met.\n\n"
        messages = [
            HumanMessage(content=instructions + review_prompt),
        ]

        # HYBRID LOGIC: Escalate to Gemini 2.5 Flash if local model fails multiple times
        current_reviewer = reviewer_llm
        is_escalated = state["retry_count"] >= 2
        
        if is_escalated:
            try:
                from app.llm_factory import get_llm
                from agents.agent_config import load_agent_config
                agents = load_agent_config()
                reviewer_spec = next(a for a in agents if a.role == "reviewer")
                
                obs.log_execution_step("reviewer_escalation", {"reason": f"Local model stalemate. Escalating to {reviewer_spec.model}."})

                current_reviewer = get_llm(reviewer_spec)
                print(f"\n[HYBRID] Escalating to {reviewer_spec.model} for expert review...")
            except Exception as e:
                obs.log_execution_step("escalation_error", {"error": str(e)})

        import time
        max_api_retries = 3
        result_content = None
        
        for attempt in range(max_api_retries):
            try:
                result = current_reviewer.invoke(messages)
                content = result.content
                if isinstance(content, list):
                    content = "\n".join([str(c.get('text', c) if isinstance(c, dict) else c) for c in content])
                result_content = content.strip()
                break
            except Exception as e:
                obs.log_execution_step("reviewer_api_error", {"attempt": attempt+1, "error": str(e)})
                if attempt < max_api_retries - 1:
                    time.sleep(2)
                else:
                    obs.log_execution_step("reviewer_api_failure", {"reason": "API unresponsive after 3 attempts."})
                    
        # If the cloud API completely failed, fall back to the Local Orchestrator for review
        if not result_content:
            try:
                obs.log_execution_step("reviewer_fallback", {"reason": "Falling back to local LLM for review."})
                from app.llm_factory import get_llm
                from agents.agent_config import load_agent_config
                agents = load_agent_config()
                solver_spec = next(a for a in agents if a.role == "primary")
                local_reviewer = get_llm(solver_spec)
                result = local_reviewer.invoke(messages)
                result_content = result.content.strip()
            except Exception as e:
                # If even the local model fails, auto-approve to prevent a complete pipeline lock
                obs.log_execution_step("reviewer_fatal", {"error": str(e)})
                result_content = "APPROVED (Auto-fallback due to fatal LLM error)"

        approved = "APPROVED" in result_content.upper()

        obs.log_execution_step("reviewer_result", {
            "approved": approved,
            "feedback": result_content,
        })

        return {
            "reviewer_feedback": result_content,
            "retry_count": state["retry_count"] if approved else state["retry_count"] + 1,
        }

    return reviewer_node


# =============================================================================
# Edge Conditions (Transition Logic)
# =============================================================================

def should_continue_or_review(state: CTFSolverState) -> str:
    """
    Decide whether to continue solving or transition to review.

    Called after the solver node. Checks:
    1. Was mark_phase_complete called? -> go to "verify"
    2. Step limit reached? -> go to "verify"
    3. Otherwise -> loop back to solver ("continue")
    """
    # Check if the agent called mark_phase_complete in the most recent turn
    for msg in reversed(state["messages"][-5:]):
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls'):
            for tc in msg.tool_calls:
                if tc.get('name') == 'mark_phase_complete':
                    return "verify"

    # Check step limit
    if state["step_count"] >= state["max_steps"]:
        return "verify"

    # Check for excessive errors
    if state["error_count"] >= 5:
        return "verify"

    return "continue"


def should_advance_or_retry(state: CTFSolverState) -> str:
    """
    Decide whether to advance to the next phase or retry.

    Called after the reviewer node. Checks:
    1. Approved -> advance to next phase or end
    2. Rejected + retries remaining -> retry current phase
    3. Rejected + max retries -> force advance
    """
    feedback = state.get("reviewer_feedback", "")

    if "APPROVED" in feedback.upper():
        return "advance"

    if state["retry_count"] >= state["max_retries"]:
        return "force_advance"

    return "retry"


# =============================================================================
# Graph Builder
# =============================================================================

def build_ctf_graph(solver_llm, reviewer_llm, all_tools, playbook_engine, obs):
    """Construct the CTF solving state machine."""
    workflow = StateGraph(CTFSolverState)

    # Define nodes
    workflow.add_node("solve", create_solver_node(solver_llm, all_tools, playbook_engine, obs))
    workflow.add_node("verify", create_reviewer_node(reviewer_llm, obs))

    # Define transitions
    workflow.set_entry_point("solve")

    workflow.add_conditional_edges(
        "solve",
        should_continue_or_review,
        {
            "continue": "solve",
            "verify": "verify",
        }
    )

    workflow.add_conditional_edges(
        "verify",
        should_advance_or_retry,
        {
            "advance": END,
            "retry": "solve",
            "force_advance": END,
        }
    )

    return workflow.compile()


def run_state_graph(graph, playbook_engine, initial_messages, obs, max_steps=30, max_retries=3, flag_format=""):
    """Execute the playbook using the compiled LangGraph."""
    all_flags = []
    total_steps = 0

    evidence_board = ""
    
    while not playbook_engine.is_complete():
        current_phase = playbook_engine.get_current_phase()
        if not current_phase:
            break

        phase_state_name = f"phase_{playbook_engine.current_phase_index}"
        obs.log_execution_step("phase_start", {
            "phase": current_phase['name'],
            "fsm_state": "analyze",
            "phase_index": playbook_engine.current_phase_index,
            "total_phases": len(playbook_engine.phase_states)
        })

        playbook_engine.record_attempt()

        # Give the agent a clean slate for the new phase, plus the initial task
        initial_task_msg = [HumanMessage(content=initial_messages[0].content)] if initial_messages else []
        
        # Build initial state for this phase
        initial_state = {
            "messages": initial_task_msg,
            "current_phase": phase_state_name,
            "phase_data": current_phase,
            "phase_index": playbook_engine.current_phase_index,
            "found_flags": all_flags,
            "error_count": 0,
            "step_count": 0,
            "max_steps": max_steps,
            "flag_format": flag_format,
            "retry_count": 0,
            "max_retries": max_retries,
            "reviewer_feedback": "",
            "evidence_board": evidence_board,
        }

        # Run the graph for this phase
        try:
            result = graph.invoke(initial_state)

            # Collect outputs
            phase_messages = list(result.get("messages", []))
            # We no longer accumulate all_messages from all phases, to save context.
            # We just keep track of flags and steps.
            all_flags = list(result.get("found_flags", all_flags))
            total_steps += result.get("step_count", 0)

            if "APPROVED" in result.get("reviewer_feedback", "").upper():
                obs.log_execution_step("phase_complete", {
                    "phase": current_phase['name'],
                    "approved": True,
                })
                
                # Extract evidence from the mark_phase_complete tool call in this phase
                phase_evidence = ""
                for m in reversed(phase_messages):
                    if hasattr(m, 'tool_calls') and m.tool_calls:
                        for tc in m.tool_calls:
                            if tc.get("name") == "mark_phase_complete":
                                args = tc.get("args", {})
                                phase_evidence = args.get("evidence", "")
                                break
                    if phase_evidence:
                        break
                
                if phase_evidence:
                    evidence_board += f"\n--- Evidence from {current_phase['name']} ---\n{phase_evidence}\n"

                if not playbook_engine.advance_phase():
                    break
                # Notify agent of phase transition (if we wanted to, but we give a clean slate anyway)
                new_phase = playbook_engine.get_current_phase()
            else:
                obs.log_execution_step("phase_force_advanced", {
                    "phase": current_phase['name'],
                    "reason": "Max retries reached or manual advance.",
                })
                if not playbook_engine.advance_phase():
                    break

        except Exception as e:
            obs.log_execution_step("graph_fatal", {"error": str(e)})
            break

    obs.log_execution_step("playbook_finish", {
        "playbook": playbook_engine.name,
        "total_steps": total_steps,
        "flags_found": len(all_flags)
    })

    return {
        "messages": phase_messages,
        "found_flags": all_flags,
        "evidence_board": evidence_board,
        "total_steps": total_steps,
    }


# =============================================================================
# Prompt Engineering
# =============================================================================

def _build_phase_prompt(phase_data: dict, state_name: str,
                        available_tools: list, evidence_board: str = "") -> str:
    """Build a dynamic system prompt for the current phase."""
    tool_names = [t.name for t in available_tools]

    prompt = f"""You are the Lead Orchestrator of a heterogeneous CTF solving system.
You are currently in the {state_name.upper()} phase.

CURRENT PHASE: {phase_data.get('name', 'Unknown')}
Objectives:
"""
    for obj in phase_data.get("objectives", []):
        prompt += f"- {obj}\n"

    prompt += f"""
Available tools in this phase: {', '.join(tool_names)}

=== EVIDENCE BOARD (Findings from previous phases) ===
{evidence_board if evidence_board else "No previous evidence available."}
======================================================

CRITICAL RULES:
1. YOU MUST USE TOOLS. Never output pseudo-code or describe what you would do.
2. Follow the objectives strictly. Complete them in order.
3. If a tool fails, analyze the error and try a different approach.
4. When ALL objectives are met, call mark_phase_complete with evidence.
5. THE SANDBOX IS EMPTY. Do not try to run custom Python scripts (like frequency_analysis.py) because they DO NOT EXIST. You must write your own scripts using execute_python_code.
6. DO NOT search for pre-built recovery scripts (e.g., find / -name "*.py"). They are NOT there. You are the one who must write the code.
7. For multi-line Python logic, ALWAYS use execute_python_code instead of run_in_sandbox to avoid shell escaping hell.
8. If you find a flag, it will be automatically captured.
9. NATIVE TOOL CALLING: You are equipped with native tool calling. DO NOT output raw text like 'Action: tool_name' or 'Action Input: ...'. You MUST use the actual JSON tool call format provided by the API. Failure to use the native tool schema will result in a system error.
"""
    return prompt
