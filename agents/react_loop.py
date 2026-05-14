import os
import sys
import argparse
from typing import TypedDict, Annotated, Sequence
import operator
import time
import re
import traceback

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from sandbox.sandbox_manager import SandboxManager
from app.llm_factory import get_llm, get_npu_summarizer
from app.observability import Observability
from playbooks.engine import PlaybookEngine
from agents.agent_config import load_agent_config, get_agent_by_role, get_agent_by_name
from agents.guardrails import run_input_guardrails, run_output_guardrails, Verdict
from agents.handoff import build_handoff_context, execute_handoff

from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END

# Global state
sm = None
obs = None
container_id = None
playbook_engine = None
npu_model = None
agent_requested_review = False
agent_reasoning = ""
flag_format_regex = None
found_flags = []
_solver_llm_for_fallback = None  # Global reference for NPU fallback
executed_commands = []
agent_specs = []
active_guardrails = []
step_count = 0
max_steps = 50


def _check_for_flags(text):
    global flag_format_regex, found_flags
    if flag_format_regex and text:
        matches = flag_format_regex.findall(text)
        for match in matches:
            if match not in found_flags:
                found_flags.append(match)
                obs.log_execution_step("flag_found", {"flag": match})
                print(f"\n[FLAG] FLAG FOUND: {match}")


# =============================================================================
# Tools
# =============================================================================

@tool
def run_in_sandbox(command: str = None, **kwargs) -> str:
    """Executes a shell command inside the isolated Linux sandbox and returns stdout/stderr.
    REQUIRED PARAMETER: 'command' (str).
    Example: {"command": "ls -la"}
    """
    global container_id, executed_commands, step_count
    
    # Improved fallback for hallucinated parameters
    if not command and kwargs:
        if 'script_path' in kwargs:
            command = f"python3 {kwargs['script_path']}"
        elif 'code' in kwargs:
            command = f"python3 -c \"{kwargs['code']}\""
        else:
            # Stitch all kwargs together into a single command attempt
            # e.g., {"script": "python3", "args": "-c '...'"} -> "python3 -c '...'"
            parts = []
            for k, v in kwargs.items():
                if isinstance(v, list):
                    parts.append(" ".join(str(x) for x in v))
                else:
                    parts.append(str(v))
            command = " ".join(parts)
            
    if not command:
        return "ERROR: You must pass a single string under the key 'command'."

    step_count += 1
    if not container_id:
        container_id = sm.start_container()

    if step_count > max_steps:
        return f"STOP: Step limit ({max_steps}) reached. Call mark_phase_complete now."

    guard_result = run_input_guardrails(command, active_guardrails)
    if guard_result.verdict == Verdict.BLOCK:
        obs.log_execution_step("guardrail_block", {"command": command, "reason": guard_result.reason})
        return f"BLOCKED: {guard_result.reason}"
    if guard_result.verdict == Verdict.WARN:
        obs.log_execution_step("guardrail_warn", {"command": command, "reason": guard_result.reason})

    if command in executed_commands:
        return f"ERROR: You already ran this exact command. Try a DIFFERENT command."
    executed_commands.append(command)

    obs.log_execution_step("tool_call", {"tool": "run_in_sandbox", "command": command})
    result = sm.execute_command(container_id, command)

    if "Error: Container not found" in result['output']:
        container_id = None
        return "Error: Sandbox container was lost. Restarting. Please retry."

    _check_for_flags(result['output'])

    output_guard = run_output_guardrails(result['output'], ["sanitize_output"])
    output_text = output_guard.sanitized_content or result['output']

    obs.log_execution_step("tool_result", {"tool": "run_in_sandbox", "result": result})

    # Self-correction loop: structured error nudge on failure
    if result['exit_code'] != 0:
        error_msg = output_text
        if "No such file or directory" in error_msg:
            error_msg += "\nTIP: Use 'ls' or 'find' to verify the file exists before running it."
            
        return (f"COMMAND FAILED (exit code {result['exit_code']}):\n"
                f"stderr: {error_msg}\n\n"
                f"ACTION REQUIRED: Analyze this error and try a DIFFERENT approach. "
                f"Do NOT repeat the exact same command.")

    # Sync workspace after successful execution to provide evidence to user
    if result['exit_code'] == 0:
        sm.sync_workspace(container_id, "app/evidence_storage")

    return f"Exit Code: {result['exit_code']}\nOutput:\n{output_text}"


@tool
def execute_python_code(code: str) -> str:
    """Executes a block of Python code inside the sandbox.
    Use this instead of run_in_sandbox for multi-line Python scripts to avoid escaping issues.
    REQUIRED PARAMETER: 'code' (str) containing the raw python script.
    """
    global container_id
    if not container_id:
        container_id = sm.start_container()
        
    # Save the raw code to a temporary file in the sandbox and execute it
    import base64
    encoded_code = base64.b64encode(code.encode('utf-8')).decode('utf-8')
    cmd = f"echo '{encoded_code}' | base64 -d > /tmp/agent_script.py && python3 /tmp/agent_script.py"
    
    result = sm.execute_command(container_id, f'bash -c "{cmd}"')
    return f"Exit Code: {result['exit_code']}\nOutput:\n{result['output']}"


@tool
def write_file_in_sandbox(file_path: str, content: str) -> str:
    """Writes a file with given content inside the sandbox.
    Use to create Python scripts, config files, or text files for analysis.
    """
    global container_id, step_count
    step_count += 1
    if not container_id:
        container_id = sm.start_container()

    obs.log_execution_step("tool_call", {"tool": "write_file_in_sandbox", "file_path": file_path, "content_length": len(content)})

    import base64
    encoded = base64.b64encode(content.encode()).decode()
    cmd = f"echo '{encoded}' | base64 -d > {file_path}"
    result = sm.execute_command(container_id, f'bash -c "{cmd}"')

    if result['exit_code'] == 0:
        msg = f"Successfully wrote {len(content)} bytes to {file_path}"
        sm.sync_workspace(container_id, "app/evidence_storage")
    else:
        msg = f"Failed to write file: {result['output']}"

    obs.log_execution_step("tool_result", {"tool": "write_file_in_sandbox", "result": msg})
    return msg


@tool
def read_file_from_sandbox(file_path: str, max_bytes: int = 4096) -> str:
    """Reads contents of a file inside the sandbox.
    For binary files, use xxd or hexdump instead.
    """
    global container_id, step_count
    step_count += 1
    if not container_id:
        container_id = sm.start_container()

    obs.log_execution_step("tool_call", {"tool": "read_file_from_sandbox", "file_path": file_path})
    result = sm.execute_command(container_id, f"head -c {max_bytes} {file_path}")
    output = result['output']
    _check_for_flags(output)
    obs.log_execution_step("tool_result", {"tool": "read_file_from_sandbox", "result": output[:1000]})
    return f"Exit Code: {result['exit_code']}\nContent:\n{output}"


@tool
def mark_phase_complete(evidence: str = None, **kwargs) -> str:
    """Marks the current playbook phase as complete.
    REQUIRED PARAMETER: 'evidence' (str).
    Call when you have fulfilled all objectives of the current phase.
    """
    global agent_requested_review, agent_reasoning
    
    # Handle hallucinated parameters (message, note, reasoning, etc.)
    final_evidence = evidence or ""
    if not final_evidence and kwargs:
        # Stitch all kwargs into a single evidence string
        final_evidence = "\n".join([f"{k}: {v}" for k, v in kwargs.items()])
    
    if not final_evidence:
        return "ERROR: You must provide 'evidence' of what you accomplished. Example: {'evidence': 'I found the XOR key 0x42'}"

    agent_requested_review = True
    agent_reasoning = final_evidence
    obs.log_execution_step("tool_call", {"tool": "mark_phase_complete", "evidence": final_evidence})
    msg = "Review requested. The Reviewer agent will now evaluate your work."
    obs.log_execution_step("tool_result", {"tool": "mark_phase_complete", "result": msg})
    return msg


@tool
def multimodal_analysis(file_path: str, prompt: str) -> str:
    """Uses a multimodal vision/audio model to analyze a file in the sandbox.
    Use for images (.png, .jpg), audio (.wav, .mp3), or PDFs.
    """
    global container_id, step_count
    step_count += 1
    if not container_id:
        container_id = sm.start_container()

    obs.log_execution_step("tool_call", {"tool": "multimodal_analysis", "file_path": file_path, "prompt": prompt})
    check = sm.execute_command(container_id, f"ls {file_path}")
    if check['exit_code'] != 0:
        return f"Error: File {file_path} not found in sandbox."

    local_path = os.path.join(project_root, "temp_multimodal_" + os.path.basename(file_path))
    success = sm.copy_from_sandbox(container_id, file_path, local_path)
    if not success:
        return f"Error: Failed to copy {file_path} from sandbox."

    try:
        from app.llm_factory import get_llm_for_vision
        from langchain_core.messages import HumanMessage as HM
        import base64

        vision_model = get_llm_for_vision()
        if not vision_model:
            return "Error: Vision model not available."

        with open(local_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        ext = file_path.split('.')[-1].lower()
        mime_map = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
                    'wav': 'audio/wav', 'mp3': 'audio/mpeg', 'pdf': 'application/pdf'}
        mime_type = mime_map.get(ext, 'image/jpeg')

        message = HM(content=[
            {"type": "text", "text": f"You are a CTF Forensics expert. Analyze this file.\nQuestion: {prompt}"},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}}
        ])
        response = vision_model.invoke([message])
        result = response.content

        if os.path.exists(local_path):
            os.remove(local_path)

        obs.log_execution_step("tool_result", {"tool": "multimodal_analysis", "result": result})
        return f"Multimodal Analysis Result:\n{result}"
    except Exception as e:
        if os.path.exists(local_path):
            os.remove(local_path)
        return f"Error performing multimodal analysis: {str(e)}"


@tool
def search_knowledge_base(query: str, **kwargs) -> str:
    """Search the CTF knowledge base for cheat sheets, payloads, and techniques.
    REQUIRED PARAMETER: 'query' (str).
    """
    global step_count
    
    # Handle hallucinated parameters
    if not query and kwargs:
        query = next(iter(kwargs.values())) if kwargs else ""
        
    if not query:
        return "ERROR: Missing 'query' parameter. What do you want to search for?"

    step_count += 1
    obs.log_execution_step("tool_call", {"tool": "search_knowledge_base", "query": query})

    try:
        from rag.knowledge_base import get_knowledge_base
        kb = get_knowledge_base()
        result = kb.search_formatted(query)
        obs.log_execution_step("tool_result", {"tool": "search_knowledge_base", "result": result[:500]})
        return result
    except Exception as e:
        msg = f"Knowledge base unavailable: {str(e)}. Proceeding without RAG."
        obs.log_execution_step("tool_result", {"tool": "search_knowledge_base", "result": msg})
        return msg


@tool
def send_to_npu(data: str, task: str) -> str:
    """Delegate heavy text parsing to the NPU for condensed JSON summary.
    Use for large terminal output (nmap scans, directory listings, logs).
    The NPU returns a condensed JSON summary of key findings.
    """
    global step_count, npu_model
    step_count += 1
    obs.log_execution_step("tool_call", {"tool": "send_to_npu", "task": task, "text_length": len(data)})

    # Truncate very large input to prevent NPU overload
    if len(data) > 8000:
        data = data[:8000] + "\n[TRUNCATED]"

    try:
        if npu_model is None:
            npu_model = get_npu_summarizer()

        if npu_model is None:
            # High-quality fallback: use the GPU model (Solver LLM) if NPU is down
            return _llm_fallback_parse(data, task)

        messages = [
            SystemMessage(content="You are a utility parser. Return ONLY valid JSON. "
                          "Extract: open ports, services, files, credentials, vulnerabilities, flags."),
            HumanMessage(content=f"Task: {task}\n\nRaw output:\n{data}")
        ]
        result = npu_model.invoke(messages)
        content = result.content.strip()
        obs.log_execution_step("tool_result", {"tool": "send_to_npu", "result": content[:500]})
        return f"NPU Summary:\n{content}"

    except Exception as e:
        obs.log_execution_step("tool_result", {"tool": "send_to_npu", "error": str(e)})
        return _llm_fallback_parse(data, task)


def _llm_fallback_parse(raw_text: str, task: str) -> str:
    """GPU-based LLM fallback when NPU is unavailable: Uses the primary Brain."""
    global _solver_llm_for_fallback
    
    if _solver_llm_for_fallback is None:
        return f"Fallback Error: NPU and GPU fallbacks both unavailable."

    obs.log_execution_step("npu_gpu_fallback", {"task": task})
    
    messages = [
        SystemMessage(content="You are a utility parser. Extract key technical details from the raw data. "
                      "Focus on: open ports, interesting files, potential vulnerabilities, and credentials. "
                      "Output a concise summary."),
        HumanMessage(content=f"Task: {task}\n\nRaw output:\n{raw_text}")
    ]
    
    try:
        result = _solver_llm_for_fallback.invoke(messages)
        content = result.content.strip()
        return f"GPU-Summarized Findings (NPU Unavailable):\n{content}"
    except Exception as e:
        return f"Error during GPU fallback: {str(e)}"


# =============================================================================
# Playbook helpers
# =============================================================================

def _get_fallback_playbook(question, filename=None):
    q_lower = question.lower()
    if any(kw in q_lower for kw in ['xor', 'decrypt', 'encrypt', 'key', 'cipher', 'crypto']):
        return 'crypto_analysis.yaml'
    elif any(kw in q_lower for kw in ['forensic', 'image', 'hidden', 'stego', 'embedded']):
        return 'forensics.yaml'
    elif any(kw in q_lower for kw in ['pcap', 'packet', 'wireshark', 'network']):
        return 'network_analysis.yaml'
    elif any(kw in q_lower for kw in ['reverse', 'binary', 'elf', 'executable']):
        return 'binary_analysis.yaml'
    elif filename:
        return 'forensics.yaml'
    else:
        return 'web_exploitation.yaml'


def get_system_prompt(filename=None):
    global playbook_engine
    try:
        from tools.loader import load_tool_descriptors, generate_tools_prompt
        descriptors = load_tool_descriptors()
        tools_text = generate_tools_prompt(descriptors)
    except Exception:
        tools_text = "Available tools: nmap, curl, file, strings, xxd, binwalk, python3, bash"

    base_prompt = f"""You are the Lead Orchestrator of a heterogeneous CTF solving system.
You run on the GPU and handle strategic reasoning, attack planning, and exploit writing.

{tools_text}

Your agent tools:
- run_in_sandbox: Execute any shell command
- execute_python_code: Execute multi-line Python scripts reliably
- write_file_in_sandbox: Create files inside the sandbox
- read_file_from_sandbox: Read file contents
- search_knowledge_base: Query the RAG knowledge base for payloads/syntax
- send_to_npu: Delegate large output parsing to the NPU for condensed JSON
- multimodal_analysis: Analyze images/audio/PDFs via vision model
- mark_phase_complete: Call when you finish a phase

CRITICAL RULES:
1. YOU MUST USE TOOLS. Never output pseudo-code tool calls as text.
2. For large terminal output (nmap, dirb), delegate to send_to_npu.
3. For payload/syntax lookups, use search_knowledge_base.
4. BINARY SAFETY: Never read_file_from_sandbox on .enc/.jpg/.bin files. Use xxd.
5. If a command fails, analyze the error and try a corrected approach.
6. When all objectives are met, call mark_phase_complete with evidence.
7. THE SANDBOX IS EMPTY. Do not try to run custom Python scripts (like frequency_analysis.py) because they DO NOT EXIST. If you need a script, you MUST create it first using write_file_in_sandbox, OR use `python3 -c "inline code"`.
8. For multi-line Python logic, ALWAYS use execute_python_code instead of run_in_sandbox to avoid shell escaping hell."""

    if filename:
        base_prompt += f"\n\nFile uploaded at /workspace/{filename}. Start by analyzing it."

    current_phase = playbook_engine.get_current_phase()
    if current_phase:
        phase_prompt = f"\n\n--- CURRENT PHASE: {current_phase['name']} ---\nObjectives:\n"
        for obj in current_phase.get('objectives', []):
            phase_prompt += f"  - {obj}\n"
        phase_prompt += "Recommended Tools:\n"
        for rt in current_phase.get('recommended_tools', []):
            phase_prompt += f"  - {rt}\n"
        phase_prompt += "\nWhen you have completed ALL objectives, call mark_phase_complete."
        return base_prompt + phase_prompt
    else:
        return base_prompt + "\n\nAll playbook phases are complete. Summarize findings."


class OverdriveState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# =============================================================================
# Main
# =============================================================================

def main():
    global sm, obs, container_id, playbook_engine, agent_requested_review
    global flag_format_regex, found_flags, agent_specs, active_guardrails
    global step_count, max_steps, npu_model

    parser = argparse.ArgumentParser(description="Project Overdrive - Heterogeneous CTF Solver")
    parser.add_argument("target", nargs="?", default="scanme.nmap.org", help="Target IP or Domain")
    parser.add_argument("--playbook", default=None, help="Playbook file name")
    parser.add_argument("--question", default=None, help="CTF question for playbook auto-selection")
    parser.add_argument("--flag-format", default="", help="Expected flag format (e.g., CTF{)")
    parser.add_argument("--file", default=None, help="Path to uploaded CTF challenge file")
    parser.add_argument("--legacy", action="store_true", help="Use legacy while-loop instead of state graph")
    parser.add_argument("--use-state-graph", action="store_true", default=True,
                        help="Use LangGraph state graph FSM (default)")
    args = parser.parse_args()

    obs = Observability()

    # Load agent config
    agent_specs = load_agent_config()
    solver_spec = get_agent_by_role(agent_specs, "primary")
    reviewer_spec = get_agent_by_role(agent_specs, "reviewer")
    npu_spec = get_agent_by_role(agent_specs, "support")

    if solver_spec:
        active_guardrails = solver_spec.guardrails
        max_steps = solver_spec.max_steps

    if args.flag_format:
        escaped = re.escape(args.flag_format)
        flag_format_regex = re.compile(escaped + r'[^}]*\}')

    try:
        print("Initializing Project Overdrive (Heterogeneous Compute)...")
        print(f"  GPU: {solver_spec.provider}/{solver_spec.model if solver_spec else 'default'}")
        if npu_spec:
            print(f"  NPU: {npu_spec.provider}/{npu_spec.model}")
        print(f"  RAG: {'enabled' if os.environ.get('RAG_ENABLED', 'true').lower() == 'true' else 'disabled'}")

        sm = SandboxManager()

        # Create LLMs from agent specs
        solver_llm = get_llm(solver_spec)
        reviewer_llm = get_llm(reviewer_spec) if reviewer_spec else solver_llm

        # Initialize NPU model lazily (on first send_to_npu call)
        npu_model = None

        all_tools = [run_in_sandbox, execute_python_code, write_file_in_sandbox, read_file_from_sandbox,
                     multimodal_analysis, search_knowledge_base, send_to_npu,
                     mark_phase_complete]

        # Handle file upload
        filename = None
        if args.file and os.path.exists(args.file):
            filename = os.path.basename(args.file)
            if not container_id:
                container_id = sm.start_container()
            sm.copy_file(container_id, args.file, "/workspace")
            obs.log_execution_step("file_injected", {"filename": filename, "path": f"/workspace/{filename}"})

        # Select playbook
        playbook_path = ""
        if args.question:
            print("\nGenerating Dynamic Playbook from Question...")
            obs.log_execution_step("planner_start", {"question": args.question})

            # Retrieve RAG context
            rag_context = ""
            try:
                from rag.knowledge_base import get_knowledge_base
                kb = get_knowledge_base()
                rag_context = kb.search_formatted(args.question)
            except Exception as e:
                print(f"RAG unavailable for planning: {e}")

            prompt = f"""You are an expert CTF planner. Generate a custom YAML playbook to solve the user's request.
Target: {args.target}
Question: {args.question}
"""
            if filename:
                prompt += f"File provided: /workspace/{filename}\n"
            
            if rag_context:
                prompt += f"\nRelevant Knowledge Base Context:\n{rag_context}\n"

            prompt += """
You must output ONLY valid YAML. No markdown formatting, no ```yaml code blocks, no explanations.
The YAML MUST follow this EXACT structure:
name: "Dynamic Custom Playbook"
description: "Generated plan"
metadata:
  difficulty: medium
  category: custom
phases:
  - id: phase_1
    name: "Phase 1 Name"
    depends_on: []
    allowed_tools: [run_in_sandbox, search_knowledge_base]
    max_attempts: 3
    objectives:
      - "Specific command to run or objective to achieve"
    recommended_tools: [tool_name]
"""
            import time
            max_retries = 3
            yaml_content = ""
            
            for attempt in range(max_retries):
                try:
                    # Use HumanMessage for Gemini compatibility
                    planner_result = reviewer_llm.invoke([HumanMessage(content=prompt)])
                    content = planner_result.content
                    if isinstance(content, list):
                        content = "\n".join([str(c.get('text', c) if isinstance(c, dict) else c) for c in content])
                    yaml_content = content.strip()
                    break
                except Exception as e:
                    print(f"Planner API error (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                    else:
                        print("Primary Planner failed after 3 attempts. Falling back to Local Coder...")
                        try:
                            # Fallback to local Qwen model for planning
                            planner_result = solver_llm.invoke([HumanMessage(content=prompt)])
                            content = planner_result.content
                            if isinstance(content, list):
                                content = "\n".join([str(c.get('text', c) if isinstance(c, dict) else c) for c in content])
                            yaml_content = content.strip()
                            print("Local Coder successfully generated the dynamic playbook.")
                        except Exception as e2:
                            print(f"Local fallback planner also failed: {e2}")

            if yaml_content:
                # Robust extraction: find the start of the YAML block
                if "name:" in yaml_content:
                    yaml_content = yaml_content[yaml_content.find("name:"):]
                
                # Strip markdown code blocks if the model ignored instructions
                if yaml_content.startswith("```"):
                    lines = yaml_content.split('\n')
                    if len(lines) > 2:
                        yaml_content = '\n'.join(lines[1:-1])
                
                # Save dynamic playbook
                ts = int(time.time())
                selected = f"dynamic_{ts}.yaml"
                playbook_path = os.path.join(project_root, "playbooks", selected)
                with open(playbook_path, "w", encoding="utf-8") as f:
                    f.write(yaml_content)
                
                print(f"Generated dynamic playbook: {selected}")
                obs.log_execution_step("planner_result", {"playbook_selected": selected})
            else:
                print("Failed to generate dynamic playbook. Using static fallback.")
                selected = _get_fallback_playbook(args.question, filename)
                playbook_path = os.path.join(project_root, "playbooks", selected)
        else:
            playbook_name = args.playbook if args.playbook else "web_enum.yaml"
            playbook_path = os.path.join(project_root, "playbooks", playbook_name)

        playbook_engine = PlaybookEngine(playbook_path)
        print(f"Loaded playbook: {playbook_engine.name}")
        obs.log_execution_step("playbook_loaded", {
            "name": playbook_engine.name,
            "total_phases": len(playbook_engine.phase_states),
        })

        # Build task message
        task_parts = []
        if args.question:
            task_parts.append(f"User Request: {args.question}")
        if args.target:
            task_parts.append(f"Target: {args.target}")
        if filename:
            task_parts.append(f"File to analyze: /workspace/{filename}")
            
        task_parts.append(f"Execute the {playbook_engine.name} playbook to accomplish this.")
        task = "\n".join(task_parts)
        print(f"\nTask: {task}\n")
        obs.log_execution_step("agent_start", {"task": task, "playbook": playbook_engine.name})

        # ── State Graph execution (default) ──
        if args.use_state_graph and not args.legacy:
            print("Using LangGraph State Graph FSM...\n")
            from agents.state_graph import build_ctf_graph, run_state_graph

            global _solver_llm_for_fallback
            _solver_llm_for_fallback = solver_llm
            
            graph = build_ctf_graph(solver_llm, reviewer_llm, all_tools, playbook_engine, obs)
            initial_messages = [HumanMessage(content=task)]

            result = run_state_graph(
                graph, playbook_engine, initial_messages, obs,
                max_steps=max_steps, max_retries=3,
                flag_format=args.flag_format
            )

            found_flags.extend(result.get("found_flags", []))
            
            # Print the accumulated evidence board as the final summary
            final_answer = result.get("evidence_board", "").strip()
            
            # Fallback: if no evidence found, use the last substantial AI message
            if not final_answer:
                final_msgs = result.get("messages", [])
                for m in reversed(final_msgs):
                    if isinstance(m, AIMessage) and m.content:
                        final_answer = m.content
                        break

            if final_answer:
                print("\n" + "="*40)
                print("FINAL SUMMARY OF FINDINGS:")
                print("="*40)
                print(final_answer)
                print("="*40 + "\n")
                
                # Update the live UI report with the final summary
                from agents.live_reporter import add_final_summary
                add_final_summary(final_answer)

        # ── Legacy while-loop execution ──
        else:
            print("Using legacy while-loop execution...\n")
            _run_legacy_loop(solver_llm, reviewer_llm, all_tools, task, filename,
                             solver_spec, reviewer_spec)

        # Final report and Evidence Archive
        print("\n" + "="*40)
        print("MISSION COMPLETE - EVIDENCE ARCHIVED")
        print("="*40)
        
        evidence_path = os.path.abspath("app/evidence_storage")
        print(f"📁 Workspace Artifacts: {evidence_path}")
        
        # Create a ZIP archive for easy download/sharing
        try:
            import shutil
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_name = f"mission_results_{timestamp}"
            shutil.make_archive(zip_name, 'zip', "app/evidence_storage")
            print(f"📦 Results Zipped: {os.path.abspath(zip_name + '.zip')}")
        except Exception as e:
            print(f"Warning: Could not create zip archive: {e}")

        if found_flags:
            print(f"\n🚩 Flags Captured: {found_flags}")
            obs.log_execution_step("agent_finish", {"response": f"Flags found: {found_flags}"})
        else:
            obs.log_execution_step("agent_finish", {"response": "Playbook fully executed."})
        
        print("\nAll tasks completed.")


    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        if obs:
            obs.log_error(str(e))
        sys.exit(1)
    finally:
        if container_id and sm:
            sm.teardown(container_id)


def _run_legacy_loop(solver_llm, reviewer_llm, all_tools, task, filename,
                     solver_spec, reviewer_spec):
    """Legacy while-loop execution path (--legacy flag)."""
    global agent_requested_review, agent_reasoning, step_count

    def build_prompt(state):
        return get_system_prompt(filename)

    phase_tool_outputs = []
    messages = [HumanMessage(content=task)]
    max_phase_retries = 3

    while not playbook_engine.is_complete():
        agent_requested_review = False
        phase_tool_outputs.clear()
        step_count = 0

        current_phase = playbook_engine.get_current_phase()
        phase_state = playbook_engine.get_current_phase_state()
        phase_num = playbook_engine.current_phase_index + 1
        total_phases = len(playbook_engine.phase_states)
        playbook_engine.record_attempt()

        print(f"\n--- Phase {phase_num}/{total_phases}: {current_phase['name']} ---")
        obs.log_execution_step("phase_start", {
            "phase": current_phase['name'], "phase_num": phase_num,
            "total_phases": total_phases,
        })

        retries = 0
        while not agent_requested_review and retries < max_phase_retries:
            if step_count >= max_steps:
                print(f"\n[WARN] Step limit ({max_steps}) reached.")
                break

            agent = create_react_agent(solver_llm, all_tools, prompt=build_prompt)
            config = {"recursion_limit": 300}
            inputs = {"messages": messages, "remaining_steps": 250}

            try:
                result = agent.invoke(inputs, config)
            except Exception as e:
                print(f"\n[WARN] LangGraph error: {e}. Retrying...")
                inputs["messages"] = messages[-5:]
                result = agent.invoke(inputs, config)

            messages = result["messages"]

            if not agent_requested_review:
                nudge = (f"[Attempt {retries+2}/{max_phase_retries}] Phase '{current_phase['name']}'. "
                         f"Please complete objectives or call mark_phase_complete.")
                messages.append(HumanMessage(content=nudge))
                retries += 1

        if not agent_requested_review:
            if not playbook_engine.advance_phase():
                break
            continue

        # Reviewer handoff
        handoff_ctx = build_handoff_context(
            from_agent=solver_spec.name if solver_spec else "lead_orchestrator",
            to_agent=reviewer_spec.name if reviewer_spec else "flag_verifier",
            phase_name=current_phase['name'],
            phase_objectives=current_phase.get('objectives', []),
            reasoning=agent_reasoning,
            tool_evidence=phase_tool_outputs[-10:],
            found_flags=found_flags,
            steps_taken=step_count,
        )
        obs.log_execution_step("handoff", handoff_ctx.to_dict())

        handoff_result = execute_handoff(reviewer_llm, handoff_ctx,
                                          reviewer_instructions=reviewer_spec.instructions if reviewer_spec else "")
        obs.log_execution_step("reviewer_feedback", {"critique": handoff_result.feedback,
                                                      "approved": handoff_result.approved})

        if handoff_result.approved:
            if playbook_engine.advance_phase():
                new_phase = playbook_engine.get_current_phase()
                msg = f"Reviewer APPROVED. Next phase: {new_phase['name']}."
            else:
                msg = "Reviewer APPROVED. Playbook complete!"
            print(f"\n{msg}")
            messages.append(HumanMessage(content=msg))
        else:
            can_retry = playbook_engine.rollback_phase()
            if can_retry:
                msg = f"Reviewer REJECTED: {handoff_result.feedback}. Retry phase."
            else:
                msg = f"Reviewer REJECTED, max attempts. Force-advancing."
                playbook_engine.advance_phase()
            print(f"\n{msg}")
            messages.append(HumanMessage(content=msg))


if __name__ == "__main__":
    main()
