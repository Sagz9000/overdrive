import streamlit as st
import os
import sys
import json
import time
import subprocess

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from playbooks.engine import PlaybookEngine

def update_env_file(key, value):
    """Helper to persist API keys to the .env file."""
    env_path = os.path.join(project_root, ".env")
    if not os.path.exists(env_path):
        with open(env_path, "w") as f:
            f.write(f"{key}={value}\n")
        return
    
    with open(env_path, "r") as f:
        lines = f.readlines()
    
    updated = False
    new_lines = []
    for line in lines:
        if line.startswith(f"{key}="):
            new_lines.append(f"{key}={value}\n")
            updated = True
        else:
            new_lines.append(line)
    
    if not updated:
        new_lines.append(f"{key}={value}\n")
    
    with open(env_path, "w") as f:
        f.writelines(new_lines)
    
    # Update current process environment
    os.environ[key] = value

st.set_page_config(page_title="Project Overdrive", layout="wide", page_icon="🎯")

st.title("🎯 Project Overdrive: CTF MAS")

# State
if "run_started" not in st.session_state:
    st.session_state.run_started = False
if "log_file" not in st.session_state:
    st.session_state.log_file = None
if "cmd_proc" not in st.session_state:
    st.session_state.cmd_proc = None

col1, col2 = st.columns([1, 2])

with col1:
    st.header("⚙️ Configuration")
    target = st.text_input("Target IP/Domain (leave blank for file-only challenges)", placeholder="e.g. scanme.nmap.org")
    
    # Model selector — query Ollama for available models
    available_models = ["mistral:7b", "gemma4:e2b", "qwen2.5-coder:7b", "gemma4:e4b", "gemma:7b"]
    try:
        result = subprocess.run(
            ["docker", "exec", "project_overdrive_ollama", "ollama", "list"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            pulled_models = [line.split()[0] for line in lines if line.strip()]
            if pulled_models:
                available_models = pulled_models
    except Exception:
        pass
    
    selected_model = st.selectbox("🤖 Local Model", available_models, index=0)
    
    # Challenge category
    category = st.selectbox("🏷️ Challenge Category", [
        "Auto-detect", "Crypto", "Forensics", "Web", "Reversing", 
        "Steganography", "Network", "OSINT", "Miscellaneous"
    ])
    
    mode = st.radio("Execution Mode", ["Pre-defined Playbook", "Dynamic CTF Question"])
    
    playbook_dir = os.path.join(project_root, "playbooks")
    playbooks = [f for f in os.listdir(playbook_dir) if f.endswith(".yaml")]
    selected_playbook = None
    question = ""
    flag_format = ""
    
    if mode == "Pre-defined Playbook":
        selected_playbook = st.selectbox("Select Playbook", playbooks)
    else:
        question = st.text_area("CTF Challenge Question", "Find the hidden flag on port 80.")
        flag_format = st.text_input("Known Flag Format (Optional)", "HTB{")
        uploaded_file = st.file_uploader("Upload CTF File (Optional)")
    
    with st.expander("🔑 API Settings"):
        st.caption("Enable cloud features like the Gemini Reviewer or Vision tools.")
        gemini_key = st.text_input("Gemini API Key", type="password", value=os.environ.get("GOOGLE_API_KEY", ""))
        if st.button("Save API Key"):
            update_env_file("GOOGLE_API_KEY", gemini_key)
            st.success("API Key saved to .env!")
    
    if st.button("🚀 Start Execution", disabled=st.session_state.run_started):
        st.session_state.run_started = True
        log_dir = os.path.join(project_root, "app", "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        cmd = [sys.executable, os.path.join(project_root, "agents", "react_loop.py"), target]
        if mode == "Pre-defined Playbook":
            cmd.extend(["--playbook", selected_playbook])
        else:
            # Inject category into question if specified
            q = question
            if category != "Auto-detect":
                q = f"[Category: {category}] {question}"
            cmd.extend(["--question", q])
            if flag_format:
                cmd.extend(["--flag-format", flag_format])
            if uploaded_file is not None:
                upload_dir = os.path.join(project_root, "app", "uploads")
                os.makedirs(upload_dir, exist_ok=True)
                file_path = os.path.join(upload_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                cmd.extend(["--file", file_path])
                
        # Get existing logs to detect the new one
        existing_logs = set(os.listdir(log_dir)) if os.path.exists(log_dir) else set()
        
        env = os.environ.copy()
        env["LLM_PROVIDER"] = "ollama"
        env["USE_LOCAL_LLM"] = "true"
        env["LOCAL_MODEL_NAME"] = selected_model
        env["LOCAL_API_BASE"] = "http://localhost:11434"
        st.session_state.cmd_proc = subprocess.Popen(cmd, env=env)
        
        # Poll for new log file
        new_log = None
        for _ in range(30):  # wait up to 30 seconds
            current_logs = set(os.listdir(log_dir)) if os.path.exists(log_dir) else set()
            new_logs = current_logs - existing_logs
            if new_logs:
                new_log = os.path.join(log_dir, list(new_logs)[0])
                break
            time.sleep(1)
            
        if new_log:
            st.session_state.log_file = new_log
        else:
            st.error("Timed out waiting for the execution log file to be created. Please try again.")
            st.session_state.run_started = False

def _render_log_line(line):
    """Parse and render a single JSONL log line to the Streamlit UI."""
    try:
        data = json.loads(line)
        step_type = data.get("type", "unknown")
        
        if step_type == "tool_call":
            tool_name = data['data'].get('tool', 'unknown')
            st.write(f"🔧 **Tool Call:** `{tool_name}`")
            detail = data['data'].get('command', data['data'].get('reasoning', data['data'].get('path', '')))
            if detail:
                st.code(detail)
                
        elif step_type == "tool_result":
            st.write(f"✅ **Tool Result:**")
            result = data['data'].get('result', {})
            if isinstance(result, dict) and 'output' in result:
                output = result['output']
                st.code(output[:2000] + ("..." if len(output) > 2000 else ""))
            elif isinstance(result, str):
                st.code(result[:2000])
            else:
                st.write(result)
                
        elif step_type == "agent_start":
            st.info(f"🚀 **Agent Started:** {data['data'].get('task')}")
            
        elif step_type == "agent_thought":
            thought = data['data'].get('thought', '')
            if len(thought) > 1500:
                thought = thought[:1500] + "..."
            st.markdown(f"🤔 **Agent Reasoning:**\n> {thought}")
            
        elif step_type == "agent_finish":
            st.success(f"🏁 **Agent Finished:**\n{data['data'].get('response')}")
            
        elif step_type == "reviewer_feedback":
            st.warning(f"👀 **Reviewer Feedback:**\n{data['data'].get('critique')}")
            
        elif step_type == "phase_start":
            phase_num = data['data'].get('phase_num', '?')
            total = data['data'].get('total_phases', '?')
            phase_name = data['data'].get('phase')
            objectives = data['data'].get('objectives', [])
            st.divider()
            st.info(f"📋 **Phase {phase_num}/{total}: {phase_name}**")
            if objectives:
                obj_text = "\n".join(f"  • {o}" for o in objectives)
                st.caption(f"Objectives:\n{obj_text}")
            
        elif step_type == "phase_complete":
            st.success(f"✅ **Phase Complete:** {data['data'].get('phase')} → Next: {data['data'].get('next_phase')}")
            
        elif step_type == "playbook_complete":
            st.balloons()
            st.success(f"🎉 **Playbook Complete!** Final phase: {data['data'].get('final_phase')}")
            
        elif step_type == "planner_start":
            st.info(f"🧠 **Planner Selecting Playbook...**\nQuestion: {data['data'].get('question')}")
            
        elif step_type == "planner_result":
            st.success(f"📝 **Selected Static Playbook:** `{data['data'].get('playbook_selected')}`")
            
        elif step_type == "file_injected":
            st.info(f"📁 **File Injected:** `{data['data'].get('filename')}` → `{data['data'].get('path')}`")
            
        elif step_type == "flag_found":
            st.balloons()
            st.success(f"🚩🚩🚩 **FLAG FOUND:** `{data['data'].get('flag')}`")
            
        elif step_type == "error":
            st.error(f"❌ **Execution Error:**\n{data['data'].get('error')}")
            
    except Exception:
        pass


with col2:
    st.header("📡 Live Observability Feed")
    
    if st.session_state.run_started and st.session_state.log_file:
        with open(st.session_state.log_file, 'r') as f:
            f.seek(0)
            # Main polling loop — runs while agent subprocess is alive
            while st.session_state.cmd_proc and st.session_state.cmd_proc.poll() is None:
                line = f.readline()
                if not line:
                    time.sleep(0.5)
                    continue
                _render_log_line(line)
            
            # CRITICAL: Read any remaining buffered lines after process exits
            remaining = f.readlines()
            for line in remaining:
                _render_log_line(line)
                
        st.success("✅ Execution Complete.")
        st.session_state.run_started = False

    try:
        data = json.loads(line)
        step_type = data.get("type", "unknown")
        
        if step_type == "tool_call":
            tool_name = data['data'].get('tool', 'unknown')
            st.write(f"🔧 **Tool Call:** `{tool_name}`")
            detail = data['data'].get('command', data['data'].get('reasoning', data['data'].get('path', '')))
            if detail:
                st.code(detail)
                
        elif step_type == "tool_result":
            st.write(f"✅ **Tool Result:**")
            result = data['data'].get('result', {})
            if isinstance(result, dict) and 'output' in result:
                output = result['output']
                st.code(output[:2000] + ("..." if len(output) > 2000 else ""))
            elif isinstance(result, str):
                st.code(result[:2000])
            else:
                st.write(result)
                
        elif step_type == "agent_start":
            st.info(f"🚀 **Agent Started:** {data['data'].get('task')}")
            
        elif step_type == "agent_thought":
            thought = data['data'].get('thought', '')
            # Truncate very long thoughts
            if len(thought) > 1500:
                thought = thought[:1500] + "..."
            st.markdown(f"🤔 **Agent Reasoning:**\n> {thought}")
            
        elif step_type == "agent_finish":
            st.success(f"🏁 **Agent Finished:**\n{data['data'].get('response')}")
            
        elif step_type == "reviewer_feedback":
            st.warning(f"👀 **Reviewer Feedback:**\n{data['data'].get('critique')}")
            
        elif step_type == "phase_complete":
            st.success(f"✅ **Phase Complete:** {data['data'].get('phase')} → Next: {data['data'].get('next_phase')}")
            
        elif step_type == "playbook_complete":
            st.balloons()
            st.success(f"🎉 **Playbook Complete!** Final phase: {data['data'].get('final_phase')}")
            
        elif step_type == "planner_start":
            st.info(f"🧠 **Planner Formulating Playbook...**\nQuestion: {data['data'].get('question')}")
            
        elif step_type == "planner_result":
            st.success(f"📝 **Dynamic Playbook Generated:**")
            st.code(data['data'].get('playbook'), language="yaml")
            
        elif step_type == "file_injected":
            st.info(f"📁 **File Injected:** `{data['data'].get('filename')}` → `{data['data'].get('path')}`")
            
        elif step_type == "flag_found":
            st.balloons()
            st.success(f"🚩🚩🚩 **FLAG FOUND:** `{data['data'].get('flag')}`")
            
        elif step_type == "error":
            st.error(f"❌ **Execution Error:**\n{data['data'].get('error')}")
            
    except Exception:
        pass
