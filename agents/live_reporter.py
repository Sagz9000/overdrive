import os
from datetime import datetime

REPORT_PATH = "live_report.md"

def update_live_report(phase: str, step: int, reasoning: str, action: str, result: str):
    """Updates the live_report.md dashboard with the latest agent activity."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Create the report if it doesn't exist
    if not os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("# Project Overdrive: Live Progress Dashboard\n")
            f.write("Status: 🚀 Initializing \"COMMANDER\" mode...\n\n")

    # Read existing content
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Keep header (first 3 lines), prepend new step so newest is at top
    header = lines[:3]
    content = lines[3:]
    
    new_entry = [
        f"## [{timestamp}] Step {step} | Phase: {phase}\n\n",
        f"### Reasoning\n{reasoning}\n\n",
        f"### Action\n`{action}`\n\n",
        f"### Result\n```\n{result[:1000]}{'...' if len(result) > 1000 else ''}\n```\n\n",
        "---\n\n"
    ]
    
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.writelines(header)
        f.writelines(new_entry)
        f.writelines(content)

def update_report_status(status_text: str):
    """Updates the status line at the top of the report."""
    if not os.path.exists(REPORT_PATH):
        return
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if lines:
        lines[1] = f"Status: {status_text}\n"
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.writelines(lines)

def add_final_summary(summary: str):
    """Adds a MISSION COMPLETE block with the final evidence board."""
    update_live_report("COMPLETED", 0, "Mission Objectives Met.", "FINAL REPORT", summary)
    update_report_status("✅ MISSION COMPLETE")
