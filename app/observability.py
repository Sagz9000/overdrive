import json
import os
import time

class Observability:
    def __init__(self, log_dir="logs"):
        self.log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.run_id = f"run_{int(time.time())}"
        self.log_file = os.path.join(self.log_dir, f"{self.run_id}.jsonl")
        
        # Touch the file immediately so it exists
        open(self.log_file, "a").close()
        
    def log_execution_step(self, step_type, data):
        log_entry = {
            "timestamp": time.time(),
            "run_id": self.run_id,
            "type": step_type,
            "data": data
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
    def log_error(self, error_details):
        self.log_execution_step("error", {"error": error_details})
