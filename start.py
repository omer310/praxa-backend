"""
Startup script that runs both the FastAPI server and LiveKit agent worker.
Used for single-service deployment on Railway.
"""

import subprocess
import sys
import os
import signal

def main():
    port = os.getenv("PORT", "8000")
    
    # Start FastAPI server
    web_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "main:app", 
        "--host", "0.0.0.0", 
        "--port", port
    ])
    
    # Start LiveKit agent worker
    # Use the praxa_agent module directly
    agent_process = subprocess.Popen([
        sys.executable, "agent/praxa_agent.py", "dev"
    ])
    
    def signal_handler(signum, frame):
        """Handle shutdown gracefully."""
        print("Shutting down...")
        web_process.terminate()
        agent_process.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Wait for either process to exit
    try:
        while True:
            web_exit = web_process.poll()
            agent_exit = agent_process.poll()
            
            if web_exit is not None:
                print(f"Web server exited with code {web_exit}")
                agent_process.terminate()
                sys.exit(web_exit)
            
            if agent_exit is not None:
                print(f"Agent worker exited with code {agent_exit}")
                web_process.terminate()
                sys.exit(agent_exit)
            
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()
