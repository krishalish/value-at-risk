import subprocess
import sys
import os
from pathlib import Path

def check_directories():
    """Check if required directories exist."""
    required_dirs = ['config', 'logs', 'data', 'reports', 'plots', 'output']
    for dir_name in required_dirs:
        Path(dir_name).mkdir(exist_ok=True)

def check_config():
    """Check if config file exists."""
    if not Path('config/config.yaml').exists():
        print("Error: config/config.yaml not found!")
        sys.exit(1)

def main():
    """Run the VaR system."""
    # Check setup
    check_directories()
    check_config()
    
    try:
        # Start FastAPI server
        api_process = subprocess.Popen(
            [sys.executable, '-m', 'uvicorn', 'var_system.api:app', '--port', '8000'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("API server started on http://localhost:8000")
        
        # Start scheduler
        scheduler_process = subprocess.Popen(
            [sys.executable, '-m', 'var_system.scheduler'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("Scheduler started")
        
        # Keep running until interrupted
        api_process.wait()
        scheduler_process.wait()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
        api_process.terminate()
        scheduler_process.terminate()
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 