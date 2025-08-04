import os
import sys
import subprocess
from dotenv import load_dotenv

# load env
load_dotenv()

def main():
    host = os.getenv("API_HOST", "localhost")
    backend_port = os.getenv("API_PORT", "8000")
    frontend_port = os.getenv("FRONTEND_PORT", "8501")
    
    print(f"Starting frontend...")
    print(f"Make sure backend is running on: http://{host}:{backend_port}")
    print(f"Frontend will be available at: http://localhost:{frontend_port}")

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "frontend/app.py",
            "--server.port", f"{frontend_port}",
            "--server.address", "localhost"
        ])
    except Exception as e:
        print(f"Error in starting frontend: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()