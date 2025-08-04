import os
import sys
import subprocess
from dotenv import load_dotenv

# load env
load_dotenv()

def main():
    host = os.getenv("API_HOST", "localhost")
    port = os.getenv("API_PORT", "8000")
    
    print(f"Starting frontend...")
    print(f"Make sure backend is running on: http://{host}:{port}")
    print(f"Frontend will be available at: http://localhost:8501")

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "frontend/app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except Exception as e:
        print(f"Error in starting frontend: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()