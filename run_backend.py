import os
import sys
import uvicorn
from dotenv import load_dotenv

# load env
load_dotenv()

def main():
    host = os.getenv("API_HOST", "localhost")
    port = int(os.getenv("API_PORT", 8000))
    
    print(f"Starting Backend API server...")
    print(f"Server will run on: http://{host}:{port}")
    
    try:
        uvicorn.run("backend.main:app", 
            host=host, port=port,
            reload=True, log_level="info"
        )
    except Exception as e:
        print(f"Error in starting server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()