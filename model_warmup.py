import requests
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3.2:1b")

def warm_up_model():
    """
    Send a simple prompt to initialize the model.
    This first call can take much longer than subsequent calls.
    """
    print(f"Warming up model {DEFAULT_MODEL}...")
    start_time = time.time()
    
    try:
        # Simple prompt to warm up the model
        prompt = "Hello, please respond with a short greeting."
        
        # Request body
        data = {
            "model": DEFAULT_MODEL,
            "prompt": prompt,
            "stream": False
        }
        
        # Send request and measure time
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=data)
        
        if response.status_code == 200:
            elapsed = time.time() - start_time
            print(f"✅ Model warmed up successfully in {elapsed:.2f} seconds")
            return True
        else:
            print(f"❌ Warmup failed. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error during warmup: {str(e)}")
        return False

if __name__ == "__main__":
    warm_up_model()