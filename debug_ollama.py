"""
Debug script to test Ollama connection and model availability.
Run this script to diagnose issues with Ollama before running the main application.
"""
import requests
import json
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3:21b")

def print_separator():
    print("\n" + "=" * 50 + "\n")

def test_ollama_connection():
    """Test basic connectivity to Ollama server."""
    print(f"Testing connection to Ollama at {OLLAMA_BASE_URL}...")
    
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        
        if response.status_code == 200:
            print("✅ Successfully connected to Ollama!")
            return True, response.json()
        else:
            print(f"❌ Failed to connect to Ollama. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False, None
    except requests.RequestException as e:
        print(f"❌ Connection error: {str(e)}")
        return False, None

def check_model_availability(data, model_name):
    """Check if the specified model is available in Ollama."""
    if not data or "models" not in data:
        print("❌ No models information available")
        return False
    
    models = data.get("models", [])
    if not models:
        print("❌ No models found in Ollama")
        return False
    
    # Check if our model exists
    for model in models:
        if model.get("name") == model_name:
            print(f"✅ Model '{model_name}' is available!")
            return True
    
    print(f"❌ Model '{model_name}' NOT found. Available models:")
    for model in models:
        print(f"  - {model.get('name')}")
    
    return False

def test_model_generation():
    """Test if the model can generate simple text."""
    print(f"Testing text generation with model '{DEFAULT_MODEL}'...")
    
    try:
        # Simple prompt to test generation
        prompt = "Write a short 'Hello World' function in Python."
        
        # Request body
        data = {
            "model": DEFAULT_MODEL,
            "prompt": prompt,
            "stream": False
        }
        
        # Send request
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Model generated text successfully!")
            print("\nGenerated text sample:")
            print("------------------------")
            print(result.get("response", "")[:200] + "..." if len(result.get("response", "")) > 200 else result.get("response", ""))
            return True
        else:
            print(f"❌ Generation failed. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except requests.RequestException as e:
        print(f"❌ Error during generation: {str(e)}")
        return False

def check_ollama_service():
    """Check if Ollama service is running."""
    print("Checking if Ollama service is running...")
    
    import platform
    import subprocess
    
    system = platform.system()
    
    if system == "Linux":
        try:
            result = subprocess.run(["systemctl", "is-active", "ollama"], capture_output=True, text=True)
            if "active" in result.stdout:
                print("✅ Ollama service is active (systemd)")
                return True
            else:
                print("❌ Ollama service is not active (systemd)")
                return False
        except:
            print("⚠️ Could not check Ollama service status via systemd")
    
    # Generic check by trying to connect
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/", timeout=2)
        print("✅ Ollama server is responding to requests")
        return True
    except:
        print("❌ Ollama server is not responding to requests")
        return False

def main():
    """Run all diagnostic tests."""
    print_separator()
    print("OLLAMA DIAGNOSTIC TOOL")
    print_separator()
    
    # Check if Ollama service is running
    service_status = check_ollama_service()
    print_separator()
    
    if not service_status:
        print("⚠️ Ollama service doesn't appear to be running!")
        print("\nTo start Ollama:")
        print("  - Linux/Mac: Run 'ollama serve' in a terminal")
        print("  - Windows: Open the Ollama application")
        print_separator()
        return
    
    # Test connection to Ollama
    connection_success, data = test_ollama_connection()
    print_separator()
    
    if not connection_success:
        print("⚠️ Cannot connect to Ollama. Please check:")
        print(f"  1. Ollama is running (ollama serve)")
        print(f"  2. The URL is correct: {OLLAMA_BASE_URL}")
        print(f"  3. No firewall is blocking the connection")
        print_separator()
        return
    
    # Check model availability
    model_available = check_model_availability(data, DEFAULT_MODEL)
    print_separator()
    
    if not model_available:
        print(f"⚠️ Model '{DEFAULT_MODEL}' is not available. To install it:")
        print(f"  Run: ollama pull {DEFAULT_MODEL}")
        print_separator()
        return
    
    # Test model generation
    generation_success = test_model_generation()
    print_separator()
    
    if not generation_success:
        print("⚠️ Model generation test failed. Check Ollama logs for more details.")
        print_separator()
        return
    
    # All tests passed
    print("🎉 All diagnostic tests passed! You can run the main application now.")
    print_separator()

if __name__ == "__main__":
    main()