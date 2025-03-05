# test_app.py
import streamlit as st
from llm_manager import LLMManager
from model_setup_ui import ModelSetupUI

def main():
    st.title("LLM Test App")
    
    # Initialize LLM Manager
    llm_manager = LLMManager()
    model_setup_ui = ModelSetupUI(llm_manager)
    
    # Check Ollama connection
    st.header("Ollama Connection Check")
    import requests
    try:
        response = requests.get(f"{llm_manager.ollama_base_url}/api/tags")
        if response.status_code == 200:
            st.success(f"Successfully connected to Ollama at {llm_manager.ollama_base_url}")
            models = response.json().get("models", [])
            if models:
                st.write("Available models:")
                for model in models:
                    st.write(f"- {model.get('name')}")
            else:
                st.warning("No models found. You may need to pull a model first.")
                st.code("ollama pull llama3.2:1b", language="bash")
        else:
            st.error(f"Failed to connect to Ollama. Status code: {response.status_code}")
    except Exception as e:
        st.error(f"Error connecting to Ollama: {str(e)}")
        st.info("Make sure Ollama is running with 'ollama serve'")
    
    # Model setup
    st.header("Model Setup")
    model_setup_ui.render_ollama_setup()
    
    # Test generation if model is initialized
    st.header("Test Generation")
    if "main_llm_initialized" in st.session_state and st.session_state.main_llm_initialized:
        model_name = st.session_state.main_llm_name
        st.write(f"Using model: {model_name}")
        
        prompt = st.text_area("Enter prompt:", "Write a simple Python function to calculate factorial.")
        
        if st.button("Generate"):
            with st.spinner("Generating..."):
                llm = llm_manager.initialized_models.get(model_name)
                if llm:
                    response = llm.invoke(prompt)
                    st.code(response, language="python")
                else:
                    st.error("Model not found in initialized models. Please reinitialize.")

if __name__ == "__main__":
    main()