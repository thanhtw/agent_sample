"""
UI Components for model setup and management in the Peer Review System.
"""

import streamlit as st
import os
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Import LLM Manager
from llm_manager import LLMManager

class ModelSetupUI:
    """UI components for setting up and managing LLM models."""
    
    def __init__(self, llm_manager: LLMManager):
        """
        Initialize the Model Setup UI.
        
        Args:
            llm_manager (LLMManager): LLM Manager instance
        """
        self.llm_manager = llm_manager
        load_dotenv()
    
    def render_ollama_setup(self, key_prefix: str = "main") -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Render Ollama-specific setup with a dropdown for all models.
        
        Args:
            key_prefix (str): Prefix for Streamlit widget keys
            
        Returns:
            Optional[Tuple[str, Dict[str, Any]]]: (Model name, model parameters) if a model is selected
                                                 and initialized, None otherwise
        """
        st.write("### Ollama Models")
        st.write("Select a model from the dropdown. Models with ✅ are already pulled.")
        
        # Get available models
        ollama_models = self.llm_manager.get_available_models()
        
        if not ollama_models:
            st.warning("No Ollama models found. Please make sure Ollama is running.")
            if st.button("Refresh Models", key=f"{key_prefix}_refresh"):
                st.rerun()
            return None
        
        # Format model names for dropdown - adding indicators for pulled models
        model_display_names = {}
        for model in ollama_models:
            if model.get("pulled", False):
                display_name = f"✅ {model['name']} - {model['description']}"
            else:
                display_name = f"⬇️ {model['name']} - {model['description']}"
            model_display_names[display_name] = model["id"]
        
        # Model selection dropdown
        selected_display_name = st.selectbox(
            "Select Model:",
            list(model_display_names.keys()),
            key=f"{key_prefix}_model_select"
        )
        
        if selected_display_name:
            model_id = model_display_names[selected_display_name]
            selected_model = next((m for m in ollama_models if m["id"] == model_id), None)
            
            # Show model information
            st.write(f"**Selected Model:** {selected_model['name']}")
            
            # Check if model is already pulled
            is_pulled = selected_model.get("pulled", False)
            
            if not is_pulled:
                # Show pull button for models that haven't been pulled yet
                if st.button(f"Pull {selected_model['name']} Model", key=f"{key_prefix}_pull"):
                    with st.spinner(f"Pulling {selected_model['name']}..."):
                        if self.llm_manager.download_ollama_model(model_id):
                            st.success(f"Successfully pulled {selected_model['name']}")
                            st.rerun()
                        else:
                            st.error(f"Failed to pull {selected_model['name']}")
            
            # Model parameters
            st.write("### Model Parameters")
            
            # Temperature slider
            temperature = st.slider(
                "Temperature:",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                key=f"{key_prefix}_temp",
                help="Higher values make output more random, lower values more deterministic."
            )
            
            # Store parameters
            model_params = {
                "temperature": temperature
            }
            
            # Initialize button - only enable if model is pulled
            if st.button("Initialize Selected Model", disabled=not is_pulled, key=f"{key_prefix}_init"):
                if not is_pulled:
                    st.warning("Please pull the model first before initializing.")
                    return None
                else:
                    with st.spinner(f"Initializing {selected_model['name']}..."):
                        llm = self.llm_manager.initialize_model(
                            model_id, 
                            model_params
                        )
                        
                        if llm:
                            st.session_state[f"{key_prefix}_llm_name"] = model_id
                            st.session_state[f"{key_prefix}_llm_params"] = model_params
                            st.session_state[f"{key_prefix}_llm_initialized"] = True
                            
                            # Store in session state for retrieval
                            st.success(f"Successfully initialized {selected_model['name']}!")
                            
                            return model_id, model_params
                        else:
                            st.error(f"Failed to initialize {selected_model['name']}.")
                            return None
            
            # If already initialized, return the model info
            if st.session_state.get(f"{key_prefix}_llm_initialized"):
                return st.session_state.get(f"{key_prefix}_llm_name"), st.session_state.get(f"{key_prefix}_llm_params")
                
        return None
    
    def render_model_config_tabs(self) -> Dict[str, Any]:
        """
        Render tabs for configuring different models in the workflow.
        
        Returns:
            Dict[str, Any]: Dictionary of model configurations
        """
        st.header("Model Configuration")
        st.write("Configure models for each stage of the peer review process.")
        
        # Initialize config dict
        model_configs = {
            "generative": None,
            "review": None,
            "summary": None,
            "compare": None
        }
        
        # Create tabs for each model role
        tabs = st.tabs(["Generative", "Review", "Summary", "Compare"])
        
        with tabs[0]:
            st.write("### Generative Model")
            st.write("This model generates code with intentional problems.")
            result = self.render_ollama_setup("generative")
            if result:
                model_configs["generative"] = {
                    "model_name": result[0],
                    "model_params": result[1]
                }
        
        with tabs[1]:
            st.write("### Review Model")
            st.write("This model analyzes student reviews.")
            result = self.render_ollama_setup("review")
            if result:
                model_configs["review"] = {
                    "model_name": result[0],
                    "model_params": result[1]
                }
        
        with tabs[2]:
            st.write("### Summary Model")
            st.write("This model summarizes review comments.")
            result = self.render_ollama_setup("summary")
            if result:
                model_configs["summary"] = {
                    "model_name": result[0],
                    "model_params": result[1]
                }
        
        with tabs[3]:
            st.write("### Compare Model")
            st.write("This model compares student reviews with known problems.")
            result = self.render_ollama_setup("compare")
            if result:
                model_configs["compare"] = {
                    "model_name": result[0],
                    "model_params": result[1]
                }
        
        # Option to use the same model for all roles
        st.write("### Quick Setup")
        if st.checkbox("Use the same model for all roles"):
            st.info("This will override any individual model settings above.")
            result = self.render_ollama_setup("all")
            if result:
                for role in model_configs:
                    model_configs[role] = {
                        "model_name": result[0],
                        "model_params": result[1]
                    }
        
        # Save configuration to environment
        if st.button("Save Configuration to Environment"):
            self._save_model_config_to_env(model_configs)
            st.success("Configuration saved! Restart the application to apply changes.")
        
        return model_configs
    
    def _save_model_config_to_env(self, model_configs: Dict[str, Any]):
        """
        Save model configurations to .env file.
        
        Args:
            model_configs (Dict[str, Any]): Dictionary of model configurations
        """
        # Load existing env vars
        env_vars = {}
        if os.path.exists(".env"):
            with open(".env", "r") as f:
                for line in f:
                    if line.strip() and "=" in line:
                        key, value = line.strip().split("=", 1)
                        env_vars[key] = value
        
        # Update with new values
        for role, config in model_configs.items():
            if config:
                env_vars[f"{role.upper()}_MODEL"] = config["model_name"]
                env_vars[f"{role.upper()}_TEMPERATURE"] = str(config["model_params"]["temperature"])
        
        # Write back to .env file
        with open(".env", "w") as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")