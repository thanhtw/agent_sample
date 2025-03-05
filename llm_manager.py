"""
LLM Manager for Peer Review Agent.

This module handles LLM initialization, configuration, and management
for the peer code review agent, integrating with LangChainLLM.
"""

import os
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_core.language_models import BaseLanguageModel

from langchain_llm import LangChainLLM, LangChainManager


class LLMManager:
    """
    LLM Manager for handling model initialization, configuration and management.
    """
    
    def __init__(self):
        """Initialize the LLM Manager with environment variables."""
        load_dotenv()
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.default_model = os.getenv("DEFAULT_MODEL", "llama3.2:1b")
        
        # Initialize the LangChain manager
        self.langchain_manager = LangChainManager()
        
        # Track initialized models
        self.initialized_models = {}
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of available Ollama models.
        
        Returns:
            List[Dict[str, Any]]: List of model information dictionaries
        """
        return self.langchain_manager.get_ollama_models()
    
    def initialize_model(self, model_name: str, provider: str = "ollama", 
                          model_params: Optional[Dict[str, Any]] = None) -> Optional[BaseLanguageModel]:
        """
        Initialize a model using LangChainLLM.
        
        Args:
            model_name (str): Name of the model to initialize
            provider (str): Provider name (default: "ollama")
            model_params (Dict[str, Any], optional): Model parameters
            
        Returns:
            Optional[BaseLanguageModel]: Initialized LLM or None if initialization fails
        """
        if model_name in self.initialized_models:
            print(f"Using cached model: {model_name}")
            return self.initialized_models[model_name]
            
        # Apply default model parameters if none provided
        if model_params is None:
            model_params = self._get_default_params(model_name)
        
        # Initialize via LangChainManager
        try:
            llm = self.langchain_manager.initialize_llm(model_name, provider, model_params)
            if llm:
                # Cache the initialized model
                self.initialized_models[model_name] = llm
                return llm
            else:
                print(f"Failed to initialize {model_name}")
                return None
        except Exception as e:
            print(f"Error initializing model {model_name}: {str(e)}")
            return None
    
    def initialize_model_from_env(self, model_key: str, temperature_key: str) -> Optional[BaseLanguageModel]:
        """
        Initialize a model using environment variables.
        
        Args:
            model_key (str): Environment variable key for model name
            temperature_key (str): Environment variable key for temperature
            
        Returns:
            Optional[BaseLanguageModel]: Initialized LLM or None if initialization fails
        """
        model_name = os.getenv(model_key, self.default_model)
        temperature = float(os.getenv(temperature_key, "0.7"))
        
        model_params = {
            "temperature": temperature
        }
        
        return self.initialize_model(model_name, "ollama", model_params)
    
    def fallback_to_ollama(self, model_name: str, temperature: float = 0.7) -> Optional[BaseLanguageModel]:
        """
        Fallback to direct Ollama initialization if LangChainLLM fails.
        
        Args:
            model_name (str): Name of the model to initialize
            temperature (float): Temperature setting
            
        Returns:
            Optional[BaseLanguageModel]: Initialized Ollama LLM or None if initialization fails
        """
        try:
            llm = Ollama(
                base_url=self.ollama_base_url,
                model=model_name,
                temperature=temperature
            )
            return llm
        except Exception as e:
            print(f"Fallback to direct Ollama failed: {str(e)}")
            return None
    
    def _get_default_params(self, model_name: str) -> Dict[str, Any]:
        """
        Get default parameters for a specific model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Dict[str, Any]: Default parameters for the model
        """
        # Basic defaults
        params = {
            "temperature": 0.7,
            "max_tokens": 512
        }
        
        # Adjust based on model name
        if "generative" in model_name or any(gen in model_name for gen in ["llama3", "llama-3"]):
            params["temperature"] = 0.8  # Slightly higher creativity for generative tasks
            
        elif "review" in model_name or any(rev in model_name for gen in ["mistral", "deepseek"]):
            params["temperature"] = 0.3  # Lower temperature for review tasks
            
        elif "summary" in model_name:
            params["temperature"] = 0.4  # Moderate temperature for summary tasks
            
        elif "compare" in model_name:
            params["temperature"] = 0.5  # Balanced temperature for comparison tasks
        
        return params