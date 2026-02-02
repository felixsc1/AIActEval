"""
Ollama utilities for EU AI Act Compliance Testing POC.

Handles Ollama model inference and connection management.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
import ollama


class OllamaConnectionError(Exception):
    """Raised when Ollama is not running or accessible."""
    pass


def check_ollama_connection(base_url: str = "http://localhost:11434") -> bool:
    """
    Check if Ollama is running and accessible.

    Args:
        base_url: Ollama server URL

    Returns:
        True if connection successful, False otherwise
    """
    try:
        client = ollama.Client(host=base_url)
        # Try to list models as a connection test
        client.list()
        return True
    except Exception as e:
        print(f"Ollama connection check failed: {e}")
        return False


def get_ollama_models(base_url: str = "http://localhost:11434") -> List[Any]:
    """
    Get list of available Ollama models.

    Args:
        base_url: Ollama server URL

    Returns:
        List of model objects (from ollama library, with attributes like 'name', 'size', etc.)

    Raises:
        OllamaConnectionError: If Ollama is not accessible
    """
    try:
        client = ollama.Client(host=base_url)
        response = client.list()
        # Response object has 'models' attribute, not dictionary key
        if hasattr(response, 'models'):
            return response.models
        # Fallback for dict-like responses
        elif isinstance(response, dict):
            return response.get('models', [])
        else:
            # Try to access as attribute or return empty list
            return getattr(response, 'models', [])
    except Exception as e:
        raise OllamaConnectionError(f"Cannot connect to Ollama at {base_url}: {e}")


def generate_llm_response(
    model: str,
    prompt: str,
    base_url: str = "http://localhost:11434",
    **kwargs
) -> str:
    """
    Generate a response from an Ollama model.

    Args:
        model: Ollama model name (e.g., 'llama3.2:3b')
        prompt: Input prompt to send to the model
        base_url: Ollama server URL
        **kwargs: Additional parameters for ollama.generate()

    Returns:
        Generated response text

    Raises:
        OllamaConnectionError: If Ollama is not accessible
    """
    try:
        client = ollama.Client(host=base_url)
        response = client.generate(
            model=model,
            prompt=prompt,
            stream=False,  # Get full response at once
            **kwargs
        )
        return response.get('response', '').strip()
    except Exception as e:
        raise OllamaConnectionError(f"Failed to generate response with {model}: {e}")