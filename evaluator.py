"""
Evaluation engine for EU AI Act Compliance Testing POC.

Handles Ollama model inference and DeepEval metric evaluation.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
import ollama
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BiasMetric
from config import create_judge_model, get_enabled_metrics


class OllamaConnectionError(Exception):
    """Raised when Ollama is not running or accessible."""
    pass


class APIKeyMissingError(Exception):
    """Raised when required API keys are missing."""
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


def check_api_keys() -> Tuple[bool, List[str]]:
    """
    Check if required API keys are available.

    Returns:
        Tuple of (all_keys_present, list_of_missing_keys)
    """
    required_keys = ['OPENAI_API_KEY', 'CONFIDENT_API_KEY']
    missing_keys = []

    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)

    return len(missing_keys) == 0, missing_keys


def run_bias_evaluation(
    dataset: List[Dict[str, Any]],
    ollama_model: str,
    judge_model_key: str = "gpt-5-mini",
    threshold: float = 0.5,
    strict_mode: bool = False,
    ollama_base_url: str = "http://localhost:11434"
) -> Dict[str, Any]:
    """
    Run bias evaluation on dataset using Ollama for responses and DeepEval for analysis.

    Args:
        dataset: List of test cases with 'input' field
        ollama_model: Ollama model name to test
        judge_model_key: GPT model to use as judge
        threshold: Bias detection threshold (0.0-1.0)
        strict_mode: Binary scoring if True
        ollama_base_url: Ollama server URL

    Returns:
        Dictionary with evaluation results and metadata

    Raises:
        APIKeyMissingError: If required API keys are missing
        OllamaConnectionError: If Ollama is not accessible
    """
    # Check API keys
    keys_ok, missing_keys = check_api_keys()
    if not keys_ok:
        raise APIKeyMissingError(f"Missing required API keys: {', '.join(missing_keys)}")

    # Check Ollama connection
    if not check_ollama_connection(ollama_base_url):
        raise OllamaConnectionError("Ollama is not running or not accessible")

    # Create the judge model (e.g., gpt-5-mini, gpt-5-nano)
    judge_model = create_judge_model(judge_model_key)

    # Generate test cases
    test_cases = []
    generation_errors = []

    print(f"Generating responses for {len(dataset)} test cases using {ollama_model}...")

    for i, item in enumerate(dataset):
        try:
            # Step 1: Generate response from local Ollama model (model under test)
            response = generate_llm_response(
                model=ollama_model,
                prompt=item["input"],
                base_url=ollama_base_url
            )

            # Step 2: Create test case with input and actual_output
            test_case = LLMTestCase(
                input=item["input"],
                actual_output=response
            )
            test_cases.append(test_case)

            if (i + 1) % 5 == 0:  # Progress update every 5 cases
                print(f"Generated {i + 1}/{len(dataset)} responses...")

        except Exception as e:
            error_msg = f"Failed to generate response for case {i}: {e}"
            print(error_msg)
            generation_errors.append(error_msg)
            continue

    if not test_cases:
        return {
            "success": False,
            "error": "No test cases could be generated",
            "generation_errors": generation_errors
        }

    print(f"Running bias evaluation on {len(test_cases)} test cases...")

    try:
        # Step 3: Create BiasMetric with custom judge model (overrides default GPT-4.1)
        metric = BiasMetric(
            model=judge_model,       # Use selected GPT-5.x model as judge
            threshold=threshold,     # User-configurable threshold
            strict_mode=strict_mode, # Binary scoring if True
            include_reason=True      # Always include explanation
        )

        # Step 4: Run evaluation - auto-logs to Confident AI dashboard
        results = evaluate(test_cases=test_cases, metrics=[metric])

        return {
            "success": True,
            "test_cases_evaluated": len(test_cases),
            "ollama_model": ollama_model,
            "judge_model": judge_model_key,
            "threshold": threshold,
            "strict_mode": strict_mode,
            "generation_errors": generation_errors,
            "results": results
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Evaluation failed: {e}",
            "test_cases_evaluated": len(test_cases),
            "generation_errors": generation_errors
        }


def get_evaluation_summary(eval_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract summary statistics from evaluation results.

    Args:
        eval_results: Results from run_bias_evaluation

    Returns:
        Dictionary with summary statistics
    """
    if not eval_results.get("success"):
        return {"status": "failed", "error": eval_results.get("error", "Unknown error")}

    # This would need to be adapted based on actual DeepEval results structure
    # For now, return basic info
    return {
        "status": "completed",
        "test_cases": eval_results["test_cases_evaluated"],
        "model_tested": eval_results["ollama_model"],
        "judge_used": eval_results["judge_model"],
        "threshold": eval_results["threshold"],
        "strict_mode": eval_results["strict_mode"],
        "generation_errors": len(eval_results.get("generation_errors", []))
    }


def get_confident_ai_dashboard_url() -> str:
    """
    Get the URL for viewing results on Confident AI dashboard.

    Returns:
        Dashboard URL
    """
    # This would ideally get the specific evaluation URL from DeepEval
    # For now, return the general dashboard URL
    return "https://app.confident-ai.com"