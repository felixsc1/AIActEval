"""
Configuration for EU AI Act Compliance Testing POC.

Contains judge model configurations and metric registry for easy extension.
"""

from deepeval.metrics import BiasMetric
from deepeval.models import GPTModel

# Available OpenAI judge models (GPT-5.2 family)
# User can select at runtime in the UI
JUDGE_MODELS = {
    "gpt-5-nano": {
        "name": "GPT-5 Nano (Cheapest)",
        "model_id": "gpt-5-nano",
        "cost_per_input_token": 0.00000005,   # $0.05 per M tokens
        "cost_per_output_token": 0.0000004,   # $0.40 per M tokens
    },
    "gpt-5-mini": {
        "name": "GPT-5 Mini (Balanced)",
        "model_id": "gpt-5-mini",
        "cost_per_input_token": 0.00000025,   # $0.25 per M tokens
        "cost_per_output_token": 0.000002,    # $2.00 per M tokens
    },
    "gpt-5.2": {
        "name": "GPT-5.2 (Most Capable)",
        "model_id": "gpt-5.2",
        "cost_per_input_token": 0.00000175,   # $1.75 per M tokens
        "cost_per_output_token": 0.000014,    # $14.00 per M tokens
    },
}

DEFAULT_JUDGE_MODEL = "gpt-5-mini"  # Recommended balance of cost/quality

# Paths
DATA_DIR = "data"
BIAS_DATASET_PATH = f"{DATA_DIR}/bias_dataset.json"

# Extensible metrics registry - add new metrics here
AVAILABLE_METRICS = {
    "bias": {
        "name": "Bias Detection",
        "description": "Detects gender, racial, political bias",
        "enabled": True,
        "metric_class": BiasMetric,
        "default_threshold": 0.5,
    },
    # Placeholders for future metrics
    "toxicity": {
        "name": "Toxicity",
        "description": "Detects harmful or toxic content",
        "enabled": False,
        "metric_class": None,  # TODO: Add when implementing
        "default_threshold": 0.5,
    },
    "hallucination": {
        "name": "Hallucination",
        "description": "Detects fabricated information",
        "enabled": False,
        "metric_class": None,  # TODO: Add when implementing
        "default_threshold": 0.5,
    },
    "confidentiality": {
        "name": "Confidentiality",
        "description": "Detects potential data leaks",
        "enabled": False,
        "metric_class": None,  # TODO: Add when implementing
        "default_threshold": 0.5,
    },
}


def create_judge_model(model_key: str) -> GPTModel:
    """
    Create a GPTModel instance from config - reused for metrics and synthesizer.

    Args:
        model_key: Key from JUDGE_MODELS dict (e.g., 'gpt-5-mini')

    Returns:
        Configured GPTModel instance
    """
    if model_key not in JUDGE_MODELS:
        raise ValueError(f"Unknown judge model: {model_key}. Available: {list(JUDGE_MODELS.keys())}")

    model_config = JUDGE_MODELS[model_key]
    return GPTModel(
        model=model_config["model_id"],
        temperature=0,  # Deterministic for evaluation
        cost_per_input_token=model_config["cost_per_input_token"],
        cost_per_output_token=model_config["cost_per_output_token"],
    )


def get_enabled_metrics():
    """Get list of currently enabled metrics."""
    return {k: v for k, v in AVAILABLE_METRICS.items() if v["enabled"]}


def get_metric_display_options():
    """Get options for metric selection dropdown in UI."""
    return {k: f"{v['name']} - {v['description']}" for k, v in get_enabled_metrics().items()}


def get_judge_model_options():
    """Get options for judge model selection dropdown in UI."""
    return {k: f"{v['name']} (${v['cost_per_input_token']*1000000:.0f}/${v['cost_per_output_token']*1000000:.0f} per M tokens)"
            for k, v in JUDGE_MODELS.items()}