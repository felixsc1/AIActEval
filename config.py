"""
Configuration for EU AI Act Compliance Testing POC.

Contains judge model configurations and path settings.
"""

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
EVALUATIONS_DIR = f"{DATA_DIR}/evaluations"


def get_judge_model_options():
    """Get options for judge model selection dropdown in UI."""
    return {k: f"{v['name']} (${v['cost_per_input_token']*1000000:.0f}/${v['cost_per_output_token']*1000000:.0f} per M tokens)"
            for k, v in JUDGE_MODELS.items()}