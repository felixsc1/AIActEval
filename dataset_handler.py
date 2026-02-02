"""
Dataset management for EU AI Act Compliance Testing POC.

Handles loading and saving of datasets.
"""

import json
import os
from typing import List, Dict, Any
from config import BIAS_DATASET_PATH


def load_dataset(filepath: str = BIAS_DATASET_PATH) -> List[Dict[str, Any]]:
    """
    Load dataset from JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        List of test case dictionaries, each with at least 'input' key
    """
    if not os.path.exists(filepath):
        # Return empty dataset if file doesn't exist
        return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Validate structure - each item should have 'input'
            validated_data = []
            for item in data:
                if isinstance(item, dict) and 'input' in item:
                    validated_data.append(item)
                else:
                    print(f"Warning: Skipping invalid test case: {item}")
            return validated_data
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading dataset from {filepath}: {e}")
        return []


def save_dataset(dataset: List[Dict[str, Any]], filepath: str = BIAS_DATASET_PATH) -> None:
    """
    Save dataset to JSON file.

    Args:
        dataset: List of test case dictionaries
        filepath: Path to save the JSON file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"Dataset saved to {filepath} ({len(dataset)} test cases)")
    except Exception as e:
        print(f"Error saving dataset to {filepath}: {e}")
        raise