"""
Utility Bias Test Run Storage

Handles persistent storage of utility bias test runs as JSON files.
Provides functions for saving, listing, and loading test results.
"""

import json
import os
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime


def get_utility_bias_runs_dir() -> Path:
    """
    Get the directory for storing utility bias test runs.
    Creates the directory if it doesn't exist.

    Returns:
        Path to the utility bias runs directory
    """
    runs_dir = Path("data/utility_bias_runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def serialize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Serialize a pandas DataFrame to a JSON-serializable format.

    Args:
        df: DataFrame to serialize

    Returns:
        Dict containing 'columns' and 'data' for reconstruction
    """
    return {
        'columns': df.columns.tolist(),
        'data': df.to_dict('records')
    }


def deserialize_dataframe(obj: Dict[str, Any]) -> pd.DataFrame:
    """
    Deserialize a DataFrame from the JSON format created by serialize_dataframe.

    Args:
        obj: Dict containing 'columns' and 'data'

    Returns:
        Reconstructed DataFrame
    """
    return pd.DataFrame(obj['data'], columns=obj['columns'])


def generate_run_id(model_name: str) -> str:
    """
    Generate a unique run ID based on timestamp and model name.

    Args:
        model_name: Name of the model used in the test

    Returns:
        Unique run identifier string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Clean model name for filename (remove special chars, shorten if needed)
    clean_model = "".join(c for c in model_name.split(':')[0] if c.isalnum() or c in ('_', '-')).strip()
    if len(clean_model) > 20:
        clean_model = clean_model[:20]
    unique_suffix = str(uuid.uuid4())[:8]
    return f"{timestamp}_{clean_model}_{unique_suffix}"


def save_utility_bias_run(payload: Dict[str, Any]) -> Path:
    """
    Save a utility bias test run to a JSON file.

    Args:
        payload: Complete run data including config, results, and metadata

    Returns:
        Path to the saved JSON file
    """
    runs_dir = get_utility_bias_runs_dir()

    # Generate run ID if not provided
    if 'run_id' not in payload:
        model_name = payload.get('model_info', {}).get('ollama_model', 'unknown_model')
        payload['run_id'] = generate_run_id(model_name)

    # Set creation timestamp if not provided
    if 'created_at' not in payload:
        payload['created_at'] = datetime.now().isoformat()

    # Set schema version
    payload['version'] = 1

    # Ensure run_id is a valid filename
    run_id = payload['run_id']
    safe_filename = "".join(c for c in run_id if c.isalnum() or c in ('_', '-')).strip()
    if not safe_filename:
        safe_filename = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    filepath = runs_dir / f"{safe_filename}.json"

    # Write JSON with nice formatting
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    return filepath


def list_utility_bias_runs() -> List[Dict[str, Any]]:
    """
    List all available utility bias test runs with lightweight metadata.

    Returns:
        List of dicts with basic run information (no heavy dataframes loaded)
    """
    runs_dir = get_utility_bias_runs_dir()
    runs = []

    for json_file in runs_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                # Read only the metadata fields we need for listing
                data = json.load(f)

                run_info = {
                    'run_id': data.get('run_id', json_file.stem),
                    'filepath': str(json_file),
                    'created_at': data.get('created_at', 'Unknown'),
                    'model': data.get('model_info', {}).get('ollama_model', 'Unknown'),
                    'anchor_key': data.get('test_config', {}).get('anchor_key', 'Unknown'),
                    'num_ethnicities': len(data.get('test_config', {}).get('ethnicities', [])),
                    'num_n_values': len(data.get('test_config', {}).get('n_values', [])),
                    'num_anchor_variations': data.get('test_config', {}).get('num_anchor_variations', 0),
                    'status': data.get('run_metadata', {}).get('status', 'Unknown'),
                    'total_queries': data.get('run_metadata', {}).get('total_queries', 0),
                    'notes': data.get('notes', '')
                }

                runs.append(run_info)

        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            # Skip corrupted or inaccessible files
            continue

    # Sort by creation date (newest first)
    runs.sort(key=lambda x: x.get('created_at', ''), reverse=True)

    return runs


def load_utility_bias_run(run_id_or_path: str) -> Dict[str, Any]:
    """
    Load a complete utility bias test run from JSON file.

    Args:
        run_id_or_path: Either a run_id (filename without .json) or full path to JSON file

    Returns:
        Complete run data with reconstructed DataFrames

    Raises:
        FileNotFoundError: If the run file doesn't exist
        ValueError: If the JSON is malformed or missing required fields
    """
    runs_dir = get_utility_bias_runs_dir()

    # Determine filepath
    if os.path.isabs(run_id_or_path) or run_id_or_path.startswith('data/'):
        filepath = Path(run_id_or_path)
    else:
        # Assume it's a run_id, find the matching file
        matching_files = list(runs_dir.glob(f"*{run_id_or_path}*.json"))
        if not matching_files:
            raise FileNotFoundError(f"No run file found for run_id: {run_id_or_path}")
        filepath = matching_files[0]  # Take the first match

    if not filepath.exists():
        raise FileNotFoundError(f"Run file not found: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Validate schema version
        version = data.get('version', 0)
        if version != 1:
            raise ValueError(f"Unsupported schema version: {version}")

        # Reconstruct DataFrames from serialized format
        if 'results' in data:
            results = data['results']

            # Reconstruct results_df
            if 'results_df' in results and results['results_df']:
                data['results']['results_df'] = deserialize_dataframe(results['results_df'])

            # Reconstruct preference_curves
            if 'preference_curves' in results and results['preference_curves']:
                data['results']['preference_curves'] = deserialize_dataframe(results['preference_curves'])

            # Reconstruct summary_table
            if 'summary_table' in results and results['summary_table']:
                data['results']['summary_table'] = deserialize_dataframe(results['summary_table'])

        return data

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in run file {filepath}: {e}")
    except KeyError as e:
        raise ValueError(f"Missing required field in run file {filepath}: {e}")


def delete_utility_bias_run(run_id: str) -> bool:
    """
    Delete a utility bias test run file.

    Args:
        run_id: Run identifier

    Returns:
        True if successfully deleted, False otherwise
    """
    runs_dir = get_utility_bias_runs_dir()

    # Find the matching file
    matching_files = list(runs_dir.glob(f"*{run_id}*.json"))
    if not matching_files:
        return False

    try:
        matching_files[0].unlink()  # Delete the first match
        return True
    except OSError:
        return False


def update_run_notes(run_id: str, notes: str) -> bool:
    """
    Update the notes field for a utility bias test run.

    Args:
        run_id: Run identifier
        notes: New notes content

    Returns:
        True if successfully updated, False otherwise
    """
    try:
        # Load the run
        run_data = load_utility_bias_run(run_id)

        # Update notes
        run_data['notes'] = notes

        # Save back (this will overwrite the file)
        save_utility_bias_run(run_data)

        return True

    except (FileNotFoundError, ValueError):
        return False