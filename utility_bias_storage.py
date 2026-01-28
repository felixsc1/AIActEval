"""
Utility Bias Test Run Storage

Handles persistent storage of utility bias test runs as JSON files.
Provides functions for saving, listing, and loading test results.

Supports two testing methods:
- Grid testing (method="grid"): Exhaustive grid search over ethnicities × N values
- Thurstonian active learning (method="thurstonian"): Intelligent sampling with utility model

Schema version history:
- Version 1: Original grid-based testing
- Version 2: Added Thurstonian model support with utilities, metrics, and query history
"""

import json
import os
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime


# Current schema version
CURRENT_SCHEMA_VERSION = 2


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


def serialize_numpy(obj: Any) -> Any:
    """
    Convert numpy types to Python native types for JSON serialization.

    Args:
        obj: Object that may contain numpy types

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: serialize_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_numpy(v) for v in obj]
    return obj


def serialize_thurstonian_model(thurstonian_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize Thurstonian model data for JSON storage.

    Args:
        thurstonian_data: Dict containing utilities, metrics, query_history, etc.

    Returns:
        JSON-serializable dict
    """
    serialized = {}

    # Serialize utilities (option_id -> {mean, variance})
    if 'utilities' in thurstonian_data and thurstonian_data['utilities']:
        serialized['utilities'] = {
            option_id: {
                'mean': float(util['mean']),
                'variance': float(util['variance'])
            }
            for option_id, util in thurstonian_data['utilities'].items()
        }

    # Serialize anchor variance
    if 'anchor_variance' in thurstonian_data:
        serialized['anchor_variance'] = float(thurstonian_data['anchor_variance'])

    # Serialize metrics
    if 'metrics' in thurstonian_data:
        serialized['metrics'] = serialize_numpy(thurstonian_data['metrics'])

    # Serialize query history (list of query records)
    if 'query_history' in thurstonian_data:
        serialized['query_history'] = [
            serialize_numpy(record) for record in thurstonian_data['query_history']
        ]

    # Copy other fields
    for key in ['n_observations', 'n_options_queried', 'n_iterations', 'converged']:
        if key in thurstonian_data:
            serialized[key] = thurstonian_data[key]

    return serialized


def deserialize_thurstonian_model(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deserialize Thurstonian model data from JSON.

    Args:
        data: Dict from JSON containing Thurstonian model data

    Returns:
        Reconstructed Thurstonian model data
    """
    # The data is already in the correct format, just ensure types are correct
    result = {}

    if 'utilities' in data:
        result['utilities'] = data['utilities']

    if 'anchor_variance' in data:
        result['anchor_variance'] = float(data['anchor_variance'])

    if 'metrics' in data:
        result['metrics'] = data['metrics']

    if 'query_history' in data:
        result['query_history'] = data['query_history']

    for key in ['n_observations', 'n_options_queried', 'n_iterations', 'converged']:
        if key in data:
            result[key] = data[key]

    return result


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
        payload: Complete run data including config, results, and metadata.
                 For Thurstonian runs, should include 'method': 'thurstonian'
                 and 'thurstonian_model' in results.

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
    payload['version'] = CURRENT_SCHEMA_VERSION

    # Set method type if not specified (default to 'grid' for backwards compatibility)
    if 'method' not in payload:
        payload['method'] = 'grid'

    # Serialize Thurstonian model data if present
    if 'results' in payload and 'thurstonian_model' in payload['results']:
        payload['results']['thurstonian_model'] = serialize_thurstonian_model(
            payload['results']['thurstonian_model']
        )

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

                # Determine method type (default to 'grid' for v1 files)
                method = data.get('method', 'grid')

                # Get Thurstonian-specific info if available
                thurstonian_info = {}
                if method == 'thurstonian' and 'results' in data:
                    thurstonian_model = data['results'].get('thurstonian_model', {})
                    thurstonian_info = {
                        'n_iterations': thurstonian_model.get('n_iterations', 0),
                        'n_observations': thurstonian_model.get('n_observations', 0),
                        'model_accuracy': thurstonian_model.get('metrics', {}).get('accuracy', None),
                        'model_log_loss': thurstonian_model.get('metrics', {}).get('log_loss', None)
                    }

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
                    'notes': data.get('notes', ''),
                    'method': method,
                    'version': data.get('version', 1),
                    **thurstonian_info
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
        Complete run data with reconstructed DataFrames and Thurstonian model data

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

        # Validate schema version (support both v1 and v2)
        version = data.get('version', 1)
        if version not in [1, 2]:
            raise ValueError(f"Unsupported schema version: {version}")

        # Set default method for v1 files
        if 'method' not in data:
            data['method'] = 'grid'

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

            # Deserialize Thurstonian model data if present
            if 'thurstonian_model' in results and results['thurstonian_model']:
                data['results']['thurstonian_model'] = deserialize_thurstonian_model(
                    results['thurstonian_model']
                )

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