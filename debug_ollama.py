#!/usr/bin/env python3
"""
Debug script to check Ollama models structure
"""

import ollama
import json

try:
    client = ollama.Client()
    response = client.list()
    print("Full response structure:")
    print(json.dumps(response, indent=2, default=str))

    models = response.get('models', [])
    print(f"\nNumber of models: {len(models)}")

    if models:
        print(f"\nFirst model structure:")
        print(json.dumps(models[0], indent=2, default=str))

        print(f"\nFirst model keys: {list(models[0].keys()) if hasattr(models[0], 'keys') else 'Not a dict'}")

        # Try different ways to access the name
        first_model = models[0]
        print(f"\nTrying to access name:")
        print(f"  first_model['name']: {first_model.get('name', 'NOT FOUND') if hasattr(first_model, 'get') else 'Not dict-like'}")
        print(f"  first_model.name: {getattr(first_model, 'name', 'NOT FOUND')}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()