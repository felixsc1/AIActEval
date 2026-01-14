#!/usr/bin/env python3
"""
Quick test to verify all imports work correctly.
"""

try:
    import sys
    print(f"Python version: {sys.version}")

    # Test basic imports
    import streamlit as st
    print("✅ streamlit imported")

    import deepeval
    print("✅ deepeval imported")

    import ollama
    print("✅ ollama imported")

    import pandas as pd
    print("✅ pandas imported")

    from dotenv import load_dotenv
    print("✅ python-dotenv imported")

    # Test our modules
    from config import JUDGE_MODELS, create_judge_model
    print("✅ config module imported")

    from dataset_handler import load_dataset, save_dataset
    print("✅ dataset_handler module imported")

    from evaluator import check_api_keys, check_ollama_connection
    print("✅ evaluator module imported")

    print("\n🎉 All imports successful! The application should run correctly.")
    print("To start the app, run: streamlit run app.py")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install missing dependencies with: pip install -r requirements.txt")
    sys.exit(1)