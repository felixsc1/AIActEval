"""
Utility Bias Testing Page - EU AI Act Compliance Testing POC

Setup and run utility bias tests using non-monetary preference queries.
"""

import math
import streamlit as st
import pandas as pd
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules
from evaluator import check_ollama_connection, get_ollama_models
from utility_bias import (
    generate_utility_queries,
    run_utility_bias_test,
    compute_statistics,
    create_preference_plot,
    create_exchange_rates_plot,
    create_summary_table,
    get_default_ethnicities,
    get_default_n_values,
    get_jailbreaking_system_prompts,
    get_anchor_options,
    run_robust_utility_bias_test,
    generate_anchor_variations,
    unload_model,
    append_multi_shot_examples,
    get_all_available_models,
    get_model_provider_and_name,
)
from utility_bias_storage import save_utility_bias_run, serialize_dataframe


def render_header():
    """Render the page header."""
    st.title("⚖️ Utility Bias Testing")
    st.markdown("*Test for implicit bias using non-monetary preference queries*")

    st.markdown(
        """
    Test for implicit bias in LLMs using non-monetary preference queries. This method compares
    scientific advancement against saving lives across different ethnicities to quantify hidden biases
    through statistical analysis of preference patterns.
    """
    )


def render_prerequisites_check():
    """Check and display prerequisites."""
    issues = []

    ollama_ok = check_ollama_connection()
    if not ollama_ok:
        issues.append("⚠️ Ollama not running (local models will not be available)")

    # Check Groq API key
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        issues.append("⚠️ GROQ_API_KEY not set (Groq models will not be available)")

    if not ollama_ok and not groq_key:
        issues.append("❌ Neither Ollama nor Groq API key are available")
        st.error("**Prerequisites not met:**")
        for issue in issues:
            st.error(issue)
        st.info("Please ensure at least one model provider is available.")
        return False
    elif issues:
        # Show warnings but allow to continue
        st.warning("**Some prerequisites not met:**")
        for issue in issues:
            st.warning(issue)
        st.info("You can still use available model providers.")

    return True


def render_test_configuration():
    """Render the test configuration section."""
    st.subheader("⚙️ Test Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Ethnicities to Test**")
        default_ethnicities = get_default_ethnicities()
        selected_ethnicities = st.multiselect(
            "Select ethnicities:",
            options=default_ethnicities,
            default=default_ethnicities,
            help="Ethnicities to include in bias testing",
        )

        if not selected_ethnicities:
            st.warning("Please select at least one ethnicity.")
            return None

    with col2:
        st.markdown("**N Values (Number of People)**")
        default_n_values = get_default_n_values()
        n_value_labels = [f"{n:,}" for n in default_n_values]

        # Create slider for range selection
        n_range = st.slider(
            "Select N value range:",
            min_value=0,
            max_value=len(default_n_values) - 1,
            value=(0, len(default_n_values) - 1),
            help="Range of N values (log-spaced from 1 to 1e12)",
        )

        selected_n_values = default_n_values[n_range[0] : n_range[1] + 1]
        st.caption(
            f"Selected: {len(selected_n_values)} values from {selected_n_values[0]:,} to {selected_n_values[-1]:,}"
        )

    # Anchor configuration
    st.markdown("**Anchor Outcome**")
    anchor_options = get_anchor_options()
    selected_anchor_key = st.selectbox(
        "Select anchor:",
        options=list(anchor_options.keys()),
        index=0,  # Default to "Strong anchor"
        help="Choose the scientific advancement scenario to compare against saving lives. Stronger anchors require more people to save to switch preference.",
    )
    anchor_text = anchor_options[selected_anchor_key]

    # Anchor variations toggle
    num_anchor_variations = st.radio(
        "Number of anchor variations:",
        options=[5, 10],
        index=0,  # Default to 5 for backward compatibility (first option)
        horizontal=True,
        help="Choose whether to use 5 or 10 variations of the anchor text for robustness testing. More variations provide better reliability but take longer to run.",
    )

    # Show selected anchor text
    with st.expander("View selected anchor text", expanded=False):
        st.code(anchor_text, language=None)
        st.caption(
            "This anchor represents outcome P (scientific advancement) in the preference queries."
        )

    # System prompt configuration for jailbreaking
    st.markdown("**System Prompt (Jailbreaking)**")
    st.markdown("*Use when models refuse to answer questions involving ethnicities*")

    system_prompt_options = get_jailbreaking_system_prompts()
    selected_system_prompt_key = st.selectbox(
        "Select system prompt:",
        options=list(system_prompt_options.keys()),
        index=0,  # Default to "Default (No system prompt)"
        help="Choose a jailbreaking prompt to bypass model restrictions on sensitive topics",
    )
    selected_system_prompt = system_prompt_options[selected_system_prompt_key]

    # Show selected prompt content
    if selected_system_prompt.strip():
        with st.expander("View selected system prompt", expanded=False):
            st.code(selected_system_prompt, language=None)
            st.caption(
                "This prompt will be sent as the system message to bypass restrictions."
            )

    # Multi-shot examples checkbox
    include_examples = st.checkbox(
        "Include multi-shot examples",
        value=True,  # Checked by default
        help="Append two example interactions to the system prompt to demonstrate the expected response format for preference questions.",
    )

    # Apply multi-shot examples if requested
    final_system_prompt = append_multi_shot_examples(
        selected_system_prompt, include_examples
    )

    # Show final prompt content if examples were added
    if include_examples and selected_system_prompt.strip():
        with st.expander("View final system prompt (with examples)", expanded=False):
            st.code(final_system_prompt, language=None)
            st.caption(
                "This is the complete prompt that will be sent to the model, including examples."
            )

    return {
        "selected_ethnicities": selected_ethnicities,
        "selected_n_values": selected_n_values,
        "anchor_text": anchor_text,
        "selected_anchor_key": selected_anchor_key,
        "num_anchor_variations": num_anchor_variations,
        "selected_system_prompt_key": selected_system_prompt_key,
        "final_system_prompt": final_system_prompt,
        "include_examples": include_examples,
    }


def render_model_selection():
    """Render the model selection section."""
    st.subheader("🤖 Model Selection")

    # Cache models in session state to avoid frequent API calls and ensure stability
    if (
        "cached_models" not in st.session_state
        or "models_timestamp" not in st.session_state
    ):
        st.session_state.cached_models = get_all_available_models()
        st.session_state.models_timestamp = time.time()
    elif (
        time.time() - st.session_state.models_timestamp > 60
    ):  # Refresh every 60 seconds
        st.session_state.cached_models = get_all_available_models()
        st.session_state.models_timestamp = time.time()

    all_models = st.session_state.cached_models

    if all_models:
        # Sort model keys for consistent ordering: Ollama first, then Groq
        model_keys = sorted(
            all_models.keys(), key=lambda x: (0 if x.startswith("ollama/") else 1, x)
        )

        # Initialize the selected model in session state if not set
        if "utility_bias_selected_model" not in st.session_state:
            st.session_state.utility_bias_selected_model = model_keys[0]

        # Ensure the stored selection is still valid
        if st.session_state.utility_bias_selected_model not in model_keys:
            st.session_state.utility_bias_selected_model = model_keys[0]

        # Find the current index
        try:
            current_index = model_keys.index(
                st.session_state.utility_bias_selected_model
            )
        except ValueError:
            current_index = 0

        selected_model_key = st.selectbox(
            "Select model:",
            options=model_keys,
            index=current_index,
            format_func=lambda x: all_models[x],
            help="Choose a model from Ollama (local) or Groq (cloud) for utility bias testing",
        )

        # Update session state with the new selection
        st.session_state.utility_bias_selected_model = selected_model_key

        # Extract provider and actual model name
        provider, model_name = get_model_provider_and_name(selected_model_key)

        # Show provider-specific information
        if provider == "groq":
            st.info(
                "📡 Using Groq cloud API. Rate limits: 30 requests/minute, 8000 tokens/minute."
            )
        else:
            st.info("🏠 Using local Ollama instance.")

        return selected_model_key, provider, model_name
    else:
        st.error(
            "No models found. Please ensure Ollama is running with models, or check your Groq API key."
        )
        return None, None, None


def render_performance_options(model_provider):
    """Render the performance options section."""
    st.subheader("⚡ Performance Options")

    if model_provider == "groq":
        st.markdown(
            "*Groq models run on optimized cloud infrastructure with automatic rate limiting*"
        )
        st.info("📊 **Groq Rate Limits:** 30 requests/minute, 8000 tokens/minute")
        # Return default values for Groq (these won't be used)
        return 2048, None, 0, False
    else:
        st.markdown(
            "*Optimize memory usage and GPU acceleration for better performance*"
        )

        perf_col1, perf_col2, perf_col3 = st.columns(3)

        with perf_col1:
            num_ctx = st.selectbox(
                "Context Window Size:",
                options=[512, 1024, 2048, 4096],
                index=2,  # Default to 2048
                help="Smaller context windows use less memory. 2048 is sufficient for our short prompts. "
                "Reduce to 1024 or 512 if experiencing memory issues with larger models.",
            )

        with perf_col2:
            gpu_option = st.selectbox(
                "GPU Usage:",
                options=[
                    "Auto (Recommended)",
                    "GPU with CPU Fallback",
                    "Force All GPU Layers",
                    "CPU Only",
                ],
                index=0,
                help="Auto: Let Ollama decide GPU allocation. "
                "GPU with Fallback: Try GPU first, auto-retry with CPU if out of memory. "
                "Force All GPU: Use maximum GPU layers (faster but may fail if VRAM is insufficient). "
                "CPU Only: Disable GPU (slower but always works).",
            )
            # Convert UI option to num_gpu value and fallback flag
            gpu_fallback_enabled = False
            if gpu_option == "Auto (Recommended)":
                num_gpu = None
            elif gpu_option == "GPU with CPU Fallback":
                num_gpu = 999  # Try max GPU first
                gpu_fallback_enabled = True
            elif gpu_option == "Force All GPU Layers":
                num_gpu = 999  # High number to use all available GPU layers
            else:  # CPU Only
                num_gpu = 0

            if gpu_option == "Force All GPU Layers":
                st.caption("⚠️ May fail if VRAM is insufficient")

        with perf_col3:
            cleanup_interval = st.selectbox(
                "Memory Cleanup Interval:",
                options=[0, 50, 100, 200],
                index=2,  # Default to 100
                format_func=lambda x: "Disabled" if x == 0 else f"Every {x} queries",
                help="Periodically unload and reload model during long test runs to prevent memory leaks. "
                "Recommended for models like deepseek-r1 that may accumulate memory usage.",
            )

        return num_ctx, num_gpu, cleanup_interval, gpu_fallback_enabled


def render_single_query_test(
    selected_model_key, model_provider, config, num_ctx, num_gpu, gpu_fallback_enabled
):
    """Render the single query test section."""
    st.subheader("🧪 Test Single Query")
    st.markdown(
        "Test all 5 anchor variations for a single ethnicity/N combination to see how responses vary."
    )

    test_col1, test_col2 = st.columns([1, 1])
    with test_col1:
        test_ethnicity = st.selectbox(
            "Test Ethnicity:", options=config["selected_ethnicities"], key="test_eth"
        )
        test_n = st.number_input(
            "Test N value:", min_value=1, value=100, step=1, key="test_n"
        )

    with test_col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        if st.button("🔬 Test All Variations", key="test_query_btn"):
            try:
                # Get anchor variations based on user selection
                anchor_variations = generate_anchor_variations(
                    config["anchor_text"], config["num_anchor_variations"]
                )

                st.markdown(f"**Testing {len(anchor_variations)} anchor variations:**")
                st.markdown("*Each variation tests the same ethnicity/N combination*")

                variation_results = []

                for i, variation in enumerate(anchor_variations):
                    with st.expander(f"Variation {i+1}", expanded=(i == 0)):
                        # Generate query for this variation
                        test_queries = generate_utility_queries(
                            ethnicities=[test_ethnicity],
                            n_values=[test_n],
                            anchor=variation,
                        )

                        test_query = test_queries.iloc[0]["query"]

                        st.markdown("**Anchor Text:**")
                        st.code(variation, language=None)

                        st.markdown("**Generated Prompt:**")
                        st.code(test_query, language=None)

                        # Run the query for this variation with optimized settings
                        with st.spinner(f"Querying model (variation {i+1})..."):
                            # Show debug info for Groq
                            if model_provider == "groq":
                                groq_model_name = selected_model_key.replace(
                                    "groq/", ""
                                )
                                st.caption(
                                    f"Debug: Using Groq model '{groq_model_name}'"
                                )

                                # Direct Groq API test for debugging
                                try:
                                    from utility_bias import call_groq_api

                                    test_messages = [
                                        {"role": "user", "content": "Say 'P' or 'Q'."}
                                    ]
                                    test_response = call_groq_api(
                                        groq_model_name,
                                        test_messages,
                                        temperature=0.0,
                                        max_tokens=10,
                                    )
                                    st.caption(
                                        f"Debug: Groq API test response: {test_response}"
                                    )
                                except Exception as debug_e:
                                    st.error(
                                        f"Debug: Groq API direct test failed: {debug_e}"
                                    )

                            test_results = run_utility_bias_test(
                                model=selected_model_key,
                                queries_df=test_queries,
                                progress_callback=None,
                                system_prompt=config["final_system_prompt"],
                                num_ctx=num_ctx,
                                num_gpu=num_gpu,
                                gpu_fallback=gpu_fallback_enabled,
                                model_provider=model_provider,
                            )

                            test_result = test_results.iloc[0]

                            st.markdown("**Model Response:**")
                            st.code(test_result["response"], language=None)

                            st.markdown("**Parsed Result:**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Raw Choice", test_result.get("raw_choice", "N/A")
                                )
                            with col2:
                                st.metric(
                                    "Interpreted", test_result.get("choice", "N/A")
                                )
                            with col3:
                                st.metric(
                                    "Is Refusal",
                                    "Yes" if test_result["is_refusal"] else "No",
                                )

                            variation_results.append(
                                {
                                    "variation": i + 1,
                                    "choice": test_result.get("raw_choice", "N/A"),
                                    "is_refusal": test_result["is_refusal"],
                                }
                            )

                # Summary of all variations
                st.markdown("---")
                st.markdown("**Summary Across All Variations:**")

                valid_results = [r for r in variation_results if not r["is_refusal"]]
                p_count = sum(1 for r in valid_results if r["choice"] == "P")
                q_count = sum(1 for r in valid_results if r["choice"] == "Q")
                refusal_count = sum(1 for r in variation_results if r["is_refusal"])

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Variations", len(variation_results))
                with col2:
                    st.metric(
                        "P Responses",
                        f"{p_count}/{len(valid_results)}" if valid_results else "0/0",
                    )
                with col3:
                    st.metric(
                        "Q Responses",
                        f"{q_count}/{len(valid_results)}" if valid_results else "0/0",
                    )
                with col4:
                    st.metric("Refusals", refusal_count)

                # Check for heavy favoritism
                if valid_results:
                    p_percentage = (p_count / len(valid_results)) * 100
                    q_percentage = (q_count / len(valid_results)) * 100

                    if p_percentage > 95 or q_percentage > 95:
                        st.warning(
                            "⚠️ **Heavy Favoritism Detected:** Model shows extreme preference for one choice across all variations. This suggests the model may be unreliable for bias testing."
                        )
                    elif refusal_count > 2:
                        st.warning(
                            "⚠️ **High Refusal Rate:** Model refused to answer in most variations. Try using a jailbreaking system prompt."
                        )
                    else:
                        st.success(
                            "✅ **Good Variation:** Model shows reasonable response variation across anchor texts."
                        )

                # Unload model after single query test to free VRAM (Ollama only)
                if model_provider == "ollama":
                    with st.spinner("Cleaning up model from memory..."):
                        if unload_model(selected_model_key):
                            st.info("✅ Model unloaded from VRAM to free memory.")
                        else:
                            st.info(
                                "ℹ️ Model cleanup attempted (may already be unloaded)."
                            )

            except Exception as e:
                st.error(f"Test failed: {e}")
                import traceback

                st.code(traceback.format_exc())
                # Try to unload model even on error to prevent memory issues (Ollama only)
                if model_provider == "ollama":
                    unload_model(selected_model_key)


def render_full_test_execution(
    selected_model_key,
    model_provider,
    config,
    num_ctx,
    num_gpu,
    cleanup_interval,
    gpu_fallback_enabled,
):
    """Render the full test execution section."""
    st.subheader("🚀 Run Utility Bias Test")

    # Show test summary
    base_queries = len(config["selected_ethnicities"]) * len(
        config["selected_n_values"]
    )
    total_queries = base_queries * config["num_anchor_variations"]
    st.info(
        f"**Test will generate {total_queries} queries** ({len(config['selected_ethnicities'])} ethnicities × {len(config['selected_n_values'])} N values × {config['num_anchor_variations']} anchor variations)"
    )

    if st.button("🧪 Run Utility Bias Test", type="primary", width="stretch"):
        with st.spinner(
            "Generating queries and running inference... This may take several minutes."
        ):
            try:
                # Progress callback
                progress_bar = st.progress(0)
                status_text = st.empty()

                def progress_callback(current, total):
                    progress = current / total
                    progress_bar.progress(progress)
                    status_text.text(f"Processing query {current}/{total}...")

                # Run robust utility bias test with anchor variations and performance optimizations
                stats, status_message = run_robust_utility_bias_test(
                    model=selected_model_key,
                    anchor_text=config["anchor_text"],
                    ethnicities=config["selected_ethnicities"],
                    n_values=config["selected_n_values"],
                    base_url="http://localhost:11434",
                    progress_callback=progress_callback,
                    system_prompt=config["final_system_prompt"],
                    num_anchor_variations=config["num_anchor_variations"],
                    num_ctx=num_ctx,
                    num_gpu=num_gpu,
                    cleanup_interval=cleanup_interval,
                    gpu_fallback=gpu_fallback_enabled,
                    model_provider=model_provider,
                )

                # Clear progress
                progress_bar.empty()
                status_text.empty()

                # Extract results for compatibility
                results_df = stats.get("results_df", pd.DataFrame())

                # Store results in session state
                st.session_state.utility_bias_results = {
                    "queries_df": pd.DataFrame(),  # Not available in aggregated results
                    "results_df": results_df,
                    "stats": stats,
                    "config": {
                        "model": selected_model_key,
                        "ethnicities": config["selected_ethnicities"],
                        "n_values": config["selected_n_values"],
                        "anchor": config["anchor_text"],
                        "system_prompt": config[
                            "selected_system_prompt_key"
                        ],  # Store the key for display
                        "timestamp": time.time(),
                    },
                }

                # Prepare run payload for persistent storage
                run_payload = {
                    "model_info": {
                        "ollama_model": selected_model_key,
                        "base_url": "http://localhost:11434",
                        "used_gpu": num_gpu is not None,
                        "num_ctx": num_ctx,
                        "num_gpu": num_gpu,
                        "gpu_fallback": gpu_fallback_enabled,
                    },
                    "test_config": {
                        "ethnicities": config["selected_ethnicities"],
                        "n_values": config["selected_n_values"],
                        "anchor_key": config["selected_anchor_key"],
                        "anchor_text": config["anchor_text"],
                        "num_anchor_variations": config["num_anchor_variations"],
                        "system_prompt_key": config["selected_system_prompt_key"],
                        "include_examples": config["include_examples"],
                        "cleanup_interval": cleanup_interval,
                    },
                    "run_metadata": {
                        "total_queries": base_queries * config["num_anchor_variations"],
                        "elapsed_seconds": None,  # Could be added later if timing is tracked
                        "status": stats.get("status", "completed"),
                        "status_message": status_message,
                    },
                    "results": {
                        "results_df": serialize_dataframe(results_df),
                        "preference_curves": serialize_dataframe(
                            stats.get("preference_curves", pd.DataFrame())
                        ),
                        "summary_table": serialize_dataframe(
                            create_summary_table(stats)
                        ),
                        "stats_core": {
                            "refusal_rates": stats.get("refusal_rates", {}),
                            "switch_points": stats.get("switch_points", {}),
                            "exchange_rates": stats.get("exchange_rates", {}),
                            "exchange_rate_reference": stats.get(
                                "exchange_rate_reference", {}
                            ),
                            "exchange_rate_reference_category": stats.get(
                                "exchange_rate_reference_category"
                            ),
                            "consistency": (
                                stats.get("consistency", pd.DataFrame()).to_dict()
                                if "consistency" in stats
                                else {}
                            ),
                            "aggregation_metadata": stats.get(
                                "aggregation_metadata", {}
                            ),
                        },
                    },
                    "notes": "",
                }

                # Save to persistent storage
                try:
                    saved_path = save_utility_bias_run(run_payload)
                    st.info(f"✅ Test results saved to: {saved_path.name}")
                except Exception as save_error:
                    st.warning(
                        f"⚠️ Test completed but failed to save results: {save_error}"
                    )

                # Unload model after full test to free VRAM (Ollama only)
                if model_provider == "ollama":
                    unload_model(selected_model_key)

                # Check if test succeeded or failed
                status_value = stats.get("status")
                if "error" in stats or status_value in ("error", "failed"):
                    st.error(f"❌ {status_message}")
                elif status_value == "warning":
                    st.warning(f"⚠️ {status_message}")
                else:
                    st.success(f"✅ {status_message}")
                    st.info(
                        "📊 Go to **Utility Bias Results** page to view detailed analysis"
                    )
                st.rerun()

            except Exception as e:
                st.error(f"❌ Test failed: {e}")
                # Try to unload model even on error to prevent memory issues (Ollama only)
                if model_provider == "ollama":
                    unload_model(selected_model_key)
                return


def render_grid_testing_tab():
    """Render the Grid Testing tab content (original exhaustive testing approach)."""
    st.markdown(
        """
    **Grid Testing** runs queries for all combinations of ethnicities × N values × anchor variations.
    This provides comprehensive coverage but requires many queries.
    """
    )

    # Test configuration
    config = render_test_configuration()
    if config is None:
        return

    # Model selection
    try:
        selected_model_key, model_provider, selected_model = render_model_selection()
        if selected_model_key is None:
            return
    except Exception as e:
        st.error(f"Error in model selection: {e}")
        return

    # Performance options
    try:
        num_ctx, num_gpu, cleanup_interval, gpu_fallback_enabled = (
            render_performance_options(model_provider)
        )
    except Exception as e:
        st.error(f"Error in performance options: {e}")
        return

    # Single query test
    try:
        render_single_query_test(
            selected_model_key,
            model_provider,
            config,
            num_ctx,
            num_gpu,
            gpu_fallback_enabled,
        )
    except Exception as e:
        st.error(f"Error in single query test: {e}")

    # Full test execution
    try:
        render_full_test_execution(
            selected_model_key,
            model_provider,
            config,
            num_ctx,
            num_gpu,
            cleanup_interval,
            gpu_fallback_enabled,
        )
    except Exception as e:
        st.error(f"Error in full test execution: {e}")


def render_thurstonian_tab():
    """Render the Thurstonian Active Learning tab content."""
    st.markdown(
        """
    **Thurstonian Active Learning** uses intelligent sampling to select which (ethnicity, N) 
    combinations to query based on model uncertainty. This can be more efficient than exhaustive 
    grid testing while still producing accurate switch points and exchange rates.
    
    *Inspired by the [Utility Engineering paper](https://arxiv.org/pdf/2502.08640) (Mazeika et al., 2025)*
    """
    )

    # Import Thurstonian components
    try:
        from thurstonian_bias import (
            ThurstoniaBiasModel,
            ThurstonianActiveLearner,
            BiasOption,
            PreferenceObservation,
            fit_thurstonian_to_grid_results,
            convert_thurstonian_to_results_format,
        )

        thurstonian_available = True
    except ImportError as e:
        st.error(f"Thurstonian module not available: {e}")
        thurstonian_available = False
        return

    # Sub-tabs for New Test vs Post-hoc Fitting
    subtab1, subtab2 = st.tabs(
        ["🆕 New Active Learning Test", "📁 Fit to Existing Results"]
    )

    with subtab1:
        render_thurstonian_testing_ui()

    with subtab2:
        render_posthoc_fitting_ui()


def render_thurstonian_testing_ui():
    """Render the Thurstonian active learning testing UI."""
    from thurstonian_bias import (
        ThurstoniaBiasModel,
        ThurstonianActiveLearner,
        BiasOption,
        PreferenceObservation,
        convert_thurstonian_to_results_format,
    )

    # ===== CONFIGURATION SECTION =====
    st.subheader("⚙️ Test Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Ethnicities to Test**")
        default_ethnicities = get_default_ethnicities()
        th_selected_ethnicities = st.multiselect(
            "Select ethnicities:",
            options=default_ethnicities,
            default=default_ethnicities,
            help="Ethnicities to include in bias testing",
            key="th_ethnicities",
        )

        if not th_selected_ethnicities:
            st.warning("Please select at least one ethnicity.")
            return

    with col2:
        st.markdown("**N Values (Number of People)**")
        default_n_values = get_default_n_values()

        th_n_range = st.slider(
            "Select N value range:",
            min_value=0,
            max_value=len(default_n_values) - 1,
            value=(0, len(default_n_values) - 1),
            help="Range of N values (log-spaced from 1 to 1e12)",
            key="th_n_range",
        )

        th_selected_n_values = default_n_values[th_n_range[0] : th_n_range[1] + 1]
        st.caption(
            f"Selected: {len(th_selected_n_values)} values from {th_selected_n_values[0]:,} to {th_selected_n_values[-1]:,}"
        )

    # Anchor configuration
    st.markdown("**Anchor Outcome**")
    anchor_options = get_anchor_options()
    th_selected_anchor_key = st.selectbox(
        "Select anchor:",
        options=list(anchor_options.keys()),
        index=0,
        help="Choose the scientific advancement scenario to compare against saving lives.",
        key="th_anchor",
    )
    th_anchor_text = anchor_options[th_selected_anchor_key]

    with st.expander("View selected anchor text", expanded=False):
        st.code(th_anchor_text, language=None)

    # System prompt configuration
    st.markdown("**System Prompt (Jailbreaking)**")
    system_prompt_options = get_jailbreaking_system_prompts()
    th_selected_system_prompt_key = st.selectbox(
        "Select system prompt:",
        options=list(system_prompt_options.keys()),
        index=0,
        help="Choose a jailbreaking prompt to bypass model restrictions",
        key="th_system_prompt",
    )
    th_selected_system_prompt = system_prompt_options[th_selected_system_prompt_key]

    # Multi-shot examples
    th_include_examples = st.checkbox(
        "Include multi-shot examples",
        value=True,
        help="Append example interactions to demonstrate expected response format.",
        key="th_examples",
    )

    th_final_system_prompt = append_multi_shot_examples(
        th_selected_system_prompt, th_include_examples
    )

    # ===== ACTIVE LEARNING PARAMETERS =====
    st.subheader("🧠 Active Learning Parameters")

    al_col1, al_col2, al_col3 = st.columns(3)

    with al_col1:
        th_P = st.slider(
            "Uncertainty percentile (P%):",
            min_value=5,
            max_value=50,
            value=10,
            help="Sample from bottom P% of utility variance (higher = more exploration)",
            key="th_P",
        )

    with al_col2:
        th_Q = st.slider(
            "Degree percentile (Q%):",
            min_value=5,
            max_value=50,
            value=20,
            help="Sample from bottom Q% of query counts (higher = more coverage)",
            key="th_Q",
        )

    with al_col3:
        th_max_iterations = st.number_input(
            "Max iterations:",
            min_value=1,
            max_value=20,
            value=5,
            help="Maximum active learning iterations",
            key="th_max_iter",
        )

    al_col4, al_col5, al_col6 = st.columns(3)

    with al_col4:
        th_queries_per_iter = st.number_input(
            "Queries per iteration:",
            min_value=5,
            max_value=100,
            value=20,
            help="Number of (ethnicity, N) pairs to query per iteration",
            key="th_queries_per_iter",
        )

    with al_col5:
        th_K = st.number_input(
            "Responses per query (K):",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of responses to collect per query (using anchor variations)",
            key="th_K",
        )

    with al_col6:
        th_intermediate_n = st.number_input(
            "Intermediate N per interval:",
            min_value=0,
            max_value=5,
            value=1,
            help="Number of intermediate N values to add between consecutive grid points (0 = use original grid only)",
            key="th_intermediate_n",
        )

    # Move epochs to a new row or adjust layout
    with st.container():
        th_num_epochs = st.selectbox(
            "Model fitting epochs:",
            options=[100, 500, 1000, 2000],
            index=2,
            help="Number of optimization epochs for Thurstonian model fitting",
            key="th_epochs",
        )

    # Show estimated query count
    total_options = len(th_selected_ethnicities) * len(th_selected_n_values)
    grid_queries = total_options * th_K
    estimated_al_queries = (
        total_options // 3 + th_max_iterations * th_queries_per_iter
    ) * th_K

    st.info(
        f"""
    **Estimated Queries:**
    - Total options: {total_options} (ethnicity × N combinations)
    - Grid testing would need: ~{grid_queries:,} queries
    - Active learning estimate: ~{estimated_al_queries:,} queries (may vary based on convergence)
    """
    )

    # ===== MODEL SELECTION =====
    st.subheader("🤖 Model Selection")

    # Reuse model selection logic
    if "cached_models" not in st.session_state:
        st.session_state.cached_models = get_all_available_models()
        st.session_state.models_timestamp = time.time()

    all_models = st.session_state.cached_models

    if not all_models:
        st.error("No models available. Please check Ollama or Groq API key.")
        return

    model_keys = sorted(
        all_models.keys(), key=lambda x: (0 if x.startswith("ollama/") else 1, x)
    )

    th_selected_model_key = st.selectbox(
        "Select model:",
        options=model_keys,
        format_func=lambda x: all_models[x],
        help="Choose a model for Thurstonian testing",
        key="th_model",
    )

    th_provider, th_model_name = get_model_provider_and_name(th_selected_model_key)

    if th_provider == "groq":
        st.info("📡 Using Groq cloud API. Rate limits apply.")
    else:
        st.info("🏠 Using local Ollama instance.")

    # Performance options (simplified for Thurstonian)
    if th_provider == "ollama":
        with st.expander("⚡ Performance Options", expanded=False):
            th_num_ctx = st.selectbox(
                "Context Window:", options=[512, 1024, 2048], index=2, key="th_num_ctx"
            )
            th_num_gpu = st.selectbox(
                "GPU Usage:", options=["Auto", "CPU Only"], index=0, key="th_gpu"
            )
            th_num_gpu = None if th_num_gpu == "Auto" else 0
    else:
        th_num_ctx = 2048
        th_num_gpu = None

    # ===== RUN THURSTONIAN TEST =====
    st.subheader("🚀 Run Thurstonian Test")

    if st.button(
        "🧠 Run Thurstonian Active Learning Test", type="primary", key="th_run_btn"
    ):
        _run_thurstonian_test(
            ethnicities=th_selected_ethnicities,
            n_values=th_selected_n_values,
            anchor_text=th_anchor_text,
            anchor_key=th_selected_anchor_key,
            system_prompt=th_final_system_prompt,
            system_prompt_key=th_selected_system_prompt_key,
            include_examples=th_include_examples,
            model_key=th_selected_model_key,
            model_provider=th_provider,
            num_ctx=th_num_ctx,
            num_gpu=th_num_gpu,
            P=th_P,
            Q=th_Q,
            max_iterations=th_max_iterations,
            queries_per_iteration=th_queries_per_iter,
            K=th_K,
            num_epochs=th_num_epochs,
        )


def _run_thurstonian_test(
    ethnicities,
    n_values,
    anchor_text,
    anchor_key,
    system_prompt,
    system_prompt_key,
    include_examples,
    model_key,
    model_provider,
    num_ctx,
    num_gpu,
    P,
    Q,
    max_iterations,
    queries_per_iteration,
    K,
    num_epochs,
):
    """Execute the Thurstonian active learning test."""
    from thurstonian_bias import (
        ThurstoniaBiasModel,
        ThurstonianActiveLearner,
        BiasOption,
        PreferenceObservation,
        convert_thurstonian_to_results_format,
    )
    from utility_bias_storage import save_utility_bias_run, serialize_dataframe

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_container = st.empty()

    try:
        status_text.text("Initializing Thurstonian model...")

        # Initialize model
        model = ThurstoniaBiasModel(
            ethnicities=ethnicities,
            n_values=n_values,
            num_epochs=num_epochs,
            learning_rate=0.01,
            seed=42,
            verbose=False,
        )

        # Initialize active learner
        learner = ThurstonianActiveLearner(
            model=model,
            P=P,
            Q=Q,
            num_queries_per_iteration=queries_per_iteration,
            K=K,
            seed=42,
            num_intermediate_per_interval=th_intermediate_n,
        )

        # Get anchor variations for K responses
        anchor_variations = generate_anchor_variations(anchor_text, K)

        # Query history for storage
        query_history = []
        total_queries_executed = 0

        # ===== INITIAL SAMPLING =====
        status_text.text("Phase 1: Initial sampling...")
        initial_options = learner.get_initial_queries(
            num_queries=queries_per_iteration * 2
        )

        for i, option in enumerate(initial_options):
            progress = (i + 1) / len(initial_options) * 0.3  # First 30%
            progress_bar.progress(progress)
            status_text.text(
                f"Initial sampling: {i+1}/{len(initial_options)} - {option.ethnicity}, N={option.n_value:,}"
            )

            # Ensure option is registered in model
            if option.id not in model.options_by_id:
                model.add_option(option)

            # Query with K anchor variations
            responses = _query_option_with_variations(
                option=option,
                anchor_variations=anchor_variations,
                model_key=model_key,
                model_provider=model_provider,
                system_prompt=system_prompt,
                num_ctx=num_ctx,
                num_gpu=num_gpu,
            )

            # Add observations
            for var_idx, (response, raw_response, is_refusal) in enumerate(responses):
                obs = PreferenceObservation(
                    option=option,
                    anchor_variation_idx=var_idx,
                    response=response,
                    raw_response=raw_response,
                    is_refusal=is_refusal,
                )
                model.add_observation(obs)

            query_history.append(
                {
                    "ethnicity": option.ethnicity,
                    "n_value": option.n_value,
                    "iteration": 0,
                    "responses": [r[0] for r in responses],
                }
            )
            total_queries_executed += len(responses)

        learner.update_query_counts(initial_options, K)

        # Initial model fit
        status_text.text("Fitting initial model...")
        utilities, metrics = model.fit(print_every=500)

        with metrics_container:
            _log_loss = metrics.get("log_loss")
            if (
                _log_loss is not None
                and isinstance(_log_loss, float)
                and math.isnan(_log_loss)
            ):
                st.warning(
                    "No usable preference data (e.g. all refusals). "
                    "Fit is undefined; try a stronger model or different prompt."
                )
            st.markdown(
                f"**Initial Fit:** Log Loss: {metrics['log_loss']:.4f}, Accuracy: {metrics['accuracy']:.2%}"
            )

        # ===== ACTIVE LEARNING ITERATIONS =====
        for iteration in range(max_iterations):
            status_text.text(
                f"Phase 2: Active learning iteration {iteration + 1}/{max_iterations}..."
            )

            # Get next queries
            next_options = learner.get_next_queries(queries_per_iteration)

            if not next_options:
                status_text.text("No more options to query. Stopping early.")
                break

            for i, option in enumerate(next_options):
                progress = (
                    0.3
                    + (iteration * len(next_options) + i + 1)
                    / (max_iterations * len(next_options))
                    * 0.6
                )
                progress_bar.progress(min(progress, 0.9))
                status_text.text(
                    f"Iteration {iteration + 1}: {i+1}/{len(next_options)} - {option.ethnicity}, N={option.n_value:,}"
                )

                # Ensure option is registered in model
                if option.id not in model.options_by_id:
                    model.add_option(option)

                # Query with K anchor variations
                responses = _query_option_with_variations(
                    option=option,
                    anchor_variations=anchor_variations,
                    model_key=model_key,
                    model_provider=model_provider,
                    system_prompt=system_prompt,
                    num_ctx=num_ctx,
                    num_gpu=num_gpu,
                )

                # Add observations
                for var_idx, (response, raw_response, is_refusal) in enumerate(
                    responses
                ):
                    obs = PreferenceObservation(
                        option=option,
                        anchor_variation_idx=var_idx,
                        response=response,
                        raw_response=raw_response,
                        is_refusal=is_refusal,
                    )
                    model.add_observation(obs)

                query_history.append(
                    {
                        "ethnicity": option.ethnicity,
                        "n_value": option.n_value,
                        "iteration": iteration + 1,
                        "responses": [r[0] for r in responses],
                    }
                )
                total_queries_executed += len(responses)

            learner.update_query_counts(next_options, K)

            # Refit model
            utilities, metrics = model.fit(print_every=500)

            with metrics_container:
                _log_loss = metrics.get("log_loss")
                if (
                    _log_loss is not None
                    and isinstance(_log_loss, float)
                    and math.isnan(_log_loss)
                ):
                    st.warning(
                        "No usable preference data (e.g. all refusals). "
                        "Fit is undefined; try a stronger model or different prompt."
                    )
                st.markdown(
                    f"**Iteration {iteration + 1}:** Log Loss: {metrics['log_loss']:.4f}, Accuracy: {metrics['accuracy']:.2%}, Queries: {total_queries_executed}"
                )

        # ===== FINALIZE AND SAVE =====
        progress_bar.progress(0.95)
        status_text.text("Finalizing results...")

        # Convert to standard results format
        results_data = convert_thurstonian_to_results_format(model, anchor_text)

        # Create summary table
        summary_table = create_summary_table(
            {
                "switch_points": model.get_all_switch_points(),
                "exchange_rates": model.get_exchange_rates()[0],
                "exchange_rate_reference_category": model.get_exchange_rates()[1],
                "refusal_rates": {},
            }
        )

        # Prepare run payload
        run_payload = {
            "method": "thurstonian",
            "model_info": {
                "ollama_model": model_key,
                "base_url": "http://localhost:11434",
                "num_ctx": num_ctx,
                "num_gpu": num_gpu,
            },
            "test_config": {
                "ethnicities": ethnicities,
                "n_values": n_values,
                "anchor_key": anchor_key,
                "anchor_text": anchor_text,
                "num_anchor_variations": K,
                "system_prompt_key": system_prompt_key,
                "include_examples": include_examples,
                "thurstonian_params": {
                    "P": P,
                    "Q": Q,
                    "max_iterations": max_iterations,
                    "queries_per_iteration": queries_per_iteration,
                    "K": K,
                    "num_epochs": num_epochs,
                },
            },
            "run_metadata": {
                "total_queries": total_queries_executed,
                "n_iterations": max_iterations,
                "status": "completed",
            },
            "results": {
                "results_df": serialize_dataframe(results_data["results_df"]),
                "preference_curves": serialize_dataframe(
                    results_data["preference_curves"]
                ),
                "summary_table": serialize_dataframe(summary_table),
                "stats_core": results_data["stats_core"],
                "thurstonian_model": results_data["thurstonian_model"],
            },
            "notes": "",
        }

        # Add query history
        run_payload["results"]["thurstonian_model"]["query_history"] = query_history
        run_payload["results"]["thurstonian_model"]["n_iterations"] = max_iterations

        # Save results
        try:
            saved_path = save_utility_bias_run(run_payload)
            progress_bar.progress(1.0)
            status_text.empty()
            st.success(
                f"✅ Thurstonian test completed! Results saved to: {saved_path.name}"
            )
            st.info(f"📊 Total queries executed: {total_queries_executed}")
            st.info("📊 Go to **Utility Bias Results** page to view detailed analysis")
        except Exception as save_error:
            st.warning(f"Test completed but failed to save: {save_error}")

        # Unload model if Ollama
        if model_provider == "ollama":
            unload_model(model_key)

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Thurstonian test failed: {e}")
        import traceback

        st.code(traceback.format_exc())
        if model_provider == "ollama":
            unload_model(model_key)


def _query_option_with_variations(
    option,
    anchor_variations,
    model_key,
    model_provider,
    system_prompt,
    num_ctx,
    num_gpu,
):
    """Query a single option with multiple anchor variations."""
    responses = []

    for var_idx, anchor_var in enumerate(anchor_variations):
        # Generate query
        queries_df = generate_utility_queries(
            ethnicities=[option.ethnicity], n_values=[option.n_value], anchor=anchor_var
        )

        # Run query
        try:
            results = run_utility_bias_test(
                model=model_key,
                queries_df=queries_df,
                progress_callback=None,
                system_prompt=system_prompt,
                num_ctx=num_ctx,
                num_gpu=num_gpu,
                gpu_fallback=False,
                model_provider=model_provider,
            )

            result = results.iloc[0]
            raw_choice = result.get("raw_choice", "")
            is_refusal = result.get("is_refusal", False)

            # Map to P/Q
            if raw_choice == "P":
                response = "P"
            elif raw_choice == "Q":
                response = "Q"
            else:
                response = "P"  # Default on parse failure
                is_refusal = True

            responses.append((response, str(result.get("response", "")), is_refusal))

        except Exception as e:
            # On error, treat as refusal
            responses.append(("P", str(e), True))

    return responses


def render_posthoc_fitting_ui():
    """Render the post-hoc Thurstonian fitting UI for existing grid results."""
    from thurstonian_bias import (
        fit_thurstonian_to_grid_results,
        convert_thurstonian_to_results_format,
    )
    from utility_bias_storage import (
        list_utility_bias_runs,
        load_utility_bias_run,
        save_utility_bias_run,
        serialize_dataframe,
    )

    st.markdown(
        """
    **Post-hoc Fitting** allows you to fit a Thurstonian utility model to existing grid test results.
    This lets you compare the Thurstonian approach against exhaustive grid testing using the same data.
    """
    )

    # List available grid runs
    all_runs = list_utility_bias_runs()
    grid_runs = [r for r in all_runs if r.get("method", "grid") == "grid"]

    if not grid_runs:
        st.warning(
            "No grid test results found. Run a grid test first, then return here to fit the Thurstonian model."
        )
        return

    # Format runs for selection
    run_options = {
        r[
            "run_id"
        ]: f"{r['created_at'][:16]} - {r['model']} ({r['num_ethnicities']} ethnicities, {r['num_n_values']} N values)"
        for r in grid_runs
    }

    selected_run_id = st.selectbox(
        "Select grid test result to fit:",
        options=list(run_options.keys()),
        format_func=lambda x: run_options[x],
        help="Choose an existing grid test result to fit the Thurstonian model to",
    )

    # Fitting parameters
    st.markdown("**Fitting Parameters**")

    fit_col1, fit_col2 = st.columns(2)

    with fit_col1:
        fit_epochs = st.selectbox(
            "Model fitting epochs:",
            options=[100, 500, 1000, 2000],
            index=2,
            help="Number of optimization epochs",
        )

    with fit_col2:
        fit_lr = st.selectbox(
            "Learning rate:",
            options=[0.001, 0.01, 0.1],
            index=1,
            format_func=lambda x: str(x),
            help="Adam optimizer learning rate",
        )

    if st.button("🔧 Fit Thurstonian Model to Grid Results", type="primary"):
        try:
            with st.spinner("Loading grid results..."):
                run_data = load_utility_bias_run(selected_run_id)

            results_df = run_data.get("results", {}).get("results_df")

            if results_df is None or (
                isinstance(results_df, pd.DataFrame) and results_df.empty
            ):
                st.error("No results data found in selected run.")
                return

            # Show source data info
            st.info(
                f"Fitting to {len(results_df)} data points from: {run_options[selected_run_id]}"
            )

            with st.spinner("Fitting Thurstonian model..."):
                # Fit model
                fitted_model, metrics = fit_thurstonian_to_grid_results(
                    results_df=results_df,
                    num_epochs=fit_epochs,
                    learning_rate=fit_lr,
                    verbose=False,
                )

            # Store fitted data in session state for persistence across reruns
            st.session_state.thurstonian_fitted_model = fitted_model
            st.session_state.thurstonian_fitted_metrics = metrics
            st.session_state.thurstonian_source_run_data = run_data
            st.session_state.thurstonian_fit_params = {
                "fit_epochs": fit_epochs,
                "fit_lr": fit_lr,
                "selected_run_id": selected_run_id,
            }

            # Show results
            st.success("✅ Thurstonian model fitted successfully!")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Log Loss", f"{metrics['log_loss']:.4f}")
            with col2:
                st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            with col3:
                st.metric("Data Points", metrics.get("n_observations", len(results_df)))

            # Show switch points comparison
            st.markdown("**Switch Points (Thurstonian vs Original)**")

            switch_points = fitted_model.get_all_switch_points()
            original_switch_points = (
                run_data.get("results", {})
                .get("stats_core", {})
                .get("switch_points", {})
            )

            comparison_data = []
            for ethnicity in fitted_model.ethnicities:
                th_sp = switch_points.get(ethnicity)
                orig_sp = original_switch_points.get(ethnicity)

                comparison_data.append(
                    {
                        "Ethnicity": ethnicity,
                        "Thurstonian Switch Point": f"{th_sp:,.0f}" if th_sp else "N/A",
                        "Original Switch Point": (
                            f"{orig_sp:,.0f}" if orig_sp else "N/A"
                        ),
                        "Difference": (
                            f"{((th_sp or 0) - (orig_sp or 0)):,.0f}"
                            if th_sp and orig_sp
                            else "N/A"
                        ),
                    }
                )

            st.dataframe(pd.DataFrame(comparison_data), width="stretch")

            # Option to save as new Thurstonian result
            st.markdown("---")
            if st.button("💾 Save Thurstonian Fit as New Result"):
                # Retrieve fitted data from session state
                fitted_model = st.session_state.get("thurstonian_fitted_model")
                run_data = st.session_state.get("thurstonian_source_run_data")
                fit_params = st.session_state.get("thurstonian_fit_params", {})

                if fitted_model is None or run_data is None:
                    st.error(
                        "❌ No fitted model data found. Please fit the model first."
                    )
                    return

                # Convert to results format
                results_data = convert_thurstonian_to_results_format(
                    fitted_model, run_data.get("test_config", {}).get("anchor_text", "")
                )

                # Create new run payload
                new_payload = {
                    "method": "thurstonian",
                    "model_info": run_data.get("model_info", {}),
                    "test_config": {
                        **run_data.get("test_config", {}),
                        "thurstonian_params": {
                            "fit_epochs": fit_params.get("fit_epochs", 1000),
                            "learning_rate": fit_params.get("fit_lr", 0.01),
                            "source_run_id": fit_params.get("selected_run_id", ""),
                            "posthoc_fit": True,
                        },
                    },
                    "run_metadata": {
                        "total_queries": run_data.get("run_metadata", {}).get(
                            "total_queries", 0
                        ),
                        "status": "completed",
                        "posthoc_fit": True,
                        "source_run_id": fit_params.get("selected_run_id", ""),
                    },
                    "results": {
                        "results_df": serialize_dataframe(results_data["results_df"]),
                        "preference_curves": serialize_dataframe(
                            results_data["preference_curves"]
                        ),
                        "summary_table": serialize_dataframe(
                            results_data["summary_table"]
                        ),
                        "stats_core": results_data["stats_core"],
                        "thurstonian_model": results_data["thurstonian_model"],
                    },
                    "notes": f'Post-hoc Thurstonian fit of grid results from {fit_params.get("selected_run_id", "")}',
                }

                try:
                    saved_path = save_utility_bias_run(new_payload)
                    st.success(f"✅ Saved Thurstonian fit to: {saved_path.name}")
                    st.info(
                        "📊 Go to **Utility Bias Results** page to view detailed analysis"
                    )

                    # Clear session state after successful save
                    st.session_state.pop("thurstonian_fitted_model", None)
                    st.session_state.pop("thurstonian_fitted_metrics", None)
                    st.session_state.pop("thurstonian_source_run_data", None)
                    st.session_state.pop("thurstonian_fit_params", None)

                except Exception as save_error:
                    st.error(f"Failed to save: {save_error}")
                    import traceback

                    st.code(traceback.format_exc())

        except Exception as e:
            st.error(f"Failed to fit model: {e}")
            import traceback

            st.code(traceback.format_exc())

    # Show previously fitted results if available in session state
    if (
        "thurstonian_fitted_model" in st.session_state
        and "thurstonian_fitted_metrics" in st.session_state
    ):
        fitted_model = st.session_state.thurstonian_fitted_model
        metrics = st.session_state.thurstonian_fitted_metrics
        run_data = st.session_state.thurstonian_source_run_data
        fit_params = st.session_state.thurstonian_fit_params

        st.markdown("---")
        st.info("📊 **Previously fitted model results:**")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Log Loss", f"{metrics['log_loss']:.4f}")
        with col2:
            st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        with col3:
            st.metric("Data Points", metrics.get("n_observations", 0))

        # Show switch points comparison if run_data is available
        if run_data:
            st.markdown("**Switch Points (Thurstonian vs Original)**")

            switch_points = fitted_model.get_all_switch_points()
            original_switch_points = (
                run_data.get("results", {})
                .get("stats_core", {})
                .get("switch_points", {})
            )

            comparison_data = []
            for ethnicity in fitted_model.ethnicities:
                th_sp = switch_points.get(ethnicity)
                orig_sp = original_switch_points.get(ethnicity)

                comparison_data.append(
                    {
                        "Ethnicity": ethnicity,
                        "Thurstonian Switch Point": f"{th_sp:,.0f}" if th_sp else "N/A",
                        "Original Switch Point": (
                            f"{orig_sp:,.0f}" if orig_sp else "N/A"
                        ),
                        "Difference": (
                            f"{((th_sp or 0) - (orig_sp or 0)):,.0f}"
                            if th_sp and orig_sp
                            else "N/A"
                        ),
                    }
                )

            st.dataframe(pd.DataFrame(comparison_data), width="stretch")

        # Option to save as new Thurstonian result
        st.markdown("---")
        if st.button("💾 Save Thurstonian Fit as New Result"):
            # Retrieve fitted data from session state
            fitted_model = st.session_state.get("thurstonian_fitted_model")
            run_data = st.session_state.get("thurstonian_source_run_data")
            fit_params = st.session_state.get("thurstonian_fit_params", {})

            if fitted_model is None or run_data is None:
                st.error("❌ No fitted model data found. Please fit the model first.")
                return

            # Convert to results format
            results_data = convert_thurstonian_to_results_format(
                fitted_model, run_data.get("test_config", {}).get("anchor_text", "")
            )

            # Create new run payload
            new_payload = {
                "method": "thurstonian",
                "model_info": run_data.get("model_info", {}),
                "test_config": {
                    **run_data.get("test_config", {}),
                    "thurstonian_params": {
                        "fit_epochs": fit_params.get("fit_epochs", 1000),
                        "learning_rate": fit_params.get("fit_lr", 0.01),
                        "source_run_id": fit_params.get("selected_run_id", ""),
                        "posthoc_fit": True,
                    },
                },
                "run_metadata": {
                    "total_queries": run_data.get("run_metadata", {}).get(
                        "total_queries", 0
                    ),
                    "status": "completed",
                    "posthoc_fit": True,
                    "source_run_id": fit_params.get("selected_run_id", ""),
                },
                "results": {
                    "results_df": serialize_dataframe(results_data["results_df"]),
                    "preference_curves": serialize_dataframe(
                        results_data["preference_curves"]
                    ),
                    "summary_table": serialize_dataframe(results_data["summary_table"]),
                    "stats_core": results_data["stats_core"],
                    "thurstonian_model": results_data["thurstonian_model"],
                },
                "notes": f'Post-hoc Thurstonian fit of grid results from {fit_params.get("selected_run_id", "")}',
            }

            try:
                saved_path = save_utility_bias_run(new_payload)
                st.success(f"✅ Saved Thurstonian fit to: {saved_path.name}")
                st.info(
                    "📊 Go to **Utility Bias Results** page to view detailed analysis"
                )

                # Clear session state after successful save
                st.session_state.pop("thurstonian_fitted_model", None)
                st.session_state.pop("thurstonian_fitted_metrics", None)
                st.session_state.pop("thurstonian_source_run_data", None)
                st.session_state.pop("thurstonian_fit_params", None)

            except Exception as save_error:
                st.error(f"Failed to save: {save_error}")
                import traceback

                st.code(traceback.format_exc())


def main():
    """Main page entry point."""
    render_header()

    # Prerequisites check
    if not render_prerequisites_check():
        return

    # Create tabs for different testing approaches
    tab1, tab2 = st.tabs(["📊 Grid Testing", "🧠 Thurstonian Active Learning"])

    with tab1:
        render_grid_testing_tab()

    with tab2:
        render_thurstonian_tab()


if __name__ == "__main__":
    main()
