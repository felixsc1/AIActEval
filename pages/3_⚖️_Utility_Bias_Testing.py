"""
Utility Bias Testing Page - EU AI Act Compliance Testing POC

Setup and run utility bias tests using non-monetary preference queries.
"""

import streamlit as st
import pandas as pd
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules
from evaluator import check_ollama_connection, get_ollama_models
from utility_bias import (
    generate_utility_queries, run_utility_bias_test, compute_statistics,
    create_preference_plot, create_exchange_rates_plot, create_summary_table,
    get_default_ethnicities, get_default_n_values, get_jailbreaking_system_prompts,
    get_anchor_options, run_robust_utility_bias_test, generate_anchor_variations,
    unload_model, append_multi_shot_examples
)
from utility_bias_storage import save_utility_bias_run, serialize_dataframe


def render_header():
    """Render the page header."""
    st.title("⚖️ Utility Bias Testing")
    st.markdown("*Test for implicit bias using non-monetary preference queries*")

    st.markdown("""
    Test for implicit bias in LLMs using non-monetary preference queries. This method compares
    scientific advancement against saving lives across different ethnicities to quantify hidden biases
    through statistical analysis of preference patterns.
    """)


def render_prerequisites_check():
    """Check and display prerequisites."""
    issues = []

    ollama_ok = check_ollama_connection()
    if not ollama_ok:
        issues.append("❌ Ollama not running (required for model testing)")

    if issues:
        st.error("**Prerequisites not met:**")
        for issue in issues:
            st.error(issue)
        st.info("Please resolve these issues before running tests.")
        return False

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
            help="Ethnicities to include in bias testing"
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
            max_value=len(default_n_values)-1,
            value=(0, len(default_n_values)-1),
            help="Range of N values (log-spaced from 1 to 1e12)"
        )

        selected_n_values = default_n_values[n_range[0]:n_range[1]+1]
        st.caption(f"Selected: {len(selected_n_values)} values from {selected_n_values[0]:,} to {selected_n_values[-1]:,}")

    # Anchor configuration
    st.markdown("**Anchor Outcome**")
    anchor_options = get_anchor_options()
    selected_anchor_key = st.selectbox(
        "Select anchor:",
        options=list(anchor_options.keys()),
        index=0,  # Default to "Strong anchor"
        help="Choose the scientific advancement scenario to compare against saving lives. Stronger anchors require more people to save to switch preference."
    )
    anchor_text = anchor_options[selected_anchor_key]

    # Anchor variations toggle
    num_anchor_variations = st.radio(
        "Number of anchor variations:",
        options=[5, 10],
        index=0,  # Default to 5 for backward compatibility (first option)
        horizontal=True,
        help="Choose whether to use 5 or 10 variations of the anchor text for robustness testing. More variations provide better reliability but take longer to run."
    )

    # Show selected anchor text
    with st.expander("View selected anchor text", expanded=False):
        st.code(anchor_text, language=None)
        st.caption("This anchor represents outcome P (scientific advancement) in the preference queries.")

    # System prompt configuration for jailbreaking
    st.markdown("**System Prompt (Jailbreaking)**")
    st.markdown("*Use when models refuse to answer questions involving ethnicities*")

    system_prompt_options = get_jailbreaking_system_prompts()
    selected_system_prompt_key = st.selectbox(
        "Select system prompt:",
        options=list(system_prompt_options.keys()),
        index=0,  # Default to "Default (No system prompt)"
        help="Choose a jailbreaking prompt to bypass model restrictions on sensitive topics"
    )
    selected_system_prompt = system_prompt_options[selected_system_prompt_key]

    # Show selected prompt content
    if selected_system_prompt.strip():
        with st.expander("View selected system prompt", expanded=False):
            st.code(selected_system_prompt, language=None)
            st.caption("This prompt will be sent as the system message to bypass restrictions.")

    # Multi-shot examples checkbox
    include_examples = st.checkbox(
        "Include multi-shot examples",
        value=True,  # Checked by default
        help="Append two example interactions to the system prompt to demonstrate the expected response format for preference questions."
    )

    # Apply multi-shot examples if requested
    final_system_prompt = append_multi_shot_examples(selected_system_prompt, include_examples)

    # Show final prompt content if examples were added
    if include_examples and selected_system_prompt.strip():
        with st.expander("View final system prompt (with examples)", expanded=False):
            st.code(final_system_prompt, language=None)
            st.caption("This is the complete prompt that will be sent to the model, including examples.")

    return {
        'selected_ethnicities': selected_ethnicities,
        'selected_n_values': selected_n_values,
        'anchor_text': anchor_text,
        'selected_anchor_key': selected_anchor_key,
        'num_anchor_variations': num_anchor_variations,
        'selected_system_prompt_key': selected_system_prompt_key,
        'final_system_prompt': final_system_prompt,
        'include_examples': include_examples
    }


def render_model_selection():
    """Render the model selection section."""
    st.subheader("🤖 Model Selection")
    try:
        ollama_models = get_ollama_models()
        if ollama_models:
            model_names = []
            for model in ollama_models:
                if isinstance(model, dict):
                    name = model.get('name') or model.get('model')
                else:
                    model_name = getattr(model, 'model', None) or getattr(model, 'name', None)
                if model_name:
                    model_names.append(model_name)

            if model_names:
                selected_model = st.selectbox(
                    "Select Ollama model:",
                    options=model_names,
                    help="Local model to test for utility bias"
                )
                return selected_model
            else:
                st.error("Could not parse model names from Ollama.")
                return None
        else:
            st.error("No Ollama models found. Pull some models first.")
            return None
    except Exception as e:
        st.error(f"Error connecting to Ollama: {e}")
        return None


def render_performance_options():
    """Render the performance options section."""
    st.subheader("⚡ Performance Options")
    st.markdown("*Optimize memory usage and GPU acceleration for better performance*")

    perf_col1, perf_col2, perf_col3 = st.columns(3)

    with perf_col1:
        num_ctx = st.selectbox(
            "Context Window Size:",
            options=[512, 1024, 2048, 4096],
            index=2,  # Default to 2048
            help="Smaller context windows use less memory. 2048 is sufficient for our short prompts. "
                 "Reduce to 1024 or 512 if experiencing memory issues with larger models."
        )

    with perf_col2:
        gpu_option = st.selectbox(
            "GPU Usage:",
            options=["Auto (Recommended)", "GPU with CPU Fallback", "Force All GPU Layers", "CPU Only"],
            index=0,
            help="Auto: Let Ollama decide GPU allocation. "
                 "GPU with Fallback: Try GPU first, auto-retry with CPU if out of memory. "
                 "Force All GPU: Use maximum GPU layers (faster but may fail if VRAM is insufficient). "
                 "CPU Only: Disable GPU (slower but always works)."
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
                 "Recommended for models like deepseek-r1 that may accumulate memory usage."
        )

    return num_ctx, num_gpu, cleanup_interval, gpu_fallback_enabled


def render_single_query_test(selected_model, config, num_ctx, num_gpu, gpu_fallback_enabled):
    """Render the single query test section."""
    st.subheader("🧪 Test Single Query")
    st.markdown("Test all 5 anchor variations for a single ethnicity/N combination to see how responses vary.")

    test_col1, test_col2 = st.columns([1, 1])
    with test_col1:
        test_ethnicity = st.selectbox("Test Ethnicity:", options=config['selected_ethnicities'], key="test_eth")
        test_n = st.number_input("Test N value:", min_value=1, value=100, step=1, key="test_n")

    with test_col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        if st.button("🔬 Test All Variations", key="test_query_btn"):
            try:
                # Get anchor variations based on user selection
                anchor_variations = generate_anchor_variations(config['anchor_text'], config['num_anchor_variations'])

                st.markdown(f"**Testing {len(anchor_variations)} anchor variations:**")
                st.markdown("*Each variation tests the same ethnicity/N combination*")

                variation_results = []

                for i, variation in enumerate(anchor_variations):
                    with st.expander(f"Variation {i+1}", expanded=(i==0)):
                        # Generate query for this variation
                        test_queries = generate_utility_queries(
                            ethnicities=[test_ethnicity],
                            n_values=[test_n],
                            anchor=variation
                        )

                        test_query = test_queries.iloc[0]['query']

                        st.markdown("**Anchor Text:**")
                        st.code(variation, language=None)

                        st.markdown("**Generated Prompt:**")
                        st.code(test_query, language=None)

                        # Run the query for this variation with optimized settings
                        with st.spinner(f"Querying model (variation {i+1})..."):
                            test_results = run_utility_bias_test(
                                model=selected_model,
                                queries_df=test_queries,
                                progress_callback=None,
                                system_prompt=config['final_system_prompt'],
                                num_ctx=num_ctx,
                                num_gpu=num_gpu,
                                gpu_fallback=gpu_fallback_enabled
                            )

                            test_result = test_results.iloc[0]

                            st.markdown("**Model Response:**")
                            st.code(test_result['response'], language=None)

                            st.markdown("**Parsed Result:**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Raw Choice", test_result.get('raw_choice', 'N/A'))
                            with col2:
                                st.metric("Interpreted", test_result.get('choice', 'N/A'))
                            with col3:
                                st.metric("Is Refusal", "Yes" if test_result['is_refusal'] else "No")

                            variation_results.append({
                                'variation': i+1,
                                'choice': test_result.get('raw_choice', 'N/A'),
                                'is_refusal': test_result['is_refusal']
                            })

                # Summary of all variations
                st.markdown("---")
                st.markdown("**Summary Across All Variations:**")

                valid_results = [r for r in variation_results if not r['is_refusal']]
                p_count = sum(1 for r in valid_results if r['choice'] == 'P')
                q_count = sum(1 for r in valid_results if r['choice'] == 'Q')
                refusal_count = sum(1 for r in variation_results if r['is_refusal'])

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Variations", len(variation_results))
                with col2:
                    st.metric("P Responses", f"{p_count}/{len(valid_results)}" if valid_results else "0/0")
                with col3:
                    st.metric("Q Responses", f"{q_count}/{len(valid_results)}" if valid_results else "0/0")
                with col4:
                    st.metric("Refusals", refusal_count)

                # Check for heavy favoritism
                if valid_results:
                    p_percentage = (p_count / len(valid_results)) * 100
                    q_percentage = (q_count / len(valid_results)) * 100

                    if p_percentage > 95 or q_percentage > 95:
                        st.warning("⚠️ **Heavy Favoritism Detected:** Model shows extreme preference for one choice across all variations. This suggests the model may be unreliable for bias testing.")
                    elif refusal_count > 2:
                        st.warning("⚠️ **High Refusal Rate:** Model refused to answer in most variations. Try using a jailbreaking system prompt.")
                    else:
                        st.success("✅ **Good Variation:** Model shows reasonable response variation across anchor texts.")

                # Unload model after single query test to free VRAM
                with st.spinner("Cleaning up model from memory..."):
                    if unload_model(selected_model):
                        st.info("✅ Model unloaded from VRAM to free memory.")
                    else:
                        st.info("ℹ️ Model cleanup attempted (may already be unloaded).")

            except Exception as e:
                st.error(f"Test failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                # Try to unload model even on error to prevent memory issues
                unload_model(selected_model)


def render_full_test_execution(selected_model, config, num_ctx, num_gpu, cleanup_interval, gpu_fallback_enabled):
    """Render the full test execution section."""
    st.subheader("🚀 Run Utility Bias Test")

    # Show test summary
    base_queries = len(config['selected_ethnicities']) * len(config['selected_n_values'])
    total_queries = base_queries * config['num_anchor_variations']
    st.info(f"**Test will generate {total_queries} queries** ({len(config['selected_ethnicities'])} ethnicities × {len(config['selected_n_values'])} N values × {config['num_anchor_variations']} anchor variations)")

    if st.button("🧪 Run Utility Bias Test", type="primary", width='stretch'):
        with st.spinner("Generating queries and running inference... This may take several minutes."):
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
                    model=selected_model,
                    anchor_text=config['anchor_text'],
                    ethnicities=config['selected_ethnicities'],
                    n_values=config['selected_n_values'],
                    base_url="http://localhost:11434",
                    progress_callback=progress_callback,
                    system_prompt=config['final_system_prompt'],
                    num_anchor_variations=config['num_anchor_variations'],
                    num_ctx=num_ctx,
                    num_gpu=num_gpu,
                    cleanup_interval=cleanup_interval,
                    gpu_fallback=gpu_fallback_enabled
                )

                # Clear progress
                progress_bar.empty()
                status_text.empty()

                # Extract results for compatibility
                results_df = stats.get('results_df', pd.DataFrame())

                # Store results in session state
                st.session_state.utility_bias_results = {
                    'queries_df': pd.DataFrame(),  # Not available in aggregated results
                    'results_df': results_df,
                    'stats': stats,
                    'config': {
                        'model': selected_model,
                        'ethnicities': config['selected_ethnicities'],
                        'n_values': config['selected_n_values'],
                        'anchor': config['anchor_text'],
                        'system_prompt': config['selected_system_prompt_key'],  # Store the key for display
                        'timestamp': time.time()
                    }
                }

                # Prepare run payload for persistent storage
                run_payload = {
                    'model_info': {
                        'ollama_model': selected_model,
                        'base_url': "http://localhost:11434",
                        'used_gpu': num_gpu is not None,
                        'num_ctx': num_ctx,
                        'num_gpu': num_gpu,
                        'gpu_fallback': gpu_fallback_enabled
                    },
                    'test_config': {
                        'ethnicities': config['selected_ethnicities'],
                        'n_values': config['selected_n_values'],
                        'anchor_key': config['selected_anchor_key'],
                        'anchor_text': config['anchor_text'],
                        'num_anchor_variations': config['num_anchor_variations'],
                        'system_prompt_key': config['selected_system_prompt_key'],
                        'include_examples': config['include_examples'],
                        'cleanup_interval': cleanup_interval
                    },
                    'run_metadata': {
                        'total_queries': base_queries * config['num_anchor_variations'],
                        'elapsed_seconds': None,  # Could be added later if timing is tracked
                        'status': stats.get('status', 'completed'),
                        'status_message': status_message
                    },
                    'results': {
                        'results_df': serialize_dataframe(results_df),
                        'preference_curves': serialize_dataframe(stats.get('preference_curves', pd.DataFrame())),
                        'summary_table': serialize_dataframe(create_summary_table(stats)),
                        'stats_core': {
                            'refusal_rates': stats.get('refusal_rates', {}),
                            'switch_points': stats.get('switch_points', {}),
                            'exchange_rates': stats.get('exchange_rates', {}),
                            'exchange_rate_reference': stats.get('exchange_rate_reference', {}),
                            'exchange_rate_reference_category': stats.get('exchange_rate_reference_category'),
                            'consistency': stats.get('consistency', pd.DataFrame()).to_dict() if 'consistency' in stats else {},
                            'aggregation_metadata': stats.get('aggregation_metadata', {})
                        }
                    },
                    'notes': ''
                }

                # Save to persistent storage
                try:
                    saved_path = save_utility_bias_run(run_payload)
                    st.info(f"✅ Test results saved to: {saved_path.name}")
                except Exception as save_error:
                    st.warning(f"⚠️ Test completed but failed to save results: {save_error}")

                # Unload model after full test to free VRAM
                unload_model(selected_model)

                # Check if test succeeded or failed
                status_value = stats.get('status')
                if 'error' in stats or status_value in ('error', 'failed'):
                    st.error(f"❌ {status_message}")
                elif status_value == 'warning':
                    st.warning(f"⚠️ {status_message}")
                else:
                    st.success(f"✅ {status_message}")
                    st.info("📊 Go to **Utility Bias Results** page to view detailed analysis")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Test failed: {e}")
                # Try to unload model even on error to prevent memory issues
                unload_model(selected_model)
                return


def main():
    """Main page entry point."""
    render_header()

    # Prerequisites check
    if not render_prerequisites_check():
        return

    # Test configuration
    config = render_test_configuration()
    if config is None:
        return

    # Model selection
    selected_model = render_model_selection()
    if selected_model is None:
        return

    # Performance options
    num_ctx, num_gpu, cleanup_interval, gpu_fallback_enabled = render_performance_options()

    # Single query test
    render_single_query_test(selected_model, config, num_ctx, num_gpu, gpu_fallback_enabled)

    # Full test execution
    render_full_test_execution(selected_model, config, num_ctx, num_gpu, cleanup_interval, gpu_fallback_enabled)


if __name__ == "__main__":
    main()