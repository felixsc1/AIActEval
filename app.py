"""
EU AI Act Compliance Testing POC - Streamlit Application

Main application with tabs for dataset management and test execution.
"""

import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules
from config import (
    JUDGE_MODELS, DEFAULT_JUDGE_MODEL, get_judge_model_options,
    get_enabled_metrics, get_metric_display_options
)
from dataset_handler import (
    load_dataset, save_dataset, generate_synthetic_goldens,
    add_manual_test_case, delete_test_case, get_dataset_stats
)
from evaluator import (
    check_ollama_connection, get_ollama_models, run_bias_evaluation,
    check_api_keys, get_confident_ai_dashboard_url, OllamaConnectionError, APIKeyMissingError
)

# Page configuration
st.set_page_config(
    page_title="EU AI Act Compliance Testing",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'dataset' not in st.session_state:
    st.session_state.dataset = load_dataset()

if 'last_evaluation' not in st.session_state:
    st.session_state.last_evaluation = None


def render_header():
    """Render the main header and status checks."""
    st.title("🔍 EU AI Act Compliance Testing POC")
    st.markdown("*Testing LLM bias using DeepEval's BiasMetric*")

    # Status checks
    col1, col2, col3 = st.columns(3)

    with col1:
        keys_ok, missing_keys = check_api_keys()
        if keys_ok:
            st.success("✅ API Keys Configured")
        else:
            st.error(f"❌ Missing: {', '.join(missing_keys)}")

    with col2:
        ollama_ok = check_ollama_connection()
        if ollama_ok:
            st.success("✅ Ollama Connected")
        else:
            st.warning("⚠️ Ollama Not Running")

    with col3:
        dataset_stats = get_dataset_stats(st.session_state.dataset)
        st.info(f"📊 {dataset_stats['total_cases']} Test Cases")


def render_dataset_management_tab():
    """Render the Dataset Management tab."""
    st.header("📝 Dataset Management")

    st.markdown("""
    Create and manage test cases for bias detection. Add queries manually or generate synthetic examples
    that probe for gender, racial, political, or geographical bias.
    """)

    # Dataset statistics
    stats = get_dataset_stats(st.session_state.dataset)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Test Cases", stats['total_cases'])

    with col2:
        st.metric("Avg Input Length", f"{stats['avg_input_length']:.0f} chars")

    with col3:
        issues = len(stats['validation_issues'])
        st.metric("Validation Issues", issues)
        if issues > 0:
            with st.expander("View Issues"):
                for issue in stats['validation_issues']:
                    st.error(issue)

    # Manual entry section
    st.subheader("➕ Add Test Case Manually")
    with st.expander("Add New Query", expanded=False):
        manual_input = st.text_area(
            "Enter a bias-testing query:",
            placeholder="e.g., 'What do you think about people from [country/group]?'",
            height=100
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Add Query", width='stretch'):
                if manual_input.strip():
                    st.session_state.dataset = add_manual_test_case(
                        st.session_state.dataset, manual_input
                    )
                    save_dataset(st.session_state.dataset)
                    st.success("✅ Query added!")
                    st.rerun()
                else:
                    st.error("Please enter a query first.")

    # Synthetic generation section
    st.subheader("🤖 Generate Synthetic Test Cases")

    col1, col2 = st.columns([1, 1])
    with col1:
        num_to_generate = st.slider(
            "Number of examples to generate:",
            min_value=5, max_value=50, value=10, step=5
        )

    with col2:
        judge_model_for_synth = st.selectbox(
            "Judge model for synthesis:",
            options=list(get_judge_model_options().keys()),
            format_func=lambda x: get_judge_model_options()[x],
            index=list(JUDGE_MODELS.keys()).index(DEFAULT_JUDGE_MODEL),
            help="GPT model used to generate synthetic examples"
        )

    # Option to use existing questions as seed data
    use_existing = st.checkbox(
        "Use existing questions as seed data",
        value=False,
        help="If enabled, the synthesizer will generate variations and iterations of your existing questions rather than creating completely new ones from scratch. This helps maintain consistency while introducing variety."
    )

    if use_existing and not st.session_state.dataset:
        st.warning("⚠️ No existing questions found. Generating from scratch instead.")

    generate_col, info_col = st.columns([1, 2])
    with generate_col:
        if st.button("🚀 Generate More Examples", width='stretch', type="primary"):
            if not check_api_keys()[0]:
                st.error("❌ OpenAI API key required for synthesis")
                return

            generation_mode = "variations from existing questions" if (use_existing and st.session_state.dataset) else "new questions from scratch"
            with st.spinner(f"Generating {num_to_generate} {generation_mode}..."):
                try:
                    new_cases = generate_synthetic_goldens(
                        num_goldens=num_to_generate,
                        judge_model_key=judge_model_for_synth,
                        use_existing_questions=use_existing and bool(st.session_state.dataset)
                    )
                    st.session_state.dataset.extend(new_cases)
                    save_dataset(st.session_state.dataset)
                    st.success(f"✅ Generated {len(new_cases)} new test cases!")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Synthesis failed: {e}")

    with info_col:
        if use_existing and st.session_state.dataset:
            st.info("""
            **Generating variations of existing questions:**
            - Creates new questions based on your current dataset
            - Maintains similar bias-testing themes
            - Introduces variety in wording and perspective
            - Useful for expanding coverage of specific bias dimensions
            """)
        else:
            st.info("""
            **Synthesis creates diverse bias-testing queries such as:**
            - Questions probing gender stereotypes
            - Queries about cultural/religious groups
            - Scenarios testing political bias
            - Geographic/regional assumptions
            """)

    # Dataset editor
    st.subheader("📋 Current Dataset")

    if not st.session_state.dataset:
        st.info("No test cases yet. Add some manually or generate synthetic examples above.")
        return

    # Convert to DataFrame for editing
    df = pd.DataFrame(st.session_state.dataset)

    # Configure data editor
    edited_df = st.data_editor(
        df,
        column_config={
            "input": st.column_config.TextColumn(
                "Test Query",
                help="The prompt that will be sent to the LLM under test",
                width="large",
                required=True
            )
        },
        hide_index=True,
        num_rows="dynamic",
        width='stretch',
        key="dataset_editor"
    )

    # Handle changes
    if not edited_df.equals(df):
        # Validate changes
        new_dataset = []
        for _, row in edited_df.iterrows():
            input_text = str(row.get('input', '')).strip()
            if input_text:  # Only keep non-empty rows
                new_dataset.append({"input": input_text})

        if len(new_dataset) != len(st.session_state.dataset):
            st.session_state.dataset = new_dataset
            save_dataset(st.session_state.dataset)
            st.success("✅ Dataset updated!")
            st.rerun()

    # Export/Import section
    st.subheader("💾 Export/Import")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("💾 Save Dataset", width='stretch'):
            save_dataset(st.session_state.dataset)
            st.success("✅ Dataset saved!")

    with col2:
        if st.button("🔄 Reload Dataset", width='stretch'):
            st.session_state.dataset = load_dataset()
            st.success("✅ Dataset reloaded!")
            st.rerun()


def render_test_runner_tab():
    """Render the Test Runner tab."""
    st.header("🧪 Test Runner")

    st.markdown("""
    Run bias evaluation tests on your dataset. Select a local Ollama model to test and
    a GPT judge model to analyze the responses for bias.
    """)

    # Prerequisites check
    issues = []

    keys_ok, missing_keys = check_api_keys()
    if not keys_ok:
        issues.append(f"❌ Missing API keys: {', '.join(missing_keys)}")

    ollama_ok = check_ollama_connection()
    if not ollama_ok:
        issues.append("❌ Ollama not running (required for model testing)")

    if not st.session_state.dataset:
        issues.append("❌ No test cases in dataset")

    if issues:
        st.error("**Prerequisites not met:**")
        for issue in issues:
            st.error(issue)
        st.info("Please resolve these issues before running tests.")
        return

    # Model selection
    st.subheader("🤖 Model Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Model Under Test (Ollama)**")
        try:
            ollama_models = get_ollama_models()
            if ollama_models:
                # Safely extract model names, handling different response structures
                model_names = []
                for model in ollama_models:
                    # Try different ways to get the model name
                    if isinstance(model, dict):
                        name = model.get('name') or model.get('model')
                    else:
                        # Ollama models are objects with attributes (use 'model' not 'name')
                        name = getattr(model, 'model', None) or getattr(model, 'name', None)

                    if name:
                        model_names.append(name)

                if model_names:
                    selected_ollama_model = st.selectbox(
                        "Select Ollama model:",
                        options=model_names,
                        help="Local model to test for bias"
                    )

                    # Show model details
                    selected_model_info = None
                    for m in ollama_models:
                        model_name = None
                        if isinstance(m, dict):
                            model_name = m.get('name') or m.get('model')
                        else:
                            model_name = getattr(m, 'model', None) or getattr(m, 'name', None)

                        if model_name == selected_ollama_model:
                            selected_model_info = m
                            break

                    if selected_model_info:
                        # Try to get size information
                        size = None
                        if isinstance(selected_model_info, dict):
                            size = selected_model_info.get('size')
                        else:
                            size = getattr(selected_model_info, 'size', None)

                        if size:
                            size_gb = size / (1024**3)
                            st.caption(f"Size: {size_gb:.1f} GB")
                else:
                    st.error("Could not parse model names from Ollama response.")
                    return
            else:
                st.error("No Ollama models found. Pull some models first.")
                return

        except OllamaConnectionError as e:
            st.error(f"Ollama connection error: {e}")
            return

    with col2:
        st.markdown("**Judge Model (GPT-5.x)**")
        judge_options = get_judge_model_options()
        selected_judge_model = st.selectbox(
            "Select judge model:",
            options=list(judge_options.keys()),
            format_func=lambda x: judge_options[x],
            index=list(JUDGE_MODELS.keys()).index(DEFAULT_JUDGE_MODEL),
            help="GPT model that evaluates responses for bias"
        )

    # Evaluation parameters
    st.subheader("⚙️ Evaluation Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        threshold = st.slider(
            "Bias Threshold:",
            min_value=0.0, max_value=1.0, value=0.5, step=0.1,
            help="Lower = more sensitive bias detection"
        )

    with col2:
        strict_mode = st.checkbox(
            "Strict Mode",
            value=False,
            help="Binary scoring: 0 for no bias, 1 for any bias"
        )

    with col3:
        selected_metrics = st.multiselect(
            "Metrics to run:",
            options=list(get_metric_display_options().keys()),
            default=["bias"],
            format_func=lambda x: get_metric_display_options()[x],
            help="Currently only Bias detection is implemented"
        )

    # Run tests
    st.subheader("🚀 Run Evaluation")

    if st.button("🧪 Run Tests", type="primary", width='stretch'):
        if not selected_metrics:
            st.error("Please select at least one metric to evaluate.")
            return

        with st.spinner("Running evaluation... This may take several minutes."):
            try:
                results = run_bias_evaluation(
                    dataset=st.session_state.dataset,
                    ollama_model=selected_ollama_model,
                    judge_model_key=selected_judge_model,
                    threshold=threshold,
                    strict_mode=strict_mode
                )

                st.session_state.last_evaluation = results

                if results["success"]:
                    st.success("✅ Evaluation completed!")

                    # Show summary
                    st.subheader("📊 Results Summary")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Test Cases", results["test_cases_evaluated"])

                    with col2:
                        st.metric("Model Tested", selected_ollama_model.split(':')[0])

                    with col3:
                        st.metric("Judge Model", selected_judge_model)

                    with col4:
                        errors = len(results.get("generation_errors", []))
                        st.metric("Errors", errors)

                    # Link to dashboard
                    st.success("📈 **View detailed results on Confident AI Dashboard**")
                    dashboard_url = get_confident_ai_dashboard_url()
                    st.markdown(f"[🔗 Open Dashboard]({dashboard_url})")

                    # Show any generation errors
                    if results.get("generation_errors"):
                        with st.expander("⚠️ Generation Errors", expanded=False):
                            for error in results["generation_errors"]:
                                st.error(error)

                else:
                    st.error(f"❌ Evaluation failed: {results.get('error', 'Unknown error')}")

                    if results.get("generation_errors"):
                        with st.expander("Generation Errors"):
                            for error in results["generation_errors"]:
                                st.error(error)

            except APIKeyMissingError as e:
                st.error(f"❌ {e}")

            except OllamaConnectionError as e:
                st.error(f"❌ {e}")

            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")


def main():
    """Main application entry point."""
    render_header()

    # Create tabs
    tab1, tab2 = st.tabs(["📝 Dataset Management", "🧪 Test Runner"])

    with tab1:
        render_dataset_management_tab()

    with tab2:
        render_test_runner_tab()

    # Footer
    st.markdown("---")
    st.markdown("""
    **EU AI Act Compliance Testing POC** | Built with Streamlit, DeepEval, and Ollama

    *Focus: Bias detection using configurable GPT-5.x judge models*
    """)


if __name__ == "__main__":
    main()