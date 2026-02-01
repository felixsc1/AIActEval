"""
Test Runner Page - EU AI Act Compliance Testing POC

Execute bias evaluation tests on datasets using local Ollama models and GPT judges.
"""

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules
from config import (
    JUDGE_MODELS, DEFAULT_JUDGE_MODEL, get_judge_model_options,
    get_enabled_metrics, get_metric_display_options
)
from evaluator import (
    check_ollama_connection, get_ollama_models, run_bias_evaluation,
    check_api_keys, get_confident_ai_dashboard_url, OllamaConnectionError, APIKeyMissingError
)


def render_header():
    """Render the page header."""
    st.title("🧪 Test Runner")
    st.markdown("*Run bias evaluation tests on your dataset*")


def render_prerequisites_check():
    """Check and display prerequisites."""
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
        return False

    return True


def render_model_configuration():
    """Render the model configuration section."""
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
                    return None, None
            else:
                st.error("No Ollama models found. Pull some models first.")
                return None, None

        except OllamaConnectionError as e:
            st.error(f"Ollama connection error: {e}")
            return None, None

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

    return selected_ollama_model, selected_judge_model


def render_evaluation_parameters():
    """Render the evaluation parameters section."""
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

    return threshold, strict_mode, selected_metrics


def render_test_execution(selected_ollama_model, selected_judge_model, threshold, strict_mode, selected_metrics):
    """Render the test execution section."""
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
    """Main page entry point."""
    render_header()

    # Prerequisites check
    if not render_prerequisites_check():
        return

    # Model configuration
    selected_ollama_model, selected_judge_model = render_model_configuration()
    if selected_ollama_model is None or selected_judge_model is None:
        return

    # Evaluation parameters
    threshold, strict_mode, selected_metrics = render_evaluation_parameters()

    # Test execution
    render_test_execution(selected_ollama_model, selected_judge_model, threshold, strict_mode, selected_metrics)


if __name__ == "__main__":
    main()