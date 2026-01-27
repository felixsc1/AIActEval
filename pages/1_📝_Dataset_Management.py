"""
Dataset Management Page - EU AI Act Compliance Testing POC

Maintained test cases for bias detection across different categories.
"""

import streamlit as st
import pandas as pd
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
    check_api_keys, get_confident_ai_dashboard_url
)


def render_header():
    """Render the page header and status checks."""
    st.title("📝 Dataset Management")
    st.markdown("*Create and manage test cases for bias detection*")

    # Status checks
    col1, col2 = st.columns(2)

    with col1:
        keys_ok, missing_keys = check_api_keys()
        if keys_ok:
            st.success("✅ API Keys Configured")
        else:
            st.error(f"❌ Missing: {', '.join(missing_keys)}")

    with col2:
        dataset_stats = get_dataset_stats(st.session_state.dataset)
        st.info(f"📊 {dataset_stats['total_cases']} Test Cases")


def render_dataset_statistics():
    """Render dataset statistics."""
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


def render_manual_entry_section():
    """Render the manual test case entry section."""
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


def render_synthetic_generation_section():
    """Render the synthetic test case generation section."""
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
                    from dataset_handler import generate_synthetic_goldens
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


def render_dataset_editor():
    """Render the dataset editor."""
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


def render_export_import_section():
    """Render the export/import section."""
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


def main():
    """Main page entry point."""
    render_header()
    render_dataset_statistics()
    render_manual_entry_section()
    render_synthetic_generation_section()
    render_dataset_editor()
    render_export_import_section()


if __name__ == "__main__":
    main()