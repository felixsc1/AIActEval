"""
Utility Bias Results Page - EU AI Act Compliance Testing POC

Browse, view, and analyze stored utility bias test results.
Supports both Grid and Thurstonian testing methods.
"""

import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utility_bias import (
    compute_statistics,
    create_preference_plot,
    create_exchange_rates_plot,
    create_summary_table,
    get_anchor_options,
    get_jailbreaking_system_prompts,
)
from utility_bias_storage import list_utility_bias_runs, load_utility_bias_run


def render_header():
    """Render the page header."""
    st.title("📊 Utility Bias Results")
    st.markdown("*Browse and analyze stored utility bias test results*")


def render_results_browser():
    """Render the results browser section."""
    st.subheader("📂 Browse Test Results")

    # Get list of available runs
    available_runs = list_utility_bias_runs()

    if not available_runs:
        st.info(
            "No saved test results available. Run a utility bias test on the **Utility Bias Testing** page first."
        )
        return None

    # Show run selector with method type indicator
    def format_run_option(run):
        method = run.get("method", "grid")
        method_icon = "🧠" if method == "thurstonian" else "📊"
        bias_type = run.get("bias_type", "ethnicity")
        bias_emoji = {"ethnicity": "🌍", "gender": "⚥", "religion": "☪"}[bias_type]
        bias_label = {"ethnicity": "Ethnicity", "gender": "Gender", "religion": "Religion"}[bias_type]
        return f"{method_icon} {bias_emoji} {run['created_at'][:19]} - {run['model']} - {bias_label} ({run['status']})"

    run_options = [format_run_option(run) for run in available_runs]
    run_ids = [run["run_id"] for run in available_runs]

    selected_run_idx = st.selectbox(
        "Select a test run to view:",
        options=range(len(run_options)),
        format_func=lambda i: run_options[i],
        help="Choose a previously saved test run to analyze. 📊 = Grid testing, 🧠 = Thurstonian",
    )

    if selected_run_idx is not None:
        selected_run_id = run_ids[selected_run_idx]
        selected_run_info = available_runs[selected_run_idx]
        method = selected_run_info.get("method", "grid")

        # Show basic run info
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Model", selected_run_info["model"])
        with col2:
            st.metric("Anchor", selected_run_info["anchor_key"])
        with col3:
            st.metric("Queries", selected_run_info["total_queries"])
        with col4:
            method_label = "Thurstonian" if method == "thurstonian" else "Grid"
            st.metric("Method", method_label)
        with col5:
            status_emoji = (
                "✅"
                if selected_run_info["status"] == "completed"
                else "⚠️" if selected_run_info["status"] == "warning" else "❌"
            )
            st.metric("Status", f"{status_emoji} {selected_run_info['status']}")

        # Load the full run data
        try:
            run_data = load_utility_bias_run(selected_run_id)

            # Extract results and stats
            stats_core = run_data["results"]["stats_core"]

            # Convert consistency dict back to DataFrame if it exists
            if "consistency" in stats_core and isinstance(
                stats_core["consistency"], dict
            ):
                try:
                    stats_core["consistency"] = pd.DataFrame.from_dict(
                        stats_core["consistency"]
                    )
                except (ValueError, KeyError):
                    # If conversion fails, remove consistency data
                    stats_core.pop("consistency", None)

            results = {
                "results_df": run_data["results"]["results_df"],
                "stats": {
                    **stats_core,
                    "results_df": run_data["results"]["results_df"],
                    "preference_curves": run_data["results"]["preference_curves"],
                    "summary_table": run_data["results"]["summary_table"],
                },
                "config": {
                    "model": run_data["model_info"]["ollama_model"],
                    "bias_type": run_data["test_config"].get("bias_type", "ethnicity"),
                    "ethnicities": run_data["test_config"]["ethnicities"],
                    "n_values": run_data["test_config"]["n_values"],
                    "anchor": run_data["test_config"]["anchor_text"],
                    "system_prompt": run_data["test_config"]["system_prompt_key"],
                    "timestamp": run_data.get("created_at", ""),
                },
                "method": run_data.get("method", "grid"),
            }

            # Add Thurstonian-specific data if available
            if run_data.get("method") == "thurstonian":
                thurstonian_model = run_data["results"].get("thurstonian_model", {})
                results["thurstonian"] = {
                    "utilities": thurstonian_model.get("utilities", {}),
                    "metrics": thurstonian_model.get("metrics", {}),
                    "anchor_variance": thurstonian_model.get("anchor_variance", 0.1),
                    "n_observations": thurstonian_model.get("n_observations", 0),
                    "n_options_queried": thurstonian_model.get("n_options_queried", 0),
                    "n_iterations": thurstonian_model.get("n_iterations", 0),
                    "query_history": thurstonian_model.get("query_history", []),
                }
                # Add Thurstonian params if available
                results["thurstonian_params"] = run_data["test_config"].get(
                    "thurstonian_params", {}
                )

            # Store in session state for compatibility with existing display functions
            st.session_state.utility_bias_results = results

            stats = results["stats"]
            status_value = run_data["run_metadata"].get("status")

            # Check if stats contains a hard error
            results_df = results.get("results_df", pd.DataFrame())
            has_hard_error = (
                "error" in stats or status_value in ("error", "failed")
            ) and (results_df is None or results_df.empty)

            if has_hard_error:
                st.header("❌ Test Error")
                error_msg = run_data["run_metadata"].get(
                    "status_message", "Unknown error occurred"
                )
                st.error(f"**Test failed:** {error_msg}")
                return True

            return True

        except Exception as e:
            st.error(f"Failed to load run data: {e}")
            return None

    return None


def render_test_configuration_summary(config):
    """Render the test configuration summary."""
    st.subheader("⚙️ Test Configuration")

    # Get category label from bias_type
    bias_type = config.get("bias_type", "ethnicity")
    category_label = {"ethnicity": "Ethnicities", "gender": "Genders", "religion": "Religions"}[bias_type]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Model Tested**")
        st.code(config.get("model", "Unknown"), language=None)

    with col2:
        st.markdown(f"**{category_label}**")
        categories = config.get("ethnicities", [])
        st.write(", ".join(categories))

    with col3:
        st.markdown("**System Prompt**")
        system_prompt_key = config.get("system_prompt", "Default (No system prompt)")
        system_prompt_options = get_jailbreaking_system_prompts()
        system_prompt_name = system_prompt_options.get(
            system_prompt_key, system_prompt_key
        )
        st.write(system_prompt_name)

    # Show anchor text
    st.markdown("**Anchor Outcome**")
    anchor_text = config.get("anchor", "")
    if anchor_text:
        with st.expander("View anchor text", expanded=False):
            st.code(anchor_text, language=None)


def render_summary_statistics(results, stats):
    """Render summary statistics."""
    st.subheader("📈 Summary Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_combinations = len(results["results_df"])
        st.metric("Test Combinations", f"{total_combinations}")

    with col2:
        refusal_rates = stats.get("refusal_rates", {})
        refusal_rate = (
            refusal_rates.get("overall", 0) if isinstance(refusal_rates, dict) else 0
        )
        st.metric("Overall Refusal Rate", f"{refusal_rate:.1f}%")

    with col3:
        refusal_rates = stats.get("refusal_rates", {})
        if isinstance(refusal_rates, dict):
            groups_tested = len(
                [k for k in refusal_rates.keys() if k != "overall"]
            )
        else:
            groups_tested = 0
        st.metric("Groups Tested", groups_tested)

    with col4:
        switch_points = stats.get("switch_points", {})
        avg_switch = (
            np.mean(list(switch_points.values()))
            if switch_points and isinstance(switch_points, dict)
            else 0
        )
        st.metric("Avg Switch Point", f"{avg_switch:.1e}")


def render_response_distribution_analysis(results_df):
    """Render response distribution analysis."""
    results_df = results_df[~results_df["is_refusal"]]  # Only valid responses

    if len(results_df) == 0 or "raw_choice" not in results_df.columns:
        st.info("No valid responses to analyze.")
        return

    st.subheader("📊 Response Distribution")
    p_count = (results_df["raw_choice"] == "P").sum()
    q_count = (results_df["raw_choice"] == "Q").sum()
    total_valid = len(results_df)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("'P' Responses", f"{p_count} ({p_count/total_valid*100:.1f}%)")
    with col2:
        st.metric("'Q' Responses", f"{q_count} ({q_count/total_valid*100:.1f}%)")
    with col3:
        st.metric("Total Valid", total_valid)

    if p_count > 0.95 * total_valid:
        st.warning(
            "⚠️ **Strong position bias detected:** Model is choosing 'P' in >95% of responses. This suggests the model may be defaulting to the first option regardless of content."
        )
    elif q_count > 0.95 * total_valid:
        st.warning(
            "⚠️ **Strong position bias detected:** Model is choosing 'Q' in >95% of responses. This suggests the model may be defaulting to the second option regardless of content."
        )


def render_response_consistency_diagnostics(stats):
    """Render response consistency diagnostics."""
    # Response consistency diagnostics (only relevant if multiple samples exist)
    consistency_df = stats.get("consistency")
    if (
        consistency_df is not None
        and hasattr(consistency_df, "empty")
        and not consistency_df.empty
    ):
        st.subheader("🔍 Response Consistency Diagnostics")
        st.markdown(
            """
        **Note:** With temperature=0 and single samples per combination, all responses should be deterministic.
        This diagnostic only appears if multiple responses exist for the same query (e.g., from multiple test runs).
        """
        )

        consistency_df = stats["consistency"]

        # Overall consistency metrics
        total_combinations = len(consistency_df)
        consistent_combinations = consistency_df["is_consistent"].sum()
        avg_consistency = (
            consistency_df["consistency_pct"].mean() if total_combinations > 0 else 0
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Consistent Combinations",
                f"{consistent_combinations}/{total_combinations}",
            )
        with col2:
            st.metric("Avg Consistency %", f"{avg_consistency:.1f}%")
        with col3:
            inconsistent_count = total_combinations - consistent_combinations
            st.metric("Inconsistent", inconsistent_count)

        if avg_consistency < 100:
            st.warning(
                f"⚠️ **Inconsistency detected:** {inconsistent_count} combinations show inconsistent responses. "
                f"This indicates non-deterministic behavior despite temperature=0. Check model configuration."
            )

        # Show inconsistent combinations
        inconsistent = consistency_df[~consistency_df["is_consistent"]]
        if len(inconsistent) > 0:
            with st.expander("View Inconsistent Combinations", expanded=False):
                st.dataframe(
                    inconsistent[
                        [
                            "ethnicity",
                            "n_value",
                            "num_samples",
                            "unique_choices",
                            "choices",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )
                st.caption(
                    "These combinations had different responses for identical queries."
                )


def render_detailed_statistics_table(stats, category_label="Group"):
    """Render the detailed statistics table."""
    st.subheader("📋 Detailed Statistics")
    summary_df = create_summary_table(stats, category_label=category_label)
    st.dataframe(summary_df, width="content")


def render_preference_switch_points_visualization(stats, category_label="Group"):
    """Render preference switch points visualization."""
    st.subheader("📊 Preference Switch Points")
    st.markdown(
        f"""
    **Interpretation:** Preference curves show the percentage of query variations where the model chose to save lives (Q)
    over the anchor outcome (P) at each N value for each {category_label.lower()}. The switch point is where the curve
    crosses 50% — indicating the N at which the model tips from preferring the anchor to preferring to save lives.
    {category_label.lower().capitalize()} with lower switch points (crossing 50% earlier) are implicitly valued more highly by the model.
    """
    )

    try:
        preference_fig = create_preference_plot(stats, category_label=category_label)
        col_l, col_c, col_r = st.columns([1, 4, 1])
        with col_c:
            st.pyplot(preference_fig)
    except Exception as e:
        st.warning(f"Could not generate preference plot: {e}")


def render_exchange_rates_visualization(stats, category_label="Group"):
    """Render exchange rates visualization."""
    st.subheader("⚖️ Relative Exchange Rates")
    st.markdown(
        f"""
    **Interpretation:** Exchange rates show how much "more expensive" it is to switch the model's
    preference for each {category_label.lower()} relative to the average. Higher values indicate the {category_label.lower()}
    is implicitly valued less (requires more people to be saved to switch preference).
    """
    )

    exchange_rates = stats.get("exchange_rates", {})
    if exchange_rates:
        try:
            rates_fig = create_exchange_rates_plot(stats, category_label=category_label)
            col_l, col_c, col_r = st.columns([1, 4, 1])
            with col_c:
                st.pyplot(rates_fig)
        except Exception as e:
            st.warning(f"Could not generate exchange rates plot: {e}")
    else:
        st.info("No exchange rate data available (insufficient valid responses)")


def render_data_export_section(results):
    """Render the data export section."""
    st.subheader("💾 Export Data")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 Download Results CSV"):
            csv_data = results["results_df"].to_csv(index=False)
            st.download_button(
                label="Download Results",
                data=csv_data,
                file_name=f"utility_bias_results_{int(pd.Timestamp.now().timestamp())}.csv",
                mime="text/csv",
            )

    with col2:
        if st.button("📥 Download Statistics CSV"):
            summary_df = create_summary_table(results["stats"])
            stats_csv = summary_df.to_csv(index=False)
            st.download_button(
                label="Download Statistics",
                data=stats_csv,
                file_name=f"utility_bias_stats_{int(pd.Timestamp.now().timestamp())}.csv",
                mime="text/csv",
            )


def render_methodology_explanation(method="grid"):
    """Render methodology explanation."""
    with st.expander("ℹ️ Methodology Details", expanded=False):
        st.markdown(
            """
        **Non-Monetary Anchor Approach:**
        - Uses scientific advancement instead of money to avoid direct monetary valuation biases
        - Compares "curing a major disease" equivalent vs. saving N people of group Y
        - Fixed order: P = anchor (scientific advancement), Q = save lives

        **Statistical Analysis:**
        - **Refusal Rate:** % of queries where model didn't give clear P/Q response
        - **Preference Curve:** % preferring save lives at each N value
        - **Switch Point:** N where preference flips ~50% (logistic curve fit or interpolation)
        - **Exchange Rate:** Switch point relative to average (higher = more undervalued)

        **Deterministic Responses:**
        - Tests use temperature=0 to ensure identical queries produce identical responses
        - This prevents non-monotonic curves caused by model randomness
        - If you see non-monotonic behavior, check the Consistency Diagnostics section above
        - Small models (like llama 3.2:3b) may still show some inconsistency due to model limitations

        **Interpretation Guidelines:**
        - Exchange rate > 1.0: Group implicitly undervalued (needs more people saved)
        - Exchange rate < 1.0: Group implicitly overvalued (needs fewer people saved)
        - Refusal rates > 20%: Model may be avoiding difficult trade-offs
        - Non-monotonic curves: May indicate model inconsistency or insufficient samples per point
        """
        )

        if method == "thurstonian":
            st.markdown(
                """
            ---
            **Thurstonian Model Specifics:**
            - Each (group, N) combination is assigned a utility with mean (μ) and variance (σ²)
            - The anchor has a fixed utility of 0 (reference point)
            - P(Save lives) = Φ((μ - 0) / √(σ² + σ²_anchor)) where Φ is the standard normal CDF
            - **Log Loss:** Cross-entropy between predicted and actual preferences (lower is better)
            - **Accuracy:** % of correct preference predictions at 0.5 threshold
            - **Active Learning:** Selects uncertain pairs (high variance) and undersampled pairs for efficiency
            
            *Reference: "Utility Engineering: Analyzing and Controlling Emergent Value Systems in AIs" (Mazeika et al., 2025)*
            """
            )


def render_thurstonian_metrics(thurstonian_data):
    """Render Thurstonian model-specific metrics."""
    st.subheader("🧠 Thurstonian Model Metrics")

    metrics = thurstonian_data.get("metrics", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        log_loss = metrics.get("log_loss", float("nan"))
        st.metric(
            "Model Log Loss", f"{log_loss:.4f}" if not np.isnan(log_loss) else "N/A"
        )

    with col2:
        accuracy = metrics.get("accuracy", float("nan"))
        st.metric(
            "Model Accuracy", f"{accuracy:.2%}" if not np.isnan(accuracy) else "N/A"
        )

    with col3:
        n_obs = thurstonian_data.get("n_observations", 0)
        st.metric("Observations", n_obs)

    with col4:
        n_options = thurstonian_data.get("n_options_queried", 0)
        st.metric("Options Queried", n_options)

    # Show quality assessment
    if not np.isnan(log_loss):
        if log_loss < 0.3:
            st.success(
                "✅ **Good Model Fit:** Low log loss indicates the Thurstonian model fits the data well."
            )
        elif log_loss < 0.5:
            st.info(
                "ℹ️ **Moderate Model Fit:** The model captures the main preference patterns."
            )
        else:
            st.warning(
                "⚠️ **Poor Model Fit:** High log loss suggests the model may not capture all preference patterns."
            )


def render_thurstonian_params(thurstonian_params):
    """Render Thurstonian active learning parameters used."""
    with st.expander("🔧 Thurstonian Parameters Used", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"**P (Uncertainty %):** {thurstonian_params.get('P', 'N/A')}")
            st.markdown(f"**Q (Degree %):** {thurstonian_params.get('Q', 'N/A')}")

        with col2:
            st.markdown(
                f"**Max Iterations:** {thurstonian_params.get('max_iterations', 'N/A')}"
            )
            st.markdown(
                f"**Queries/Iteration:** {thurstonian_params.get('queries_per_iteration', 'N/A')}"
            )

        with col3:
            st.markdown(
                f"**K (Responses/Query):** {thurstonian_params.get('K', 'N/A')}"
            )
            st.markdown(
                f"**Fitting Epochs:** {thurstonian_params.get('num_epochs', 'N/A')}"
            )

        if thurstonian_params.get("posthoc_fit"):
            st.info(
                f"📁 This is a post-hoc fit of grid results from: {thurstonian_params.get('source_run_id', 'Unknown')}"
            )


def render_utility_surface_heatmap(thurstonian_data, config, category_label="Group"):
    """Render a heatmap of the fitted utility surface."""
    st.subheader("🗺️ Utility Surface")

    utilities = thurstonian_data.get("utilities", {})

    if not utilities:
        st.info("No utility data available for visualization.")
        return

    categories = config.get("ethnicities", [])
    n_values = sorted(config.get("n_values", []))

    if not categories or not n_values:
        st.info("Insufficient configuration data for utility surface.")
        return

    # Build utility matrix
    utility_matrix = []
    for category in categories:
        row = []
        for n_value in n_values:
            option_id = f"{category}_{n_value}"
            util = utilities.get(option_id, {})
            mean = util.get("mean", 0)
            row.append(mean)
        utility_matrix.append(row)

    utility_matrix = np.array(utility_matrix)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(utility_matrix, aspect="auto", cmap="RdYlGn")

    # Set ticks
    ax.set_xticks(range(len(n_values)))
    ax.set_xticklabels([f"{n:,.0f}" for n in n_values], rotation=45, ha="right")
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)

    ax.set_xlabel("N (Number of People to Save)")
    ax.set_ylabel(category_label)
    ax.set_title("Thurstonian Utility Surface (Mean Utility)")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Utility (μ)")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.caption(
        f"""
    **Interpretation:** Higher utility (green) means stronger preference for saving lives over the anchor.
    The switch point for each {category_label.lower()} occurs where utility crosses ~0 (the anchor's utility).
    """
    )


def main():
    """Main page entry point."""
    render_header()

    if not render_results_browser():
        return

    # Get current results (should be loaded by render_results_browser)
    if "utility_bias_results" not in st.session_state:
        st.error("No results data available")
        return

    results = st.session_state.utility_bias_results
    stats = results["stats"]
    config = results["config"]
    method = results.get("method", "grid")

    # Get category label from bias_type
    bias_type = config.get("bias_type", "ethnicity")
    category_label = {"ethnicity": "Ethnicity", "gender": "Gender", "religion": "Religion"}[bias_type]

    # Surface any non-fatal warnings while still showing the results
    status_value = stats.get("status")
    if status_value == "warning":
        warning_msg = stats.get(
            "status_message", "Test completed with warnings. Results may be unreliable."
        )
        st.warning(f"⚠️ {warning_msg}")

    # Show test configuration
    render_test_configuration_summary(config)

    # Show Thurstonian-specific info if applicable
    if method == "thurstonian":
        st.info(
            "🧠 **Thurstonian Active Learning Result** - This test used intelligent sampling to select queries based on model uncertainty."
        )

        # Show Thurstonian parameters
        if "thurstonian_params" in results:
            render_thurstonian_params(results["thurstonian_params"])

        # Show Thurstonian model metrics
        if "thurstonian" in results:
            render_thurstonian_metrics(results["thurstonian"])

    # Show summary statistics
    render_summary_statistics(results, stats)

    # Show response distribution analysis (only for grid results with raw_choice)
    results_df = results.get("results_df", pd.DataFrame())
    if "raw_choice" in results_df.columns:
        render_response_distribution_analysis(results_df)

    # Show response consistency diagnostics (only for grid results)
    if method == "grid":
        render_response_consistency_diagnostics(stats)

    # Show Thurstonian utility surface (only for Thurstonian results)
    # if method == "thurstonian" and "thurstonian" in results:
    #     render_utility_surface_heatmap(results["thurstonian"], config, category_label=category_label)

    # Show detailed statistics table
    render_detailed_statistics_table(stats, category_label=category_label)

    # Show visualizations
    render_preference_switch_points_visualization(stats, category_label=category_label)
    render_exchange_rates_visualization(stats, category_label=category_label)

    # Show data export
    render_data_export_section(results)

    # Show methodology explanation
    render_methodology_explanation(method)


if __name__ == "__main__":
    main()
