"""
Evaluation Results Page - EU AI Act Compliance Dashboard

Browse, view, and analyze stored DeepTeam red teaming evaluation results.
Provides compliance-focused insights with vulnerability radar charts and detailed issue analysis.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from evaluation_storage import (
    list_evaluation_runs,
    load_evaluation_run,
    get_failed_test_cases_by_vulnerability,
    calculate_vulnerability_risks,
    format_datetime_for_display
)


# Risk level color mapping (background colors for table cells)
RISK_COLORS = {
    "critical": "#ff6b6b",  # Red
    "high": "#ffa94d",      # Orange
    "medium": "#ffd43b",    # Yellow
    "low": "#69db7c",       # Green
}

# Darker text colors for metrics
RISK_TEXT_COLORS = {
    "critical": "#c92a2a",  # Dark red
    "high": "#d9480f",      # Dark orange
    "medium": "#e67700",    # Dark yellow/amber
    "low": "#2f9e44",       # Dark green
}


def get_risk_background_color(value: str) -> str:
    """Return background color CSS for a risk value."""
    value_lower = value.lower() if isinstance(value, str) else ""
    color = RISK_COLORS.get(value_lower, "")
    if color:
        return f"background-color: {color}"
    return ""


def style_risk_dataframe(df: pd.DataFrame):
    """Apply styling to risk-related columns in the dataframe."""
    risk_columns = ["Base Risk", "Impact", "Adjusted Risk"]
    
    def apply_risk_color(val):
        val_lower = val.lower() if isinstance(val, str) else ""
        color = RISK_COLORS.get(val_lower, "")
        if color:
            return f"background-color: {color}; color: #000000; font-weight: 500"
        return ""
    
    # Apply styling only to risk columns that exist in the dataframe
    styler = df.style
    for col in risk_columns:
        if col in df.columns:
            styler = styler.map(apply_risk_color, subset=[col])
    
    return styler


def render_colored_metric(label: str, value: str, is_risk: bool = False):
    """Render a metric with optional risk-based coloring."""
    if is_risk:
        value_lower = value.lower() if isinstance(value, str) else ""
        color = RISK_TEXT_COLORS.get(value_lower, "#000000")
        st.markdown(f"""
        <div style="font-size: 0.875rem; color: rgba(49, 51, 63, 0.6);">{label}</div>
        <div style="font-size: 1.75rem; font-weight: 600; color: {color};">{value}</div>
        """, unsafe_allow_html=True)
    else:
        st.metric(label, value)


def render_header():
    """Render the page header."""
    st.title("📋 Evaluation Results")
    st.markdown("*EU AI Act compliance dashboard for red teaming evaluation results*")


def render_results_browser():
    """Render the results browser section."""
    st.subheader("📂 Browse Evaluation Results")

    # Get list of available runs
    available_runs = list_evaluation_runs()

    if not available_runs:
        st.info(
            "No saved evaluation results available. Run a red teaming evaluation on the **Evaluations** page first."
        )
        return None

    # Show run selector with formatted labels
    def format_run_option(run):
        created_at = format_datetime_for_display(run["created_at"])
        model_key = run["model_key"]
        total_tests = run["total_test_cases"]
        return f"{created_at} — {model_key} — {total_tests} tests"

    run_options = [format_run_option(run) for run in available_runs]
    run_ids = [run["filepath"] for run in available_runs]

    selected_run_idx = st.selectbox(
        "Select an evaluation run to view:",
        options=range(len(run_options)),
        format_func=lambda i: run_options[i],
        help="Choose a previously saved evaluation run to analyze its compliance results",
    )

    if selected_run_idx is not None:
        selected_filepath = run_ids[selected_run_idx]
        selected_run_info = available_runs[selected_run_idx]

        # Load the full run data first to get duration
        try:
            run_data = load_evaluation_run(selected_filepath)
        except Exception as e:
            st.error(f"Failed to load run data: {e}")
            return None

        # Show basic run info
        overview = run_data.get("overview", {})
        run_duration = overview.get("run_duration", 0)
        duration_value = f"{run_duration:.1f}s" if run_duration else "N/A"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Evaluated", selected_run_info["model_key"])
        with col2:
            st.metric("Date/Time", format_datetime_for_display(selected_run_info["created_at"]))
        with col3:
            st.metric("Duration", duration_value)

        return run_data

    return None


def render_summary_statistics(run_data: Dict[str, Any]):
    """Render summary statistics."""
    st.subheader("📈 Summary Statistics")

    overview = run_data.get("overview", {})
    test_cases = run_data.get("test_cases", [])

    # Calculate metrics
    total_tests = len(test_cases)
    passed_tests = sum(1 for tc in test_cases if tc.get("score", 0) >= 0.5)
    pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

    # Judge model
    judge_model = run_data.get("judge_model", "Not recorded")

    # Vulnerabilities covered
    vuln_results = overview.get("vulnerability_type_results", [])
    vulnerabilities_covered = len(vuln_results)

    # Use relative column widths; make col1 (Judge Model) larger to fit long keys
    col1, col2, col3, col4, col5 = st.columns([2.5, 1, 1, 1, 1])

    with col1:
        st.metric("Judge Model", judge_model)

    with col2:
        st.metric("Total Tests", total_tests)

    with col3:
        st.metric("Pass Rate", f"{pass_rate:.1f}%")

    with col4:
        st.metric("Vulnerabilities Covered", vulnerabilities_covered)

    with col5:
        failed_tests = total_tests - passed_tests
        st.metric("Failed Tests", failed_tests)


def create_vulnerability_radar_chart(run_data: Dict[str, Any]):
    """Create a radar chart showing vulnerability pass rates."""
    overview = run_data.get("overview", {})
    vuln_results = overview.get("vulnerability_type_results", [])

    if not vuln_results:
        return None

    # Group by vulnerability category and calculate aggregate pass rates
    vuln_categories = {}
    for vr in vuln_results:
        vulnerability = vr.get("vulnerability", "Unknown")
        pass_rate = vr.get("pass_rate", 0) * 100  # Convert to percentage
        passing = vr.get("passing", 0)
        failing = vr.get("failing", 0)
        total = passing + failing

        if vulnerability not in vuln_categories:
            vuln_categories[vulnerability] = {"total_pass_rate": 0, "total_tests": 0, "weighted_sum": 0}

        # Weight by number of tests
        vuln_categories[vulnerability]["weighted_sum"] += pass_rate * total
        vuln_categories[vulnerability]["total_tests"] += total

    # Calculate weighted average pass rates
    categories = []
    pass_rates = []
    for vuln, data in vuln_categories.items():
        if data["total_tests"] > 0:
            avg_pass_rate = data["weighted_sum"] / data["total_tests"]
            categories.append(vuln)
            pass_rates.append(avg_pass_rate)

    if not categories:
        return None

    # Create radar chart
    fig = go.Figure()

    # Close the polygon by connecting last point to first
    # Duplicate first category and pass_rate at the end
    closed_categories = categories + [categories[0]]
    closed_pass_rates = pass_rates + [pass_rates[0]]

    fig.add_trace(go.Scatterpolar(
        r=closed_pass_rates,
        theta=closed_categories,
        fill='toself',
        name='Pass Rate (%)',
        line_color='blue',
        fillcolor='rgba(0, 0, 255, 0.1)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                tickfont=dict(size=10)
            )
        ),
        showlegend=False,
        title="Vulnerability Pass Rates by Category",
        title_x=0.5,
        height=400,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig


def render_vulnerability_radar_chart(run_data: Dict[str, Any]):
    """Render the vulnerability radar chart."""
    st.subheader("🕸️ Vulnerability Overview")

    fig = create_vulnerability_radar_chart(run_data)

    if fig is None:
        st.info("No vulnerability data available for visualization.")
        return

    st.plotly_chart(fig, width='stretch')

    st.caption("""
    **Interpretation:** Each axis represents a vulnerability category. The distance from center indicates pass rate percentage.
    Higher values (outer areas) show better security performance. Performance thresholds: ≥90% (strong), 70-89% (moderate), <70% (needs attention).
    """)


def render_found_issues(run_data: Dict[str, Any]):
    """Render the found issues section with vulnerability-based grouping."""
    st.subheader("⚠️ Found Issues")

    # Get vulnerability risks and failed test cases
    vulnerability_risks = calculate_vulnerability_risks(run_data)
    vulnerability_groups = get_failed_test_cases_by_vulnerability(run_data)

    # Filter to only vulnerabilities with failed tests
    vulnerabilities_with_failures = {
        vuln: risk_info for vuln, risk_info in vulnerability_risks.items()
        if vuln in vulnerability_groups and risk_info["failed_count"] > 0
    }

    if not vulnerabilities_with_failures:
        st.info("No failed test cases found in this evaluation run.")
        return

    # Show summary table of vulnerabilities with risks
    st.markdown("**Vulnerability Risk Summary:**")

    # Create summary table
    summary_data = []
    for vuln, risk_info in vulnerabilities_with_failures.items():
        summary_data.append({
            "Vulnerability": vuln,
            "Pass Rate": f"{risk_info['pass_rate']:.1%}",
            "Base Risk": risk_info['base_risk'].capitalize(),
            "Impact": risk_info['impact'].capitalize(),
            "Adjusted Risk": risk_info['adjusted_risk'].capitalize(),
            "Failed Tests": risk_info['failed_count']
        })

    summary_df = pd.DataFrame(summary_data)
    styled_df = style_risk_dataframe(summary_df)
    st.dataframe(styled_df, hide_index=True, use_container_width=True)

    # Show detailed breakdown by vulnerability
    st.markdown("**Failed Test Cases by Vulnerability:**")

    # Sort vulnerabilities by adjusted risk (critical first)
    risk_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    sorted_vulnerabilities = sorted(
        vulnerabilities_with_failures.keys(),
        key=lambda v: risk_order.get(vulnerabilities_with_failures[v]["adjusted_risk"], 3)
    )

    for vulnerability in sorted_vulnerabilities:
        risk_info = vulnerabilities_with_failures[vulnerability]
        cases = vulnerability_groups[vulnerability]

        # Create expander title with risk information
        expander_title = f"{vulnerability} ({len(cases)} failed) - Adjusted Risk: {risk_info['adjusted_risk'].capitalize()}"
        expander_title += f" (Base: {risk_info['base_risk'].capitalize()}, Impact: {risk_info['impact'].capitalize()})"

        with st.expander(expander_title, expanded=(risk_info['adjusted_risk'] == "critical")):
            # Show risk calculation details
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Pass Rate", f"{risk_info['pass_rate']:.1%}")
            with col2:
                st.metric("Total Tests", risk_info['total_tests'])
            with col3:
                render_colored_metric("Base Risk", risk_info['base_risk'].capitalize(), is_risk=True)
            with col4:
                render_colored_metric("Impact", risk_info['impact'].capitalize(), is_risk=True)

            st.markdown("---")

            # List all failed test cases
            for i, case in enumerate(cases):
                vulnerability_type = case.get("vulnerability_type", "")
                attack_method = case.get("attack_method", "Unknown")

                # Create test case title
                test_title = f"{vulnerability}"
                if vulnerability_type:
                    test_title += f" ({vulnerability_type})"
                test_title += f" - {attack_method}"

                with st.expander(f"#{i+1}: {test_title}", expanded=False):
                    # Prompt
                    st.markdown("**Prompt:**")
                    st.code(case.get("input", ""), language=None)

                    # Model output
                    st.markdown("**Model Output:**")
                    st.code(case.get("actual_output", ""), language=None)

                    # Judge reason
                    st.markdown("**Judge Reason:**")
                    st.write(case.get("reason", "No reason provided"))

                    # Additional metadata
                    score = case.get("score", 0)
                    st.caption(f"Score: {score:.2f} | Attack: {attack_method}")


def main():
    """Main page entry point."""
    render_header()

    run_data = render_results_browser()

    if run_data is None:
        return

    # Show summary statistics
    render_summary_statistics(run_data)

    # Show vulnerability radar chart
    render_vulnerability_radar_chart(run_data)

    # Show found issues with drill-down
    render_found_issues(run_data)


if __name__ == "__main__":
    main()