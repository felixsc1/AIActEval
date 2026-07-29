<p align="center">
  <img src="assets/logo.jpg" alt="SafeEval Logo" width="300"/>
</p>

# SafeEval

**SafeEval** is a compliance testing tool designed to evaluate Large Language Models (LLMs) against EU AI Act requirements. It leverages the **DeepTeam** library for automated red teaming and implements advanced **Thurstonian utility bias testing** to uncover implicit biases.

## Watch the tutorials

New to SafeEval? Start here — two short demos show how to use the tool end to end:

| Tutorial | What you'll learn | Watch |
| --- | --- | --- |
| **Part 1 – Basic Evaluations** | App overview and DeepTeam red-teaming evaluations | [▶ Watch Part 1](https://github.com/felixsc1/AIActEval/releases/download/demo-videos-v1/SafeEval_Demo_part1_BasicEvaluations.mp4) |
| **Part 2 – Utility Bias Testing** | Thurstonian utility bias testing walkthrough | [▶ Watch Part 2](https://github.com/felixsc1/AIActEval/releases/download/demo-videos-v1/SafeEval_Demo_part2_UtilityBiasTesting.mp4) |

> Videos are hosted on the [Demo Videos / Tutorials release](https://github.com/felixsc1/AIActEval/releases/tag/demo-videos-v1). Click a link above to open/download the MP4 (GitHub READMEs cannot embed video players).

## Features

- **Automated Red Teaming (DeepTeam)**:

  - Uses the `deepteam` library to perform comprehensive red teaming.
  - Tests for **PII Leakage**, **Bias** (Race, Gender, Religion, Politics), and **Toxicity**.
  - Deploys various attack strategies including Prompt Injection, Roleplay, and more.

- **Utility Bias Testing**:

  - Quantifies implicit bias using non-monetary preference queries (e.g., comparing scientific advancement vs. saving lives across different demographics).
  - **Thurstonian Active Learning**: Efficiently samples query combinations to model utility functions and determine exchange rates with fewer queries.
  - **Grid Testing**: Exhaustive testing of all demographic/N-value combinations.
  - Statistical analysis of refusal rates, preference curves, and switch points.

- **Model Support**:

  - **Local**: Run tests against local models using **Ollama**.
  - **Cloud**: Support for **Groq** API for faster inference.

- **Interactive Dashboard**:
  - Streamlit-based interface for configuring tests and visualizing results.
  - Detailed reporting and historic result analysis.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Copy the example environment file and configure your API keys:

```bash
cp env.example .env
```

Edit `.env` to include your keys (e.g., `OPENAI_API_KEY`, `GROQ_API_KEY`, `CONFIDENT_API_KEY`).

### 3. Setup Ollama (Optional for Local Testing)

Install [Ollama](https://ollama.com) and pull models to test:

```bash
ollama pull llama3
ollama pull mistral
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

## Usage

1.  **Evaluations Page (🔴)**: Configure and run DeepTeam red teaming evaluations. Select vulnerabilities and the model to test.
2.  **Utility Bias Testing Page (⚖️)**: Choose between **Thurstonian Active Learning** (recommended for efficiency) or **Grid Testing** to measure implicit utility bias.
3.  **Results Pages (📋, 📊)**: View detailed reports and visualizations of your test runs.

## Running tests

Basic pytest tests are provided for the Utility Bias Testing and Utility Bias Results pages. They check that the pages render without showing error/exception boxes (catching regressions that don’t raise Python exceptions), and that one Thurstonian-style preference query runs correctly with a **mocked** API response (no real API calls or costs).

1. Install dependencies (including pytest):

   ```bash
   pip install -r requirements.txt
   ```

2. Run the tests from the project root:

   ```bash
   pytest tests/ -v
   ```

   To run only the utility-bias page tests:

   ```bash
   pytest tests/test_utility_bias_pages.py -v
   ```

   The Thurstonian one-query test mocks `utility_bias.call_groq_api`, so it verifies request/response logic without using the Groq API.
