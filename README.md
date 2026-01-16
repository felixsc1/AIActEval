# EU AI Act Compliance Testing POC

A Streamlit application for testing LLM bias using DeepEval's BiasMetric, with Ollama integration for local model testing and Confident AI dashboard logging.

## Features

- **Dataset Management**: Create, edit, and generate synthetic bias-testing queries
- **Bias Testing**: Evaluate LLMs for gender, racial, political, and geographical bias
- **Utility Bias Testing**: Quantitative bias measurement using non-monetary preference queries that compare scientific advancement vs. saving lives across different ethnicities, with statistical analysis of refusal rates, preference curves, switch points, and relative exchange rates
- **Local LLM Testing**: Use Ollama to test models locally
- **Configurable Judges**: Choose from GPT-5.2 variants (nano/mini/5.2) for evaluation
- **Results Dashboard**: View detailed results on Confident AI platform

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Environment Variables

Copy the example environment file and fill in your API keys:

```bash
cp env.example .env
```

Edit `.env` with your keys:

- `CONFIDENT_API_KEY`: Get from [app.confident-ai.com](https://app.confident-ai.com)
- `OPENAI_API_KEY`: Your OpenAI API key for GPT-5.x judge models

### 3. Setup Ollama (for local LLM testing)

Install Ollama from [ollama.com](https://ollama.com) and pull some models:

```bash
# Install Ollama (follow instructions at ollama.com)

# Pull models for testing (example models that might exhibit bias)
ollama pull llama3.2:3b
ollama pull llama3.2:1b
ollama pull mistral:7b
```

### 4. Run the Application

```bash
streamlit run app.py
```

The app will be available at http://localhost:8501

## Usage

### Dataset Management Tab

1. Add bias-testing queries manually or generate synthetic ones
2. Use the judge model dropdown to select GPT-5.x variant for synthesis
3. Click "Generate More" to create additional test cases

### Test Runner Tab

1. Select the Ollama model to test
2. Choose the judge model for evaluation
3. Configure bias detection parameters
4. Run tests and view results on Confident AI dashboard

## Architecture

- `app.py`: Main Streamlit interface with tabs
- `config.py`: Metrics and judge model configuration
- `dataset_handler.py`: Dataset CRUD and synthetic generation
- `evaluator.py`: Ollama integration and DeepEval evaluation
- `utility_bias.py`: Quantitative utility bias testing module with statistical analysis and visualization
- `data/bias_dataset.json`: Persisted test cases

## Cost Considerations

Judge model costs (per 1M tokens):

- GPT-5-nano: $0.05 input / $0.40 output
- GPT-5-mini: $0.25 input / $2.00 output (recommended default)
- GPT-5.2: $1.75 input / $14.00 output

## Extensibility

Easy to add new metrics (Toxicity, Hallucination, etc.) by updating `config.py` and extending the evaluation logic.
