# Supermarket Pydantic Agent

This project is a local AI-powered sales analyst built with Pydantic AI and Ollama. It reads the supermarket sales dataset in `supermarket_sales.csv` and answers natural-language questions about the data without sending the dataset to a hosted service.

## Features

- Loads the local supermarket sales CSV
- Uses a Pydantic AI agent with built-in analysis tools
- Supports questions about totals, averages, filters, and transactions
- Works with a local Ollama model for privacy-friendly local inference

## Project structure

- `main.py` – app entry point and agent setup
- `supermarket_sales.csv` – supermarket transaction dataset
- `pyproject.toml` – project dependencies and Python metadata
- `.env` – optional environment settings for model and API configuration

## Requirements

- Python 3.14+
- Ollama installed and running locally
- A model available in Ollama, such as `gemma4:e4b`

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
uv sync
```

3. Start Ollama and pull the model:

```powershell
ollama serve
ollama pull gemma4:e4b
```

4. Run the app:

```powershell
uv run python main.py
```

## Example questions

- Which city had the highest total sales?
- Compare gross income by product line.
- Show the transactions paid by e-wallet.
- What is the average rating by branch?
- What is the total revenue for Fashion accessories?

## Environment configuration

You can set your model and local Ollama endpoint in a `.env` file:

```env
OLLAMA_MODEL=ollama:gemma4:e4b
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_API_KEY=ollama
```

The app defaults to the local Ollama endpoint if these values are not provided.

## Notes

- The agent is restricted to the dataset in this workspace.
- It uses local tools to inspect and summarize the CSV rather than relying on external APIs for the actual analysis.
