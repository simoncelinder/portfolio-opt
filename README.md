# Portfolio Optimizer
Tool for optimizing what share each asset of a portfolio should have to optimize a performance metric e.g. Sharpe Ratio
or max returns with user specified constraint for volatility.

Live frontend at: https://portfolio-allocation-optimizer.streamlit.app

## 1. Set up environment with uv and install dependencies
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Or manually create and activate
uv venv
source .venv/bin/activate  # or source .venv/Scripts/activate on Windows
uv pip install -e .
```

## 2. Run the app locally
```bash
# Using uv (recommended)
uv run streamlit run portfolio_opt/app.py

# Or if you've activated the venv
streamlit run portfolio_opt/app.py
```

## 3. Running tests
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_portfolio_math.py
```
