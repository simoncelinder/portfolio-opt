# Portfolio Optimizer
Tool for optimizing what share each asset of a portfolio should have to optimize a performance metric e.g. Sharpe Ratio
or max returns with user specified constraint for volatility.

Live frontend at: https://portfolio-allocation-optimizer.streamlit.app

## 1. Set up environment with uv and install dependencies
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install all dependencies
uv sync
```

## 2. Run the app locally
```bash
uv run streamlit run portfolio_opt/app.py
```

## 3. Running tests
```bash
uv run pytest
```
