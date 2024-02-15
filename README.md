# Portfolio Optimizer
Tool for optimizing what share each asset of a portfolio should have to optimize a performance metric e.g. Sharpe Ratio
or max returns with user specified constraint for volatility.

Live frontend at: https://portfolio-allocation-optimizer.streamlit.app

## 1. Set up a virtualenv, activate it and install dependencies
```bash
python3 -m venv .pyenv 
source .pyenv/bin/activate  # or source .pyenv/Scripts/activate on windows
pip install -r requirements.txt  # Since streamlit looks for requirements.txt rather than setup.py etc.
```


## 2. If want to open the app / frontend locally
```bash
streamlit run portfolio_opt/app.py
```
