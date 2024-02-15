import datetime

import streamlit as st
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yfin
import scipy.optimize as spo
import cufflinks as cf
import plotly.express as px

yfin.pdr_override()
cf.go_offline()


def adjusted_std(p: pd.Series, periods: int = 52) -> float:
    # Divide with very slow rolling average to scale to stock value and trend,
    # and capture normalized remaining fluctuation
    slow_moving_avg = p.rolling(periods, center=True, min_periods=1).mean()
    scaled_series = (p / slow_moving_avg)
    return scaled_series.std()


def std_constraint(x):
    share_of_portfolio = dict(zip(companies_pretty, x))
    p = stocks_to_portfolio_time_series(
        df=df,
        share_of_portfolio=share_of_portfolio,
    )
    std = adjusted_std(p)
    std_check = max_std - std
    return std_check


def stocks_to_portfolio_time_series(
    df: pd.DataFrame,  # Just ordinary stock prices over time, not returns etc.
    share_of_portfolio: dict,
) -> pd.Series:
    assert (df.isna().sum() == 0).all(), 'Must first impute nulls in valid way'
    return (
        df
        .mul(share_of_portfolio)
        .sum(axis=1)
        .rename('portfolio')
    )


def portfolio_percent_return(p: pd.Series) -> float:
    return (p.iloc[-1] / p.iloc[0] - 1) * 100


def objective(x):
    share_of_portfolio = dict(zip(companies_pretty, x))
    p = stocks_to_portfolio_time_series(
        df=df,
        share_of_portfolio=share_of_portfolio,
    )
    r = portfolio_percent_return(p)
    std = adjusted_std(p)
    performance = r / std
    return -performance


def weights_sum_constraint(x):
    return sum(x) - 1  # Ensure shares sum to 1


def plotly_pie_from_shares_dict(shares_dict: dict):
    shares_df = pd.DataFrame(list(shares_dict.items()), columns=['Asset', 'Share'])
    return px.pie(shares_df, values='Share', names='Asset', title='Asset Share of Total Portfolio', height=400, width=670)


def plotly_portfolio_lineplot(s: pd.Series):
    portfolio_df = s.reset_index().rename(columns={'index': 'Date', 'portfolio': 'Portfolio Value'})
    return px.line(portfolio_df, x='Date', y='Portfolio Value', title='Portfolio Value Over Time', height=400, width=670)


# Streamlit app start
st.title('Stock Portfolio Optimizer')

# Defined outside the functions for global access
companies = [
    'GC=F', 'NVDA', 'INTC', 'TSM', 'ASML', 'LRCX', 'MU', 'AMAT', 'KLAC', 'AAPL', 'GOOG',
]
pretty_name_map = {
    'GC=F': 'Gold price', 'NVDA': 'Nvidia', 'INTC': 'Intel', 'TSM': 'TSMC',
    'ASML': 'ASML', 'LRCX': 'Lam Research', 'MU': 'Micron Technology',
    'AMAT': 'Applied Materials', 'KLAC': 'KLA Corporation', 'AAPL': 'Apple', 'GOOG': 'Google',
}
companies_pretty = [pretty_name_map[n] for n in companies]


with st.form("input_assumptions", clear_on_submit=False):

    start_date = st.date_input('Asset Data Start Date', value=datetime.date(2012, 5, 1), key='start_date')
    end_date = st.date_input('Asset Data End Date', value=datetime.date.today(), key='end_date')
    assert end_date > start_date, 'End date must be later than start date'
    selected_stocks = st.multiselect(
        'Select stocks for your portfolio:',
        companies_pretty,
        default=['Nvidia', 'Intel', 'Apple', 'Google', 'Applied Materials',
                 'TSMC', 'ASML', 'Micron Technology', 'Gold price']
    )

    max_std = st.slider(
        'Max allowed portfolio volatility ((portfolio time series / slow moving avg).std())',
        min_value=0.0,
        max_value=0.4,
        value=0.02,
        step=0.001
    )

    selected_symbols = [k for k, v in pretty_name_map.items() if v in selected_stocks]

    # Fetch data
    df = pd.concat(
        [
            pdr.get_data_yahoo(symbol, start=start_date, end=end_date, interval='1wk')[['Close']]
            .rename(columns={'Close': pretty_name_map[symbol]})
            for symbol in selected_symbols
        ],
        axis=1,
    )

    # Optimization process (same as before, using the selected stocks only)
    x0 = [1/len(df.columns) for _ in range(len(df.columns))]
    bounds = [(0, 1)] * len(x0)

    cons = [
        {'type': 'eq', 'fun': weights_sum_constraint},
        {'type': 'ineq', 'fun': std_constraint},
    ]

    run = st.form_submit_button('Run optimizer!')

    if run:
        result = spo.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)

        st.caption(f"Optimization outcome (success/fail to find optimum): {result.success}")

        optimized_shares_dict = {company: result.x[i] for i, company in enumerate(selected_stocks)}
        p_optimized = stocks_to_portfolio_time_series(df=df, share_of_portfolio=optimized_shares_dict)
        r_optimized = portfolio_percent_return(p_optimized)
        std = adjusted_std(p_optimized)

        performance_optimized = r_optimized / std

        pie_fig = plotly_pie_from_shares_dict(optimized_shares_dict)
        st.plotly_chart(pie_fig)

        portfolio_lineplot_fig = plotly_portfolio_lineplot(p_optimized)
        st.plotly_chart(portfolio_lineplot_fig)

        st.caption(f"Return: {r_optimized:.2f}%")
        st.caption(f"Adjusted Std: {std:.2f} (user set: {max_std}) ")
        st.caption(f"Performance ratio: {performance_optimized:.2f}")
        st.caption(f"Source code: https://github.com/simoncelinder/portfolio-opt")
