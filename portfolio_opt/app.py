import datetime
import warnings

import streamlit as st
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yfin
import scipy.optimize as spo
import cufflinks as cf
import plotly.express as px

warnings.simplefilter(action='ignore', category=FutureWarning)
yfin.pdr_override()
cf.go_offline()


def adjusted_std(p: pd.Series, periods: int = 52) -> float:
    # Divide with very slow rolling average to scale to stock value and trend,
    # and capture normalized remaining fluctuation
    slow_moving_avg = p.rolling(periods, center=True, min_periods=1).mean()
    scaled_series = (p / slow_moving_avg)
    return scaled_series.std()


def std_constraint(x):
    share_of_portfolio = dict(zip(final_stock_names, x))
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
    df = df.copy()
    return (
        df
        .mul(share_of_portfolio)
        .sum(axis=1)
        .rename('portfolio')
    )


def portfolio_percent_return(p: pd.Series) -> float:
    return (p.iloc[-1] / p.iloc[0] - 1) * 100


def objective(x):
    share_of_portfolio = dict(zip(final_stock_names, x))
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
    s = s.copy()
    s_formatted = s.to_frame('Portfolio Value').sort_index().reset_index()
    return px.line(s_formatted, x='Date', y='Portfolio Value', title='Portfolio Value Over Time', height=400, width=670)


def plot_all_stocks(df: pd.DataFrame):
    df = df.copy()
    long_df = df.reset_index().melt(id_vars='Date', var_name='Variable', value_name='Value')
    long_df = long_df.rename(columns={'index': 'Date'})
    return px.line(
        long_df,
        x='Date',
        y='Value',
        color='Variable',
        title='Reality Check Asset Input Data',
        height=700,
        width=670
    )

def plotly_stocks_lineplot(df: pd.DataFrame):
    df = df.copy()
    portfolio_df = df.reset_index().rename(columns={'index': 'Date', 'portfolio': 'Portfolio Value'})
    return px.line(portfolio_df, x='Date', y='Portfolio Value', title='Portfolio Value Over Time', height=400, width=670)


def merge_small_shares(optimized_shares_dict: dict) -> dict:
    optimized_shares_dict = optimized_shares_dict.copy()

    other_share = 0.0
    keys_to_delete = []
    for stock_name, share in optimized_shares_dict.items():
        if share < 0.01:  # Check if share is below 1%
            other_share += share  # Add share to 'Other'
            keys_to_delete.append(stock_name)  # Mark for deletion

    for key in keys_to_delete:
        del optimized_shares_dict[key]

    if other_share > 0:
        optimized_shares_dict['Other'] = other_share

    return optimized_shares_dict


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


def df_for_single_asset(
    start_date: datetime.date,
    end_date: datetime.date,
    ticker: str,
    pretty_name_map: dict
) -> pd.DataFrame:
    return pdr.get_data_yahoo(
        ticker, start=start_date, end=end_date, interval='1wk'
    )[['Close']].rename(columns={'Close': pretty_name_map.get(ticker, ticker)})


with st.form("input_assumptions", clear_on_submit=False):
    start_date = st.date_input('Asset Data Start Date', value=datetime.date(2015, 1, 1), key='start_date')
    end_date = st.date_input('Asset Data End Date', value=datetime.date.today(), key='end_date')
    assert end_date > start_date, 'End date must be later than start date'

    selected_stocks = st.multiselect(
        'Select stocks for your portfolio:',
        options=[pretty_name_map.get(n, n) for n in companies],
        # Display pretty names or symbols if pretty name not available
        default=['Nvidia', 'Intel', 'Apple', 'Google', 'Applied Materials', 'TSMC', 'ASML', 'Micron Technology',
                 'Gold price']
    )

    extra_tickers = st.text_input(
        'Add any extra asset tickers (comma-separated) not listed above matching those available in Yahoo Finance:',
        value='ARKB, ^GSPC, CL=F, GC=F, SI=F, 051910.KS, 300750.SZ'
    )
    extra_tickers_list = [ticker.strip() for ticker in extra_tickers.split(',') if ticker]

    max_std = st.slider(
        'Max allowed portfolio volatility ((portfolio time series / slow moving avg).std())',
        min_value=0.001,
        max_value=0.300,
        value=0.020,
        step=0.001
    )

    selected_symbols = [k for k, v in pretty_name_map.items() if v in selected_stocks] + extra_tickers_list
    selected_symbols = list(set(selected_symbols))
    run = st.form_submit_button('Run optimizer!')

    if run:

        # Try ty lookup pretty names from Yahoo Finance query for those we dont already have
        for s in selected_symbols:
            if s not in pretty_name_map.keys():
                try:
                    max_chars = 25  # Not to skew plot with super long names eating from lineplot space
                    pretty_name_map[s] = yfin.Ticker(s).info['longName'][0:max_chars]
                except Exception as e:
                    print(f"Could not add long name for ticker: {s}, error: {e}")

        # Since it was found that adding custom tickers can screw up the dataset by their weeks
        # not starting with exact same date, we instead merge to closest match
        df = df_for_single_asset(
                start_date=start_date,
                end_date=end_date,
                ticker=selected_symbols[0],
                pretty_name_map=pretty_name_map,
        ).reset_index()

        for ticker in selected_symbols[1:]:
            other_df = df_for_single_asset(
                start_date=start_date,
                end_date=end_date,
                ticker=ticker,
                pretty_name_map=pretty_name_map,
            ).reset_index()
            df = pd.merge_asof(df, other_df, on='Date', direction='nearest')

        df = df.set_index('Date')



        df = (
            df
            .interpolate(axis=0)
            .fillna(0)
        )

        final_stock_names = list(df.columns)

        # Optimization process (same as before, using the selected stocks only)
        x0 = [1 / len(df.columns) for _ in range(len(df.columns))]
        bounds = [(0, 1)] * len(x0)

        cons = [
            {'type': 'eq', 'fun': weights_sum_constraint},
            {'type': 'ineq', 'fun': std_constraint},
        ]

        result = spo.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)

        st.caption(f"Optimization outcome (success/fail to find optimum): {result.success}")

        optimized_shares_dict = {company: result.x[i] for i, company in enumerate(final_stock_names)}
        p_optimized = stocks_to_portfolio_time_series(df=df, share_of_portfolio=optimized_shares_dict)
        r_optimized = portfolio_percent_return(p_optimized)
        std = adjusted_std(p_optimized)

        performance_optimized = r_optimized / std

        optimized_shares_dict_w_other = merge_small_shares(optimized_shares_dict)
        pie_fig = plotly_pie_from_shares_dict(optimized_shares_dict_w_other)
        st.plotly_chart(pie_fig)

        portfolio_lineplot_fig = plotly_portfolio_lineplot(p_optimized)
        st.plotly_chart(portfolio_lineplot_fig)

        st.caption(f"Return: {r_optimized:.2f}%")
        st.caption(f"Adjusted Std: {std:.2f} (user set: {max_std}) ")
        st.caption(f"Performance ratio: {performance_optimized:.2f}")
        st.caption(f"Source code: https://github.com/simoncelinder/portfolio-opt")

        stocks_df_normalized = df / df.iloc[0]
        stocks_lineplot_fig = plot_all_stocks(stocks_df_normalized)
        st.plotly_chart(stocks_lineplot_fig)
