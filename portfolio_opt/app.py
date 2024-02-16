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


def yearly_rolling_returns(s: pd.Series) -> pd.Series:
    returns_df = s.copy().to_frame('value')
    returns_df['yr_shift'] = returns_df['value'].shift(-52)
    returns_df = returns_df.dropna(how='any')
    returns_df['return'] = (returns_df['yr_shift'] / returns_df['value'] - 1) * 100
    return returns_df['return']


def portfolio_percent_return(p: pd.Series) -> float:
    return (p.iloc[-1] / p.iloc[0] - 1) * 100


def objective(x):
    share_of_portfolio = dict(zip(final_stock_names, x))
    p = stocks_to_portfolio_time_series(
        df=df,
        share_of_portfolio=share_of_portfolio,
    )

    returns_series = yearly_rolling_returns(p)
    avg_return = float(returns_series.mean())
    return -avg_return


def weights_sum_constraint(x):
    return sum(x) - 1  # Ensure shares sum to 1


def optimize(
    df: pd.DataFrame,
    asset_max_share: float,
):
    x0 = [1 / len(df.columns) for _ in range(len(df.columns))]
    bounds = [(0, asset_max_share)] * len(x0)
    cons = [
        {'type': 'eq', 'fun': weights_sum_constraint},
        {'type': 'ineq', 'fun': std_constraint},
    ]
    result = spo.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
    return result


def plotly_pie_from_shares_dict(shares_dict: dict):
    shares_df = pd.DataFrame(list(shares_dict.items()), columns=['Asset', 'Share'])
    return px.pie(shares_df, values='Share', names='Asset', title='Asset Share of Total Portfolio', height=400, width=670)


def plotly_boxplot(df: pd.DataFrame, group_col: str, value_col: str):
    fig = px.box(df, x=group_col, y=value_col)
    return fig


def plot_all_stocks(df: pd.DataFrame):
    df = df.copy()
    long_df = df.reset_index().melt(id_vars='Date', var_name='Asset', value_name='Value')
    long_df = long_df.rename(columns={'index': 'Date'})
    return px.line(
        long_df,
        x='Date',
        y='Value',
        color='Asset',
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


st.title('Stock Portfolio Optimizer')

asset_name_max_chars = 25  # Not to skew plot with super long names eating from lineplot space
companies = [
    'GC=F', 'NVDA', 'INTC', 'TSM', 'ASML', 'LRCX', 'MU', 'AMAT', 'KLAC', 'AAPL', 'GOOG',
]
pretty_name_map = {
    'GC=F': 'Gold price', 'NVDA': 'Nvidia', 'INTC': 'Intel', 'TSM': 'TSMC',
    'ASML': 'ASML', 'LRCX': 'Lam Research', 'MU': 'Micron Technology',
    'AMAT': 'Applied Materials', 'KLAC': 'KLA Corporation', 'AAPL': 'Apple', 'GOOG': 'Google',
    '^GSPC': 'S&P 500', 'BTC-USD': 'Bitcoin USD', 'SI=F': 'Silver Mar 24',
}


def df_for_single_asset(
    start_date: datetime.date,
    end_date: datetime.date,
    ticker: str,
    pretty_name_map: dict
) -> pd.DataFrame:
    return (
        pdr.get_data_yahoo(
            ticker,
            start=start_date,
            end=end_date,
            interval='1wk'
        )
        [['Close']]
        .rename(columns={'Close': pretty_name_map.get(ticker, ticker)})
    )


def add_pretty_name_from_yf_to_dict(pretty_name_map: dict, s: str):
    pretty_name_map = pretty_name_map.copy()
    try:
        pretty_name_map[s] = yfin.Ticker(s).info['longName'][0:asset_name_max_chars]
    except Exception as e:
        print(f"Could not add long name for ticker: {s}, error: {e}")
    return pretty_name_map


def join_other_asset(
    df: pd.DataFrame,
    start_date: datetime.date,
    end_date: datetime.date,
    ticker: str,
    pretty_name_map: dict,
) -> pd.DataFrame:
    other_df = df_for_single_asset(
        start_date=start_date,
        end_date=end_date,
        ticker=ticker,
        pretty_name_map=pretty_name_map,
    ).reset_index()
    if other_df['Date'].min().date() > start_date + datetime.timedelta(days=6):
        st.caption(f"Skipping asset: {other_df.columns[-1]} due to starts to late, start: {other_df['Date'].min()}")
    else:
        df = pd.merge_asof(df, other_df, on='Date', direction='nearest')
    return df


def combine_into_compare_returns_df(
    p_optimized: pd.Series,
    benchmark: pd.Series,
    benchmark_name: str,
) -> pd.DataFrame:
    return pd.concat([
        (
            yearly_rolling_returns(p_optimized)
            .to_frame('Yearly Rolling Returns')
            .assign(Group='Portfolio')
        ),
        (
            yearly_rolling_returns(benchmark)
            .to_frame('Yearly Rolling Returns')
            .assign(Group=benchmark_name)
        )],
        axis=0
    )


with st.form("input_assumptions", clear_on_submit=False):
    start_date = st.date_input('Asset Data Start Date', value=datetime.date(2005, 1, 1), key='start_date')
    end_date = st.date_input('Asset Data End Date', value=datetime.date.today(), key='end_date')
    assert end_date > start_date, 'End date must be later than start date'

    selected_stocks = st.multiselect(
        'Select stocks for your portfolio:',
        options=[pretty_name_map.get(n, n) for n in companies],
        # Display pretty names or symbols if pretty name not available
        default=['Nvidia', 'Intel', 'Apple', 'Google', 'Applied Materials', 'TSMC', 'ASML', 'Micron Technology',
                 'Gold price']
    )

    benchmark_ticker = st.text_input(
        'Set your benchmark as a ticker (defaults to Nasdaq 100 Tech Sector):',
        value='^NDXT'
    )

    extra_tickers = st.text_input(
        'Add any extra asset tickers (comma-separated) not listed above matching those available in Yahoo Finance:',
        value='BTC-USD, ^GSPC, CL=F, GC=F, SI=F, 051910.KS, 300750.SZ'
    )
    extra_tickers_list = [ticker.strip() for ticker in extra_tickers.split(',') if ticker]

    asset_max_share = st.slider(
        'Max share of portfolio that one single asset can take up',
        min_value=0.05,
        max_value=1.0,
        value=0.20,
        step=0.02
    )

    max_std = st.slider(
        'Max allowed portfolio volatility ((portfolio time series / slow moving avg).std())',
        min_value=0.001,
        max_value=0.300,
        value=0.060,
        step=0.001
    )

    selected_symbols = list(set(
        [k for k, v in pretty_name_map.items() if v in selected_stocks] + extra_tickers_list
    ))
    run = st.form_submit_button('Run optimizer!')

    if run:

        # Try ty lookup pretty names from Yahoo Finance query for those we dont already have
        for s in selected_symbols + [benchmark_ticker]:
            if s not in pretty_name_map.keys():
                pretty_name_map = add_pretty_name_from_yf_to_dict(pretty_name_map, s)

        benchmark_df = df_for_single_asset(
            start_date=start_date,
            end_date=end_date,
            ticker=benchmark_ticker,
            pretty_name_map=pretty_name_map,
        ).reset_index()
        benchmark_name = benchmark_df.columns[-1]

        # Since it was found that adding custom tickers can screw up the dataset by their weeks
        # not starting with exact same date, we instead merge to closest match
        df = df_for_single_asset(
                start_date=start_date,
                end_date=end_date,
                ticker=selected_symbols[0],
                pretty_name_map=pretty_name_map,
        ).reset_index()

        for ticker in selected_symbols[1:]:
            df = join_other_asset(
                df=df,
                start_date=start_date,
                end_date=end_date,
                ticker=ticker,
                pretty_name_map=pretty_name_map,
            )

        # Format inputs
        df = df.set_index('Date')
        final_stock_names = list(df.columns)

        result = optimize(df, asset_max_share)
        st.caption(f"Optimization outcome (success/fail to find optimum): {result.success}")

        # Format outputs - TODO factor out so cleaner
        optimized_shares_dict = {company: result.x[i] for i, company in enumerate(final_stock_names)}
        p_optimized = stocks_to_portfolio_time_series(df=df, share_of_portfolio=optimized_shares_dict)
        r_optimized = portfolio_percent_return(p_optimized)
        std = adjusted_std(p_optimized)
        performance_optimized = r_optimized / std
        optimized_shares_dict_w_other = merge_small_shares(optimized_shares_dict)

        portf_df = p_optimized.to_frame('Value').reset_index()
        portf_df['Value'] = portf_df['Value'] / portf_df['Value'].iloc[0]  # Scale to start at 1
        portf_df['Group'] = 'Portfolio'
        benchmark_df = benchmark_df.rename(columns={benchmark_name: 'Value'})
        benchmark_df['Value'] = benchmark_df['Value'] / benchmark_df['Value'].iloc[0]
        benchmark_df['Group'] = benchmark_name

        # Pie chart
        st.plotly_chart(plotly_pie_from_shares_dict(optimized_shares_dict_w_other))

        # Return botplot
        st.plotly_chart(
                px.box(
                    combine_into_compare_returns_df(p_optimized, benchmark_df['Value'], benchmark_name),
                    x='Group',
                    y='Yearly Rolling Returns',
                    color='Group',
                    height=400,
                    width=670
                )
            )

        # Lineplots vs benchmark
        st.plotly_chart(
                px.line(
                pd.concat([portf_df, benchmark_df], axis=0),
                x='Date',
                y='Value',
                color='Group',
                title='Portfolio Vs Benchmark Value Over Time',
                height=400,
                width=670
            )
        )

        benchmark_return_percent = (benchmark_df['Value'].iloc[-1] / benchmark_df['Value'].iloc[0] - 1) * 100
        st.caption(f"Portfolio return from start to end: {r_optimized:.1f}% | {benchmark_name}: {benchmark_return_percent:.1f}%")
        st.caption(f"Adjusted Std: {std:.2f} (user set: {max_std}) ")
        st.caption(f"Performance ratio: {performance_optimized:.2f}")
        st.caption(f"Source code: https://github.com/simoncelinder/portfolio-opt")

        # Reality check lineplots of assets
        stocks_df_normalized = df / df.iloc[0]
        st.plotly_chart(plot_all_stocks(stocks_df_normalized))
