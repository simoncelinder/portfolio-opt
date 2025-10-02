import datetime
import warnings
import time

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yfin
from curl_cffi import requests as curl_requests
import scipy.optimize as spo
import cufflinks as cf
import plotly.express as px

warnings.simplefilter(action='ignore', category=FutureWarning)
cf.go_offline()

# Create session with browser spoofing to avoid rate limits
_yf_session = curl_requests.Session(impersonate='chrome')


def adjusted_std(p: pd.Series, periods: int = 52) -> float:
    # Divide with very slow rolling average to scale to stock value and trend,
    # and capture normalized remaining fluctuation using log returns
    if p.empty or len(p) < 2:
        return 0.0
    slow_moving_avg = p.rolling(periods, center=True, min_periods=1).mean()
    scaled_series = (p / slow_moving_avg)
    # Calculate log returns from the normalized series for better statistical properties
    log_returns = np.log(scaled_series / scaled_series.shift(1))
    return log_returns.std()


def std_constraint(x, df, stock_names, max_std_value):
    share_of_portfolio = dict(zip(stock_names, x))
    p = stocks_to_portfolio_time_series(
        df=df,
        share_of_portfolio=share_of_portfolio,
    )
    std = adjusted_std(p)
    std_check = max_std_value - std
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
    if s.empty or len(s) < 53:
        return pd.Series(dtype=float)
    returns_df = s.copy().to_frame('value')
    returns_df['yr_shift'] = returns_df['value'].shift(-52)
    returns_df = returns_df.dropna(how='any')
    # Use log returns for better mathematical properties (additive, more normal distribution)
    returns_df['return'] = np.log(returns_df['yr_shift'] / returns_df['value']) * 100
    return returns_df['return']


def portfolio_percent_return(p: pd.Series) -> float:
    # Use log returns for better mathematical properties
    if p.empty or len(p) < 2:
        return 0.0
    return np.log(p.iloc[-1] / p.iloc[0]) * 100


def objective(x, df, stock_names):
    share_of_portfolio = dict(zip(stock_names, x))
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
    max_std_value: float = 0.06,
):
    stock_names = list(df.columns)
    x0 = [1 / len(stock_names) for _ in range(len(stock_names))]
    bounds = [(0, asset_max_share)] * len(x0)
    cons = [
        {'type': 'eq', 'fun': weights_sum_constraint},
        {'type': 'ineq', 'fun': lambda x: std_constraint(x, df, stock_names, max_std_value)},
    ]
    result = spo.minimize(
        lambda x: objective(x, df, stock_names),
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons
    )
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

st.write(
    'This application optimizes portfolio allocation to maximize returns while respecting '
    'constraints on volatility and maximum asset concentration. The optimizer uses linear '
    'programming to determine the ideal percentage allocation for each asset, assuming '
    'continuous rebalancing to maintain constant portfolio weights. Select your assets, '
    'set your risk tolerance, and discover the optimal portfolio composition.'
)

st.caption(
    '**Mathematical approach:** The optimization employs log returns, a best practice in '
    'quantitative finance due to their superior statistical properties (time-additivity and '
    'approximate normality). Volatility is calculated using a trend-adjusted method: each '
    'asset\'s price is normalized by dividing by a slow-moving average, isolating price '
    'fluctuations around the underlying trend. This ensures that sustained price trends '
    'don\'t artificially inflate volatility metrics, allowing the optimizer to focus on '
    'true risk (swings and deviations) rather than penalizing assets with strong directional momentum.'
)

st.caption(
    '**Note on data fetching:** Yahoo Finance introduced significantly stricter rate limiting '
    'in late 2024, which may occasionally cause delays or require retries when fetching asset data. '
    'The app implements automatic retry logic to handle these limitations but it can still cause problems.'
)

st.divider()

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


@st.cache_data(ttl=3600)
def df_for_single_asset(
    start_date: datetime.date,
    end_date: datetime.date,
    ticker: str,
    pretty_name: str,
    max_retries: int = 5,
) -> pd.DataFrame:
    for attempt in range(max_retries):
        try:
            ticker_obj = yfin.Ticker(ticker, session=_yf_session)
            result = (
                ticker_obj.history(
                    start=start_date,
                    end=end_date,
                    interval='1wk'
                )
                [['Close']]
                .rename(columns={'Close': pretty_name})
            )
            if not result.empty:
                # Remove timezone info to avoid merge conflicts between different exchanges
                result.index = result.index.tz_localize(None)
                return result
            elif attempt < max_retries - 1:
                time.sleep(3)
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 3 + (attempt * 2)
                time.sleep(wait_time)
                continue
    
    return pd.DataFrame(columns=['Close']).rename(columns={'Close': pretty_name})


def add_pretty_name_from_yf_to_dict(pretty_name_map: dict, s: str):
    pretty_name_map = pretty_name_map.copy()
    try:
        ticker_obj = yfin.Ticker(s, session=_yf_session)
        info = ticker_obj.info
        if 'longName' in info and info['longName']:
            pretty_name_map[s] = info['longName'][0:asset_name_max_chars]
        elif 'shortName' in info and info['shortName']:
            pretty_name_map[s] = info['shortName'][0:asset_name_max_chars]
    except Exception:
        pass
    return pretty_name_map


def join_other_asset(
    df: pd.DataFrame,
    start_date: datetime.date,
    end_date: datetime.date,
    ticker: str,
    pretty_name_map: dict,
) -> pd.DataFrame:
    pretty_name = pretty_name_map.get(ticker, ticker)
    other_df = df_for_single_asset(
        start_date=start_date,
        end_date=end_date,
        ticker=ticker,
        pretty_name=pretty_name,
    ).reset_index()
    if other_df.empty or pd.isna(other_df['Date'].min()):
        st.caption(f"Skipping asset: {other_df.columns[-1] if len(other_df.columns) > 1 else ticker} due to no data available")
    elif other_df['Date'].min().date() > start_date + datetime.timedelta(days=6):
        st.caption(f"Skipping asset: {other_df.columns[-1]} due to starts too late, start: {other_df['Date'].min()}")
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
        'Max allowed portfolio volatility (std of weekly log returns on normalized series)',
        min_value=0.001,
        max_value=0.100,
        value=0.030,
        step=0.001,
        help='Log return volatility: ~0.02 (low), ~0.03 (moderate), ~0.04+ (high)'
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
                time.sleep(0.5)

        benchmark_pretty_name = pretty_name_map.get(benchmark_ticker, benchmark_ticker)
        benchmark_df = df_for_single_asset(
            start_date=start_date,
            end_date=end_date,
            ticker=benchmark_ticker,
            pretty_name=benchmark_pretty_name,
        ).reset_index()
        benchmark_name = benchmark_df.columns[-1]
        time.sleep(0.5)

        # Since it was found that adding custom tickers can screw up the dataset by their weeks
        # not starting with exact same date, we instead merge to closest match
        first_pretty_name = pretty_name_map.get(selected_symbols[0], selected_symbols[0])
        df = df_for_single_asset(
                start_date=start_date,
                end_date=end_date,
                ticker=selected_symbols[0],
                pretty_name=first_pretty_name,
        ).reset_index()

        for ticker in selected_symbols[1:]:
            time.sleep(0.5)
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

        result = optimize(df, asset_max_share, max_std)
        st.caption(f"Optimization outcome (success/fail to find optimum): {result.success}")

        # Format outputs - TODO factor out so cleaner
        optimized_shares_dict = {company: result.x[i] for i, company in enumerate(final_stock_names)}
        p_optimized = stocks_to_portfolio_time_series(df=df, share_of_portfolio=optimized_shares_dict)
        r_optimized = portfolio_percent_return(p_optimized)
        std = adjusted_std(p_optimized)
        performance_optimized = r_optimized / std if std != 0 else 0.0
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
