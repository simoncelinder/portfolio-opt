import pytest
import numpy as np
import pandas as pd
from portfolio_opt.app import (
    adjusted_std,
    yearly_rolling_returns,
    portfolio_percent_return,
    stocks_to_portfolio_time_series,
)


def test_adjusted_std_with_constant_series():
    p = pd.Series([100.0] * 100)
    result = adjusted_std(p)
    assert np.isnan(result) or result == 0.0


def test_adjusted_std_with_trending_series():
    p = pd.Series(np.linspace(100, 200, 100))
    result = adjusted_std(p)
    assert isinstance(result, float)
    assert not np.isnan(result)
    assert result >= 0


def test_adjusted_std_with_volatile_series():
    np.random.seed(42)
    p = pd.Series(100 + np.random.randn(100) * 10)
    result = adjusted_std(p)
    assert isinstance(result, float)
    assert result > 0


def test_adjusted_std_detrending_property():
    np.random.seed(42)
    trend = np.linspace(100, 200, 100)
    noise = np.random.randn(100) * 5
    p_with_trend = pd.Series(trend + noise)
    p_without_trend = pd.Series(100 + noise)
    
    std_with_trend = adjusted_std(p_with_trend)
    std_without_trend = adjusted_std(p_without_trend)
    
    assert abs(std_with_trend - std_without_trend) < std_without_trend * 0.5


def test_yearly_rolling_returns_basic():
    p = pd.Series(np.linspace(100, 200, 104))
    returns = yearly_rolling_returns(p)
    
    assert len(returns) == 104 - 52
    assert all(returns > 0)


def test_yearly_rolling_returns_log_property():
    initial_price = 100.0
    final_price = 110.0
    p = pd.Series([initial_price] + [0] * 51 + [final_price])
    
    returns = yearly_rolling_returns(p)
    expected_log_return = np.log(final_price / initial_price) * 100
    
    assert len(returns) == 1
    assert np.isclose(returns.iloc[0], expected_log_return, rtol=1e-5)


def test_yearly_rolling_returns_decreasing_prices():
    p = pd.Series(np.linspace(200, 100, 104))
    returns = yearly_rolling_returns(p)
    
    assert len(returns) == 104 - 52
    assert all(returns < 0)


def test_portfolio_percent_return_increase():
    p = pd.Series([100.0, 110.0, 120.0, 130.0])
    result = portfolio_percent_return(p)
    expected_log_return = np.log(130.0 / 100.0) * 100
    
    assert np.isclose(result, expected_log_return, rtol=1e-5)
    assert result > 0


def test_portfolio_percent_return_decrease():
    p = pd.Series([100.0, 90.0, 80.0, 70.0])
    result = portfolio_percent_return(p)
    expected_log_return = np.log(70.0 / 100.0) * 100
    
    assert np.isclose(result, expected_log_return, rtol=1e-5)
    assert result < 0


def test_portfolio_percent_return_no_change():
    p = pd.Series([100.0, 100.0, 100.0, 100.0])
    result = portfolio_percent_return(p)
    
    assert np.isclose(result, 0.0, atol=1e-10)


def test_portfolio_percent_return_doubling():
    p = pd.Series([100.0, 200.0])
    result = portfolio_percent_return(p)
    expected_log_return = np.log(2.0) * 100
    
    assert np.isclose(result, expected_log_return, rtol=1e-5)


def test_stocks_to_portfolio_time_series_equal_weights():
    df = pd.DataFrame({
        'Stock_A': [100.0, 110.0, 120.0],
        'Stock_B': [200.0, 210.0, 220.0],
    })
    share_of_portfolio = {'Stock_A': 0.5, 'Stock_B': 0.5}
    
    result = stocks_to_portfolio_time_series(df, share_of_portfolio)
    expected = pd.Series([150.0, 160.0, 170.0], name='portfolio')
    
    pd.testing.assert_series_equal(result, expected)


def test_stocks_to_portfolio_time_series_weighted():
    df = pd.DataFrame({
        'Stock_A': [100.0, 110.0, 120.0],
        'Stock_B': [200.0, 210.0, 220.0],
    })
    share_of_portfolio = {'Stock_A': 0.3, 'Stock_B': 0.7}
    
    result = stocks_to_portfolio_time_series(df, share_of_portfolio)
    expected = pd.Series([170.0, 180.0, 190.0], name='portfolio')
    
    pd.testing.assert_series_equal(result, expected)


def test_stocks_to_portfolio_time_series_single_stock():
    df = pd.DataFrame({
        'Stock_A': [100.0, 110.0, 120.0],
    })
    share_of_portfolio = {'Stock_A': 1.0}
    
    result = stocks_to_portfolio_time_series(df, share_of_portfolio)
    expected = pd.Series([100.0, 110.0, 120.0], name='portfolio')
    
    pd.testing.assert_series_equal(result, expected)


def test_log_returns_additivity():
    p = pd.Series([100.0, 110.0, 121.0, 133.1])
    
    total_return = np.log(p.iloc[-1] / p.iloc[0]) * 100
    
    step_returns = []
    for i in range(len(p) - 1):
        step_returns.append(np.log(p.iloc[i + 1] / p.iloc[i]) * 100)
    
    sum_of_steps = sum(step_returns)
    
    assert np.isclose(total_return, sum_of_steps, rtol=1e-5)


def test_adjusted_std_with_short_series():
    p = pd.Series([100.0, 105.0, 110.0])
    result = adjusted_std(p, periods=2)
    
    assert isinstance(result, float)
    assert result >= 0 or np.isnan(result)


def test_yearly_rolling_returns_exact_52_weeks():
    p = pd.Series([100.0] * 52 + [110.0])
    returns = yearly_rolling_returns(p)
    
    assert len(returns) == 1
    expected = np.log(110.0 / 100.0) * 100
    assert np.isclose(returns.iloc[0], expected, rtol=1e-5)

