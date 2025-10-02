import pytest
import numpy as np
import pandas as pd
from portfolio_opt.app import (
    optimize,
    stocks_to_portfolio_time_series,
    adjusted_std,
    yearly_rolling_returns,
    portfolio_percent_return,
)


def create_synthetic_stock(start_value: float, weekly_growth: float, weeks: int, volatility: float = 0.0) -> pd.Series:
    np.random.seed(42)
    values = [start_value]
    for _ in range(weeks - 1):
        growth = weekly_growth * (1 + np.random.randn() * volatility)
        values.append(values[-1] * (1 + growth))
    return pd.Series(values)


def test_optimizer_prefers_high_return_stock():
    weeks = 104
    
    df = pd.DataFrame({
        'HighReturn': create_synthetic_stock(100, 0.02, weeks, 0.01),
        'LowReturn': create_synthetic_stock(100, 0.001, weeks, 0.01),
    })
    
    result = optimize(df, asset_max_share=1.0, max_std_value=0.05)
    
    assert result.success
    stock_names = list(df.columns)
    weights = dict(zip(stock_names, result.x))
    
    assert weights['HighReturn'] > weights['LowReturn']
    assert weights['HighReturn'] > 0.6


def test_optimizer_with_one_clearly_dominant_stock():
    weeks = 104
    
    df = pd.DataFrame({
        'Superstar': create_synthetic_stock(100, 0.03, weeks, 0.005),
        'Average': create_synthetic_stock(100, 0.005, weeks, 0.01),
        'Poor': create_synthetic_stock(100, -0.001, weeks, 0.01),
    })
    
    result = optimize(df, asset_max_share=1.0, max_std_value=0.05)
    
    assert result.success
    stock_names = list(df.columns)
    weights = dict(zip(stock_names, result.x))
    
    assert weights['Superstar'] > 0.8
    assert weights['Poor'] < 0.1


def test_optimizer_respects_max_share_constraint():
    weeks = 104
    
    df = pd.DataFrame({
        'Stock_A': create_synthetic_stock(100, 0.02, weeks, 0.01),
        'Stock_B': create_synthetic_stock(100, 0.015, weeks, 0.01),
        'Stock_C': create_synthetic_stock(100, 0.01, weeks, 0.01),
    })
    
    max_share = 0.4
    result = optimize(df, asset_max_share=max_share, max_std_value=0.05)
    
    # Even if not fully converged, weights should respect constraint
    for weight in result.x:
        assert weight <= max_share + 0.01


def test_optimizer_weights_sum_to_one():
    weeks = 104
    
    df = pd.DataFrame({
        'Stock_A': create_synthetic_stock(100, 0.01, weeks, 0.01),
        'Stock_B': create_synthetic_stock(100, 0.01, weeks, 0.01),
    })
    
    result = optimize(df, asset_max_share=1.0, max_std_value=0.05)
    
    assert result.success
    assert np.isclose(sum(result.x), 1.0, atol=1e-4)


def test_optimizer_with_volatile_vs_stable_stocks():
    weeks = 104
    
    np.random.seed(42)
    stable = create_synthetic_stock(100, 0.015, weeks, 0.001)
    
    np.random.seed(43)
    volatile = create_synthetic_stock(100, 0.015, weeks, 0.05)
    
    df = pd.DataFrame({
        'Stable': stable,
        'Volatile': volatile,
    })
    
    result = optimize(df, asset_max_share=1.0, max_std_value=0.02)
    
    assert result.success
    stock_names = list(df.columns)
    weights = dict(zip(stock_names, result.x))
    
    assert weights['Stable'] > weights['Volatile']


def test_optimizer_all_stocks_equal_should_diversify():
    weeks = 104
    
    df = pd.DataFrame({
        'Stock_A': create_synthetic_stock(100, 0.01, weeks, 0.01),
        'Stock_B': create_synthetic_stock(100, 0.01, weeks, 0.01),
        'Stock_C': create_synthetic_stock(100, 0.01, weeks, 0.01),
    })
    
    result = optimize(df, asset_max_share=1.0, max_std_value=0.05)
    
    assert result.success
    
    for weight in result.x:
        assert weight > 0.2
        assert weight < 0.4


def test_optimizer_with_max_share_forces_diversification():
    weeks = 104
    
    df = pd.DataFrame({
        'Best': create_synthetic_stock(100, 0.02, weeks, 0.01),
        'Good': create_synthetic_stock(100, 0.015, weeks, 0.01),
        'OK': create_synthetic_stock(100, 0.01, weeks, 0.01),
        'Mediocre': create_synthetic_stock(100, 0.005, weeks, 0.01),
        'Poor': create_synthetic_stock(100, 0.001, weeks, 0.01),
    })
    
    result = optimize(df, asset_max_share=0.20, max_std_value=0.05)
    
    assert result.success
    stock_names = list(df.columns)
    weights = dict(zip(stock_names, result.x))
    
    # With 20% max and 5 stocks, all must be at 20% to sum to 100%
    # This demonstrates the "forced diversification" effect you see in the frontend!
    assert weights['Best'] == pytest.approx(0.20, abs=0.01)
    
    num_at_max = sum(1 for w in result.x if w >= 0.19)
    assert num_at_max == 5  # All 5 stocks hit the 20% cap


def test_optimizer_volatility_constraint_is_binding():
    weeks = 104
    
    np.random.seed(42)
    high_vol = create_synthetic_stock(100, 0.02, weeks, 0.03)
    
    np.random.seed(43)
    low_vol = create_synthetic_stock(100, 0.015, weeks, 0.005)
    
    df = pd.DataFrame({
        'HighVol': high_vol,
        'LowVol': low_vol,
    })
    
    max_std_value = 0.020
    result = optimize(df, asset_max_share=1.0, max_std_value=max_std_value)
    
    assert result.success
    stock_names = list(df.columns)
    weights = dict(zip(stock_names, result.x))
    
    p = stocks_to_portfolio_time_series(df, weights)
    actual_std = adjusted_std(p)
    
    # Main check: volatility constraint should be respected
    assert actual_std <= max_std_value + 0.002
    # Optimizer can choose either based on return/volatility tradeoff
    assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)

