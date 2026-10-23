import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import yaml
from var_system.data_handler import DataHandler
from var_system.var_calculator import VaRCalculator
from var_system.model_monitor import ModelMonitor

# Test configuration
@pytest.fixture
def config():
    """Load test configuration."""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

@pytest.fixture
def test_returns():
    """Generate test return data."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=252, freq='B')
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    
    returns = pd.DataFrame(
        np.random.normal(0.001, 0.02, size=(252, len(tickers))),
        columns=tickers,
        index=dates
    )
    
    return returns

@pytest.fixture
def test_weights(test_returns):
    """Generate test portfolio weights."""
    return {ticker: 1/len(test_returns.columns) for ticker in test_returns.columns}

@pytest.fixture
def data_handler(tmp_path):
    """Initialize DataHandler with a throwaway database."""
    return DataHandler('config/config.yaml', db_path=tmp_path / 'test.db')

@pytest.fixture
def var_calculator(test_returns, test_weights):
    """Initialize VaRCalculator for testing."""
    return VaRCalculator(
        returns=test_returns,
        weights=test_weights,
        confidence_levels=[0.95, 0.99]
    )

@pytest.fixture
def model_monitor(config):
    """Initialize ModelMonitor for testing."""
    return ModelMonitor(config)

# DataHandler Tests
def test_data_validation(data_handler, test_returns):
    """Test data validation functionality."""
    # Test valid data
    valid, issues = data_handler.validate_data(test_returns)
    assert valid
    assert len(issues) == 0
    
    # Test invalid data (missing values)
    invalid_returns = test_returns.copy()
    invalid_returns.iloc[:30, 0] = np.nan  # ~12% missing, above the 5% limit
    valid, issues = data_handler.validate_data(invalid_returns)
    assert not valid
    assert len(issues) > 0
    
    # Test invalid data (insufficient points)
    short_returns = test_returns.iloc[:10]
    valid, issues = data_handler.validate_data(short_returns)
    assert not valid
    assert len(issues) > 0

def test_data_cleaning(data_handler, test_returns):
    """Test data cleaning functionality."""
    # Add some missing values and outliers
    dirty_returns = test_returns.copy()
    dirty_returns.iloc[0, 0] = np.nan
    dirty_returns.iloc[1, 1] = 0.5  # Outlier
    
    cleaned_returns = data_handler.clean_data(dirty_returns)
    
    assert not cleaned_returns.isnull().any().any()
    assert (cleaned_returns.abs() < 0.1).all().all()

# VaRCalculator Tests
def test_historical_var(var_calculator):
    """Test historical VaR calculation."""
    var_results = var_calculator.historical_var()
    
    assert isinstance(var_results, dict)
    assert 'Portfolio' in var_results
    assert all(0 < var < 1 for var in var_results['Portfolio'].values())

def test_parametric_var(var_calculator):
    """Test parametric VaR calculation."""
    var_results = var_calculator.parametric_var()
    
    assert isinstance(var_results, dict)
    assert 'Portfolio' in var_results
    assert all(0 < var < 1 for var in var_results['Portfolio'].values())

def test_monte_carlo_var(var_calculator):
    """Test Monte Carlo VaR calculation."""
    var_results = var_calculator.monte_carlo_var()
    
    assert isinstance(var_results, dict)
    assert 'Portfolio' in var_results
    assert all(0 < var < 1 for var in var_results['Portfolio'].values())

def test_var_consistency(var_calculator):
    """Test consistency between VaR methods."""
    hist_var = var_calculator.historical_var()
    param_var = var_calculator.parametric_var()
    mc_var = var_calculator.monte_carlo_var()
    
    # VaR estimates should be reasonably close
    for conf_level in [0.95, 0.99]:
        hist = hist_var['Portfolio'][conf_level]
        param = param_var['Portfolio'][conf_level]
        mc = mc_var['Portfolio'][conf_level]
        
        assert abs(hist - param) < 0.01
        assert abs(hist - mc) < 0.01

# ModelMonitor Tests
def test_violation_ratio(model_monitor, test_returns):
    """Test VaR violation ratio calculation."""
    # Create synthetic VaR predictions
    var_predictions = pd.Series(
        np.std(test_returns) * 1.645,  # 95% VaR assuming normal distribution
        index=test_returns.index,
        name=0.95
    )
    
    violation_ratio, valid = model_monitor.calculate_violation_ratio(
        test_returns.iloc[:, 0],
        var_predictions
    )
    
    assert 0 <= violation_ratio <= 1
    assert isinstance(valid, bool)

def test_distribution_drift(model_monitor, test_returns):
    """Test distribution drift detection."""
    # Split data into historical and recent
    split_point = len(test_returns) - model_monitor.drift_window
    historical = test_returns.iloc[:split_point]
    recent = test_returns.iloc[split_point:]
    
    drift_results = model_monitor.detect_distribution_drift(
        historical,
        recent
    )
    
    assert isinstance(drift_results, dict)
    assert all(isinstance(v, bool) for v in drift_results.values())

def test_performance_metrics(model_monitor, test_returns):
    """Test model performance metrics calculation."""
    # Create synthetic VaR predictions
    var_predictions = pd.Series(
        np.std(test_returns) * 1.645,
        index=test_returns.index,
        name=0.95
    )
    
    metrics = model_monitor.calculate_performance_metrics(
        test_returns.iloc[:, 0],
        var_predictions
    )
    
    assert isinstance(metrics, dict)
    assert 'violation_ratio' in metrics
    assert 'avg_violation_magnitude' in metrics
    assert 'max_violation' in metrics

# Integration Tests
def test_end_to_end_calculation(data_handler, config, test_returns, test_weights):
    """Test end-to-end VaR calculation process."""
    # Validate and clean data
    valid, issues = data_handler.validate_data(test_returns)
    assert valid
    
    cleaned_returns = data_handler.clean_data(test_returns)
    
    # Calculate VaR
    var_calc = VaRCalculator(
        returns=cleaned_returns,
        weights=test_weights,
        confidence_levels=[0.95, 0.99]
    )
    
    var_results = var_calc.historical_var()
    
    # Monitor results
    monitor = ModelMonitor(config)
    portfolio_returns = var_calc.portfolio_returns
    metrics = monitor.calculate_performance_metrics(
        portfolio_returns,
        pd.Series(var_results['Portfolio'][0.95], index=portfolio_returns.index, name=0.95)
    )
    
    assert isinstance(metrics, dict)
    assert all(0 < var < 1 for var in var_results['Portfolio'].values())

def test_database_operations(data_handler, test_returns):
    """Test database operations."""
    # Store returns
    data_handler.store_returns(test_returns)
    
    # Load returns
    loaded_returns = data_handler.load_returns(
        start_date=test_returns.index[0],
        end_date=test_returns.index[-1]
    )
    
    assert len(loaded_returns) == len(test_returns)
    assert all(loaded_returns.columns == test_returns.columns)

def test_var_storage(data_handler):
    """Test VaR calculation storage and retrieval."""
    # Store VaR calculation
    data_handler.store_var_calculation(
        calculation_date=datetime.now(),
        portfolio_id="test_portfolio",
        method="historical",
        confidence_level=0.95,
        time_horizon=1,
        var_value=0.02
    )
    
    # Retrieve VaR history
    var_history = data_handler.get_var_history(
        portfolio_id="test_portfolio",
        start_date=datetime.now() - timedelta(days=1),
        end_date=datetime.now()
    )
    
    assert len(var_history) > 0
    assert "test_portfolio" in var_history['portfolio_id'].values 