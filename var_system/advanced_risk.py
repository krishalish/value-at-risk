import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from typing import Dict, List, Tuple
from sklearn.covariance import EmpiricalCovariance
import cvxopt
from cvxopt import matrix, solvers
import warnings
warnings.filterwarnings('ignore')

class AdvancedRiskModel:
    """Advanced risk modeling features for institutional-grade VaR analysis."""
    
    def __init__(self, returns: pd.DataFrame, weights: Dict[str, float],
                 confidence_levels: List[float], target_return: float = 0.001):
        """
        Initialize AdvancedRiskModel with return data and parameters.
        
        Args:
            returns (pd.DataFrame): Daily returns for each stock
            weights (Dict[str, float]): Portfolio weights
            confidence_levels (List[float]): Confidence levels for VaR
            target_return (float): Target portfolio return for optimization
        """
        self.returns = returns
        self.weights = weights
        self.confidence_levels = confidence_levels
        self.target_return = target_return
        self.tickers = list(returns.columns)
        
    def fit_garch(self, ticker: str) -> Tuple[float, float]:
        """
        Fit GARCH(1,1) model to return series and forecast volatility.
        
        Args:
            ticker (str): Stock ticker
            
        Returns:
            Tuple[float, float]: Forecasted mean and volatility
        """
        if ticker == 'Portfolio':
            returns = self.returns.dot(pd.Series(self.weights))
        else:
            returns = self.returns[ticker]
        # arch's optimiser is better behaved on percentage returns
        model = arch_model(returns.values * 100, mean='Constant', vol='Garch', p=1, q=1)
        results = model.fit(disp='off')
        forecast = results.forecast(horizon=1)
        
        return (
            forecast.mean.iloc[-1, 0] / 100,
            np.sqrt(forecast.variance.iloc[-1, 0]) / 100
        )
    
    def calculate_garch_var(self) -> Dict[str, Dict[float, float]]:
        """
        Calculate VaR using GARCH(1,1) volatility forecasts.
        
        Returns:
            Dict[str, Dict[float, float]]: GARCH-based VaR estimates
        """
        results = {}
        
        # Calculate individual stock VaR
        for ticker in self.tickers:
            mean, vol = self.fit_garch(ticker)
            stock_vars = {}
            
            for conf_level in self.confidence_levels:
                z_score = stats.norm.ppf(conf_level)
                stock_vars[conf_level] = z_score * vol - mean
            
            results[ticker] = stock_vars
        
        # Calculate portfolio VaR (treat the portfolio return series as a single asset)
        mean, vol = self.fit_garch('Portfolio')
        
        portfolio_vars = {}
        for conf_level in self.confidence_levels:
            z_score = stats.norm.ppf(conf_level)
            portfolio_vars[conf_level] = z_score * vol - mean
        
        results['Portfolio'] = portfolio_vars
        return results
    
    def scale_var(self, var_values: Dict[str, Dict[float, float]], 
                  horizons: List[int]) -> Dict[int, Dict[str, Dict[float, float]]]:
        """
        Scale VaR to different time horizons using square root of time rule.
        
        Args:
            var_values (Dict[str, Dict[float, float]]): Base VaR values
            horizons (List[int]): Time horizons in days
            
        Returns:
            Dict[int, Dict[str, Dict[float, float]]]: Scaled VaR values
        """
        scaled_vars = {}
        
        for horizon in horizons:
            scale_factor = np.sqrt(horizon)
            horizon_vars = {}
            
            for asset, vars in var_values.items():
                horizon_vars[asset] = {
                    conf: var * scale_factor
                    for conf, var in vars.items()
                }
            
            scaled_vars[horizon] = horizon_vars
        
        return scaled_vars
    
    def calculate_stressed_var(self, stress_period: pd.DatetimeIndex = None,
                             stress_factor: float = 1.5) -> Dict[str, Dict[float, float]]:
        """
        Calculate stressed VaR using historical stress period or scaling factor.
        
        Args:
            stress_period (pd.DatetimeIndex): Historical stress period
            stress_factor (float): Scaling factor for volatility
            
        Returns:
            Dict[str, Dict[float, float]]: Stressed VaR estimates
        """
        if stress_period is not None:
            stress_returns = self.returns.loc[stress_period]
        else:
            # Use volatility scaling if no stress period provided
            stress_returns = self.returns * stress_factor
        
        results = {}
        
        # Calculate individual stock stressed VaR
        for ticker in self.tickers:
            stock_vars = {}
            for conf_level in self.confidence_levels:
                var = np.percentile(stress_returns[ticker],
                                  (1 - conf_level) * 100)
                stock_vars[conf_level] = abs(var)
            results[ticker] = stock_vars
        
        # Calculate portfolio stressed VaR
        portfolio_returns = pd.Series(
            np.dot(stress_returns, pd.Series(self.weights)),
            index=stress_returns.index
        )
        
        portfolio_vars = {}
        for conf_level in self.confidence_levels:
            var = np.percentile(portfolio_returns, (1 - conf_level) * 100)
            portfolio_vars[conf_level] = abs(var)
        
        results['Portfolio'] = portfolio_vars
        return results
    
    def calculate_component_var(self, conf_level: float = 0.95) -> pd.Series:
        """
        Calculate component VaR showing contribution of each asset to portfolio VaR.
        
        Args:
            conf_level (float): Confidence level for VaR
            
        Returns:
            pd.Series: Component VaR for each asset
        """
        # Calculate portfolio VaR
        portfolio_returns = pd.Series(
            np.dot(self.returns, pd.Series(self.weights)),
            index=self.returns.index
        )
        portfolio_var = abs(np.percentile(portfolio_returns,
                                        (1 - conf_level) * 100))
        
        # Calculate covariance matrix
        cov_matrix = self.returns.cov()
        
        # Calculate marginal VaR
        portfolio_vol = np.sqrt(
            np.dot(pd.Series(self.weights),
                  np.dot(cov_matrix, pd.Series(self.weights)))
        )
        
        marginal_var = {}
        for ticker in self.tickers:
            marginal_var[ticker] = (
                np.dot(cov_matrix[ticker], pd.Series(self.weights)) /
                portfolio_vol
            )
        
        # Calculate component VaR
        component_var = pd.Series(
            {ticker: self.weights[ticker] * marginal_var[ticker]
             for ticker in self.tickers}
        )
        
        # Normalize to sum to portfolio VaR
        return component_var * (portfolio_var / component_var.sum())
    
    def optimize_portfolio(self) -> Dict[str, float]:
        """
        Optimize portfolio weights to minimize VaR while maintaining target return.
        
        Returns:
            Dict[str, float]: Optimized portfolio weights
        """
        n_assets = len(self.tickers)
        returns_mean = self.returns.mean()
        returns_cov = self.returns.cov()
        
        # Convert numpy arrays to cvxopt matrices
        P = matrix(returns_cov.values)
        q = matrix(np.zeros(n_assets))
        
        # Constraints
        # 1. Sum of weights = 1
        A = matrix(np.vstack([
            np.ones(n_assets),  # Sum of weights = 1
            returns_mean.values  # Target return constraint
        ]))
        b = matrix([1.0, self.target_return])
        
        # 2. Weight bounds (0 ≤ w ≤ 1)
        G = matrix(np.vstack([
            -np.eye(n_assets),  # w ≥ 0
            np.eye(n_assets)    # w ≤ 1
        ]))
        h = matrix(np.hstack([
            np.zeros(n_assets),
            np.ones(n_assets)
        ]))
        
        # Solve optimization problem
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        
        if sol['status'] != 'optimal':
            raise Exception("Optimization failed to converge")
        
        # Extract optimal weights
        opt_weights = np.array(sol['x']).flatten()
        return dict(zip(self.tickers, opt_weights))
    
    def run_scenario_analysis(self, scenarios: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Run scenario analysis with custom shock scenarios.
        
        Args:
            scenarios (Dict[str, Dict[str, float]]): Shock scenarios
            
        Returns:
            Dict[str, float]: Portfolio returns under each scenario
        """
        results = {}
        weights = pd.Series(self.weights)
        
        for scenario_name, shocks in scenarios.items():
            # Portfolio return if every asset moves by its shock at once
            shock = pd.Series(shocks).reindex(weights.index).fillna(0.0)
            results[scenario_name] = float(weights @ shock)
            
        return results
    
    def calculate_var_confidence_intervals(self, conf_level: float = 0.95,
                                        n_bootstrap: int = 1000) -> Dict[str, Dict[str, float]]:
        """
        Calculate confidence intervals around VaR estimates using bootstrap.
        
        Args:
            conf_level (float): VaR confidence level
            n_bootstrap (int): Number of bootstrap samples
            
        Returns:
            Dict[str, Dict[str, float]]: VaR confidence intervals
        """
        results = {}
        
        for ticker in self.tickers + ['Portfolio']:
            if ticker == 'Portfolio':
                returns = pd.Series(
                    np.dot(self.returns, pd.Series(self.weights)),
                    index=self.returns.index
                )
            else:
                returns = self.returns[ticker]
            
            # Bootstrap VaR estimates
            bootstrap_vars = []
            for _ in range(n_bootstrap):
                sample = returns.sample(n=len(returns), replace=True)
                var = abs(np.percentile(sample, (1 - conf_level) * 100))
                bootstrap_vars.append(var)
            
            # Calculate confidence intervals
            results[ticker] = {
                'VaR': abs(np.percentile(returns, (1 - conf_level) * 100)),
                'CI_Lower': np.percentile(bootstrap_vars, 2.5),
                'CI_Upper': np.percentile(bootstrap_vars, 97.5)
            }
            
        return results
    
    def export_to_excel(self, filename: str, var_results: Dict[str, Dict[float, float]],
                       garch_var: Dict[str, Dict[float, float]],
                       stressed_var: Dict[str, Dict[float, float]],
                       component_var: pd.Series,
                       scenario_results: Dict[str, float],
                       confidence_intervals: Dict[str, Dict[str, float]]):
        """
        Export all results to Excel file with multiple sheets.
        
        Args:
            filename (str): Output Excel file name
            var_results: Standard VaR results
            garch_var: GARCH-based VaR results
            stressed_var: Stressed VaR results
            component_var: Component VaR results
            scenario_results: Scenario analysis results
            confidence_intervals: VaR confidence intervals
        """
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Standard VaR results
            pd.DataFrame(var_results).round(4).to_excel(
                writer, sheet_name='Standard VaR'
            )
            
            # GARCH VaR results
            pd.DataFrame(garch_var).round(4).to_excel(
                writer, sheet_name='GARCH VaR'
            )
            
            # Stressed VaR results
            pd.DataFrame(stressed_var).round(4).to_excel(
                writer, sheet_name='Stressed VaR'
            )
            
            # Component VaR results
            component_var.round(4).to_frame('Component VaR').to_excel(
                writer, sheet_name='Component VaR'
            )
            
            # Scenario analysis results
            pd.Series(scenario_results).round(4).to_frame('Return').to_excel(
                writer, sheet_name='Scenario Analysis'
            )
            
            # Confidence intervals
            pd.DataFrame(confidence_intervals).round(4).to_excel(
                writer, sheet_name='Confidence Intervals'
            ) 