import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy import stats

class VaRCalculator:
    """Calculate Value at Risk using different methodologies."""
    
    def __init__(self, returns: pd.DataFrame, weights: Dict[str, float],
                 confidence_levels: list = [0.95, 0.99]):
        """
        Initialize VaRCalculator with return data and parameters.
        
        Args:
            returns (pd.DataFrame): Daily returns for each stock
            weights (Dict[str, float]): Portfolio weights for each stock
            confidence_levels (list): List of confidence levels for VaR calculation
        """
        self.returns = returns
        self.weights = weights
        self.confidence_levels = confidence_levels
        self.portfolio_returns = self._calculate_portfolio_returns()
        
    def _calculate_portfolio_returns(self) -> pd.Series:
        """Calculate portfolio returns using weights."""
        return self.returns.dot(pd.Series(self.weights))
    
    def historical_var(self) -> Dict[str, Dict[float, float]]:
        """
        Calculate VaR using historical simulation method.
        
        Returns:
            Dict[str, Dict[float, float]]: VaR values for each asset and portfolio
        """
        try:
            results = {}
            
            # Calculate VaR for individual stocks
            for stock in self.returns.columns:
                stock_vars = {}
                for conf_level in self.confidence_levels:
                    var = np.percentile(self.returns[stock], (1 - conf_level) * 100)
                    stock_vars[conf_level] = abs(var)
                results[stock] = stock_vars
            
            # Calculate VaR for portfolio
            portfolio_vars = {}
            for conf_level in self.confidence_levels:
                var = np.percentile(self.portfolio_returns, (1 - conf_level) * 100)
                portfolio_vars[conf_level] = abs(var)
            results['Portfolio'] = portfolio_vars
            
            return results
        
        except Exception as e:
            raise Exception(f"Error calculating historical VaR: {str(e)}")
    
    def parametric_var(self) -> Dict[str, Dict[float, float]]:
        """
        Calculate VaR using parametric (normal distribution) method.
        
        Returns:
            Dict[str, Dict[float, float]]: VaR values for each asset and portfolio
        """
        try:
            results = {}
            
            # Calculate VaR for individual stocks
            for stock in self.returns.columns:
                stock_vars = {}
                mean = np.mean(self.returns[stock])
                std = np.std(self.returns[stock])
                for conf_level in self.confidence_levels:
                    z_score = stats.norm.ppf(conf_level)
                    var = z_score * std - mean
                    stock_vars[conf_level] = var
                results[stock] = stock_vars
            
            # Calculate VaR for portfolio
            portfolio_vars = {}
            mean = np.mean(self.portfolio_returns)
            std = np.std(self.portfolio_returns)
            for conf_level in self.confidence_levels:
                z_score = stats.norm.ppf(conf_level)
                var = z_score * std - mean
                portfolio_vars[conf_level] = var
            results['Portfolio'] = portfolio_vars
            
            return results
        
        except Exception as e:
            raise Exception(f"Error calculating parametric VaR: {str(e)}")
    
    def monte_carlo_var(self, simulations: int = 10000) -> Dict[str, Dict[float, float]]:
        """
        Calculate VaR using Monte Carlo simulation.
        
        Args:
            simulations (int): Number of Monte Carlo simulations
            
        Returns:
            Dict[str, Dict[float, float]]: VaR values for each asset and portfolio
        """
        try:
            results = {}
            np.random.seed(42)
            
            # Calculate VaR for individual stocks
            for stock in self.returns.columns:
                stock_vars = {}
                mean = np.mean(self.returns[stock])
                std = np.std(self.returns[stock])
                
                # Generate simulated returns
                sim_returns = np.random.normal(mean, std, simulations)
                
                for conf_level in self.confidence_levels:
                    var = -np.percentile(sim_returns, (1 - conf_level) * 100)
                    stock_vars[conf_level] = var
                results[stock] = stock_vars
            
            # Calculate VaR for portfolio
            portfolio_vars = {}
            mean = np.mean(self.portfolio_returns)
            std = np.std(self.portfolio_returns)
            
            # Generate simulated portfolio returns
            sim_returns = np.random.normal(mean, std, simulations)
            
            for conf_level in self.confidence_levels:
                var = -np.percentile(sim_returns, (1 - conf_level) * 100)
                portfolio_vars[conf_level] = var
            results['Portfolio'] = portfolio_vars
            
            return results
        
        except Exception as e:
            raise Exception(f"Error calculating Monte Carlo VaR: {str(e)}") 