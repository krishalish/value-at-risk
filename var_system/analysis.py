import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, Tuple, List
import matplotlib.dates as mdates
from .plots import use_theme

class RiskAnalyzer:
    """Analyze and visualize VaR results and portfolio risk metrics."""
    
    def __init__(self, returns: pd.DataFrame, portfolio_returns: pd.Series,
                 var_results: Dict[str, Dict[float, float]], confidence_levels: List[float]):
        """
        Initialize RiskAnalyzer with return data and VaR results.
        
        Args:
            returns (pd.DataFrame): Daily returns for each stock
            portfolio_returns (pd.Series): Portfolio returns
            var_results (Dict[str, Dict[float, float]]): VaR calculation results
            confidence_levels (List[float]): Confidence levels used for VaR
        """
        self.returns = returns
        self.portfolio_returns = portfolio_returns
        self.var_results = var_results
        self.confidence_levels = confidence_levels
        use_theme()
    
    def plot_return_distributions(self, save_path: str = None):
        """Plot return distributions with VaR thresholds."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot individual stock distributions
        for i, stock in enumerate(self.returns.columns):
            sns.histplot(self.returns[stock], kde=True, ax=axes[i])
            axes[i].set_title(f'{stock} Return Distribution')
            axes[i].set_xlabel('Daily Returns')
            
            # Add VaR lines
            for conf_level in self.confidence_levels:
                var_value = self.var_results[stock][conf_level]
                axes[i].axvline(-var_value, color='r', linestyle='--',
                              label=f'{conf_level*100:.0f}% VaR')
            axes[i].legend()
        
        # Plot portfolio distribution
        sns.histplot(self.portfolio_returns, kde=True, ax=axes[-1])
        axes[-1].set_title('Portfolio Return Distribution')
        axes[-1].set_xlabel('Daily Returns')
        
        # Add portfolio VaR lines
        for conf_level in self.confidence_levels:
            var_value = self.var_results['Portfolio'][conf_level]
            axes[-1].axvline(-var_value, color='r', linestyle='--',
                           label=f'{conf_level*100:.0f}% VaR')
        axes[-1].legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_correlation_heatmap(self, save_path: str = None):
        """Plot correlation matrix heatmap."""
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.returns.corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   vmin=-1, vmax=1, fmt='.2f')
        plt.title('Stock Returns Correlation Matrix')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def calculate_rolling_var(self, window: int = 63) -> pd.DataFrame:
        """
        Calculate rolling VaR for the portfolio.
        
        Args:
            window (int): Rolling window size (default: 63 days = 1 quarter)
            
        Returns:
            pd.DataFrame: Rolling VaR values
        """
        rolling_vars = pd.DataFrame(index=self.returns.index)
        
        for conf_level in self.confidence_levels:
            # quantile of the previous `window` days (shift so day t is not in its own window)
            rolling_var = -self.portfolio_returns.rolling(window).quantile(1 - conf_level).shift(1)
            rolling_vars[f'{conf_level*100:.0f}% VaR'] = rolling_var
            
        return rolling_vars
    
    def plot_rolling_var(self, window: int = 63, save_path: str = None):
        """Plot rolling VaR over time."""
        rolling_vars = self.calculate_rolling_var(window)
        
        plt.figure(figsize=(12, 6))
        for conf_level in self.confidence_levels:
            plt.plot(rolling_vars.index,
                    rolling_vars[f'{conf_level*100:.0f}% VaR'],
                    label=f'{conf_level*100:.0f}% VaR')
            
        plt.plot(self.portfolio_returns.index,
                abs(self.portfolio_returns),
                'r.',
                alpha=0.3,
                label='Absolute Returns')
        
        plt.title('Rolling VaR vs Actual Returns')
        plt.xlabel('Date')
        plt.ylabel('Value at Risk')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def calculate_expected_shortfall(self) -> Dict[str, Dict[float, float]]:
        """
        Calculate Expected Shortfall (Conditional VaR).
        
        Returns:
            Dict[str, Dict[float, float]]: ES values for each asset and portfolio
        """
        results = {}
        
        # Calculate ES for individual stocks
        for stock in self.returns.columns:
            stock_es = {}
            for conf_level in self.confidence_levels:
                var = self.var_results[stock][conf_level]
                losses = self.returns[stock][self.returns[stock] < -var]
                es = abs(np.mean(losses)) if len(losses) > 0 else var
                stock_es[conf_level] = es
            results[stock] = stock_es
        
        # Calculate ES for portfolio
        portfolio_es = {}
        for conf_level in self.confidence_levels:
            var = self.var_results['Portfolio'][conf_level]
            losses = self.portfolio_returns[self.portfolio_returns < -var]
            es = abs(np.mean(losses)) if len(losses) > 0 else var
            portfolio_es[conf_level] = es
        results['Portfolio'] = portfolio_es
        
        return results
    
    def backtest_var(self) -> pd.DataFrame:
        """
        Backtest VaR model by calculating violation ratios.
        
        Returns:
            pd.DataFrame: Violation ratios for each confidence level
        """
        results = []
        
        for conf_level in self.confidence_levels:
            var = self.var_results['Portfolio'][conf_level]
            violations = sum(self.portfolio_returns < -var)
            violation_ratio = violations / len(self.portfolio_returns)
            expected_ratio = 1 - conf_level
            
            results.append({
                'Confidence Level': f'{conf_level*100:.0f}%',
                'Expected Violations': f'{expected_ratio*100:.2f}%',
                'Actual Violations': f'{violation_ratio*100:.2f}%',
                'Violation Ratio': f'{(violation_ratio/expected_ratio):.2f}'
            })
            
        return pd.DataFrame(results)
    
    def test_normality(self) -> pd.DataFrame:
        """
        Perform Jarque-Bera test for normality.
        
        Returns:
            pd.DataFrame: Test statistics and p-values
        """
        results = []
        
        # Test individual stocks
        for stock in self.returns.columns:
            statistic, p_value = stats.jarque_bera(self.returns[stock])
            results.append({
                'Asset': stock,
                'JB Statistic': f'{statistic:.2f}',
                'P-value': f'{p_value:.4f}',
                'Normal at 5%': 'Yes' if p_value > 0.05 else 'No'
            })
        
        # Test portfolio
        statistic, p_value = stats.jarque_bera(self.portfolio_returns)
        results.append({
            'Asset': 'Portfolio',
            'JB Statistic': f'{statistic:.2f}',
            'P-value': f'{p_value:.4f}',
            'Normal at 5%': 'Yes' if p_value > 0.05 else 'No'
        })
        
        return pd.DataFrame(results) 