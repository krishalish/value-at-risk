import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Dict
from datetime import datetime, timedelta

class DataGenerator:
    """Generate or fetch stock return data for VaR calculations."""
    
    def __init__(self, tickers: List[str], days: int = 252,
                 mean: float = 0.001, std: float = 0.02,
                 use_real_data: bool = True):
        """
        Initialize DataGenerator with stock parameters.
        
        Args:
            tickers (List[str]): List of stock tickers
            days (int): Number of trading days to generate/fetch
            mean (float): Mean daily return (for synthetic data)
            std (float): Standard deviation of daily returns (for synthetic data)
            use_real_data (bool): Whether to use real market data
        """
        self.tickers = tickers
        self.days = days
        self.mean = mean
        self.std = std
        self.use_real_data = use_real_data
    
    def fetch_real_data(self) -> pd.DataFrame:
        """Fetch real market data using yfinance."""
        try:
            # Calculate start date based on desired number of days
            # Add extra days to account for weekends and holidays
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.days * 1.5)
            
            # Fetch data for all tickers
            data = pd.DataFrame()
            for ticker in self.tickers:
                stock = yf.Ticker(ticker)
                hist = stock.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d')
                )
                # Calculate daily returns
                returns = hist['Close'].pct_change()
                data[ticker] = returns
            
            # Drop NaN values and get the last 'days' rows
            data = data.dropna()
            if len(data) > self.days:
                data = data.tail(self.days)
            
            return data
            
        except Exception as e:
            print(f"Error fetching real data: {str(e)}")
            print("Falling back to synthetic data generation...")
            return self.generate_synthetic_data()
    
    def generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic daily returns."""
        try:
            np.random.seed(42)  # For reproducibility
            returns_dict = {}
            
            for ticker in self.tickers:
                returns = np.random.normal(self.mean, self.std, self.days)
                returns_dict[ticker] = returns
                
            returns_df = pd.DataFrame(returns_dict)
            returns_df.index = pd.date_range(
                end=pd.Timestamp.today(),
                periods=self.days,
                freq='B'
            )
            
            return returns_df
            
        except Exception as e:
            raise Exception(f"Error generating synthetic data: {str(e)}")
    
    def generate_returns(self) -> pd.DataFrame:
        """
        Generate or fetch return data based on settings.
        
        Returns:
            pd.DataFrame: DataFrame with daily returns for each stock
        """
        if self.use_real_data:
            return self.fetch_real_data()
        else:
            return self.generate_synthetic_data()
    
    def get_portfolio_weights(self) -> Dict[str, float]:
        """
        Get equal weights for portfolio stocks.
        
        Returns:
            Dict[str, float]: Dictionary of stock weights
        """
        weight = 1.0 / len(self.tickers)
        return {ticker: weight for ticker in self.tickers} 