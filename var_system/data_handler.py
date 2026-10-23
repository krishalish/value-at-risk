import pandas as pd
import numpy as np
import sqlite3
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

class DataHandler:
    """Handle data loading, validation, and storage for VaR calculations."""
    
    def __init__(self, config_path: str, db_path: Optional[str] = None):
        """
        Initialize DataHandler with configuration.
        
        Args:
            config_path (str): Path to configuration YAML file
            db_path (str, optional): Override the database path from the config
        """
        self.config = self._load_config(config_path)
        self.db_path = str(db_path or self.config['database']['path'])
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._setup_database()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def _setup_database(self):
        """Set up SQLite database and required tables."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables for storing return data and VaR calculations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS return_data (
                    date TEXT,
                    ticker TEXT,
                    return REAL,
                    PRIMARY KEY (date, ticker)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS var_calculations (
                    calculation_date TEXT,
                    portfolio_id TEXT,
                    method TEXT,
                    confidence_level REAL,
                    time_horizon INTEGER,
                    var_value REAL,
                    PRIMARY KEY (calculation_date, portfolio_id, method, confidence_level, time_horizon)
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("Database setup completed")
            
        except Exception as e:
            logger.error(f"Database setup error: {str(e)}")
            raise
    
    def validate_data(self, returns: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate return data against configured criteria.
        
        Args:
            returns (pd.DataFrame): Return data to validate
            
        Returns:
            Tuple[bool, List[str]]: Validation status and list of issues
        """
        issues = []
        
        try:
            # Check minimum data points
            if len(returns) < self.config['data_validation']['min_data_points']:
                issues.append(f"Insufficient data points: {len(returns)} < "
                            f"{self.config['data_validation']['min_data_points']}")
            
            # Check for missing values
            missing_pct = returns.isnull().mean()
            if (missing_pct > self.config['data_validation']['max_missing_pct']).any():
                issues.append("Missing values exceed threshold")
            
            # Check for outliers
            z_scores = np.abs((returns - returns.mean()) / returns.std())
            threshold = self.config['data_validation']['outlier_std_threshold']
            outliers = (z_scores > threshold).sum()
            if outliers.any():
                # Fat tails are expected in returns, so this is a note, not a failure
                logger.info(f"{outliers.sum()} returns beyond {threshold} standard deviations")
            
            # Check for invalid values
            if (returns.abs() > 1).any().any():  # Returns > 100% are suspicious
                issues.append("Found suspicious return values > 100%")
            
            valid = len(issues) == 0
            if valid:
                logger.info("Data validation passed")
            else:
                logger.warning(f"Data validation issues found: {issues}")
            
            return valid, issues
            
        except Exception as e:
            logger.error(f"Data validation error: {str(e)}")
            raise
    
    def clean_data(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Clean return data by handling missing values and outliers.
        
        Args:
            returns (pd.DataFrame): Return data to clean
            
        Returns:
            pd.DataFrame: Cleaned return data
        """
        try:
            # Handle missing values
            returns_cleaned = returns.copy()
            
            # Forward fill, then backward fill missing values
            returns_cleaned = returns_cleaned.ffill()
            returns_cleaned = returns_cleaned.bfill()
            
            # Handle outliers with a robust centre and scale (median and MAD), because
            # a single huge value inflates the mean and standard deviation and hides itself
            threshold = self.config['data_validation']['outlier_std_threshold']
            centre = returns_cleaned.median()
            scale = 1.4826 * (returns_cleaned - centre).abs().median()  # MAD scaled to match sigma under normality
            
            # Clip values beyond the threshold back to the boundary
            returns_cleaned = returns_cleaned.clip(
                lower=centre - threshold * scale,
                upper=centre + threshold * scale,
                axis=1
            )
            
            logger.info("Data cleaning completed")
            return returns_cleaned
            
        except Exception as e:
            logger.error(f"Data cleaning error: {str(e)}")
            raise
    
    def store_returns(self, returns: pd.DataFrame):
        """Store return data in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Reshape data for storage
            returns_long = returns.reset_index().melt(
                id_vars='index',
                var_name='ticker',
                value_name='return'
            )
            returns_long = returns_long.rename(columns={returns_long.columns[0]: 'date'})
            returns_long['date'] = pd.to_datetime(returns_long['date']).dt.strftime('%Y-%m-%d')
            
            # Store data (replace rows that already exist for the same date and ticker)
            conn.executemany(
                "INSERT OR REPLACE INTO return_data (date, ticker, return) VALUES (?, ?, ?)",
                returns_long[['date', 'ticker', 'return']].itertuples(index=False, name=None)
            )
            conn.commit()
            conn.close()
            
            logger.info(f"Stored {len(returns_long)} return records")
            
        except Exception as e:
            logger.error(f"Error storing returns: {str(e)}")
            raise
    
    def store_var_calculation(self, calculation_date: datetime, portfolio_id: str,
                            method: str, confidence_level: float, time_horizon: int,
                            var_value: float):
        """Store VaR calculation results in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO var_calculations
                (calculation_date, portfolio_id, method, confidence_level,
                 time_horizon, var_value)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (calculation_date.strftime('%Y-%m-%d'), portfolio_id, method,
                  confidence_level, time_horizon, var_value))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Stored VaR calculation for {portfolio_id}")
            
        except Exception as e:
            logger.error(f"Error storing VaR calculation: {str(e)}")
            raise
    
    def load_returns(self, start_date: datetime, end_date: datetime,
                    tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """Load return data from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
                SELECT date, ticker, return
                FROM return_data
                WHERE date BETWEEN ? AND ?
            """
            
            if tickers:
                query += f" AND ticker IN ({','.join(['?']*len(tickers))})"
                params = (start_date.strftime('%Y-%m-%d'),
                         end_date.strftime('%Y-%m-%d')) + tuple(tickers)
            else:
                params = (start_date.strftime('%Y-%m-%d'),
                         end_date.strftime('%Y-%m-%d'))
            
            # Load data
            returns = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            # Reshape to wide format
            returns_wide = returns.pivot(
                index='date',
                columns='ticker',
                values='return'
            )
            
            logger.info(f"Loaded returns for {len(returns_wide.columns)} tickers")
            return returns_wide
            
        except Exception as e:
            logger.error(f"Error loading returns: {str(e)}")
            raise
    
    def get_var_history(self, portfolio_id: str, start_date: datetime,
                       end_date: datetime) -> pd.DataFrame:
        """Load historical VaR calculations from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
                SELECT *
                FROM var_calculations
                WHERE portfolio_id = ?
                AND calculation_date BETWEEN ? AND ?
            """
            
            var_history = pd.read_sql_query(
                query, conn,
                params=(portfolio_id,
                       start_date.strftime('%Y-%m-%d'),
                       end_date.strftime('%Y-%m-%d'))
            )
            conn.close()
            
            logger.info(f"Loaded VaR history for portfolio {portfolio_id}")
            return var_history
            
        except Exception as e:
            logger.error(f"Error loading VaR history: {str(e)}")
            raise 