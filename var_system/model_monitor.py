import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta
from scipy import stats

logger = logging.getLogger(__name__)

class ModelMonitor:
    """Monitor VaR model performance and detect distribution drift."""
    
    def __init__(self, config: dict):
        """
        Initialize ModelMonitor with configuration.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.violation_window = config['monitoring']['violation_window']
        self.drift_window = config['monitoring']['drift_detection']['window_size']
        self.drift_threshold = config['monitoring']['drift_detection']['z_score_threshold']
    
    def calculate_violation_ratio(self, returns: pd.Series,
                                var_predictions: pd.Series) -> Tuple[float, bool]:
        """
        Calculate VaR violation ratio and test if it's statistically valid.
        
        Args:
            returns (pd.Series): Actual returns
            var_predictions (pd.Series): VaR predictions
            
        Returns:
            Tuple[float, bool]: Violation ratio and validity flag
        """
        try:
            # Calculate violations
            violations = (returns < -var_predictions).astype(int)
            violation_ratio = violations.mean()
            
            # Expected violation ratio (e.g., 0.05 for 95% VaR)
            expected_ratio = 1 - var_predictions.name  # Assumes name is confidence level
            
            # Kupiec test for violation ratio
            n = len(returns)
            x = violations.sum()
            
            # Log-likelihood ratio statistic
            if x == 0:
                llr = -2 * n * np.log(1 - expected_ratio)
            elif x == n:
                llr = -2 * n * np.log(expected_ratio)
            else:
                llr = -2 * (x * np.log(expected_ratio/violation_ratio) +
                          (n-x) * np.log((1-expected_ratio)/(1-violation_ratio)))
            
            # Chi-square test with 1 degree of freedom
            p_value = 1 - stats.chi2.cdf(llr, 1)
            valid = p_value > 0.05
            
            logger.info(f"Violation ratio: {violation_ratio:.3f} (expected: {expected_ratio:.3f})")
            if not valid:
                logger.warning("VaR violations statistically significant")
            
            return float(violation_ratio), bool(valid)
            
        except Exception as e:
            logger.error(f"Error calculating violation ratio: {str(e)}")
            raise
    
    def detect_distribution_drift(self, historical_returns: pd.DataFrame,
                                recent_returns: pd.DataFrame) -> Dict[str, bool]:
        """
        Detect if return distribution has drifted significantly.
        
        Args:
            historical_returns (pd.DataFrame): Historical return data
            recent_returns (pd.DataFrame): Recent return data
            
        Returns:
            Dict[str, bool]: Drift detection results for each asset
        """
        try:
            drift_detected = {}
            
            for column in historical_returns.columns:
                # Calculate distribution statistics
                hist_mean = historical_returns[column].mean()
                hist_std = historical_returns[column].std()
                recent_mean = recent_returns[column].mean()
                recent_std = recent_returns[column].std()
                
                # Calculate z-scores for changes
                mean_z_score = abs(recent_mean - hist_mean) / (hist_std / np.sqrt(len(recent_returns)))
                std_z_score = abs(recent_std - hist_std) / (hist_std / np.sqrt(len(recent_returns)))
                
                # Check for significant drift
                drift_detected[column] = bool(mean_z_score > self.drift_threshold or
                                              std_z_score > self.drift_threshold)
                
                if drift_detected[column]:
                    logger.warning(f"Distribution drift detected for {column}")
            
            return drift_detected
            
        except Exception as e:
            logger.error(f"Error detecting distribution drift: {str(e)}")
            raise
    
    def calculate_performance_metrics(self, returns: pd.Series,
                                   var_predictions: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive model performance metrics.
        
        Args:
            returns (pd.Series): Actual returns
            var_predictions (pd.Series): VaR predictions
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        try:
            metrics = {}
            
            # Violation ratio
            violations = (returns < -var_predictions).astype(int)
            metrics['violation_ratio'] = violations.mean()
            
            # Average violation magnitude
            violation_returns = returns[violations == 1]
            if len(violation_returns) > 0:
                metrics['avg_violation_magnitude'] = abs(violation_returns.mean())
            else:
                metrics['avg_violation_magnitude'] = 0
            
            # Maximum violation
            if len(violation_returns) > 0:
                metrics['max_violation'] = abs(violation_returns.min())
            else:
                metrics['max_violation'] = 0
            
            # Clustering of violations (runs test)
            if len(violations) > 1:
                runs, n_pos, n_neg = self._runs_test(violations)
                metrics['violation_clustering'] = runs
            
            logger.info("Performance metrics calculated successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            raise
    
    def _runs_test(self, violations: pd.Series) -> Tuple[int, int, int]:
        """
        Perform runs test to detect clustering of violations.
        
        Args:
            violations (pd.Series): Series of VaR violations (0/1)
            
        Returns:
            Tuple[int, int, int]: Number of runs, positive values, negative values
        """
        # Count runs
        runs = 1
        for i in range(1, len(violations)):
            if violations.iloc[i] != violations.iloc[i-1]:
                runs += 1
        
        # Count positive and negative values
        n_pos = sum(violations == 1)
        n_neg = sum(violations == 0)
        
        return runs, n_pos, n_neg
    
    def generate_monitoring_report(self, returns: pd.DataFrame,
                                 var_predictions: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Generate comprehensive monitoring report.
        
        Args:
            returns (pd.DataFrame): Return data
            var_predictions (Dict[str, pd.Series]): VaR predictions by method
            
        Returns:
            pd.DataFrame: Monitoring report
        """
        try:
            report_data = []
            
            for method, predictions in var_predictions.items():
                # Calculate metrics
                violation_ratio, valid = self.calculate_violation_ratio(
                    returns['Portfolio'],
                    predictions
                )
                
                metrics = self.calculate_performance_metrics(
                    returns['Portfolio'],
                    predictions
                )
                
                # Detect drift
                historical = returns.iloc[:-self.drift_window]
                recent = returns.iloc[-self.drift_window:]
                drift_results = self.detect_distribution_drift(historical, recent)
                
                # Compile report
                report_data.append({
                    'Method': method,
                    'Violation_Ratio': violation_ratio,
                    'Valid_at_95': valid,
                    'Avg_Violation_Magnitude': metrics['avg_violation_magnitude'],
                    'Max_Violation': metrics['max_violation'],
                    'Violation_Clustering': metrics['violation_clustering'],
                    'Distribution_Drift': any(drift_results.values())
                })
            
            report = pd.DataFrame(report_data)
            logger.info("Monitoring report generated successfully")
            return report
            
        except Exception as e:
            logger.error(f"Error generating monitoring report: {str(e)}")
            raise 