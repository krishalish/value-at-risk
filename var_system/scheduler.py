import schedule
import time
import yaml
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from var_system.data_handler import DataHandler
from var_system.var_calculator import VaRCalculator
from var_system.model_monitor import ModelMonitor
from var_system.report_generator import RiskReport
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VaRScheduler:
    """Schedule and run automated VaR calculations and reporting."""
    
    def __init__(self, config_path: str):
        """
        Initialize scheduler with configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.data_handler = DataHandler(config_path)
        self.model_monitor = ModelMonitor(self.config)
        
        # Create output directories
        Path("reports").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def generate_daily_report(self):
        """Generate daily VaR report."""
        try:
            logger.info("Generating daily VaR report...")
            
            # Load latest data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config['monitoring']['violation_window'])
            
            returns = self.data_handler.load_returns(start_date, end_date)
            
            # Calculate VaR using all methods
            var_calc = VaRCalculator(
                returns=returns,
                weights={col: 1/len(returns.columns) for col in returns.columns},
                confidence_levels=self.config['var_calculation']['confidence_levels']
            )
            
            hist_var = var_calc.historical_var()
            param_var = var_calc.parametric_var()
            mc_var = var_calc.monte_carlo_var()
            
            # Generate monitoring report
            var_predictions = {
                'Historical': pd.Series(hist_var['Portfolio']),
                'Parametric': pd.Series(param_var['Portfolio']),
                'Monte Carlo': pd.Series(mc_var['Portfolio'])
            }
            
            monitoring_report = self.model_monitor.generate_monitoring_report(
                returns,
                var_predictions
            )
            
            # Create PDF report
            report = RiskReport()
            
            # Add VaR results
            report.add_risk_metrics(hist_var, None)
            
            # Add monitoring results
            report.add_section("Model Monitoring")
            report.add_table(monitoring_report, "Model Performance Metrics")
            
            # Save report
            report_path = f"reports/var_report_{datetime.now().strftime('%Y%m%d')}.pdf"
            report.output(report_path)
            
            # Send email if enabled
            if self.config['reporting']['email']['enabled']:
                self._send_email_report(report_path)
            
            logger.info("Daily report generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating daily report: {str(e)}")
            raise
    
    def _send_email_report(self, report_path: str):
        """Send report via email."""
        try:
            email_config = self.config['reporting']['email']
            
            msg = MIMEMultipart()
            msg['Subject'] = f"VaR Report - {datetime.now().strftime('%Y-%m-%d')}"
            msg['From'] = email_config['sender']
            msg['To'] = ", ".join(email_config['recipients'])
            
            # Add report as attachment
            with open(report_path, 'rb') as f:
                pdf = MIMEApplication(f.read(), _subtype='pdf')
                pdf.add_header(
                    'Content-Disposition',
                    'attachment',
                    filename=Path(report_path).name
                )
                msg.attach(pdf)
            
            # Send email
            with smtplib.SMTP(email_config['smtp_server'], email_config['port']) as server:
                server.starttls()
                server.send_message(msg)
            
            logger.info("Report email sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            raise
    
    def run_scheduled_tasks(self):
        """Run scheduled tasks based on configuration."""
        try:
            # Schedule daily report
            schedule.every().day.at(self.config['reporting']['schedule']['daily_report']).do(
                self.generate_daily_report
            )
            
            # Schedule weekly report (if different from daily)
            if 'weekly_report' in self.config['reporting']['schedule']:
                day, time = self.config['reporting']['schedule']['weekly_report'].split()
                schedule.every().day.at(time).do(
                    self.generate_daily_report
                )
            
            logger.info("Scheduled tasks initialized")
            
            # Run continuously
            while True:
                schedule.run_pending()
                time.sleep(60)
                
        except Exception as e:
            logger.error(f"Error in scheduler: {str(e)}")
            raise
    
def main():
    """Main function to start the scheduler."""
    try:
        scheduler = VaRScheduler('config/config.yaml')
        scheduler.run_scheduled_tasks()
    except Exception as e:
        logger.error(f"Scheduler failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 