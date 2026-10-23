from fpdf import FPDF
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64

class RiskReport(FPDF):
    """Generate PDF risk reports with VaR analysis results."""
    
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        self.set_font('Arial', 'B', 16)
        
    def header(self):
        """Add report header with title and date."""
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Value at Risk (VaR) Analysis Report', 0, 1, 'C')
        self.set_font('Arial', '', 10)
        self.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        self.ln(10)
        
    def footer(self):
        """Add page numbers in footer."""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
    def add_section(self, title: str):
        """Add a section header."""
        self.set_font('Arial', 'B', 14)
        self.ln(10)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)
        
    def add_paragraph(self, text: str):
        """Add a paragraph of text."""
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 5, text)
        self.ln(5)
        
    def add_table(self, df: pd.DataFrame, title: str = None):
        """Add a DataFrame as a table."""
        if title:
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, title, 0, 1, 'L')
            
        self.set_font('Arial', '', 10)
        
        # Table header
        line_height = 7
        col_width = self.w / (len(df.columns) + 1)
        
        self.set_font('Arial', 'B', 10)
        self.cell(col_width, line_height, 'Index', 1)
        for col in df.columns:
            self.cell(col_width, line_height, str(col), 1)
        self.ln()
        
        # Table data
        self.set_font('Arial', '', 10)
        for idx, row in df.iterrows():
            self.cell(col_width, line_height, str(idx), 1)
            for item in row:
                self.cell(col_width, line_height, str(item), 1)
            self.ln()
        
        self.ln(5)
        
    def add_plot(self, fig: plt.Figure, title: str = None):
        """Add a matplotlib figure."""
        if title:
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, title, 0, 1, 'L')
        
        # Save plot to memory
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        
        # Add plot to PDF
        img_width = self.w - 40  # Leave margins
        self.image(buf, x=20, w=img_width)
        self.ln(5)
        
    def add_risk_metrics(self, var_results: dict, es_results: dict = None):
        """Add VaR and Expected Shortfall metrics."""
        self.add_section('Risk Metrics')
        
        # Format VaR results
        var_df = pd.DataFrame(var_results).round(4) * 100
        var_df = var_df.applymap(lambda x: f'{x:.2f}%')
        self.add_table(var_df, 'Value at Risk (VaR)')
        
        if es_results:
            es_df = pd.DataFrame(es_results).round(4) * 100
            es_df = es_df.applymap(lambda x: f'{x:.2f}%')
            self.add_table(es_df, 'Expected Shortfall (ES)')
            
    def add_portfolio_composition(self, weights: dict):
        """Add portfolio composition section."""
        self.add_section('Portfolio Composition')
        
        weights_df = pd.DataFrame({
            'Asset': list(weights.keys()),
            'Weight': [f'{w*100:.1f}%' for w in weights.values()]
        }).set_index('Asset')
        
        self.add_table(weights_df, 'Asset Weights')
        
    def add_risk_decomposition(self, component_var: pd.Series):
        """Add risk decomposition analysis."""
        self.add_section('Risk Decomposition')
        
        comp_var_df = component_var.round(4) * 100
        comp_var_df = pd.DataFrame({
            'Component VaR': comp_var_df.map(lambda x: f'{x:.2f}%'),
            'Contribution': (comp_var_df / comp_var_df.sum() * 100).map(lambda x: f'{x:.1f}%')
        })
        
        self.add_table(comp_var_df, 'Component VaR Analysis')
        
    def add_scenario_analysis(self, scenario_results: dict):
        """Add scenario analysis results."""
        self.add_section('Scenario Analysis')
        
        scenario_df = pd.DataFrame({
            'Scenario': list(scenario_results.keys()),
            'Return': [f'{r*100:.2f}%' for r in scenario_results.values()]
        }).set_index('Scenario')
        
        self.add_table(scenario_df, 'Scenario Returns') 