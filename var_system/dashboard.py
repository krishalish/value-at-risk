import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from var_system.data_generator import DataGenerator
from var_system.var_calculator import VaRCalculator
from typing import Dict, Tuple

# Set page config
st.set_page_config(
    page_title="VaR Dashboard",
    page_icon="📊",
    layout="wide"
)

# Define tooltips
TOOLTIPS = {
    'var': """Value at Risk (VaR) represents the potential loss that will not be exceeded
    with a given confidence level over a specified time horizon.""",
    'historical_var': """Historical VaR uses actual historical returns to estimate potential
    losses, making no assumptions about the distribution of returns.""",
    'parametric_var': """Parametric VaR assumes returns follow a normal distribution and
    uses mean and standard deviation to estimate potential losses.""",
    'monte_carlo_var': """Monte Carlo VaR simulates many possible return scenarios to
    estimate potential losses, providing a more robust estimate."""
}

def plot_return_distribution(returns: pd.DataFrame, var_results: dict):
    """Plot return distributions with VaR thresholds using plotly."""
    fig = go.Figure()
    
    for col in returns.columns:
        fig.add_trace(go.Histogram(
            x=returns[col],
            name=col,
            opacity=0.7,
            nbinsx=50
        ))
        
        # Add VaR lines
        for conf_level, var_value in var_results[col].items():
            fig.add_vline(
                x=-var_value,
                line_dash="dash",
                annotation_text=f"{conf_level*100:.0f}% VaR",
                line_color="red"
            )
    
    fig.update_layout(
        title="Return Distributions with VaR Thresholds",
        xaxis_title="Returns",
        yaxis_title="Frequency",
        barmode='overlay'
    )
    
    return fig

def plot_returns_line(returns: pd.DataFrame, var_results: dict):
    """Plot returns as line graph over time."""
    fig = go.Figure()
    
    # Add cumulative returns for each stock
    for col in returns.columns:
        cumulative_returns = (1 + returns[col]).cumprod()
        fig.add_trace(go.Scatter(
            x=returns.index,
            y=cumulative_returns,
            name=col,
            mode='lines'
        ))
        
        # Add VaR bands
        for conf_level, var_value in var_results[col].items():
            # Lower VaR band
            fig.add_trace(go.Scatter(
                x=returns.index,
                y=[1 - var_value] * len(returns),
                name=f'{col} {conf_level*100:.0f}% VaR',
                line=dict(dash='dash', color='red'),
                opacity=0.5
            ))
    
    fig.update_layout(
        title="Cumulative Returns with VaR Bands",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        showlegend=True
    )
    
    return fig

def calculate_performance_metrics(returns: pd.DataFrame) -> Dict[str, Tuple[float, str, str]]:
    """Calculate performance metrics for each stock."""
    metrics = {}
    for ticker in returns.columns:
        # Calculate cumulative return
        cum_return = (1 + returns[ticker]).prod() - 1
        # Determine color and symbol
        color = "green" if cum_return > 0 else "red"
        symbol = "↑" if cum_return > 0 else "↓"
        # Store metrics
        metrics[ticker] = (cum_return, color, symbol)
    return metrics

def main():
    st.title("Value at Risk (VaR) Analysis Dashboard")
    
    # Sidebar controls
    st.sidebar.header("Parameters")
    
    # Extended stock selection
    tickers = [
        # Original stocks
        'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN',
        # Technology
        'NVDA', 'AMD', 'INTC', 'CRM', 'ADBE',
        # Financial
        'JPM', 'BAC', 'GS', 'MS', 'V',
        # Healthcare
        'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK',
        # Consumer
        'PG', 'KO', 'PEP', 'WMT', 'HD',
        # Industrial
        'CAT', 'BA', 'GE', 'MMM', 'HON',
        # Energy
        'XOM', 'CVX', 'COP', 'SLB', 'EOG',
        # Communication
        'META', 'NFLX', 'DIS', 'CMCSA', 'VZ',
        # Materials
        'LIN', 'APD', 'ECL', 'NEM', 'FCX',
        # Real Estate
        'AMT', 'PLD', 'CCI', 'EQIX', 'PSA',
        # Utilities
        'NEE', 'DUK', 'SO', 'D', 'AEP'
    ]
    
    # Group stocks by sector for easier selection
    sectors = {
        'Technology': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'AMD', 'INTC', 'CRM', 'ADBE'],
        'Financial': ['JPM', 'BAC', 'GS', 'MS', 'V'],
        'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK'],
        'Consumer': ['PG', 'KO', 'PEP', 'WMT', 'HD'],
        'Industrial': ['CAT', 'BA', 'GE', 'MMM', 'HON'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
        'Communication': ['META', 'NFLX', 'DIS', 'CMCSA', 'VZ'],
        'Materials': ['LIN', 'APD', 'ECL', 'NEM', 'FCX'],
        'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA'],
        'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP']
    }
    
    # Sector selection
    selected_sectors = st.sidebar.multiselect(
        "Select Sectors",
        options=list(sectors.keys()),
        default=['Technology'],
        help="Choose sectors to analyze"
    )
    
    # Get stocks from selected sectors
    available_tickers = []
    for sector in selected_sectors:
        available_tickers.extend(sectors[sector])
    
    # Stock selection within chosen sectors
    selected_tickers = st.sidebar.multiselect(
        "Select Stocks",
        options=available_tickers,
        default=available_tickers[:5] if available_tickers else [],
        help="Choose stocks for your portfolio"
    )
    
    if not selected_tickers:
        st.warning("Please select at least one stock.")
        st.stop()
        
    # Add sector information to the sidebar
    st.sidebar.markdown("### Selected Sectors Overview")
    for sector in selected_sectors:
        sector_stocks = [tick for tick in selected_tickers if tick in sectors[sector]]
        if sector_stocks:
            st.sidebar.markdown(f"**{sector}**: {', '.join(sector_stocks)}")

    # Data source selection
    use_real_data = st.sidebar.checkbox(
        "Use Real Market Data",
        value=True,
        help="Toggle between real market data and synthetic data"
    )
    
    if use_real_data:
        st.sidebar.info("""
        Using real market data from Yahoo Finance.
        Data is fetched for the last 252 trading days.
        """)
    else:
        st.sidebar.info("""
        Using synthetic data with:
        - Mean daily return: 0.1%
        - Daily volatility: 2%
        """)
    
    # Generate data
    data_gen = DataGenerator(
        tickers=selected_tickers,
        use_real_data=use_real_data
    )
    returns = data_gen.generate_returns()
    
    # Calculate performance metrics
    performance_metrics = calculate_performance_metrics(returns)
    
    # Portfolio weights input with performance indicators
    st.sidebar.subheader("Portfolio Weights and Performance")
    weights = {}
    
    # Equal weights by default
    default_weight = 1.0 / len(selected_tickers)
    
    # Create two columns for each stock entry
    for ticker in selected_tickers:
        col1, col2 = st.sidebar.columns([3, 1])
        
        # Weight input in first column
        with col1:
            weights[ticker] = st.number_input(
                f"{ticker} Weight",
                min_value=0.0,
                max_value=1.0,
                value=default_weight,
                step=0.05,
                format="%.2f"
            )
        
        # Performance indicator in second column
        with col2:
            return_val, color, symbol = performance_metrics[ticker]
            st.markdown(
                f'<div style="color: {color}; font-size: 20px; text-align: right;">'
                f'{symbol} {return_val:.1%}</div>',
                unsafe_allow_html=True
            )
    
    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight != 0:
        weights = {k: v/total_weight for k, v in weights.items()}
    
    # VaR parameters
    st.sidebar.subheader("VaR Parameters")
    confidence_level = st.sidebar.slider(
        "Confidence Level",
        min_value=0.9,
        max_value=0.99,
        value=0.95,
        step=0.01,
        help=TOOLTIPS['var']
    )
    
    var_method = st.sidebar.selectbox(
        "VaR Calculation Method",
        ["Historical", "Parametric", "Monte Carlo"],
        help="Choose the method for calculating VaR"
    )
    
    # Visualization type selector
    viz_type = st.sidebar.selectbox(
        "Return Visualization Type",
        ["Distribution (Histogram)", "Time Series (Line)"],
        help="Choose how to visualize the returns"
    )
    
    # Calculate VaR
    var_calc = VaRCalculator(
        returns=returns,
        weights=weights,
        confidence_levels=[confidence_level]
    )
    
    if var_method == "Historical":
        var_results = var_calc.historical_var()
        method_tooltip = TOOLTIPS['historical_var']
    elif var_method == "Parametric":
        var_results = var_calc.parametric_var()
        method_tooltip = TOOLTIPS['parametric_var']
    else:  # Monte Carlo
        var_results = var_calc.monte_carlo_var()
        method_tooltip = TOOLTIPS['monte_carlo_var']
    
    # Display VaR results
    st.header("VaR Analysis Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Portfolio VaR",
            f"{var_results['Portfolio'][confidence_level]*100:.2f}%",
            help=method_tooltip
        )
    
    with col2:
        st.metric(
            "Confidence Level",
            f"{confidence_level*100:.0f}%"
        )
    
    # Display visualizations based on selected type
    if viz_type == "Distribution (Histogram)":
        st.subheader("Return Distributions")
        st.plotly_chart(
            plot_return_distribution(returns, var_results),
            width='stretch'
        )
    else:  # Time Series (Line)
        st.subheader("Returns Over Time")
        st.plotly_chart(
            plot_returns_line(returns, var_results),
            width='stretch'
        )
    
    # Display individual asset VaR
    st.subheader("Individual Asset VaR")
    asset_vars = pd.DataFrame({
        'Asset': list(var_results.keys()),
        f'{confidence_level*100:.0f}% VaR': [
            f"{var_results[asset][confidence_level]*100:.2f}%"
            for asset in var_results.keys()
        ]
    })
    st.dataframe(asset_vars)

if __name__ == "__main__":
    main() 