from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
from datetime import datetime
from .data_generator import DataGenerator
from .var_calculator import VaRCalculator

app = FastAPI(
    title="VaR Calculation API",
    description="API for Value at Risk calculations",
    version="1.0.0"
)

class PortfolioWeights(BaseModel):
    weights: Dict[str, float]

class VaRRequest(BaseModel):
    tickers: List[str]
    weights: Dict[str, float]
    confidence_level: float = 0.95
    method: str = "historical"

class VaRResponse(BaseModel):
    var_value: float
    method: str
    confidence_level: float
    calculation_date: str

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "VaR Calculation API"}

@app.post("/api/v1/var/calculate", response_model=VaRResponse)
async def calculate_var(request: VaRRequest):
    """Calculate VaR for a given portfolio."""
    try:
        # Generate sample data
        data_gen = DataGenerator(tickers=request.tickers)
        returns = data_gen.generate_returns()
        
        # Initialize VaR calculator
        var_calc = VaRCalculator(
            returns=returns,
            weights=request.weights,
            confidence_levels=[request.confidence_level]
        )
        
        # Calculate VaR using specified method
        if request.method == "historical":
            var_result = var_calc.historical_var()
        elif request.method == "parametric":
            var_result = var_calc.parametric_var()
        elif request.method == "monte_carlo":
            var_result = var_calc.monte_carlo_var()
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid VaR calculation method"
            )
        
        # Get portfolio VaR
        var_value = var_result['Portfolio'][request.confidence_level]
        
        return VaRResponse(
            var_value=var_value,
            method=request.method,
            confidence_level=request.confidence_level,
            calculation_date=datetime.now().strftime('%Y-%m-%d')
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        ) 