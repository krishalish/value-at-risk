from setuptools import setup, find_packages

setup(
    name="var_system",
    version="1.0.0",
    description="Value at Risk engine: historical, normal, Monte Carlo and GARCH(1,1) with backtesting",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26", "pandas>=2.1", "scipy>=1.11", "matplotlib>=3.8", "seaborn>=0.13",
        "arch>=6.2", "scikit-learn>=1.3", "cvxopt>=1.3", "yfinance>=0.2.40", "pyyaml>=6.0",
        "openpyxl>=3.1",
    ],
)
