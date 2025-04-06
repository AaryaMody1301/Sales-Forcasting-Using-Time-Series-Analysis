# Source Code Documentation

This directory contains the core modules for the Advanced Time Series Forecasting Framework.

## Module Structure

| File | Description |
|------|-------------|
| `analyze_datasets.py` | Dataset analysis, preprocessing, and preparation for forecasting |
| `forecasting.py` | Implementation of forecasting models including statistical, ML, and DL models |
| `utils.py` | Utility functions for data processing, visualization, and file operations |
| `run_forecasting.py` | Command-line interface for running forecasting models and launching the dashboard |
| `main.py` | Main entry point for the forecasting application |
| `forecast.py` | Core forecasting functionality and model evaluation |
| `train_model.py` | Model training and hyperparameter optimization |
| `data_preparation.py` | Data preparation and feature engineering for time series |

## Key Components

### Dataset Analysis (`analyze_datasets.py`)

The `DatasetAnalyzer` class provides methods for:
- Loading and preprocessing different types of datasets
- Analyzing data quality and statistics
- Preparing datasets for time series forecasting
- Visualizing time series data
- Generating statistical reports
- Preprocessing generic datasets with automatic feature detection

### Forecasting Models (`forecasting.py`)

Implements multiple forecasting models:
- Statistical Models: ARIMA, SARIMA, Exponential Smoothing
- Machine Learning Models: Random Forest, XGBoost, Gradient Boosting
- Deep Learning Models: LSTM
- Ensemble Methods: Combining predictions from multiple models

### Utilities (`utils.py`)

Provides utility functions for:
- File and directory operations
- Time series feature engineering
- Data visualization
- Error metrics calculation
- Configuration management

### Command-line Interface (`run_forecasting.py`)

Offers a comprehensive CLI for:
- Running specific forecasting models
- Training ensemble methods
- Processing and analyzing datasets
- Generating visualizations
- Launching the interactive dashboard

## Usage Flow

1. **Data Loading & Preprocessing**: 
   - `analyze_datasets.py` loads and preprocesses the raw data
   - Handles missing values, outliers, and feature engineering

2. **Model Training**:
   - `train_model.py` trains models with optional hyperparameter tuning
   - `forecasting.py` implements the model architectures

3. **Forecasting**:
   - `forecast.py` generates forecasts using trained models
   - Evaluates model performance with multiple metrics

4. **Visualization & Analysis**:
   - `dashboard.py` (in root dir) provides interactive visualization
   - Results are saved to the `results` directory

## Adding New Models

To add a new forecasting model:

1. Implement the model class in `forecasting.py`
2. Add model evaluation in `forecast.py`
3. Register the model in `run_forecasting.py`

## Adding New Datasets

The framework supports custom datasets through:

1. Automatic column detection in `analyze_datasets.py`
2. Preprocessing pipeline in `preprocess_generic_data()`
3. Command-line arguments in `run_forecasting.py` 