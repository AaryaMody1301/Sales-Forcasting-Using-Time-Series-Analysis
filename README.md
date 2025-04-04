# Sales Forecasting Using Time Series Analysis

This project provides a comprehensive framework for time series analysis and forecasting. It can work with any dataset that has a time-based component, with a special focus on sales and price data.

## Features

- **Generic Dataset Support**: Analysis and forecasting can be performed on any time series dataset
- **Data Preprocessing**: Automated data cleaning, feature extraction, and preparation for time series analysis
- **Multiple Forecasting Methods**:
  - ARIMA (AutoRegressive Integrated Moving Average)
  - SARIMA (Seasonal ARIMA)
  - Exponential Smoothing
  - Machine Learning models (Random Forest, Linear Regression)
  - Deep Learning (LSTM)
- **Model Evaluation**: Comprehensive metrics (RMSE, MAE, MAPE, R²) to compare model performance
- **Ensemble Forecasting**: Combine multiple models for improved accuracy
- **Visualizations**: Time series plots, forecasts, prediction intervals, and model comparisons

## Project Structure

```
.
├── data/               # Data storage
│   ├── amazon.csv      # Amazon product pricing data
│   ├── car_prices.csv  # Car sales price data
│   └── processed/      # Processed data files
├── models/             # Trained forecast models
├── notebooks/          # Jupyter notebooks for analysis
├── results/            # Forecasting results and visualizations
├── src/                # Source code
│   ├── analyze_datasets.py   # Dataset analysis and preprocessing
│   ├── forecasting.py        # Time series forecasting models
│   ├── main.py               # Main script for any dataset
│   ├── run_forecasting.py    # Example script for demo datasets
│   └── utils.py              # Helper functions
├── requirements.txt    # Required packages
└── README.md           # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Sales-Forecasting-Using-Time-Series-Analysis
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start with Demo Datasets

Run forecasting on the Amazon dataset:
```bash
python src/run_forecasting.py --dataset amazon
```

Run forecasting on the Car Prices dataset:
```bash
python src/run_forecasting.py --dataset car_prices
```

### Using Your Own Dataset

```bash
python src/main.py --data your_dataset.csv --date_col date_column --target_col target_column
```

### Command Line Arguments

- `--data`: Path to the dataset file (CSV)
- `--date_col`: Name of the date column
- `--target_col`: Name of the target column to forecast
- `--test_size`: Proportion of data to use for testing (default: 0.2)
- `--forecast_steps`: Number of steps to forecast (default: 30)
- `--models`: Models to train (arima, sarima, ets, rf, lr, lstm)
- `--output_dir`: Directory to save results (default: results)

## Example

```bash
python src/main.py --data data/sales_data.csv --date_col order_date --target_col sales_amount --forecast_steps 90 --models arima sarima rf lstm
```

This will:
1. Load the sales_data.csv file
2. Use order_date as the date column and sales_amount as the target
3. Forecast 90 steps ahead
4. Train ARIMA, SARIMA, Random Forest, and LSTM models

## Extending the Project

### Adding New Models

To add a new forecasting model, extend the TimeSeriesForecasting class in src/forecasting.py:

```python
def train_new_model(self, param1, param2):
    """
    Train a new forecasting model
    
    Parameters:
    -----------
    param1 : type
        Description
    param2 : type
        Description
        
    Returns:
    --------
    model
        Trained model
    """
    # Implementation
    return model
```

### Adding New Datasets

To add support for a new dataset format, extend the DatasetAnalyzer class in src/analyze_datasets.py:

```python
def preprocess_new_dataset(self, df):
    """
    Preprocess a new dataset format
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to preprocess
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed dataset
    """
    # Implementation
    return processed_df
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 