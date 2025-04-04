# Sales Forecasting Dashboard

This interactive dashboard visualizes time series forecasting results for sales and price data. It allows you to explore model performance, forecasts, time series analysis, and the underlying datasets.

## Features

- **Model Comparison**: Compare the performance of different forecasting models using metrics like RMSE, MAE, and MAPE
- **Forecast Visualization**: View and interact with forecasts from different models
- **Time Series Analysis**: Explore autocorrelation, stationarity tests, and other time series properties
- **Data Exploration**: Browse and understand the preprocessed datasets

## Getting Started

### Prerequisites

Make sure you have the required Python packages installed:

```bash
pip install -r requirements.txt
```

### Running the Dashboard

There are two ways to run the dashboard:

#### 1. Run forecasts and launch the dashboard in one step:

```bash
python run_forecasts.py --dataset all --mode sample
```

Options:
- `--dataset`: Choose from `amazon`, `car_prices`, or `all`
- `--mode`: Choose from `sample` (generates mock data) or `full` (runs actual forecasts)

#### 2. Run the dashboard directly (if you already have forecast results):

```bash
streamlit run dashboard.py
```

## Dashboard Navigation

The dashboard is organized into four main tabs:

### 1. Model Comparison
- Shows the best performing models according to different metrics
- Displays a comparison table with all model performance metrics
- Visualizes the RMSE comparison across models

### 2. Forecasts
- Interactive visualization of forecasts from different models
- Shows confidence intervals (where available)
- Comparison with historical data

### 3. Time Series Analysis
- Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots
- Stationarity tests
- Distribution analysis
- Additional time series visualizations

### 4. Data
- Explore the preprocessed dataset
- View summary statistics
- Examine column information and data quality metrics

## Extending the Dashboard

To add support for new datasets or forecasting models:

1. Create a forecasting script similar to `generate_forecasts.py`
2. Ensure it outputs results in the expected format in the `results/` directory
3. Update `run_forecasts.py` to include the new dataset option

## Troubleshooting

If you encounter issues:

- Make sure all required packages are installed
- Check that the `results/` directory and subdirectories exist
- Ensure forecast results have been generated before running the dashboard
- Look for error messages in the Streamlit console output 