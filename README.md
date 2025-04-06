# Advanced Time Series Forecasting Framework

A comprehensive framework for time series forecasting with multiple models, automated preprocessing, hyperparameter tuning, and an interactive dashboard.

## Features

- **Multi-Model Forecasting**: Leverage multiple forecasting methods including statistical, machine learning, and deep learning approaches
  - Statistical Models: ARIMA, SARIMA, Exponential Smoothing
  - Machine Learning Models: Random Forest, XGBoost, Gradient Boosting
  - Deep Learning Models: LSTM
  - Advanced Models: Prophet
  - Ensemble Methods: Combining predictions from multiple models

- **Advanced Data Preprocessing**:
  - Automatic date and target column detection
  - Data validation and quality assessment
  - Missing value imputation
  - Anomaly detection and handling
  - Feature engineering for time series

- **Hyperparameter Optimization**:
  - Automated model tuning
  - Cross-validation with time series split
  - Model performance comparison

- **Interactive Dashboard**:
  - Model performance comparison
  - Forecast visualization
  - Advanced time series analysis tools
  - Trend decomposition and seasonality analysis
  - Automated insights

## Requirements

The framework requires the following dependencies:

- numpy>=1.20.0
- pandas>=1.3.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- scikit-learn>=1.0.0
- statsmodels>=0.13.0
- pmdarima>=1.8.0
- prophet>=1.0
- xgboost>=1.5.0
- tensorflow>=2.8.0
- streamlit>=1.10.0
- plotly>=5.3.0

For a complete list of dependencies, see the requirements.txt file.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/advanced-time-series-forecasting.git
   cd advanced-time-series-forecasting
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Generate an updated requirements file (optional):
   ```bash
   python generate_requirements.py
   ```

## Quick Start

### Running Forecasts on Predefined Datasets

Run forecasting models on the included Amazon dataset:
```bash
python run_forecasting.py --dataset amazon --models arima prophet xgboost
```

Run forecasting models on the included Car Prices dataset:
```bash
python run_forecasting.py --dataset car_prices --models sarima prophet rf
```

### Running Forecasts on Custom Datasets

Run forecasting models on your own dataset:
```bash
python run_forecasting.py --dataset custom --data_path your_dataset.csv
```

The framework will automatically detect date and target columns, but you can also specify them:
```bash
python run_forecasting.py --dataset custom --data_path your_dataset.csv --date_col date --target_col sales
```

### Launch the Dashboard

Launch the interactive dashboard:
```bash
python run_forecasting.py --only_dashboard
```

Or run forecasts and then launch the dashboard:
```bash
python run_forecasting.py --dataset amazon --run_dashboard
```

## Project Structure

```
├── data/                  # Raw datasets
├── src/                   # Source code
│   ├── README.md          # Source code documentation
│   ├── analyze_datasets.py  # Dataset analysis & preprocessing
│   ├── forecasting.py     # Forecasting models implementation
│   ├── utils.py           # Utility functions
│   └── ...                # Other source files
├── models/                # Saved model files
├── results/               # Forecasting results & visualizations
├── dashboard.py           # Interactive Streamlit dashboard
├── run_forecasting.py     # Command-line interface
├── generate_requirements.py  # Generate requirements.txt
└── README.md              # Project documentation
```

## Data Validation

The framework includes robust data validation to ensure your datasets are suitable for time series forecasting:

- Automatic detection of date and target columns
- Validation of date formats and continuity
- Missing value analysis
- Outlier detection
- Statistical property assessment
- Data quality reporting

When loading a custom dataset, a validation report is generated with insights and recommendations.

## Datasets

### Included Datasets

- **Amazon Dataset**: Daily sales data for Amazon products
  - Date range: Recent dates
  - Features: Product prices, discounts, ratings, sales estimates

- **Car Prices Dataset**: Used car transaction data
  - Date range: Recent years
  - Features: Car make, model, year, odometer reading, selling prices

### Using Custom Datasets

You can use your own dataset by providing a CSV file with at least:
- A date/time column
- A numeric target column for forecasting

The framework will automatically:
1. Detect date and target columns
2. Validate the dataset for time series forecasting
3. Preprocess the data appropriately
4. Generate a validation report with insights

## Models

### Statistical Models
- **ARIMA**: AutoRegressive Integrated Moving Average
- **SARIMA**: Seasonal ARIMA for data with seasonality
- **Exponential Smoothing**: Simple, double, and triple (Holt-Winters)

### Machine Learning Models
- **Random Forest**: Ensemble of decision trees
- **XGBoost**: Gradient boosting implementation
- **Gradient Boosting**: Classic gradient boosting machines

### Deep Learning Models
- **LSTM**: Long Short-Term Memory networks for sequence modeling

### Advanced Models
- **Prophet**: Facebook's time series forecasting procedure

### Ensemble Methods
- Combining predictions from multiple models using weighted averaging

## Interactive Dashboard

The dashboard provides several features:

### Model Comparison
- Performance metrics visualization
- Radar charts for multi-metric comparison
- Sortable model rankings

### Forecast Visualization
- Interactive time series plots
- Forecast vs. actual comparison
- Confidence intervals

### Advanced Analysis
- Time series decomposition (trend, seasonality, residuals)
- Seasonal pattern analysis
- Autocorrelation analysis
- Statistical tests and distribution analysis

### Automated Insights
- Key statistics and findings
- Data quality assessment
- Forecasting recommendations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The datasets used in this project are for demonstration purposes
- This framework incorporates methods from various statistical and machine learning libraries
- Dashboard design inspired by modern data visualization practices 