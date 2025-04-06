import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
import json
import seaborn as sns
from pathlib import Path


def create_directory(directory_path):
    """
    Creates a directory if it does not exist.
    
    Parameters:
    -----------
    directory_path : str
        Path of the directory to create
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")


def test_stationarity(series, plot=True):
    """
    Tests the stationarity of a time series using the Augmented Dickey-Fuller test
    
    Parameters:
    -----------
    series : pd.Series
        Time series to test for stationarity
    plot : bool
        Whether to plot rolling statistics
        
    Returns:
    --------
    bool
        True if the series is stationary, False otherwise
    """
    print('\nResults of Dickey-Fuller Test:')
    
    # Set default value for the return in case of exception
    is_stationary = False
    
    try:
        # Perform the Augmented Dickey-Fuller test
        dftest = adfuller(series.dropna())
        
        # Extract and print test results
        dfoutput = pd.Series(
            dftest[0:4],
            index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
        )
        
        for key, value in dftest[4].items():
            dfoutput[f'Critical Value ({key})'] = value
        
        print(dfoutput)
        
        # Determine if the series is stationary
        is_stationary = dftest[1] <= 0.05
        
        if is_stationary:
            print("Result: The series is stationary (reject the null hypothesis)")
        else:
            print("Result: The series is not stationary (fail to reject the null hypothesis)")
        
        # Plot rolling statistics if requested
        if plot:
            plt.figure(figsize=(12, 8))
            
            # Original series
            plt.subplot(211)
            plt.plot(series, label='Original')
            plt.title('Original Time Series')
            plt.legend()
            
            # Rolling statistics
            plt.subplot(212)
            plt.plot(series.rolling(window=12).mean(), label='Rolling Mean')
            plt.plot(series.rolling(window=12).std(), label='Rolling Std')
            plt.title('Rolling Mean & Standard Deviation')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('stationarity_test.png')
            plt.close()
    
    except Exception as e:
        print(f"Error testing stationarity: {str(e)}")
    
    return is_stationary


def plot_acf_pacf(series, lags=40):
    """
    Plots the ACF and PACF for a time series
    
    Parameters:
    -----------
    series : pd.Series
        Time series to plot ACF and PACF for
    lags : int
        Number of lags to include in the plots
    """
    try:
        plt.figure(figsize=(12, 8))
        
        # ACF plot
        plt.subplot(211)
        plot_acf(series.dropna(), ax=plt.gca(), lags=lags)
        plt.title('Autocorrelation Function')
        
        # PACF plot
        plt.subplot(212)
        plot_pacf(series.dropna(), ax=plt.gca(), lags=lags)
        plt.title('Partial Autocorrelation Function')
        
        plt.tight_layout()
        plt.savefig('acf_pacf.png')
        plt.close()
        
        print("ACF and PACF plots generated successfully")
    
    except Exception as e:
        print(f"Error plotting ACF and PACF: {str(e)}")


def evaluate_forecast(actual, predicted, model_name, output_path='.'):
    """
    Evaluates forecast performance using multiple metrics
    
    Parameters:
    -----------
    actual : pd.Series
        Actual values
    predicted : pd.Series
        Predicted values
    model_name : str
        Name of the model
    output_path : str
        Path to save the evaluation plot
    
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    try:
        # Calculate metrics
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # Print metrics
        print(f'\nEvaluation metrics for {model_name}:')
        print(f'MAE: {mae:.4f}')
        print(f'MSE: {mse:.4f}')
        print(f'RMSE: {rmse:.4f}')
        print(f'R²: {r2:.4f}')
        print(f'MAPE: {mape:.4f}%')
        
        # Plot actual vs predicted
        plt.figure(figsize=(12, 6))
        plt.plot(actual.index, actual, 'b-', label='Actual')
        plt.plot(predicted.index, predicted, 'r--', label='Predicted')
        plt.title(f'Actual vs Predicted - {model_name}')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        create_directory(output_path)
        plt.savefig(os.path.join(output_path, f'{model_name}_evaluation.png'))
        plt.close()
        
        return {
            'model': model_name,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
    
    except Exception as e:
        print(f"Error evaluating forecast: {str(e)}")
        return None


def plot_forecast(historical_data, forecast_data, model_name, output_path='.'):
    """
    Plots historical data alongside forecasted data
    
    Parameters:
    -----------
    historical_data : pd.Series
        Historical time series data
    forecast_data : pd.Series
        Forecasted time series data
    model_name : str
        Name of the model
    output_path : str
        Path to save the forecast plot
    """
    try:
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(historical_data.index, historical_data, 'b-', label='Historical Data')
        
        # Plot forecasted data
        plt.plot(forecast_data.index, forecast_data, 'r--', label='Forecast')
        
        # Add a vertical line to separate historical data from forecast
        last_date = historical_data.index[-1]
        plt.axvline(x=last_date, color='k', linestyle='-', alpha=0.3)
        
        # Add labels and title
        plt.title(f'Time Series Forecast - {model_name}')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        create_directory(output_path)
        plt.savefig(os.path.join(output_path, f'{model_name}_forecast.png'))
        plt.close()
        
        print(f"Forecast plot generated for {model_name}")
    
    except Exception as e:
        print(f"Error plotting forecast: {str(e)}")


def add_time_features(df):
    """
    Adds time-based features to a DataFrame with a datetime index
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime index
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional time features
    """
    # Create a copy of the dataframe
    df_new = df.copy()
    
    # Check if index is datetime
    if not isinstance(df_new.index, pd.DatetimeIndex):
        print("Warning: Index is not DatetimeIndex. Time features not added.")
        return df_new
    
    # Add year, month, day of week features
    df_new['year'] = df_new.index.year
    df_new['month'] = df_new.index.month
    df_new['day_of_week'] = df_new.index.dayofweek
    df_new['day_of_year'] = df_new.index.dayofyear
    
    # Add quarter and is_weekend
    df_new['quarter'] = df_new.index.quarter
    df_new['is_weekend'] = df_new['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Add sin and cos features for cyclical features
    df_new['month_sin'] = np.sin(2 * np.pi * df_new['month'] / 12)
    df_new['month_cos'] = np.cos(2 * np.pi * df_new['month'] / 12)
    df_new['day_of_week_sin'] = np.sin(2 * np.pi * df_new['day_of_week'] / 7)
    df_new['day_of_week_cos'] = np.cos(2 * np.pi * df_new['day_of_week'] / 7)
    
    return df_new


def add_lag_features(df, target_column, n_lags=7):
    """
    Adds lag features to dataframe
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime index
    target_column : str
        Name of the target column
    n_lags : int
        Number of lags to add
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added lag features
    """
    # Create a copy of the dataframe
    df_new = df.copy()
    
    # Check if target column exists
    if target_column not in df_new.columns:
        print(f"Warning: Target column '{target_column}' not found in dataframe. Lag features not added.")
        return df_new
    
    # Add lag features
    for lag in range(1, n_lags + 1):
        df_new[f'{target_column}_lag_{lag}'] = df_new[target_column].shift(lag)
    
    # Add rolling mean features
    for window in [7, 14, 30]:
        if len(df_new) >= window:
            df_new[f'{target_column}_rolling_mean_{window}'] = df_new[target_column].rolling(window=window).mean()
    
    return df_new


def plot_time_series(df, column, title, output_path=None):
    """
    Plot time series data
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime index
    column : str
        Name of the column to plot
    title : str
        Title of the plot
    output_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[column])
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def plot_forecast(actual, forecast, title, output_path=None):
    """
    Plot actual vs forecast values
    
    Parameters:
    -----------
    actual : pd.Series
        Actual values with datetime index
    forecast : pd.Series
        Forecast values with datetime index
    title : str
        Title of the plot
    output_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual, label='Actual')
    plt.plot(forecast.index, forecast, label='Forecast', linestyle='--')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def calculate_metrics(actual, forecast):
    """
    Calculate error metrics for forecast
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    forecast : array-like
        Forecast values
        
    Returns:
    --------
    dict
        Dictionary with error metrics
    """
    import sklearn.metrics as metrics
    
    actual = np.array(actual)
    forecast = np.array(forecast)
    
    # Handle case when arrays are different lengths
    min_len = min(len(actual), len(forecast))
    actual = actual[:min_len]
    forecast = forecast[:min_len]
    
    # Calculate metrics
    mae = metrics.mean_absolute_error(actual, forecast)
    mse = metrics.mean_squared_error(actual, forecast)
    rmse = np.sqrt(mse)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    mask = actual != 0
    mape = np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100
    
    # Calculate SMAPE (Symmetric Mean Absolute Percentage Error)
    smape = np.mean(2 * np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast))) * 100
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'SMAPE': smape
    }


def save_results(metrics, model_name, dataset_name, output_dir):
    """
    Save model metrics to JSON file
    
    Parameters:
    -----------
    metrics : dict
        Dictionary with error metrics
    model_name : str
        Name of the model
    dataset_name : str
        Name of the dataset
    output_dir : str
        Output directory
    """
    # Create output directory if it doesn't exist
    create_directory(output_dir)
    
    # Create results file path
    results_file = os.path.join(output_dir, f'{model_name}_metrics.json')
    
    # Save metrics to JSON file
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Results saved to {results_file}")


def validate_dataset(df, date_col=None, target_col=None):
    """
    Validate a dataset for time series forecasting
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
    date_col : str, optional
        Name of the date column
    target_col : str, optional
        Name of the target column
        
    Returns:
    --------
    dict
        Dictionary with validation results and suggestions
    """
    validation_results = {
        "is_valid": True,
        "warnings": [],
        "errors": [],
        "suggestions": [],
        "statistics": {}
    }
    
    # Check if DataFrame is empty
    if df.empty:
        validation_results["is_valid"] = False
        validation_results["errors"].append("DataFrame is empty")
        return validation_results
    
    # Detect date and target columns if not provided
    if date_col is None:
        date_candidates = []
        for col in df.columns:
            # Check if column name suggests a date
            if any(keyword in col.lower() for keyword in ['date', 'time', 'day', 'year', 'month']):
                date_candidates.append(col)
            # Try to convert to datetime
            try:
                pd.to_datetime(df[col], errors='raise')
                if col not in date_candidates:
                    date_candidates.append(col)
            except:
                pass
        
        if len(date_candidates) > 0:
            date_col = date_candidates[0]
            validation_results["suggestions"].append(f"Auto-detected date column: '{date_col}'")
        else:
            validation_results["is_valid"] = False
            validation_results["errors"].append("No date column found or provided")
            return validation_results
    
    # Validate date column
    if date_col not in df.columns:
        validation_results["is_valid"] = False
        validation_results["errors"].append(f"Date column '{date_col}' not found in dataset")
        return validation_results
    
    # Try to convert date column to datetime
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors='raise')
    except Exception as e:
        validation_results["is_valid"] = False
        validation_results["errors"].append(f"Could not convert column '{date_col}' to datetime: {str(e)}")
        return validation_results
    
    # Check for duplicate dates
    if df[date_col].duplicated().any():
        duplicate_count = df[date_col].duplicated().sum()
        validation_results["warnings"].append(f"Found {duplicate_count} duplicate dates in '{date_col}'")
        validation_results["suggestions"].append("Consider aggregating data by date or using a different time granularity")
    
    # Check date range and frequency
    date_range = pd.date_range(start=df[date_col].min(), end=df[date_col].max())
    missing_dates_count = len(date_range) - len(df[date_col].unique())
    if missing_dates_count > 0:
        missing_pct = (missing_dates_count / len(date_range)) * 100
        validation_results["warnings"].append(f"Missing {missing_dates_count} dates ({missing_pct:.2f}% of time range)")
        
        if missing_pct > 20:
            validation_results["suggestions"].append("High percentage of missing dates. Consider using a different time frequency or imputation methods")
    
    # Detect or validate target column
    if target_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter out likely non-target columns
        exclude_patterns = ['id', 'year', 'month', 'day', 'week', 'index', 'code']
        filtered_cols = [col for col in numeric_cols if not any(pattern in col.lower() for pattern in exclude_patterns)]
        
        if filtered_cols:
            # Choose column with highest variance as target
            variances = df[filtered_cols].var()
            target_col = variances.idxmax()
            validation_results["suggestions"].append(f"Auto-detected target column: '{target_col}'")
        elif numeric_cols:
            target_col = numeric_cols[0]
            validation_results["suggestions"].append(f"Auto-detected target column: '{target_col}'")
        else:
            validation_results["is_valid"] = False
            validation_results["errors"].append("No numeric columns found for forecasting")
            return validation_results
    
    # Validate target column
    if target_col not in df.columns:
        validation_results["is_valid"] = False
        validation_results["errors"].append(f"Target column '{target_col}' not found in dataset")
        return validation_results
    
    # Check target column data type
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        validation_results["is_valid"] = False
        validation_results["errors"].append(f"Target column '{target_col}' is not numeric")
        return validation_results
    
    # Check for missing values in target
    missing_values = df[target_col].isnull().sum()
    if missing_values > 0:
        missing_pct = (missing_values / len(df)) * 100
        validation_results["warnings"].append(f"Target column '{target_col}' has {missing_values} missing values ({missing_pct:.2f}%)")
        
        if missing_pct > 20:
            validation_results["suggestions"].append("High percentage of missing target values. Consider imputation or filtering")
    
    # Calculate basic statistics
    validation_results["statistics"] = {
        "rows": len(df),
        "start_date": df[date_col].min().strftime('%Y-%m-%d'),
        "end_date": df[date_col].max().strftime('%Y-%m-%d'),
        "date_range_days": (df[date_col].max() - df[date_col].min()).days,
        "target_mean": df[target_col].mean(),
        "target_std": df[target_col].std(),
        "target_min": df[target_col].min(),
        "target_max": df[target_col].max(),
        "missing_dates_percentage": (missing_dates_count / len(date_range)) * 100 if len(date_range) > 0 else 0,
        "missing_target_percentage": (missing_values / len(df)) * 100
    }
    
    # Check for sufficient data
    if len(df) < 30:
        validation_results["warnings"].append(f"Dataset has only {len(df)} rows, which may be insufficient for reliable forecasting")
        validation_results["suggestions"].append("Consider using a higher frequency data or collecting more data")
    
    # Check for outliers
    z_scores = np.abs((df[target_col] - df[target_col].mean()) / df[target_col].std())
    outliers_count = (z_scores > 3).sum()
    if outliers_count > 0:
        outlier_pct = (outliers_count / len(df)) * 100
        validation_results["warnings"].append(f"Detected {outliers_count} potential outliers ({outlier_pct:.2f}%) in target column")
        
        if outlier_pct > 5:
            validation_results["suggestions"].append("Consider treating outliers using capping or transformation")
    
    return validation_results


def validate_and_report(df, date_col=None, target_col=None, output_dir=None):
    """
    Validate dataset and generate validation report
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
    date_col : str, optional
        Name of the date column
    target_col : str, optional
        Name of the target column
    output_dir : str, optional
        Directory to save validation report
        
    Returns:
    --------
    tuple
        (bool, dict) indicating if validation passed and validation results
    """
    # Run validation
    validation_results = validate_dataset(df, date_col, target_col)
    
    # Print validation summary
    print("\nDataset Validation Summary:")
    print("=" * 40)
    
    if validation_results["is_valid"]:
        print("✅ Basic validation PASSED")
    else:
        print("❌ Validation FAILED")
    
    if validation_results["errors"]:
        print("\nErrors:")
        for error in validation_results["errors"]:
            print(f"  ❌ {error}")
    
    if validation_results["warnings"]:
        print("\nWarnings:")
        for warning in validation_results["warnings"]:
            print(f"  ⚠️ {warning}")
    
    if validation_results["suggestions"]:
        print("\nSuggestions:")
        for suggestion in validation_results["suggestions"]:
            print(f"  💡 {suggestion}")
    
    if validation_results["statistics"]:
        print("\nStatistics:")
        for key, value in validation_results["statistics"].items():
            print(f"  • {key}: {value}")
    
    # Save validation report if output directory is provided
    if output_dir and validation_results["is_valid"]:
        create_directory(output_dir)
        report_path = os.path.join(output_dir, "validation_report.json")
        
        with open(report_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        print(f"\nValidation report saved to: {report_path}")
    
    return validation_results["is_valid"], validation_results 