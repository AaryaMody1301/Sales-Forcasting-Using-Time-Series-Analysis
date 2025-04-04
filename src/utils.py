import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
    df_new = df.copy()
    
    # Extract time features
    df_new['year'] = df_new.index.year
    df_new['month'] = df_new.index.month
    df_new['day'] = df_new.index.day
    df_new['dayofweek'] = df_new.index.dayofweek
    df_new['quarter'] = df_new.index.quarter
    df_new['dayofyear'] = df_new.index.dayofyear
    
    # In pandas 2.0+, isocalendar() returns a DataFrame with columns year, week, day
    # This ensures compatibility with both older and newer pandas versions
    try:
        isocal = df_new.index.isocalendar()
        if hasattr(isocal, 'week'):  # Older pandas versions
            df_new['weekofyear'] = isocal.week
        else:  # Newer pandas versions (2.0+)
            df_new['weekofyear'] = isocal['week']
    except Exception as e:
        print(f"Warning: Could not calculate week of year: {str(e)}")
        # Fallback: calculate week of year using day of year
        df_new['weekofyear'] = (df_new['dayofyear'] - 1) // 7 + 1
    
    # Add cyclical features for month, day of week, and hour
    # These help the model understand the cyclical nature of time
    df_new['month_sin'] = np.sin(2 * np.pi * df_new['month'] / 12)
    df_new['month_cos'] = np.cos(2 * np.pi * df_new['month'] / 12)
    df_new['dayofweek_sin'] = np.sin(2 * np.pi * df_new['dayofweek'] / 7)
    df_new['dayofweek_cos'] = np.cos(2 * np.pi * df_new['dayofweek'] / 7)
    
    return df_new


def add_lag_features(df, target_column=None, lag_periods=[1, 7, 14, 30], rolling_windows=[7, 14, 30]):
    """
    Adds lag and rolling window features to a DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime index
    target_column : str
        Name of the target column
    lag_periods : list
        List of lag periods to add
    rolling_windows : list
        List of rolling window sizes for average and standard deviation
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional lag features
    """
    df_new = df.copy()
    
    # If no target column is specified, use the first column
    if target_column is None:
        if isinstance(df_new, pd.Series):
            target_column = df_new.name
        else:
            target_column = df_new.columns[0]
    
    # Add lag features
    for lag in lag_periods:
        df_new[f'{target_column}_lag_{lag}'] = df_new[target_column].shift(lag)
    
    # Add rolling window features
    for window in rolling_windows:
        df_new[f'{target_column}_rolling_mean_{window}'] = df_new[target_column].rolling(window=window).mean()
        df_new[f'{target_column}_rolling_std_{window}'] = df_new[target_column].rolling(window=window).std()
    
    return df_new 