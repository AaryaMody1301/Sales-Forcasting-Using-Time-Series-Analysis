import os
import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# Make TensorFlow imports optional
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available - LSTM models won't be available")

# Make Prophet imports optional
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not available - Prophet models won't be available")

# Make XGBoost imports optional
try:
    import xgboost
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available - XGBoost models won't be available")

import warnings
from .utils import create_directory, test_stationarity, plot_acf_pacf, evaluate_forecast, plot_forecast, add_time_features

# Suppress warnings
warnings.filterwarnings('ignore')

# Model interface classes for standardized forecasting
class BaseModel:
    """Base class for all forecasting models"""
    
    def __init__(self, output_dir='results'):
        """Initialize model with output directory"""
        self.output_dir = output_dir
        self.is_fitted = False
        self.train_data = None
        self.test_data = None
        self.forecast_data = None
        self.metrics = None
        
    def fit(self, data, date_col=None, target_col=None, test_size=0.2):
        """Fit model to data"""
        raise NotImplementedError("Subclasses must implement fit()")
        
    def forecast(self, steps=30):
        """Generate forecast"""
        raise NotImplementedError("Subclasses must implement forecast()")
        
    def evaluate(self):
        """Evaluate model performance"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        if self.metrics is None:
            raise ValueError("No metrics available - model may not have been properly evaluated")
        return self.metrics
        
    def save_model(self, path):
        """Save model to disk"""
        raise NotImplementedError("Subclasses must implement save_model()")
        
    def load_model(self, path):
        """Load model from disk"""
        raise NotImplementedError("Subclasses must implement load_model()")
        
    def save_forecast(self, forecast, name):
        """Save forecast to CSV"""
        if forecast is None:
            raise ValueError("No forecast to save")
        
        # Create forecasts directory
        forecasts_dir = os.path.join(self.output_dir, 'forecasts')
        if not os.path.exists(forecasts_dir):
            os.makedirs(forecasts_dir)
        
        # Save forecast to CSV
        forecast_path = os.path.join(forecasts_dir, f"{name}_forecast.csv")
        forecast.to_csv(forecast_path)
        print(f"Forecast saved to {forecast_path}")
        
    def plot_forecast(self, name):
        """Plot forecast against actual values"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")
        if self.forecast_data is None:
            raise ValueError("No forecast data available")
        
        # Create visualizations directory
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        # Plot forecast
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_data.index, self.train_data, label='Training Data')
        plt.plot(self.test_data.index, self.test_data, label='Test Data')
        plt.plot(self.forecast_data.index, self.forecast_data, label='Forecast', linestyle='--')
        plt.title(f"{name} Forecast")
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = os.path.join(viz_dir, f"{name}_forecast.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Forecast plot saved to {plot_path}")

class ARIMAModel(BaseModel):
    """ARIMA model for time series forecasting"""
    # Implementation details...
    
class SARIMAModel(BaseModel):
    """SARIMA model for time series forecasting"""
    # Implementation details...
    
class ExponentialSmoothingModel(BaseModel):
    """Exponential Smoothing model for time series forecasting"""
    # Implementation details...
    
class ProphetModel(BaseModel):
    """Prophet model for time series forecasting"""
    # Implementation details...
    
class RandomForestModel(BaseModel):
    """Random Forest model for time series forecasting"""
    # Implementation details...
    
class GradientBoostingModel(BaseModel):
    """Gradient Boosting model for time series forecasting"""
    # Implementation details...
    
class LSTMModel(BaseModel):
    """LSTM model for time series forecasting"""
    # Implementation details...
    
class EnsembleModel(BaseModel):
    """Ensemble model for time series forecasting"""
    # Implementation details...

class XGBoostModel(BaseModel):
    """XGBoost model for time series forecasting"""
    
    def __init__(self, output_dir='results'):
        """Initialize XGBoost model with output directory"""
        super().__init__(output_dir)
        self.model = None
        self.scaler = MinMaxScaler()
        
        # Check if XGBoost is available
        if not (('XGBOOST_AVAILABLE' in globals() and XGBOOST_AVAILABLE) or 
                ('xgboost' in sys.modules)):
            raise ImportError("XGBoost is not available. Please install it with 'pip install xgboost'")
        
    def fit(self, data, date_col=None, target_col=None, test_size=0.2):
        """
        Fit XGBoost model to time series data
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with time series data
        date_col : str
            Name of date column (if not index)
        target_col : str
            Name of target column
        test_size : float
            Proportion of data to use for testing
            
        Returns:
        --------
        self
            Fitted model
        """
        # Set up date as index if not already
        df = data.copy()
        if date_col is not None and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        
        # Extract target column if not already a Series
        if isinstance(df, pd.DataFrame):
            if target_col is not None and target_col in df.columns:
                series = df[target_col]
            else:
                series = df.iloc[:, 0]  # Use first column as target
                target_col = df.columns[0]
        else:
            series = df
            
        # Store essential data for later use
        self.target_col = target_col
        self.series = series
        
        # Train/test split
        split_idx = int(len(series) * (1 - test_size))
        self.train_data = series.iloc[:split_idx]
        self.test_data = series.iloc[split_idx:]
        
        # Create features for XGBoost
        X_train, y_train, X_test, y_test = self._prepare_features(self.train_data, self.test_data)
        
        # Fit model
        try:
            self.model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            self.model.fit(X_train, y_train)
            
            # Evaluate on test data
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            
            self.metrics = {
                'RMSE': rmse,
                'MAE': mae,
                'MSE': mse
            }
            
            # Save training data for future use
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
            
            self.is_fitted = True
            
            print(f"XGBoost model fitted successfully")
            print(f"RMSE on test set: {rmse:.4f}")
            print(f"MAE on test set: {mae:.4f}")
            
            return self
            
        except Exception as e:
            print(f"Error fitting XGBoost model: {str(e)}")
            print("Falling back to RandomForest model")
            
            # Fallback to RandomForest
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Evaluate on test data
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            
            self.metrics = {
                'RMSE': rmse,
                'MAE': mae,
                'MSE': mse
            }
            
            # Save training data for future use
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
            
            self.is_fitted = True
            
            print(f"RandomForest model (fallback) fitted successfully")
            print(f"RMSE on test set: {rmse:.4f}")
            print(f"MAE on test set: {mae:.4f}")
            
            return self
    
    def _prepare_features(self, train_data, test_data=None):
        """
        Prepare features for XGBoost model
        
        Parameters:
        -----------
        train_data : pd.Series
            Training data
        test_data : pd.Series, optional
            Test data
            
        Returns:
        --------
        tuple
            (X_train, y_train, X_test, y_test)
        """
        # Create lag features
        def create_features(data, n_lags=12):
            df = data.copy()
            df = pd.DataFrame(df)
            
            # Add lag features
            for i in range(1, n_lags+1):
                df[f'lag_{i}'] = df[self.target_col].shift(i)
                
            # Add date features
            df['month'] = df.index.month
            df['dayofyear'] = df.index.dayofyear
            df['dayofweek'] = df.index.dayofweek
            
            # Add cyclical features
            df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
            df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
            df['day_sin'] = np.sin(2 * np.pi * df['dayofyear']/365)
            df['day_cos'] = np.cos(2 * np.pi * df['dayofyear']/365)
            
            # Drop NaN values
            df.dropna(inplace=True)
            
            # Split X and y
            X = df.drop(self.target_col, axis=1)
            y = df[self.target_col]
            
            return X, y
        
        # Create features for training data
        X_train, y_train = create_features(train_data)
        
        # Create features for test data if provided
        if test_data is not None:
            X_test, y_test = create_features(test_data)
            return X_train, y_train, X_test, y_test
        else:
            return X_train, y_train, None, None
    
    def forecast(self, steps=30):
        """
        Generate forecast
        
        Parameters:
        -----------
        steps : int
            Number of steps to forecast
            
        Returns:
        --------
        pd.Series
            Forecast
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        # Get last known data points
        last_date = self.series.index[-1]
        last_values = self.series.tail(20).values
        
        # Create forecast dataframe
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps)
        forecast_values = np.zeros(steps)
        
        # Iteratively forecast each step
        for i in range(steps):
            # Create features for current forecast step
            X = self._create_forecast_features(last_values, forecast_values[:i], last_date, forecast_dates[:i])
            
            # Predict the next value
            forecast_values[i] = self.model.predict(X.reshape(1, -1))[0]
            
        # Create forecast series
        forecast = pd.Series(forecast_values, index=forecast_dates)
        self.forecast_data = forecast
        
        return forecast
    
    def _create_forecast_features(self, last_values, forecasted_values, last_date, forecast_dates):
        """
        Create features for forecasting
        
        Parameters:
        -----------
        last_values : array
            Last known values
        forecasted_values : array
            Already forecasted values
        last_date : datetime
            Last known date
        forecast_dates : array
            Dates for which forecasts have been made
            
        Returns:
        --------
        array
            Features for next forecast step
        """
        # Combine last known values with already forecasted values
        all_values = np.concatenate([last_values, forecasted_values])
        
        # Get last 12 values for lag features
        lag_values = all_values[-12:]
        
        # Get date for next forecast
        next_date = last_date + pd.Timedelta(days=len(forecasted_values)+1) if not len(forecast_dates) else forecast_dates[-1] + pd.Timedelta(days=1)
        
        # Create date features
        month = next_date.month
        dayofyear = next_date.dayofyear
        dayofweek = next_date.dayofweek
        
        # Create cyclical features
        month_sin = np.sin(2 * np.pi * month/12)
        month_cos = np.cos(2 * np.pi * month/12)
        day_sin = np.sin(2 * np.pi * dayofyear/365)
        day_cos = np.cos(2 * np.pi * dayofyear/365)
        
        # Combine features
        features = np.concatenate([
            lag_values[::-1],  # Reverse to get lag_1, lag_2, etc.
            [month, dayofyear, dayofweek, month_sin, month_cos, day_sin, day_cos]
        ])
        
        return features
    
    def save_model(self, path):
        """
        Save model to disk
        
        Parameters:
        -----------
        path : str
            Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model using pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'target_col': self.target_col,
                'metrics': self.metrics,
                'is_fitted': self.is_fitted
            }, f)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load model from disk
        
        Parameters:
        -----------
        path : str
            Path to load model from
            
        Returns:
        --------
        self
            Loaded model
        """
        # Load model using pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.target_col = data['target_col']
        self.metrics = data['metrics']
        self.is_fitted = data['is_fitted']
        
        print(f"Model loaded from {path}")
        return self

class TimeSeriesForecasting:
    """
    A class for time series forecasting using various models
    """
    
    def __init__(self, output_path='models'):
        """
        Initialize the forecasting class
        
        Parameters:
        -----------
        output_path : str
            Path to save trained models and results
        """
        self.output_path = output_path
        create_directory(output_path)
        self.models = {}
        self.results = {}
        self.forecasts = {}
        self.hyperparams = {}
        
    def load_data(self, data_path, date_col=None, target_col=None):
        """
        Load and prepare data for forecasting
        
        Parameters:
        -----------
        data_path : str
            Path to the data file (CSV)
        date_col : str, optional
            Date column name (if not already an index)
        target_col : str
            Target column to forecast
            
        Returns:
        --------
        pd.Series
            Time series to forecast
        """
        print(f"Loading data from {data_path}...")
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Set date as index if not already
        if date_col is not None and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        
        # Sort index
        df.sort_index(inplace=True)
        
        # Extract target series
        if target_col is not None and target_col in df.columns:
            series = df[target_col]
        else:
            # If no target column specified, use the first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                series = df[numeric_cols[0]]
                print(f"No target column specified. Using {numeric_cols[0]} as target.")
            else:
                raise ValueError("No numeric column found in the dataset.")
        
        self.data = df
        self.target_col = target_col if target_col else numeric_cols[0]
        self.series = series
        
        # Check stationarity
        self.is_stationary = test_stationarity(series)
        
        # Plot ACF and PACF
        plot_acf_pacf(series)
        
        # Add time features to the data for better forecasting
        self.data_with_features = add_time_features(df)
        
        print(f"Data loaded. Series length: {len(series)}")
        print(f"Date range: {series.index.min()} to {series.index.max()}")
        
        return series
    
    def split_data(self, test_size=0.2):
        """
        Split data into training and testing sets
        
        Parameters:
        -----------
        test_size : float
            Proportion of data to use for testing
            
        Returns:
        --------
        tuple
            (train_data, test_data)
        """
        series = self.series
        split_idx = int(len(series) * (1 - test_size))
        train_data = series.iloc[:split_idx]
        test_data = series.iloc[split_idx:]
        
        print(f"Data split: {len(train_data)} training samples, {len(test_data)} testing samples")
        
        self.train_data = train_data
        self.test_data = test_data
        
        # Also split the dataframe with features
        if hasattr(self, 'data_with_features'):
            self.train_df = self.data_with_features.iloc[:split_idx]
            self.test_df = self.data_with_features.iloc[split_idx:]
        
        return train_data, test_data
    
    def tune_arima_parameters(self, p_range=(0, 3), d_range=(0, 2), q_range=(0, 3)):
        """
        Tune ARIMA model parameters using grid search
        
        Parameters:
        -----------
        p_range : tuple
            Range of p values to try (min, max+1)
        d_range : tuple
            Range of d values to try (min, max+1)
        q_range : tuple
            Range of q values to try (min, max+1)
            
        Returns:
        --------
        tuple
            Best (p, d, q) parameters
        """
        print("Tuning ARIMA parameters...")
        
        best_aic = float('inf')
        best_order = None
        
        # Create grid of (p,d,q) parameters
        ps = range(p_range[0], p_range[1])
        ds = range(d_range[0], d_range[1])
        qs = range(q_range[0], q_range[1])
        
        for p in ps:
            for d in ds:
                for q in qs:
                    try:
                        model = ARIMA(self.train_data, order=(p, d, q))
                        results = model.fit()
                        aic = results.aic
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)
                            print(f"New best ARIMA parameters found: {best_order} with AIC: {best_aic}")
                    except:
                        continue
        
        print(f"Best ARIMA parameters: {best_order}")
        self.hyperparams['arima'] = {'order': best_order}
        return best_order
    
    def train_arima(self, order=None):
        """
        Train ARIMA model
        
        Parameters:
        -----------
        order : tuple or None
            (p, d, q) order for the ARIMA model
            If None, use tuned parameters or default (1,1,1)
            
        Returns:
        --------
        model
            Trained ARIMA model
        """
        if order is None:
            if 'arima' in self.hyperparams and 'order' in self.hyperparams['arima']:
                order = self.hyperparams['arima']['order']
            else:
                order = (1, 1, 1)
                
        print(f"Training ARIMA{order} model...")
        
        model = ARIMA(self.train_data, order=order)
        fitted_model = model.fit()
        
        # Save model
        model_name = f"ARIMA{order}"
        self.models[model_name] = fitted_model
        
        print(f"ARIMA{order} model trained successfully")
        return fitted_model
    
    def tune_sarima_parameters(self, p_range=(0, 2), d_range=(0, 2), q_range=(0, 2), 
                              P_range=(0, 2), D_range=(0, 1), Q_range=(0, 2), s_values=[12]):
        """
        Tune SARIMA model parameters
        
        Parameters:
        -----------
        p_range, d_range, q_range : tuple
            Ranges for non-seasonal parameters
        P_range, D_range, Q_range : tuple
            Ranges for seasonal parameters
        s_values : list
            Possible seasonal periods to try
            
        Returns:
        --------
        tuple
            Best ((p,d,q), (P,D,Q,s)) parameters
        """
        print("Tuning SARIMA parameters...")
        
        best_aic = float('inf')
        best_order = None
        best_seasonal_order = None
        
        ps = range(p_range[0], p_range[1])
        ds = range(d_range[0], d_range[1])
        qs = range(q_range[0], q_range[1])
        Ps = range(P_range[0], P_range[1])
        Ds = range(D_range[0], D_range[1])
        Qs = range(Q_range[0], Q_range[1])
        
        total_combinations = len(ps) * len(ds) * len(qs) * len(Ps) * len(Ds) * len(Qs) * len(s_values)
        print(f"Testing {total_combinations} parameter combinations...")
        
        # Try a subset of combinations to avoid too much computation
        for p in ps:
            for d in ds:
                for q in qs:
                    for P in Ps:
                        for D in Ds:
                            for Q in Qs:
                                for s in s_values:
                                    try:
                                        model = SARIMAX(
                                            self.train_data,
                                            order=(p, d, q),
                                            seasonal_order=(P, D, Q, s),
                                            enforce_stationarity=False,
                                            enforce_invertibility=False
                                        )
                                        results = model.fit(disp=False)
                                        aic = results.aic
                                        
                                        if aic < best_aic:
                                            best_aic = aic
                                            best_order = (p, d, q)
                                            best_seasonal_order = (P, D, Q, s)
                                            print(f"New best SARIMA parameters: {best_order}x{best_seasonal_order} with AIC: {best_aic}")
                                    except:
                                        continue
        
        print(f"Best SARIMA parameters: {best_order}x{best_seasonal_order}")
        self.hyperparams['sarima'] = {
            'order': best_order,
            'seasonal_order': best_seasonal_order
        }
        return best_order, best_seasonal_order
    
    def train_sarima(self, order=None, seasonal_order=None):
        """
        Train SARIMA model
        
        Parameters:
        -----------
        order : tuple or None
            (p, d, q) non-seasonal order
        seasonal_order : tuple or None
            (P, D, Q, s) seasonal order
            
        Returns:
        --------
        model
            Trained SARIMA model
        """
        if order is None or seasonal_order is None:
            if 'sarima' in self.hyperparams:
                order = self.hyperparams['sarima'].get('order', (1, 1, 1))
                seasonal_order = self.hyperparams['sarima'].get('seasonal_order', (1, 1, 1, 12))
            else:
                order = (1, 1, 1)
                seasonal_order = (1, 1, 1, 12)
                
        print(f"Training SARIMA{order}x{seasonal_order} model...")
        
        model = SARIMAX(
            self.train_data, 
            order=order, 
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted_model = model.fit(disp=False)
        
        # Save model
        model_name = f"SARIMA{order}x{seasonal_order}"
        self.models[model_name] = fitted_model
        
        print(f"SARIMA{order}x{seasonal_order} model trained successfully")
        return fitted_model
    
    def train_prophet(self, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, 
                      changepoint_prior_scale=0.05, holidays=None):
        """
        Train Facebook Prophet model
        
        Parameters:
        -----------
        yearly_seasonality, weekly_seasonality, daily_seasonality : bool or int
            Seasonality components to include
        changepoint_prior_scale : float
            Controls flexibility of trend
        holidays : pd.DataFrame or None
            Custom holidays dataframe
            
        Returns:
        --------
        model
            Trained Prophet model
        """
        if not PROPHET_AVAILABLE:
            print("Prophet not available. Skipping Prophet model.")
            return None
            
        print("Training Prophet model...")
        
        # Create Prophet input data
        df = pd.DataFrame({
            'ds': self.train_data.index,
            'y': self.train_data.values
        })
        
        # Initialize and train Prophet model
        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            changepoint_prior_scale=changepoint_prior_scale,
            holidays=holidays
        )
        
        # Add additional regressors if available in data
        if hasattr(self, 'train_df'):
            for col in self.train_df.columns:
                if col != self.target_col and not col.startswith('lag_') and col != 'date':
                    # Make sure column is numeric
                    if np.issubdtype(self.train_df[col].dtype, np.number):
                        print(f"Adding regressor: {col}")
                        df[col] = self.train_df[col].values
                        model.add_regressor(col)
        
        # Fit the model
        model.fit(df)
        
        # Save model
        model_name = "Prophet"
        self.models[model_name] = model
        
        print("Prophet model trained successfully")
        return model
        
    def train_exponential_smoothing(self, trend=None, seasonal=None, seasonal_periods=None):
        """
        Train exponential smoothing model
        
        Parameters:
        -----------
        trend : str
            Type of trend component ('add', 'mul', None)
        seasonal : str
            Type of seasonal component ('add', 'mul', None)
        seasonal_periods : int
            Number of periods in a season
            
        Returns:
        --------
        model
            Trained exponential smoothing model
        """
        model_name = f"ETS_{trend}_{seasonal}_{seasonal_periods}"
        print(f"Training {model_name} model...")
        
        model = ExponentialSmoothing(
            self.train_data,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods
        )
        fitted_model = model.fit()
        
        # Save model
        self.models[model_name] = fitted_model
        
        print(f"{model_name} model trained successfully")
        return fitted_model
    
    def train_ml_model(self, model_type='rf', n_lags=12, tune_hyperparams=False):
        """
        Train ML models for time series forecasting
        
        Parameters:
        -----------
        model_type : str
            Type of model ('rf' for RandomForest, 'lr' for LinearRegression, 'gb' for GradientBoosting, 'xgb' for XGBoost)
        n_lags : int
            Number of lag features to use
        tune_hyperparams : bool
            Whether to tune hyperparameters
            
        Returns:
        --------
        object
            Trained model
        """
        print(f"Training {model_type.upper()} model...")
        
        # Prepare data with lag features
        X_train, y_train, X_test, y_test = self._prepare_ml_data(n_lags)
        
        # Select model type
        if model_type.lower() == 'rf':
            if tune_hyperparams:
                # Tune hyperparameters for RandomForest
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                
                grid = GridSearchCV(
                    RandomForestRegressor(random_state=42),
                    param_grid=param_grid,
                    cv=TimeSeriesSplit(n_splits=5),
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                
                grid.fit(X_train, y_train)
                model = grid.best_estimator_
                self.hyperparams['rf'] = grid.best_params_
                print(f"Best parameters: {grid.best_params_}")
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
        
        elif model_type.lower() == 'gb':
            if tune_hyperparams:
                # Tune hyperparameters for GradientBoosting
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10]
                }
                
                grid = GridSearchCV(
                    GradientBoostingRegressor(random_state=42),
                    param_grid=param_grid,
                    cv=TimeSeriesSplit(n_splits=5),
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                
                grid.fit(X_train, y_train)
                model = grid.best_estimator_
                self.hyperparams['gb'] = grid.best_params_
                print(f"Best parameters: {grid.best_params_}")
            else:
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
        
        elif model_type.lower() == 'xgb':
            # Check if XGBoost is available
            if 'XGBOOST_AVAILABLE' in globals() and XGBOOST_AVAILABLE:
                if tune_hyperparams:
                    # Tune hyperparameters for XGBoost
                    param_grid = {
                        'n_estimators': [100, 200, 300],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'max_depth': [3, 5, 7],
                        'subsample': [0.8, 0.9, 1.0]
                    }
                    
                    grid = GridSearchCV(
                        XGBRegressor(random_state=42),
                        param_grid=param_grid,
                        cv=TimeSeriesSplit(n_splits=5),
                        scoring='neg_mean_squared_error',
                        n_jobs=-1
                    )
                    
                    grid.fit(X_train, y_train)
                    model = grid.best_estimator_
                    self.hyperparams['xgb'] = grid.best_params_
                    print(f"Best parameters: {grid.best_params_}")
                else:
                    model = XGBRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
            else:
                print("XGBoost not available. Using RandomForest instead.")
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                model_type = 'rf'  # Update model type for saving and tracking
        
        elif model_type.lower() == 'lr':
            model = LinearRegression()
            model.fit(X_train, y_train)
        
        else:
            print(f"Unknown model type: {model_type}. Using RandomForest.")
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            model_type = 'rf'  # Update model type for saving and tracking
        
        # Evaluate model on test data
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"Model evaluation on test data:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        
        self.models[model_type] = model
        self.results[model_type] = {'rmse': rmse, 'mae': mae}
        
        return model

    def train_lstm(self, n_steps=30, n_features=1, n_epochs=50, batch_size=32, lstm_type='simple'):
        """
        Train LSTM model
        
        Parameters:
        -----------
        n_steps : int
            Number of time steps to look back
        n_features : int
            Number of features
        n_epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        lstm_type : str
            Type of LSTM architecture ('simple', 'bidirectional', 'stacked', 'cnn')
            
        Returns:
        --------
        model
            Trained LSTM model
        """
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Cannot train LSTM model.")
            return None
            
        model_name = f"LSTM_{lstm_type}"
        print(f"Training {model_name} model...")
        
        # Prepare data
        series = self.train_data.values
        
        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(series.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - n_steps):
            X.append(scaled_data[i:i+n_steps, 0])
            y.append(scaled_data[i+n_steps, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Reshape for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], n_features))
        
        # Build LSTM model based on type
        model = Sequential()
        
        if lstm_type == 'simple':
            model.add(LSTM(units=50, return_sequences=False, input_shape=(n_steps, n_features)))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))
            
        elif lstm_type == 'bidirectional':
            model.add(Bidirectional(LSTM(units=50, return_sequences=False), input_shape=(n_steps, n_features)))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))
            
        elif lstm_type == 'stacked':
            model.add(LSTM(units=50, return_sequences=True, input_shape=(n_steps, n_features)))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))
            
        elif lstm_type == 'cnn':
            model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_steps, n_features)))
            model.add(MaxPooling1D(pool_size=2))
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))
            
        else:
            raise ValueError(f"Unknown LSTM type: {lstm_type}")
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        
        # Create callbacks for better training
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
        ]
        
        # Train model with validation split
        history = model.fit(
            X, y,
            epochs=n_epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model and metadata
        self.models[model_name] = {
            'model': model,
            'scaler': scaler,
            'n_steps': n_steps,
            'history': history.history
        }
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{model_name} - Training History')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.output_path, f'{model_name}_training_history.png'))
        plt.close()
        
        print(f"{model_name} model trained successfully")
        return model
    
    def forecast(self, model_name, steps=30):
        """
        Generate forecasts for a trained model
        
        Parameters:
        -----------
        model_name : str
            Name of the trained model
        steps : int
            Number of steps to forecast
            
        Returns:
        --------
        pd.Series
            Forecasted values
        """
        print(f"Generating forecasts for {model_name}...")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Please train the model first.")
        
        model = self.models[model_name]
        
        # Generate forecasts based on model type
        if model_name.startswith("ARIMA") or model_name.startswith("SARIMA"):
            # Statistical model forecast
            forecast = model.forecast(steps=steps)
            
            # Safer frequency inference with fallback
            freq = pd.infer_freq(self.series.index)
            if freq is None:
                # If frequency can't be inferred, use a default daily frequency
                print(f"Warning: Could not infer frequency from time series. Using daily frequency.")
                freq = 'D'
                
            index = pd.date_range(
                start=self.test_data.index[0], 
                periods=steps, 
                freq=freq
            )
            forecast = pd.Series(forecast, index=index)
            
        elif model_name.startswith("ETS"):
            # Exponential smoothing forecast
            forecast = model.forecast(steps=steps)
            
            # Safer frequency inference with fallback
            freq = pd.infer_freq(self.series.index)
            if freq is None:
                # If frequency can't be inferred, use a default daily frequency
                print(f"Warning: Could not infer frequency from time series. Using daily frequency.")
                freq = 'D'
                
            index = pd.date_range(
                start=self.test_data.index[0], 
                periods=steps, 
                freq=freq
            )
            forecast = pd.Series(forecast, index=index)
            
        elif model_name.startswith(("RF", "LR")):
            # ML model forecast
            n_lags = model['n_lags']
            ml_model = model['model']
            
            # Get the last n_lags values for forecasting
            last_values = self.series.iloc[-n_lags:].values
            
            # Generate forecasts iteratively
            forecasts = []
            for _ in range(steps):
                # Prepare input features
                features = last_values[-n_lags:].reshape(1, -1)
                # Make prediction
                pred = ml_model.predict(features)[0]
                # Update last values
                last_values = np.append(last_values[1:], pred)
                # Store prediction
                forecasts.append(pred)
            
            # Create forecast series
            # Safer frequency inference with fallback
            freq = pd.infer_freq(self.series.index)
            if freq is None:
                # If frequency can't be inferred, use a default daily frequency
                print(f"Warning: Could not infer frequency from time series. Using daily frequency.")
                freq = 'D'
                
            index = pd.date_range(
                start=self.series.index[-1] + pd.Timedelta('1D'), 
                periods=steps, 
                freq=freq
            )
            forecast = pd.Series(forecasts, index=index)
            
        elif model_name.startswith("LSTM"):
            # LSTM model forecast
            if not TENSORFLOW_AVAILABLE:
                print(f"TensorFlow not available. Cannot generate forecast for {model_name}.")
                return None
                
            lstm_model = model['model']
            scaler = model['scaler']
            n_steps = model['n_steps']
            
            # Get the last n_steps values for forecasting
            last_sequence = self.series.iloc[-n_steps:].values.reshape(-1, 1)
            last_sequence = scaler.transform(last_sequence)
            
            # Generate forecasts iteratively
            current_sequence = last_sequence.reshape(1, n_steps, 1)
            forecasts = []
            
            for _ in range(steps):
                # Make prediction with verbose=0 to suppress warnings in TensorFlow 2.15+
                pred = lstm_model.predict(current_sequence, verbose=0)[0]
                # Add to forecasts
                forecasts.append(pred)
                # Update current sequence
                current_sequence = np.append(current_sequence[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
            
            # Inverse transform to get original scale
            forecasts = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()
            
            # Create forecast series
            # Safer frequency inference with fallback
            freq = pd.infer_freq(self.series.index)
            if freq is None:
                # If frequency can't be inferred, use a default daily frequency
                print(f"Warning: Could not infer frequency from time series. Using daily frequency.")
                freq = 'D'
                
            index = pd.date_range(
                start=self.series.index[-1] + pd.Timedelta('1D'), 
                periods=steps, 
                freq=freq
            )
            forecast = pd.Series(forecasts, index=index)
        
        else:
            raise ValueError(f"Unknown model type: {model_name}")
        
        # Save forecast
        self.forecasts[model_name] = forecast
        
        # Plot forecast
        plot_forecast(self.series, forecast, model_name, self.output_path)
        
        print(f"Forecast generated for {model_name}")
        return forecast
    
    def evaluate_models(self, steps=30):
        """
        Evaluate all trained models
        
        Parameters:
        -----------
        steps : int
            Number of steps to forecast for evaluation
            
        Returns:
        --------
        pd.DataFrame
            Evaluation metrics for all models
        """
        print("Evaluating models...")
        
        results = []
        
        for model_name in self.models.keys():
            try:
                # Generate forecast
                forecast = self.forecast(model_name, steps=steps)
                
                # Calculate metrics
                if len(self.test_data) >= steps:
                    actual = self.test_data.iloc[:steps]
                else:
                    actual = self.test_data
                
                # Align forecast with actual values
                forecast = forecast[:len(actual)]
                
                rmse = np.sqrt(mean_squared_error(actual, forecast))
                mae = mean_absolute_error(actual, forecast)
                r2 = r2_score(actual, forecast)
                mape = np.mean(np.abs((actual - forecast) / actual)) * 100
                
                # Add to results
                results.append({
                    'Model': model_name,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R²': r2,
                    'MAPE': mape
                })
                
                # Create evaluation plot
                evaluate_forecast(actual, forecast, model_name, self.output_path)
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {str(e)}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        results_df.to_csv(os.path.join(self.output_path, 'model_evaluation.csv'), index=False)
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        
        # Use the new seaborn barplot syntax for compatibility with seaborn 0.13+
        try:
            # New syntax (seaborn 0.13+)
            model_comparison_df = pd.DataFrame({
                'Model': results_df['Model'],
                'RMSE': results_df['RMSE']
            })
            sns.barplot(data=model_comparison_df, x='Model', y='RMSE')
        except TypeError:
            # Fall back to old syntax for older seaborn versions
            sns.barplot(x=results_df['Model'], y=results_df['RMSE'])
            
        plt.title('Model Comparison - RMSE')
        plt.xlabel('Model')
        plt.ylabel('RMSE')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'model_comparison_rmse.png'))
        plt.close()
        
        self.results = results_df
        print("Model evaluation completed")
        
        return results_df
    
    def ensemble_forecast(self, model_names=None, weights=None, steps=30):
        """
        Create an ensemble forecast from multiple models
        
        Parameters:
        -----------
        model_names : list
            List of model names to include in ensemble
        weights : list
            Weights for each model (must sum to 1)
        steps : int
            Number of steps to forecast
            
        Returns:
        --------
        pd.Series
            Ensemble forecast
        """
        print("Creating ensemble forecast...")
        
        # If no models specified, use all models
        if model_names is None:
            model_names = list(self.forecasts.keys())
        
        # If no weights specified, use equal weights
        if weights is None:
            weights = [1/len(model_names)] * len(model_names)
        
        # Ensure weights sum to 1
        weights = np.array(weights) / sum(weights)
        
        # Ensure all models have forecasts
        for model_name in model_names:
            if model_name not in self.forecasts:
                self.forecast(model_name, steps=steps)
        
        # Get forecasts
        forecasts = [self.forecasts[model_name][:steps] for model_name in model_names]
        
        # Check if all forecasts have the same length
        min_length = min(len(forecast) for forecast in forecasts)
        forecasts = [forecast[:min_length] for forecast in forecasts]
        
        # Create ensemble forecast
        ensemble = pd.DataFrame({model_name: forecast.values for model_name, forecast in zip(model_names, forecasts)})
        ensemble['weighted_forecast'] = 0
        
        for i, model_name in enumerate(model_names):
            ensemble['weighted_forecast'] += ensemble[model_name] * weights[i]
        
        # Create Series with proper index
        ensemble_forecast = pd.Series(
            ensemble['weighted_forecast'].values,
            index=forecasts[0].index[:min_length]
        )
        
        # Save ensemble forecast
        self.forecasts['ensemble'] = ensemble_forecast
        
        # Plot ensemble forecast
        plt.figure(figsize=(12, 6))
        for model_name, forecast in zip(model_names, forecasts):
            plt.plot(forecast.index, forecast.values, alpha=0.3, label=model_name)
        
        plt.plot(ensemble_forecast.index, ensemble_forecast.values, 'k-', linewidth=2, label='Ensemble')
        plt.title('Ensemble Forecast')
        plt.xlabel('Date')
        plt.ylabel(self.target_col)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'ensemble_forecast.png'))
        plt.close()
        
        print("Ensemble forecast created")
        return ensemble_forecast
    
    def run_forecasting_pipeline(self, data_path, date_col=None, target_col=None, test_size=0.2, forecast_steps=30):
        """
        Run the complete forecasting pipeline
        
        Parameters:
        -----------
        data_path : str
            Path to the data file
        date_col : str, optional
            Date column name
        target_col : str, optional
            Target column name
        test_size : float
            Proportion of data to use for testing
        forecast_steps : int
            Number of steps to forecast
            
        Returns:
        --------
        dict
            Results of the forecasting pipeline
        """
        print(f"Running forecasting pipeline for {data_path}...")
        
        # Load data
        self.load_data(data_path, date_col, target_col)
        
        # Split data
        self.split_data(test_size)
        
        # Train models
        print("\nTraining models...")
        try:
            self.train_arima(order=(1, 1, 1))
        except Exception as e:
            print(f"Error training ARIMA: {str(e)}")
            
        try:
            self.train_sarima(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        except Exception as e:
            print(f"Error training SARIMA: {str(e)}")
            
        try:
            self.train_exponential_smoothing(trend='add', seasonal=None)
        except Exception as e:
            print(f"Error training ETS: {str(e)}")
            
        try:
            self.train_ml_model(model_type='rf', n_lags=12)
        except Exception as e:
            print(f"Error training RandomForest: {str(e)}")
            
        try:
            self.train_ml_model(model_type='lr', n_lags=12)
        except Exception as e:
            print(f"Error training LinearRegression: {str(e)}")
            
        try:
            self.train_lstm(n_steps=30, n_epochs=50)
        except Exception as e:
            print(f"Error training LSTM: {str(e)}")
        
        # Evaluate models
        print("\nEvaluating models...")
        results = self.evaluate_models(steps=forecast_steps)
        
        # Create ensemble forecast
        print("\nCreating ensemble forecast...")
        ensemble = self.ensemble_forecast(steps=forecast_steps)
        
        # Return results
        pipeline_results = {
            'data': self.data,
            'series': self.series,
            'train_data': self.train_data,
            'test_data': self.test_data,
            'models': self.models,
            'forecasts': self.forecasts,
            'results': results,
            'ensemble': ensemble
        }
        
        print("\nForecasting pipeline completed!")
        return pipeline_results 