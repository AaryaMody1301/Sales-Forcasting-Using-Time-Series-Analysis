import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
# Make TensorFlow imports optional
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available - LSTM models won't be available")
import warnings
from .utils import create_directory, test_stationarity, plot_acf_pacf, evaluate_forecast, plot_forecast

# Suppress warnings
warnings.filterwarnings('ignore')

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
        
        return train_data, test_data
    
    def train_arima(self, order=(1, 1, 1)):
        """
        Train ARIMA model
        
        Parameters:
        -----------
        order : tuple
            (p, d, q) order for the ARIMA model
            
        Returns:
        --------
        model
            Trained ARIMA model
        """
        print(f"Training ARIMA{order} model...")
        
        model = ARIMA(self.train_data, order=order)
        fitted_model = model.fit()
        
        # Save model
        model_name = f"ARIMA{order}"
        self.models[model_name] = fitted_model
        
        print(f"ARIMA{order} model trained successfully")
        return fitted_model
    
    def train_sarima(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        """
        Train SARIMA model
        
        Parameters:
        -----------
        order : tuple
            (p, d, q) non-seasonal order
        seasonal_order : tuple
            (P, D, Q, s) seasonal order
            
        Returns:
        --------
        model
            Trained SARIMA model
        """
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
    
    def train_ml_model(self, model_type='rf', n_lags=12):
        """
        Train machine learning model
        
        Parameters:
        -----------
        model_type : str
            Type of model ('rf' for RandomForest, 'lr' for LinearRegression)
        n_lags : int
            Number of lag features to use
            
        Returns:
        --------
        model
            Trained ML model
        """
        model_name = f"{model_type.upper()}_Lag{n_lags}"
        print(f"Training {model_name} model...")
        
        # Prepare data with lagged features
        series = self.series.copy()
        df = series.to_frame()
        
        # Create lag features
        for i in range(1, n_lags + 1):
            df[f'lag_{i}'] = df[self.target_col].shift(i)
            
        # Drop rows with NaN values
        df = df.dropna()
        
        # Split data
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        # Prepare features and target
        X_train = train_df.drop(self.target_col, axis=1)
        y_train = train_df[self.target_col]
        X_test = test_df.drop(self.target_col, axis=1)
        y_test = test_df[self.target_col]
        
        # Train model
        if model_type == 'rf':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'lr':
            model = LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        model.fit(X_train, y_train)
        
        # Save model info
        model_data = {
            'model': model,
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'n_lags': n_lags
        }
        self.models[model_name] = model_data
        
        print(f"{model_name} model trained successfully")
        return model
    
    def train_lstm(self, n_steps=30, n_features=1, n_epochs=50, batch_size=32):
        """
        Train LSTM model
        
        Parameters:
        -----------
        n_steps : int
            Number of time steps for input sequences
        n_features : int
            Number of features
        n_epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
            
        Returns:
        --------
        model
            Trained LSTM model
        """
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Cannot train LSTM model.")
            return None
            
        model_name = f"LSTM_Steps{n_steps}_Epochs{n_epochs}"
        print(f"Training {model_name} model...")
        
        # Prepare data
        series = self.series
        values = series.values.reshape(-1, 1)
        
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_values = scaler.fit_transform(values)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_values) - n_steps):
            X.append(scaled_values[i:i+n_steps])
            y.append(scaled_values[i+n_steps])
        X, y = np.array(X), np.array(y)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(n_steps, n_features), return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        # Updated to handle optimizer changes in newer TensorFlow versions
        try:
            from tensorflow.keras.optimizers.legacy import Adam
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        except ImportError:
            # Fallback for older versions
            model.compile(optimizer='adam', loss='mse')
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=n_epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'LSTM Model Training Loss - {model_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_path, f'{model_name}_loss.png'))
        plt.close()
        
        # Save model info
        model_data = {
            'model': model,
            'scaler': scaler,
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'n_steps': n_steps,
            'n_features': n_features
        }
        self.models[model_name] = model_data
        
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