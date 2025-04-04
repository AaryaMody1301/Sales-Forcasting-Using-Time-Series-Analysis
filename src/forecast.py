import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime, timedelta
import tensorflow as tf


class SalesForecaster:
    def __init__(self, model_path='models', data_path='data/processed'):
        """
        Initialize the sales forecaster class
        
        Parameters:
        -----------
        model_path : str
            Path to trained models
        data_path : str
            Path to processed data
        """
        self.model_path = model_path
        self.data_path = data_path
        
        # Check if model directory exists
        if not os.path.exists(model_path):
            raise ValueError(f"Model directory {model_path} does not exist")
        
        # Create output directory if it doesn't exist
        output_path = 'forecasts'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        self.output_path = output_path
    
    def load_model(self, model_type):
        """
        Load a trained model
        
        Parameters:
        -----------
        model_type : str
            Type of model to load ('arima', 'sarima', 'prophet', 'lstm')
            
        Returns:
        --------
        model : object
            Loaded model
        """
        print(f"Loading {model_type} model...")
        
        if model_type == 'arima':
            model_path = os.path.join(self.model_path, 'arima_model.pkl')
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        
        elif model_type == 'sarima':
            model_path = os.path.join(self.model_path, 'sarima_model.pkl')
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        
        elif model_type == 'prophet':
            model_path = os.path.join(self.model_path, 'prophet_model.pkl')
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        
        elif model_type == 'lstm':
            model_path = os.path.join(self.model_path, 'lstm_model')
            model = tf.keras.models.load_model(model_path)
            
            # Load scaler
            scaler_path = os.path.join(self.model_path, 'lstm_scaler.pkl')
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            # Load config
            config_path = os.path.join(self.model_path, 'lstm_config.pkl')
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
            
            return model, scaler, config
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model
    
    def load_data(self, filename):
        """
        Load processed data
        
        Parameters:
        -----------
        filename : str
            Name of file to load
            
        Returns:
        --------
        pd.DataFrame
            Loaded data
        """
        file_path = os.path.join(self.data_path, filename)
        print(f"Loading data from {file_path}")
        return pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    def forecast(self, model_type, steps, train_data=None, test_data=None):
        """
        Make forecast using specified model
        
        Parameters:
        -----------
        model_type : str
            Type of model to use ('arima', 'sarima', 'prophet', 'lstm')
        steps : int
            Number of steps to forecast
        train_data : pd.DataFrame or None
            Training data (needed for some models)
        test_data : pd.DataFrame or None
            Test data (needed for reference)
            
        Returns:
        --------
        pd.DataFrame
            Forecast results
        """
        print(f"Making {steps} step forecast using {model_type} model...")
        
        # Load model
        if model_type == 'lstm':
            model, scaler, config = self.load_model(model_type)
        else:
            model = self.load_model(model_type)
        
        # Determine forecast start date
        if test_data is not None:
            # If test data is provided, start forecast from the end of test data
            last_date = test_data.index[-1]
        elif train_data is not None:
            # If only train data is provided, start forecast from the end of train data
            last_date = train_data.index[-1]
        else:
            # If no data is provided, start forecast from today
            last_date = datetime.now().date()
        
        # Create future dates
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=steps,
            freq='D'  # Daily frequency, adjust as needed
        )
        
        # Make forecast
        if model_type == 'arima' or model_type == 'sarima':
            # Get forecast
            forecast = model.forecast(steps=steps)
            forecast_values = forecast.values
            
            # Create DataFrame with dates and forecasted values
            forecast_df = pd.DataFrame({
                'date': future_dates,
                'forecast': forecast_values
            })
            forecast_df.set_index('date', inplace=True)
        
        elif model_type == 'prophet':
            # Create future dataframe for Prophet
            future = pd.DataFrame({'ds': future_dates})
            
            # Make prediction
            forecast = model.predict(future)
            
            # Create DataFrame with dates and forecasted values
            forecast_df = pd.DataFrame({
                'date': future_dates,
                'forecast': forecast['yhat'].values,
                'forecast_lower': forecast['yhat_lower'].values,
                'forecast_upper': forecast['yhat_upper'].values
            })
            forecast_df.set_index('date', inplace=True)
        
        elif model_type == 'lstm':
            sequence_length = config['sequence_length']
            
            # Get last sequence from data
            if train_data is not None and test_data is not None:
                # Use the last sequence from combined data
                combined_data = pd.concat([train_data, test_data])
                last_sequence = combined_data['sales'].values[-sequence_length:].reshape(-1, 1)
            elif train_data is not None:
                # Use the last sequence from train data
                last_sequence = train_data['sales'].values[-sequence_length:].reshape(-1, 1)
            else:
                raise ValueError("LSTM model requires training data to make a forecast")
            
            # Scale the sequence
            last_sequence = scaler.transform(last_sequence)
            
            # Make predictions one step at a time
            predictions = []
            current_seq = last_sequence.reshape(1, sequence_length, 1)
            
            for _ in range(steps):
                # Predict next value
                next_val = model.predict(current_seq)[0, 0]
                predictions.append(next_val)
                
                # Update sequence
                current_seq = np.append(current_seq[:, 1:, :], [[next_val]], axis=1)
            
            # Inverse transform to get actual values
            predictions = np.array(predictions).reshape(-1, 1)
            forecast_values = scaler.inverse_transform(predictions).flatten()
            
            # Create DataFrame with dates and forecasted values
            forecast_df = pd.DataFrame({
                'date': future_dates,
                'forecast': forecast_values
            })
            forecast_df.set_index('date', inplace=True)
        
        # Save forecast
        forecast_path = os.path.join(self.output_path, f'{model_type}_forecast.csv')
        forecast_df.to_csv(forecast_path)
        print(f"Forecast saved to {forecast_path}")
        
        # Plot forecast
        plt.figure(figsize=(12, 6))
        
        # Plot historical data if available
        if train_data is not None:
            plt.plot(train_data.index, train_data['sales'], label='Training Data')
        
        if test_data is not None:
            plt.plot(test_data.index, test_data['sales'], label='Test Data')
        
        # Plot forecast
        plt.plot(forecast_df.index, forecast_df['forecast'], label='Forecast', color='red')
        
        # Add confidence intervals for Prophet
        if model_type == 'prophet' and 'forecast_lower' in forecast_df.columns:
            plt.fill_between(
                forecast_df.index,
                forecast_df['forecast_lower'],
                forecast_df['forecast_upper'],
                color='red',
                alpha=0.2,
                label='95% Confidence Interval'
            )
        
        plt.title(f"{model_type.upper()} Sales Forecast")
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = os.path.join(self.output_path, f'{model_type}_forecast_plot.png')
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        
        return forecast_df
    
    def ensemble_forecast(self, models, steps, train_data=None, test_data=None, weights=None):
        """
        Make ensemble forecast using multiple models
        
        Parameters:
        -----------
        models : list
            List of model types to use in ensemble
        steps : int
            Number of steps to forecast
        train_data : pd.DataFrame or None
            Training data
        test_data : pd.DataFrame or None
            Test data
        weights : list or None
            Weights for each model (if None, equal weights will be used)
            
        Returns:
        --------
        pd.DataFrame
            Ensemble forecast results
        """
        print(f"Making ensemble forecast with models: {models}...")
        
        # Check if weights are provided
        if weights is None:
            weights = [1/len(models)] * len(models)
        
        # Ensure weights sum to 1
        weights = [w/sum(weights) for w in weights]
        
        # Make forecasts using each model
        forecasts = []
        for model_type in models:
            forecast_df = self.forecast(model_type, steps, train_data, test_data)
            forecasts.append(forecast_df)
        
        # Determine forecast start date
        if test_data is not None:
            # If test data is provided, start forecast from the end of test data
            last_date = test_data.index[-1]
        elif train_data is not None:
            # If only train data is provided, start forecast from the end of train data
            last_date = train_data.index[-1]
        else:
            # If no data is provided, start forecast from today
            last_date = datetime.now().date()
        
        # Create future dates
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=steps,
            freq='D'  # Daily frequency, adjust as needed
        )
        
        # Combine forecasts with weighted average
        ensemble_forecast = np.zeros(steps)
        for i, forecast_df in enumerate(forecasts):
            ensemble_forecast += weights[i] * forecast_df['forecast'].values
        
        # Create DataFrame with dates and forecasted values
        ensemble_df = pd.DataFrame({
            'date': future_dates,
            'forecast': ensemble_forecast
        })
        ensemble_df.set_index('date', inplace=True)
        
        # Save ensemble forecast
        forecast_path = os.path.join(self.output_path, 'ensemble_forecast.csv')
        ensemble_df.to_csv(forecast_path)
        print(f"Ensemble forecast saved to {forecast_path}")
        
        # Plot ensemble forecast
        plt.figure(figsize=(12, 6))
        
        # Plot historical data if available
        if train_data is not None:
            plt.plot(train_data.index, train_data['sales'], label='Training Data')
        
        if test_data is not None:
            plt.plot(test_data.index, test_data['sales'], label='Test Data')
        
        # Plot individual forecasts
        for i, (model_type, forecast_df) in enumerate(zip(models, forecasts)):
            plt.plot(forecast_df.index, forecast_df['forecast'], 
                     label=f'{model_type.upper()} Forecast', 
                     alpha=0.5, linestyle='--')
        
        # Plot ensemble forecast
        plt.plot(ensemble_df.index, ensemble_df['forecast'], 
                 label='Ensemble Forecast', 
                 color='black', linewidth=2)
        
        plt.title("Ensemble Sales Forecast")
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = os.path.join(self.output_path, 'ensemble_forecast_plot.png')
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        
        return ensemble_df


if __name__ == "__main__":
    # Example usage
    data_path = 'data/processed'
    train_file = 'train_processed_sales_data.csv'
    test_file = 'test_processed_sales_data.csv'
    
    # Check if processed data exists
    train_path = os.path.join(data_path, train_file)
    test_path = os.path.join(data_path, test_file)
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Processed data not found at {data_path}")
        print("Please run data_preparation.py first")
        exit(1)
    
    # Load data
    print(f"Loading train data from {train_path}")
    train_data = pd.read_csv(train_path, index_col=0, parse_dates=True)
    
    print(f"Loading test data from {test_path}")
    test_data = pd.read_csv(test_path, index_col=0, parse_dates=True)
    
    # Initialize forecaster
    forecaster = SalesForecaster()
    
    # Define forecast horizon
    forecast_steps = 30  # 30 days ahead
    
    # Define models to use
    models_to_use = ['arima', 'sarima', 'prophet', 'lstm']
    
    # Make individual forecasts
    for model_type in models_to_use:
        forecaster.forecast(model_type, forecast_steps, train_data, test_data)
    
    # Make ensemble forecast
    # You can adjust the weights based on model performance
    ensemble_weights = [0.2, 0.3, 0.3, 0.2]  # Example weights
    forecaster.ensemble_forecast(models_to_use, forecast_steps, train_data, test_data, ensemble_weights)
    
    print("Forecasting completed") 