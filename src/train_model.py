import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


class TimeSeriesModel:
    def __init__(self, model_path='models'):
        """
        Initialize the time series model class
        
        Parameters:
        -----------
        model_path : str
            Path to save trained models
        """
        self.model_path = model_path
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_path):
            os.makedirs(model_path)
    
    def train_arima(self, train_data, order=(5, 1, 0), target_column='sales'):
        """
        Train ARIMA model
        
        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data
        order : tuple
            (p, d, q) order for ARIMA model
        target_column : str
            Column name for the target variable
            
        Returns:
        --------
        model : ARIMA
            Trained ARIMA model
        """
        print(f"Training ARIMA model with order {order}...")
        
        # Get target variable
        y = train_data[target_column]
        
        # Train ARIMA model
        model = ARIMA(y, order=order)
        model_fit = model.fit()
        
        # Save model
        model_path = os.path.join(self.model_path, 'arima_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_fit, f)
        
        print(f"ARIMA model saved to {model_path}")
        
        return model_fit
    
    def train_sarima(self, train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), target_column='sales'):
        """
        Train SARIMA model
        
        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data
        order : tuple
            (p, d, q) order for the non-seasonal part
        seasonal_order : tuple
            (P, D, Q, s) order for the seasonal part
        target_column : str
            Column name for the target variable
            
        Returns:
        --------
        model : SARIMAX
            Trained SARIMA model
        """
        print(f"Training SARIMA model with order {order} and seasonal order {seasonal_order}...")
        
        # Get target variable
        y = train_data[target_column]
        
        # Train SARIMA model
        model = SARIMAX(y, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
        
        # Save model
        model_path = os.path.join(self.model_path, 'sarima_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_fit, f)
        
        print(f"SARIMA model saved to {model_path}")
        
        return model_fit
    
    def train_prophet(self, train_data, target_column='sales'):
        """
        Train Prophet model
        
        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data with datetime index
        target_column : str
            Column name for the target variable
            
        Returns:
        --------
        model : Prophet
            Trained Prophet model
        """
        print("Training Prophet model...")
        
        # Reset index to get date as column
        df = train_data.reset_index()
        
        # Rename columns to match Prophet requirements
        df = df.rename(columns={'date': 'ds', target_column: 'y'})
        
        # Train Prophet model
        model = Prophet()
        model.fit(df)
        
        # Save model
        model_path = os.path.join(self.model_path, 'prophet_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"Prophet model saved to {model_path}")
        
        return model
    
    def train_lstm(self, train_data, target_column='sales', sequence_length=10, epochs=50, batch_size=32):
        """
        Train LSTM model
        
        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data
        target_column : str
            Column name for the target variable
        sequence_length : int
            Number of time steps to look back
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
            
        Returns:
        --------
        model : Sequential
            Trained LSTM model
        scaler : object
            Scaler used for normalizing data
        """
        print("Training LSTM model...")
        
        from sklearn.preprocessing import MinMaxScaler
        
        # Get target variable
        data = train_data[target_column].values.reshape(-1, 1)
        
        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(len(data_scaled) - sequence_length):
            X.append(data_scaled[i:i + sequence_length, 0])
            y.append(data_scaled[i + sequence_length, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Reshape X to [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        
        # Compile model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train model
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
        
        # Save model
        model_path = os.path.join(self.model_path, 'lstm_model')
        model.save(model_path)
        
        # Save scaler
        scaler_path = os.path.join(self.model_path, 'lstm_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        print(f"LSTM model saved to {model_path}")
        print(f"LSTM scaler saved to {scaler_path}")
        
        # Save sequence length
        config_path = os.path.join(self.model_path, 'lstm_config.pkl')
        config = {'sequence_length': sequence_length}
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
        
        return model, scaler
    
    def evaluate_model(self, model_type, model, test_data, target_column='sales', forecast_steps=None):
        """
        Evaluate model performance on test data
        
        Parameters:
        -----------
        model_type : str
            Type of model ('arima', 'sarima', 'prophet', 'lstm')
        model : object
            Trained model
        test_data : pd.DataFrame
            Test data
        target_column : str
            Column name for the target variable
        forecast_steps : int or None
            Number of steps to forecast (None for all test data)
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        print(f"Evaluating {model_type} model...")
        
        # Get actual values
        y_true = test_data[target_column].values
        
        # Forecast
        if model_type == 'arima' or model_type == 'sarima':
            # Get forecast for test period
            forecast = model.forecast(steps=len(test_data))
            y_pred = forecast.values
        
        elif model_type == 'prophet':
            # Create future dataframe
            future = pd.DataFrame(test_data.index.values, columns=['ds'])
            # Make prediction
            forecast = model.predict(future)
            y_pred = forecast['yhat'].values
        
        elif model_type == 'lstm':
            model_obj, scaler = model
            
            # Load config
            config_path = os.path.join(self.model_path, 'lstm_config.pkl')
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
            
            sequence_length = config['sequence_length']
            
            # Get last sequence from training data
            # For simplicity, assume we have access to the last sequence
            # In a real scenario, you might need to load the training data
            last_sequence = test_data[target_column].values[:sequence_length].reshape(-1, 1)
            last_sequence = scaler.transform(last_sequence)
            
            # Make predictions one step at a time
            predictions = []
            current_seq = last_sequence.reshape(1, sequence_length, 1)
            
            for _ in range(len(test_data)):
                # Predict next value
                next_val = model_obj.predict(current_seq)[0, 0]
                predictions.append(next_val)
                
                # Update sequence
                current_seq = np.append(current_seq[:, 1:, :], [[next_val]], axis=1)
            
            # Inverse transform to get actual values
            predictions = np.array(predictions).reshape(-1, 1)
            y_pred = scaler.inverse_transform(predictions).flatten()
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        
        print(f"{model_type} model evaluation metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Plot actual vs predicted
        plt.figure(figsize=(12, 6))
        plt.plot(test_data.index, y_true, label='Actual')
        plt.plot(test_data.index, y_pred, label='Predicted')
        plt.title(f"{model_type.upper()} Model: Actual vs Predicted")
        plt.xlabel('Date')
        plt.ylabel(target_column)
        plt.legend()
        
        # Save plot
        plot_path = os.path.join(self.model_path, f'{model_type}_forecast.png')
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        
        return metrics


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
    
    # Initialize model
    model = TimeSeriesModel()
    
    # Define which models to train
    models_to_train = ['arima', 'sarima', 'prophet', 'lstm']
    
    # Train and evaluate models
    for model_type in models_to_train:
        if model_type == 'arima':
            # Train ARIMA model
            arima_model = model.train_arima(train_data)
            # Evaluate ARIMA model
            model.evaluate_model('arima', arima_model, test_data)
        
        elif model_type == 'sarima':
            # Train SARIMA model
            sarima_model = model.train_sarima(train_data)
            # Evaluate SARIMA model
            model.evaluate_model('sarima', sarima_model, test_data)
        
        elif model_type == 'prophet':
            # Train Prophet model
            prophet_model = model.train_prophet(train_data)
            # Evaluate Prophet model
            model.evaluate_model('prophet', prophet_model, test_data)
        
        elif model_type == 'lstm':
            # Train LSTM model
            lstm_model, scaler = model.train_lstm(train_data)
            # Evaluate LSTM model
            model.evaluate_model('lstm', (lstm_model, scaler), test_data)
    
    print("Model training and evaluation completed") 