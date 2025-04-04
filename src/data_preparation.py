import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class DataPreparation:
    def __init__(self, data_path='data/raw', output_path='data/processed'):
        """
        Initialize the data preparation class
        
        Parameters:
        -----------
        data_path : str
            Path to raw data
        output_path : str
            Path to save processed data
        """
        self.data_path = data_path
        self.output_path = output_path
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    
    def load_data(self, filename):
        """
        Load data from csv file
        
        Parameters:
        -----------
        filename : str
            Name of the file to load
            
        Returns:
        --------
        pd.DataFrame
            Loaded data
        """
        file_path = os.path.join(self.data_path, filename)
        print(f"Loading data from {file_path}")
        return pd.read_csv(file_path)
    
    def clean_data(self, df):
        """
        Clean the data by handling missing values, outliers, etc.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to clean
            
        Returns:
        --------
        pd.DataFrame
            Cleaned DataFrame
        """
        # Make a copy of the DataFrame
        df_clean = df.copy()
        
        # Convert date column to datetime
        if 'date' in df_clean.columns:
            df_clean['date'] = pd.to_datetime(df_clean['date'])
        
        # Handle missing values - using forward fill for time series
        df_clean.fillna(method='ffill', inplace=True)
        # For any remaining NaNs, use backward fill
        df_clean.fillna(method='bfill', inplace=True)
        
        # Drop duplicates
        df_clean.drop_duplicates(inplace=True)
        
        # Handle outliers using IQR method for sales column
        if 'sales' in df_clean.columns:
            Q1 = df_clean['sales'].quantile(0.25)
            Q3 = df_clean['sales'].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers
            df_clean.loc[df_clean['sales'] < lower_bound, 'sales'] = lower_bound
            df_clean.loc[df_clean['sales'] > upper_bound, 'sales'] = upper_bound
        
        return df_clean
    
    def preprocess_data(self, df):
        """
        Preprocess data for time series analysis
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to preprocess
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed DataFrame ready for time series analysis
        """
        # Make a copy of the DataFrame
        df_processed = df.copy()
        
        # Ensure data is sorted by date
        if 'date' in df_processed.columns:
            df_processed.sort_values('date', inplace=True)
            # Set date as index
            df_processed.set_index('date', inplace=True)
        
        # Add time-based features
        df_processed['year'] = df_processed.index.year
        df_processed['month'] = df_processed.index.month
        df_processed['day'] = df_processed.index.day
        df_processed['dayofweek'] = df_processed.index.dayofweek
        df_processed['quarter'] = df_processed.index.quarter
        
        # Add lag features
        if 'sales' in df_processed.columns:
            # Add 1-day, 1-week and 1-month lag features
            df_processed['sales_lag1'] = df_processed['sales'].shift(1)
            df_processed['sales_lag7'] = df_processed['sales'].shift(7)
            df_processed['sales_lag30'] = df_processed['sales'].shift(30)
            
            # Add rolling mean features
            df_processed['sales_rolling_7d'] = df_processed['sales'].rolling(window=7).mean()
            df_processed['sales_rolling_30d'] = df_processed['sales'].rolling(window=30).mean()
        
        # Drop rows with NaN values that were created by lag features
        df_processed.dropna(inplace=True)
        
        return df_processed
    
    def split_data(self, df, test_size=0.2):
        """
        Split data into train and test sets
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to split
        test_size : float
            Proportion of data to use for testing
            
        Returns:
        --------
        pd.DataFrame, pd.DataFrame
            Train and test DataFrames
        """
        # Calculate split point
        split_idx = int(len(df) * (1 - test_size))
        
        # Split data
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]
        
        return train, test
    
    def save_data(self, df, filename):
        """
        Save processed data to CSV
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to save
        filename : str
            Name to save the file as
        """
        output_path = os.path.join(self.output_path, filename)
        df.to_csv(output_path)
        print(f"Data saved to {output_path}")
    
    def process_pipeline(self, input_filename, output_filename):
        """
        Run the full data processing pipeline
        
        Parameters:
        -----------
        input_filename : str
            Name of input file
        output_filename : str
            Name of output file
        
        Returns:
        --------
        pd.DataFrame, pd.DataFrame
            Train and test DataFrames
        """
        # Load data
        df = self.load_data(input_filename)
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Preprocess data
        df_processed = self.preprocess_data(df_clean)
        
        # Split data
        train, test = self.split_data(df_processed)
        
        # Save processed datasets
        train_filename = f"train_{output_filename}"
        test_filename = f"test_{output_filename}"
        self.save_data(train, train_filename)
        self.save_data(test, test_filename)
        
        return train, test

if __name__ == "__main__":
    # Example usage
    data_prep = DataPreparation()
    
    # Check if sample data exists, if not create dummy data
    sample_file = 'sample_sales_data.csv'
    sample_path = os.path.join(data_prep.data_path, sample_file)
    
    if not os.path.exists(data_prep.data_path):
        os.makedirs(data_prep.data_path)
    
    if not os.path.exists(sample_path):
        print(f"Sample data not found at {sample_path}, creating dummy data...")
        
        # Create date range
        dates = pd.date_range(start='2020-01-01', end='2022-12-31')
        
        # Create sales with trend, seasonality and some noise
        n = len(dates)
        trend = np.linspace(100, 200, n)  # Upward trend
        seasonality = 50 * np.sin(np.linspace(0, 24*np.pi, n))  # Yearly seasonality
        noise = np.random.normal(0, 10, n)  # Random noise
        
        sales = trend + seasonality + noise
        
        # Create DataFrame
        dummy_data = pd.DataFrame({
            'date': dates,
            'sales': sales
        })
        
        # Save dummy data
        dummy_data.to_csv(sample_path, index=False)
        print(f"Dummy data created and saved to {sample_path}")
    
    # Process the data
    train, test = data_prep.process_pipeline(sample_file, 'processed_sales_data.csv')
    
    print("Data preparation completed successfully")
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}") 