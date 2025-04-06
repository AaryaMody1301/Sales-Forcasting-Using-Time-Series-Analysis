import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import sys

# Add the project directory to the path to enable absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use absolute imports instead of relative imports
from src.utils import create_directory, add_time_features, add_lag_features

class DatasetAnalyzer:
    def __init__(self, data_path='data', output_path='data/processed'):
        """
        Initialize the dataset analyzer
        
        Parameters:
        -----------
        data_path : str
            Path to raw data
        output_path : str
            Path to save processed data
        """
        self.data_path = data_path
        self.output_path = output_path
        create_directory(output_path)
    
    def load_dataset(self, filename):
        """
        Load dataset with basic information
        
        Parameters:
        -----------
        filename : str
            Name of the file to load
            
        Returns:
        --------
        pd.DataFrame
            Loaded dataset
        """
        file_path = os.path.join(self.data_path, filename)
        print(f"\nLoading {filename}...")
        
        # Read first few rows to get column names
        df_sample = pd.read_csv(file_path, nrows=5)
        print(f"Columns: {', '.join(df_sample.columns)}")
        
        # Read the full dataset
        df = pd.read_csv(file_path)
        print(f"Shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
        
        return df
    
    def preprocess_amazon_data(self, df):
        """
        Preprocess Amazon dataset
        
        Parameters:
        -----------
        df : pd.DataFrame
            Amazon dataset
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed dataset
        """
        # Create a copy of the dataset
        df_processed = df.copy()
        
        # Convert price columns to numeric
        def clean_price(price_str):
            if isinstance(price_str, str):
                # Remove currency symbol and convert to float
                try:
                    return float(price_str.replace('₹', '').replace(',', '').strip())
                except ValueError:
                    return np.nan
            return price_str
        
        print("Converting price columns to numeric...")
        df_processed['discounted_price'] = df_processed['discounted_price'].apply(clean_price)
        df_processed['actual_price'] = df_processed['actual_price'].apply(clean_price)
        
        # Convert discount percentage to numeric
        print("Converting discount percentage to numeric...")
        df_processed['discount_percentage'] = df_processed['discount_percentage'].str.replace('%', '').apply(
            lambda x: float(x) if isinstance(x, str) and x.strip() else np.nan
        )
        
        # Convert rating to numeric
        print("Converting rating to numeric...")
        df_processed['rating'] = df_processed['rating'].apply(
            lambda x: float(x) if isinstance(x, str) and x.strip() and x != '|' else np.nan
        )
        
        # Convert rating count to numeric
        print("Converting rating count to numeric...")
        df_processed['rating_count'] = df_processed['rating_count'].apply(
            lambda x: float(str(x).replace(',', '')) if pd.notnull(x) and str(x).strip() else np.nan
        )
        
        # Create a daily sales metric (using discounted price as proxy for sales)
        print("Creating daily sales metric...")
        df_processed['daily_sales'] = df_processed['discounted_price'] * df_processed['rating_count']
        
        # Create a date index (assuming reviews are from recent times)
        print("Creating date index...")
        df_processed['date'] = pd.date_range(end=datetime.now(), periods=len(df_processed), freq='D')
        df_processed.set_index('date', inplace=True)
        
        # Sort by date
        df_processed.sort_index(inplace=True)
        
        # Print summary of numeric columns
        numeric_cols = ['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count', 'daily_sales']
        print("\nSummary of numeric columns:")
        print("-" * 50)
        for col in numeric_cols:
            non_null = df_processed[col].count()
            total = len(df_processed)
            print(f"{col}: {non_null}/{total} non-null values ({non_null/total*100:.1f}%)")
        
        return df_processed
    
    def preprocess_car_prices_data(self, df):
        """
        Preprocess Car Prices dataset
        
        Parameters:
        -----------
        df : pd.DataFrame
            Car Prices dataset
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed dataset
        """
        print("\nPreprocessing Car Prices dataset...")
        
        # Create a copy of the dataset
        df_processed = df.copy()
        
        # Convert saledate to datetime
        print("Converting saledate to datetime...")
        df_processed['saledate'] = pd.to_datetime(df_processed['saledate'], errors='coerce')
        
        # Convert price columns to numeric
        print("Converting price columns to numeric...")
        price_cols = ['mmr', 'sellingprice']
        for col in price_cols:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Convert odometer to numeric
        print("Converting odometer to numeric...")
        df_processed['odometer'] = pd.to_numeric(df_processed['odometer'], errors='coerce')
        
        # Convert and validate year
        print("Converting and validating year...")
        df_processed['year'] = pd.to_numeric(df_processed['year'], errors='coerce')
        # Filter out invalid years (assuming cars are from 1900 to current year + 1)
        current_year = datetime.now().year
        df_processed.loc[
            (df_processed['year'] < 1900) | (df_processed['year'] > current_year + 1),
            'year'
        ] = np.nan
        
        # Remove rows with missing saledate
        print("Removing rows with missing saledate...")
        df_processed = df_processed.dropna(subset=['saledate'])
        
        # Set saledate as index
        print("Setting saledate as index...")
        df_processed.set_index('saledate', inplace=True)
        
        # Sort by date
        df_processed.sort_index(inplace=True)
        
        # Print summary of numeric columns
        numeric_cols = ['year', 'odometer', 'mmr', 'sellingprice']
        print("\nSummary of numeric columns:")
        print("-" * 50)
        for col in numeric_cols:
            non_null = df_processed[col].count()
            total = len(df_processed)
            print(f"{col}: {non_null}/{total} non-null values ({non_null/total*100:.1f}%)")
        
        # Print date range
        print("\nDate range:")
        print("-" * 50)
        print(f"Start date: {df_processed.index.min()}")
        print(f"End date: {df_processed.index.max()}")
        print(f"Number of days: {(df_processed.index.max() - df_processed.index.min()).days}")
        
        return df_processed
    
    def analyze_dataset(self, df, dataset_name):
        """
        Analyze dataset and generate insights
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset to analyze
        dataset_name : str
            Name of the dataset
        """
        print(f"\nAnalyzing {dataset_name} dataset...")
        
        # Basic information
        print("\nBasic Information:")
        print("-" * 50)
        print(df.info())
        
        # Missing values
        print("\nMissing Values:")
        print("-" * 50)
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0])
        
        # Numerical columns statistics
        print("\nNumerical Columns Statistics:")
        print("-" * 50)
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            print(df[numerical_cols].describe())
        
        # Save analysis results
        analysis_path = os.path.join(self.output_path, f'{dataset_name}_analysis.txt')
        with open(analysis_path, 'w', encoding='utf-8') as f:
            f.write(f"Dataset Analysis: {dataset_name}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Basic Information:\n")
            f.write("-" * 50 + "\n")
            df.info(buf=f)
            
            f.write("\nMissing Values:\n")
            f.write("-" * 50 + "\n")
            f.write(missing_values[missing_values > 0].to_string())
            
            if len(numerical_cols) > 0:
                f.write("\n\nNumerical Columns Statistics:\n")
                f.write("-" * 50 + "\n")
                f.write(df[numerical_cols].describe().to_string())
        
        print(f"\nAnalysis saved to {analysis_path}")
    
    def prepare_for_forecasting(self, df, dataset_name, date_column=None, target_column=None):
        """
        Prepare dataset for time series forecasting
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset to prepare
        dataset_name : str
            Name of the dataset
        date_column : str, optional
            Name of the date column
        target_column : str, optional
            Name of the target column for forecasting
        """
        print(f"\nPreparing {dataset_name} for forecasting...")
        
        # Create a copy of the dataset
        df_processed = df.copy()
        
        # Handle date column
        if date_column and date_column in df_processed.columns:
            df_processed[date_column] = pd.to_datetime(df_processed[date_column])
            df_processed.set_index(date_column, inplace=True)
        
        # Sort by date
        df_processed.sort_index(inplace=True)
        
        # Add time features
        df_processed = add_time_features(df_processed)
        
        # Add lag features if target column is specified
        if target_column and target_column in df_processed.columns:
            df_processed = add_lag_features(df_processed, target_column=target_column)
        
        # Save processed dataset
        processed_path = os.path.join(self.output_path, f'{dataset_name}_processed.csv')
        df_processed.to_csv(processed_path)
        print(f"Processed dataset saved to {processed_path}")
        
        return df_processed
    
    def visualize_dataset(self, df, dataset_name, target_column=None):
        """
        Create visualizations for the dataset
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset to visualize
        dataset_name : str
            Name of the dataset
        target_column : str, optional
            Name of the target column for time series plot
        """
        print(f"\nCreating visualizations for {dataset_name}...")
        
        # Create visualizations directory
        viz_path = os.path.join(self.output_path, 'visualizations')
        create_directory(viz_path)
        
        # Time series plot if target column is specified
        if target_column and target_column in df.columns:
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df[target_column])
            plt.title(f'{target_column} Over Time')
            plt.xlabel('Date')
            plt.ylabel(target_column)
            plt.grid(True)
            plt.savefig(os.path.join(viz_path, f'{dataset_name}_{target_column}_ts.png'))
            plt.close()
        
        # Distribution plots for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x=col, bins=50)
            plt.title(f'Distribution of {col}')
            plt.savefig(os.path.join(viz_path, f'{dataset_name}_{col}_dist.png'))
            plt.close()
        
        # Correlation matrix for numerical columns
        if len(numerical_cols) > 1:
            plt.figure(figsize=(12, 8))
            sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_path, f'{dataset_name}_correlation.png'))
            plt.close()
        
        print(f"Visualizations saved to {viz_path}")
    
    def preprocess_generic_data(self, df, date_col, target_col):
        """
        Preprocess a generic dataset for time series forecasting
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw dataset
        date_col : str
            Name of the date column
        target_col : str
            Name of the target column to forecast
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed dataset
        """
        print(f"Preprocessing generic dataset with date column: {date_col} and target column: {target_col}")
        
        # Create a copy of the dataset
        processed_df = df.copy()
        
        # Convert date column to datetime
        try:
            processed_df[date_col] = pd.to_datetime(processed_df[date_col])
            print(f"Converted {date_col} to datetime")
        except Exception as e:
            print(f"Warning: Error converting {date_col} to datetime: {str(e)}")
            # Try to detect and convert various date formats
            try:
                processed_df[date_col] = pd.to_datetime(processed_df[date_col], errors='coerce')
                # Drop rows with invalid dates
                processed_df = processed_df.dropna(subset=[date_col])
                print(f"Converted {date_col} to datetime with some error handling")
            except:
                raise ValueError(f"Could not convert {date_col} to datetime format")
        
        # Set date as index
        processed_df = processed_df.set_index(date_col)
        
        # Sort by date
        processed_df = processed_df.sort_index()
        
        # Handle missing values in the target column
        if processed_df[target_col].isnull().sum() > 0:
            print(f"Handling {processed_df[target_col].isnull().sum()} missing values in {target_col}")
            # For small gaps, use linear interpolation
            processed_df[target_col] = processed_df[target_col].interpolate(method='linear')
            # For any remaining NaNs at the beginning or end, use forward/backward fill
            processed_df[target_col] = processed_df[target_col].fillna(method='ffill').fillna(method='bfill')
        
        # Check for anomalies (values > 3 std from mean)
        mean = processed_df[target_col].mean()
        std = processed_df[target_col].std()
        anomalies = processed_df[(processed_df[target_col] > mean + 3*std) | (processed_df[target_col] < mean - 3*std)]
        
        if len(anomalies) > 0:
            print(f"Detected {len(anomalies)} potential anomalies in {target_col}")
            
            # If anomalies are less than 5% of the data, cap them
            if len(anomalies) < 0.05 * len(processed_df):
                print("Capping extreme values")
                upper_limit = mean + 3*std
                lower_limit = mean - 3*std
                processed_df[target_col] = processed_df[target_col].clip(lower=lower_limit, upper=upper_limit)
        
        # Add basic time features
        processed_df['year'] = processed_df.index.year
        processed_df['month'] = processed_df.index.month
        processed_df['day_of_week'] = processed_df.index.dayofweek
        processed_df['day_of_year'] = processed_df.index.dayofyear
        
        # Check data frequency and resample if needed
        inferred_freq = pd.infer_freq(processed_df.index)
        if inferred_freq is None:
            print("Warning: Could not infer data frequency. Checking for irregular timestamps...")
            
            # Check for duplicated timestamps
            if processed_df.index.duplicated().any():
                print(f"Found {processed_df.index.duplicated().sum()} duplicated timestamps")
                # Keep the last value for each timestamp
                processed_df = processed_df[~processed_df.index.duplicated(keep='last')]
            
            # Check for missing dates (significant gaps)
            date_range = pd.date_range(start=processed_df.index.min(), end=processed_df.index.max())
            missing_dates = set(date_range) - set(processed_df.index)
            
            if len(missing_dates) > 0:
                pct_missing = len(missing_dates) / len(date_range) * 100
                print(f"Missing {len(missing_dates)} dates ({pct_missing:.2f}% of the date range)")
                
                # If less than 20% of dates are missing, create a regular time series
                if pct_missing < 20:
                    # Try to infer the most likely frequency (daily, weekly, monthly)
                    diff = processed_df.index.to_series().diff().median()
                    
                    if diff.days < 2:
                        freq = 'D'  # Daily
                    elif diff.days < 8:
                        freq = 'W'  # Weekly
                    elif diff.days < 32:
                        freq = 'M'  # Monthly
                    else:
                        freq = None
                    
                    if freq:
                        print(f"Resampling data to {freq} frequency")
                        processed_df = processed_df.resample(freq).mean()
                        # Fill any new NaN values created by resampling
                        processed_df = processed_df.interpolate(method='linear')
        else:
            print(f"Detected data frequency: {inferred_freq}")
        
        # Ensure all columns have appropriate datatypes
        for col in processed_df.columns:
            if col != target_col and pd.api.types.is_numeric_dtype(processed_df[col]):
                # Convert integers to integer type for better memory usage
                if processed_df[col].fillna(0).astype(int).equals(processed_df[col].fillna(0)):
                    processed_df[col] = processed_df[col].astype('Int64')  # Nullable integer type
        
        print(f"Preprocessing complete. Processed dataframe shape: {processed_df.shape}")
        return processed_df

def main():
    """
    Main function to analyze and prepare datasets
    """
    # Initialize analyzer
    analyzer = DatasetAnalyzer()
    
    try:
        # Analyze Amazon dataset
        print("\n" + "="*50)
        print("Processing Amazon Dataset")
        print("="*50)
        
        amazon_df = analyzer.load_dataset('amazon.csv')
        amazon_processed = analyzer.preprocess_amazon_data(amazon_df)
        analyzer.analyze_dataset(amazon_processed, 'amazon')
        
        # Prepare Amazon dataset for forecasting
        amazon_forecast = analyzer.prepare_for_forecasting(
            amazon_processed,
            'amazon',
            target_column='daily_sales'
        )
        
        # Visualize Amazon dataset
        analyzer.visualize_dataset(amazon_forecast, 'amazon', target_column='daily_sales')
        
        print("\nAmazon dataset processing completed!")
    except Exception as e:
        print(f"\nError processing Amazon dataset: {str(e)}")
    
    try:
        # Analyze Car Prices dataset
        print("\n" + "="*50)
        print("Processing Car Prices Dataset")
        print("="*50)
        
        car_prices_df = analyzer.load_dataset('car_prices.csv')
        car_prices_processed = analyzer.preprocess_car_prices_data(car_prices_df)
        analyzer.analyze_dataset(car_prices_processed, 'car_prices')
        
        # Prepare Car Prices dataset for forecasting
        car_prices_forecast = analyzer.prepare_for_forecasting(
            car_prices_processed,
            'car_prices',
            target_column='sellingprice'  # Using actual selling price as target
        )
        
        # Visualize Car Prices dataset
        analyzer.visualize_dataset(car_prices_forecast, 'car_prices', target_column='sellingprice')
        
        print("\nCar Prices dataset processing completed!")
    except Exception as e:
        print(f"\nError processing Car Prices dataset: {str(e)}")
    
    print("\nDataset analysis and preparation completed!")

if __name__ == "__main__":
    main() 