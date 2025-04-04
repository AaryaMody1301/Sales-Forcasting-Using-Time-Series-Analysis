import os
import argparse
import pandas as pd
from datetime import datetime
from analyze_datasets import DatasetAnalyzer
from forecasting import TimeSeriesForecasting

def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
    --------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Time Series Forecasting')
    
    # Data arguments
    parser.add_argument('--data', type=str, required=True, 
                        help='Path to the dataset file (CSV)')
    parser.add_argument('--date_col', type=str, default=None,
                        help='Name of the date column')
    parser.add_argument('--target_col', type=str, default=None,
                        help='Name of the target column to forecast')
    
    # Forecasting arguments
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--forecast_steps', type=int, default=30,
                        help='Number of steps to forecast')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    
    # Models to train
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['arima', 'sarima', 'ets', 'rf', 'lr', 'lstm'],
                        help='Models to train (arima, sarima, ets, rf, lr, lstm)')
    
    return parser.parse_args()

def main():
    """
    Main function to run the forecasting pipeline
    """
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"forecast_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"Time Series Forecasting Pipeline")
    print(f"{'='*50}")
    
    print(f"\nInput file: {args.data}")
    print(f"Date column: {args.date_col}")
    print(f"Target column: {args.target_col}")
    print(f"Output directory: {output_dir}")
    
    # Analyze dataset
    print(f"\n{'='*50}")
    print(f"Step 1: Dataset Analysis")
    print(f"{'='*50}")
    
    analyzer = DatasetAnalyzer(output_path=os.path.join(output_dir, 'analysis'))
    
    try:
        # Load and preprocess data
        df = analyzer.load_dataset(args.data)
        
        # Check if file is processed already or needs preprocessing
        if args.data.endswith('_processed.csv'):
            # File is already processed
            processed_df = pd.read_csv(args.data)
            if args.date_col and args.date_col in processed_df.columns:
                processed_df[args.date_col] = pd.to_datetime(processed_df[args.date_col])
                processed_df.set_index(args.date_col, inplace=True)
        else:
            # Determine dataset type and preprocess accordingly
            if 'amazon' in args.data.lower():
                processed_df = analyzer.preprocess_amazon_data(df)
            elif 'car' in args.data.lower():
                processed_df = analyzer.preprocess_car_prices_data(df)
            else:
                # Generic preprocessing for unknown dataset
                processed_df = df.copy()
                if args.date_col and args.date_col in processed_df.columns:
                    processed_df[args.date_col] = pd.to_datetime(processed_df[args.date_col])
                    processed_df.set_index(args.date_col, inplace=True)
                    processed_df.sort_index(inplace=True)
        
        # Analyze processed data
        analyzer.analyze_dataset(processed_df, os.path.basename(args.data).split('.')[0])
        
        # Save processed dataset
        processed_path = os.path.join(output_dir, 'processed_data.csv')
        processed_df.to_csv(processed_path)
        print(f"Processed data saved to {processed_path}")
        
    except Exception as e:
        print(f"Error in dataset analysis: {str(e)}")
        return
    
    # Forecasting
    print(f"\n{'='*50}")
    print(f"Step 2: Time Series Forecasting")
    print(f"{'='*50}")
    
    forecaster = TimeSeriesForecasting(output_path=os.path.join(output_dir, 'forecasts'))
    
    try:
        # Load data for forecasting
        forecaster.load_data(processed_path, date_col=None, target_col=args.target_col)
        
        # Split data
        forecaster.split_data(test_size=args.test_size)
        
        # Train selected models
        for model in args.models:
            try:
                print(f"\nTraining {model} model...")
                if model.lower() == 'arima':
                    forecaster.train_arima(order=(1, 1, 1))
                elif model.lower() == 'sarima':
                    forecaster.train_sarima(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                elif model.lower() == 'ets':
                    forecaster.train_exponential_smoothing(trend='add', seasonal=None)
                elif model.lower() == 'rf':
                    forecaster.train_ml_model(model_type='rf', n_lags=12)
                elif model.lower() == 'lr':
                    forecaster.train_ml_model(model_type='lr', n_lags=12)
                elif model.lower() == 'lstm':
                    forecaster.train_lstm(n_steps=30, n_epochs=50)
                else:
                    print(f"Unknown model type: {model}")
            except Exception as e:
                print(f"Error training {model} model: {str(e)}")
        
        # Evaluate models
        results = forecaster.evaluate_models(steps=args.forecast_steps)
        
        # Print evaluation results
        print("\nModel Evaluation Results:")
        print(results)
        
        # Create ensemble forecast
        ensemble = forecaster.ensemble_forecast(steps=args.forecast_steps)
        
        print(f"\nForecasting completed successfully!")
        print(f"Results saved to {output_dir}")
        
    except Exception as e:
        print(f"Error in forecasting: {str(e)}")

if __name__ == "__main__":
    main() 