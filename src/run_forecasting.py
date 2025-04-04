"""
Time Series Forecasting Example Script for Amazon and Car Prices datasets

This script demonstrates how to use the forecasting pipeline to:
1. Analyze and preprocess datasets
2. Train multiple forecasting models
3. Generate forecasts and evaluate model performance

Usage:
    python src/run_forecasting.py --dataset amazon
    python src/run_forecasting.py --dataset car_prices
"""

import os
import argparse
from datetime import datetime
from .analyze_datasets import DatasetAnalyzer
from .forecasting import TimeSeriesForecasting

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Time Series Forecasting Example')
    
    parser.add_argument('--dataset', type=str, required=True, choices=['amazon', 'car_prices'],
                        help='Dataset to use (amazon or car_prices)')
    parser.add_argument('--forecast_steps', type=int, default=30,
                        help='Number of steps to forecast')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['arima', 'sarima', 'ets', 'rf', 'lr'],
                        help='Models to train (arima, sarima, ets, rf, lr, lstm)')
    
    return parser.parse_args()

def main():
    """Main function to run the forecasting example"""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/{args.dataset}_forecast_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Time Series Forecasting Example: {args.dataset.upper()}")
    print(f"{'='*60}")
    
    # Step 1: Analyze and preprocess dataset
    print(f"\n{'='*60}")
    print(f"Step 1: Analyze and preprocess dataset")
    print(f"{'='*60}")
    
    analyzer = DatasetAnalyzer(output_path=os.path.join(output_dir, 'analysis'))
    
    try:
        # Load dataset
        if args.dataset == 'amazon':
            # Process Amazon dataset
            data_path = 'data/amazon.csv'
            df = analyzer.load_dataset(data_path)
            processed_df = analyzer.preprocess_amazon_data(df)
            target_column = 'daily_sales'
            
        elif args.dataset == 'car_prices':
            # Process Car Prices dataset
            data_path = 'data/car_prices.csv'
            df = analyzer.load_dataset(data_path)
            processed_df = analyzer.preprocess_car_prices_data(df)
            target_column = 'sellingprice'
        
        # Analyze dataset
        analyzer.analyze_dataset(processed_df, args.dataset)
        
        # Save processed dataset
        processed_path = os.path.join(output_dir, f'{args.dataset}_processed.csv')
        processed_df.to_csv(processed_path)
        print(f"Processed data saved to {processed_path}")
        
    except Exception as e:
        print(f"Error in dataset analysis: {str(e)}")
        return
    
    # Step 2: Train forecasting models
    print(f"\n{'='*60}")
    print(f"Step 2: Train forecasting models")
    print(f"{'='*60}")
    
    forecaster = TimeSeriesForecasting(output_path=os.path.join(output_dir, 'forecasts'))
    
    try:
        # Load data for forecasting
        forecaster.load_data(processed_path, target_col=target_column)
        
        # Split data
        forecaster.split_data(test_size=args.test_size)
        
        # Train models
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
        
        # Step 3: Generate forecasts and evaluate models
        print(f"\n{'='*60}")
        print(f"Step 3: Generate forecasts and evaluate models")
        print(f"{'='*60}")
        
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