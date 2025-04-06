#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Time Series Forecasting Example Script for Amazon, Car Prices, and custom datasets

This script demonstrates how to use the forecasting pipeline to:
1. Analyze and preprocess datasets
2. Train multiple forecasting models with automatic hyperparameter tuning
3. Generate forecasts and evaluate model performance
4. Create ensemble forecasts for improved accuracy

Usage:
    python src/run_forecasting.py --dataset amazon
    python src/run_forecasting.py --dataset car_prices
    python src/run_forecasting.py --dataset custom --data_path path/to/data.csv --date_col date_column --target_col target_column
    python src/run_forecasting.py --dataset amazon --tune_hyperparams
"""

import os
import argparse
import json
import pandas as pd
from datetime import datetime
from analyze_datasets import DatasetAnalyzer
from forecasting import TimeSeriesForecasting

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Advanced Time Series Forecasting')
    
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Dataset to use (amazon, car_prices, or custom)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to custom dataset (CSV file)')
    parser.add_argument('--date_col', type=str, default=None,
                        help='Date column name for custom dataset')
    parser.add_argument('--target_col', type=str, default=None,
                        help='Target column name for custom dataset')
    parser.add_argument('--forecast_steps', type=int, default=30,
                        help='Number of steps to forecast')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['arima', 'sarima', 'ets', 'rf', 'gb', 'xgb', 'lstm', 'prophet'],
                        help='Models to train (arima, sarima, ets, prophet, rf, gb, xgb, lr, lstm)')
    parser.add_argument('--tune_hyperparams', action='store_true',
                        help='Tune model hyperparameters (takes longer)')
    parser.add_argument('--lstm_type', type=str, default='simple',
                        choices=['simple', 'bidirectional', 'stacked', 'cnn'],
                        help='Type of LSTM architecture')
    parser.add_argument('--ensemble', action='store_true',
                        help='Create an ensemble forecast from all models')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Custom output directory (default: auto-generated with timestamp)')
    
    return parser.parse_args()

def load_dataset(dataset_name, data_path=None, date_col=None, target_col=None):
    """
    Load a dataset by name or path
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset (amazon, car_prices, or custom)
    data_path : str
        Path to custom dataset
    date_col : str
        Date column name for custom dataset
    target_col : str
        Target column name for custom dataset
        
    Returns:
    --------
    tuple
        (DataFrame, date_col, target_col)
    """
    # For predefined datasets
    if dataset_name.lower() == 'amazon':
        data_path = 'data/amazon.csv'
        date_col = 'date'
        target_col = 'daily_sales'
    elif dataset_name.lower() == 'car_prices':
        data_path = 'data/car_prices.csv'
        date_col = 'saledate'
        target_col = 'sellingprice'
    elif dataset_name.lower() == 'custom':
        # For custom dataset, ensure path and required columns are provided
        if data_path is None:
            raise ValueError("For custom dataset, data_path must be provided")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
    else:
        # Treat as a path to a dataset
        if os.path.exists(dataset_name):
            data_path = dataset_name
            dataset_name = 'custom'
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Load the dataset
    df = pd.read_csv(data_path)
    
    # Auto-detect date and target columns for custom datasets if not specified
    if dataset_name.lower() == 'custom':
        if date_col is None:
            # Try to find date column automatically
            date_candidates = [col for col in df.columns if any(date_term in col.lower() 
                              for date_term in ['date', 'time', 'day', 'month', 'year'])]
            if date_candidates:
                date_col = date_candidates[0]
                print(f"Auto-detected date column: {date_col}")
            else:
                raise ValueError("Could not auto-detect date column. Please specify with --date_col")
        
        if target_col is None:
            # Try to find target column automatically - prefer numeric columns with 'price', 'sales', 'value', 'target'
            numeric_cols = df.select_dtypes(include=['number']).columns
            target_candidates = [col for col in numeric_cols if any(target_term in col.lower() 
                               for target_term in ['price', 'sale', 'value', 'target', 'amount', 'quantity'])]
            
            if target_candidates:
                target_col = target_candidates[0]
                print(f"Auto-detected target column: {target_col}")
            elif len(numeric_cols) > 0:
                target_col = numeric_cols[0]
                print(f"Using first numeric column as target: {target_col}")
            else:
                raise ValueError("Could not auto-detect target column. Please specify with --target_col")
    
    return df, data_path, date_col, target_col

def main():
    """Main function to run the forecasting example"""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"results/{args.dataset}_forecast_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\n{'='*60}")
    print(f"Advanced Time Series Forecasting: {args.dataset.upper()}")
    print(f"{'='*60}")
    
    # Step 1: Analyze and preprocess dataset
    print(f"\n{'='*60}")
    print(f"Step 1: Analyze and preprocess dataset")
    print(f"{'='*60}")
    
    analyzer = DatasetAnalyzer(output_path=os.path.join(output_dir, 'analysis'))
    
    try:
        # Load and preprocess dataset
        df, data_path, date_col, target_col = load_dataset(
            args.dataset, args.data_path, args.date_col, args.target_col
        )
        
        # Load dataset using analyzer
        df = analyzer.load_dataset(data_path)
        
        # Process dataset based on type
        if args.dataset.lower() == 'amazon':
            processed_df = analyzer.preprocess_amazon_data(df)
        elif args.dataset.lower() == 'car_prices':
            processed_df = analyzer.preprocess_car_prices_data(df)
        else:
            # Generic preprocessing for custom datasets
            processed_df = analyzer.preprocess_generic_data(df, date_col, target_col)
        
        # Save target column name for later use
        target_column = target_col
        
        # Analyze dataset with more advanced methods
        analyzer.analyze_dataset(processed_df, args.dataset, decompose=True, detect_anomalies=True)
        
        # Save processed dataset
        processed_path = os.path.join(output_dir, f'{args.dataset}_processed.csv')
        processed_df.to_csv(processed_path)
        print(f"Processed data saved to {processed_path}")
        
    except Exception as e:
        print(f"Error in dataset analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Train forecasting models
    print(f"\n{'='*60}")
    print(f"Step 2: Train forecasting models")
    print(f"{'='*60}")
    
    forecaster = TimeSeriesForecasting(output_path=os.path.join(output_dir, 'forecasts'))
    
    try:
        # Load data for forecasting
        forecaster.load_data(processed_path, date_col=date_col, target_col=target_column)
        
        # Split data
        forecaster.split_data(test_size=args.test_size)
        
        # Tune hyperparameters if requested
        if args.tune_hyperparams:
            print("\nTuning hyperparameters...")
            
            if 'arima' in args.models:
                forecaster.tune_arima_parameters()
            
            if 'sarima' in args.models:
                forecaster.tune_sarima_parameters()
        
        # Train models
        for model in args.models:
            try:
                print(f"\nTraining {model} model...")
                
                if model.lower() == 'arima':
                    forecaster.train_arima()  # Will use tuned parameters if available
                    
                elif model.lower() == 'sarima':
                    forecaster.train_sarima()  # Will use tuned parameters if available
                    
                elif model.lower() == 'ets':
                    # Try different ETS configurations based on data characteristics
                    if hasattr(forecaster, 'is_stationary') and forecaster.is_stationary:
                        # For stationary data
                        forecaster.train_exponential_smoothing(trend=None, seasonal=None)
                    else:
                        # For non-stationary data
                        forecaster.train_exponential_smoothing(trend='add', seasonal=None)
                        
                        # If we have enough data, try seasonal model as well
                        if len(forecaster.train_data) > 24:
                            forecaster.train_exponential_smoothing(
                                trend='add', 
                                seasonal='add', 
                                seasonal_periods=12
                            )
                            
                elif model.lower() == 'prophet':
                    # Train Prophet model with appropriate seasonality
                    forecaster.train_prophet(
                        yearly_seasonality=len(forecaster.train_data) >= 365,
                        weekly_seasonality=len(forecaster.train_data) >= 14,
                        daily_seasonality=len(forecaster.train_data) >= 7
                    )
                    
                elif model.lower() == 'rf':
                    # Train Random Forest with hyperparameter tuning if requested
                    forecaster.train_ml_model(
                        model_type='rf', 
                        n_lags=min(12, len(forecaster.train_data) // 10),
                        tune_hyperparams=args.tune_hyperparams
                    )
                    
                elif model.lower() == 'gb':
                    # Train Gradient Boosting with hyperparameter tuning if requested
                    forecaster.train_ml_model(
                        model_type='gb', 
                        n_lags=min(12, len(forecaster.train_data) // 10),
                        tune_hyperparams=args.tune_hyperparams
                    )
                    
                elif model.lower() == 'xgb':
                    # Train XGBoost with hyperparameter tuning if requested
                    forecaster.train_ml_model(
                        model_type='xgb', 
                        n_lags=min(12, len(forecaster.train_data) // 10),
                        tune_hyperparams=args.tune_hyperparams
                    )
                    
                elif model.lower() == 'lr':
                    # Train Linear Regression
                    forecaster.train_ml_model(
                        model_type='lr', 
                        n_lags=min(12, len(forecaster.train_data) // 10)
                    )
                    
                elif model.lower() == 'lstm':
                    # Train advanced LSTM with specified architecture
                    forecaster.train_lstm(
                        n_steps=min(30, len(forecaster.train_data) // 10),
                        n_epochs=100 if args.tune_hyperparams else 50,
                        batch_size=32,
                        lstm_type=args.lstm_type
                    )
                    
                else:
                    print(f"Unknown model type: {model}")
                    
            except Exception as e:
                print(f"Error training {model} model: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Step 3: Generate forecasts and evaluate models
        print(f"\n{'='*60}")
        print(f"Step 3: Generate forecasts and evaluate models")
        print(f"{'='*60}")
        
        # Evaluate models
        results = forecaster.evaluate_models(steps=args.forecast_steps)
        
        # Print evaluation results
        print("\nModel Evaluation Results:")
        print(results)
        
        # Create ensemble forecast if requested or if 'ensemble' is in models list
        if args.ensemble or 'ensemble' in args.models:
            print("\nCreating ensemble forecast...")
            
            ensemble = forecaster.ensemble_forecast(steps=args.forecast_steps)
            
            # Print ensemble results
            if 'ensemble' in forecaster.results:
                print("\nEnsemble Model Results:")
                print(forecaster.results['ensemble'])
        
        print(f"\nForecasting completed successfully!")
        print(f"Results saved to {output_dir}")
        
        # Generate a summary report
        with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
            f.write(f"Time Series Forecasting Summary\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Forecast Steps: {args.forecast_steps}\n")
            f.write(f"Models Trained: {', '.join(args.models)}\n\n")
            
            if results is not None:
                f.write("Model Performance:\n")
                
                # Find best model by RMSE
                best_rmse = float('inf')
                best_model = None
                
                for model, metrics in forecaster.results.items():
                    if 'RMSE' in metrics and metrics['RMSE'] < best_rmse:
                        best_rmse = metrics['RMSE']
                        best_model = model
                
                f.write(f"Best Model: {best_model} (RMSE: {best_rmse:.4f})\n\n")
                
                # Write all model results
                for model, metrics in forecaster.results.items():
                    f.write(f"{model} Model:\n")
                    for metric, value in metrics.items():
                        f.write(f"  {metric}: {value:.4f}\n")
                    f.write("\n")
        
    except Exception as e:
        print(f"Error in forecasting: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 