#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Time Series Forecasting Runner

This script provides a unified interface for:
1. Running forecasting models on various datasets (predefined or custom)
2. Training individual models or ensemble forecasts
3. Generating visualizations and evaluation metrics
4. Launching the interactive dashboard

Usage:
    python run_forecasting.py --dataset amazon --forecast_horizon 30
    python run_forecasting.py --dataset car_prices --forecast_horizon 30
    python run_forecasting.py --dataset custom --data_path data/sample_custom_data.csv --forecast_horizon 14
    python run_forecasting.py --run_dashboard  # Launches the dashboard 
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime
import pandas as pd
from pathlib import Path

# Add script directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import modules
try:
    from src.analyze_datasets import DatasetAnalyzer
    from src.forecasting import (
        ARIMAModel, SARIMAModel, ExponentialSmoothingModel, 
        ProphetModel, RandomForestModel, GradientBoostingModel,
        LSTMModel, EnsembleModel
    )
    # Import XGBoostModel if available
    try:
        from src.forecasting import XGBoostModel
        XGBOOST_AVAILABLE = True
    except ImportError:
        XGBOOST_AVAILABLE = False
        print("XGBoost not available. XGBoostModel won't be used.")
    from src.utils import validate_and_report, create_directory
except ImportError as e:
    print(f"Error importing required modules: {str(e)}")
    print("Please ensure all required packages are installed.")
    sys.exit(1)

def parse_arguments():
    """
    Parse command-line arguments
    
    Returns:
    --------
    argparse.Namespace
        Parsed arguments
    """
    # Create argument parser
    parser = argparse.ArgumentParser(description='Advanced Time Series Forecasting')
    
    # Add arguments
    parser.add_argument('--dataset', type=str, default='amazon', 
                        help='Dataset to use (amazon, car_prices, custom)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to custom dataset CSV file')
    parser.add_argument('--date_col', type=str, default=None,
                        help='Date column name for custom dataset')
    parser.add_argument('--target_col', type=str, default=None,
                        help='Target column name for custom dataset')
    parser.add_argument('--forecast_horizon', type=int, default=30,
                        help='Number of periods to forecast')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    
    # Get the available models
    available_models = ['arima', 'sarima', 'ets', 'prophet', 'rf', 'gbm', 'lstm', 'ensemble']
    if 'XGBOOST_AVAILABLE' in globals() and XGBOOST_AVAILABLE:
        available_models.append('xgboost')
    
    # Default models depend on what's available
    default_models = ['arima', 'sarima', 'ets', 'prophet', 'rf']
    if 'XGBOOST_AVAILABLE' in globals() and XGBOOST_AVAILABLE:
        default_models.append('xgboost')
    
    parser.add_argument('--models', type=str, nargs='+', 
                        default=default_models,
                        help=f'Models to run ({", ".join(available_models)})')
    parser.add_argument('--quick', action='store_true',
                        help='Run a quick version with fewer models')
    parser.add_argument('--run_dashboard', action='store_true',
                        help='Launch the dashboard after forecasting')
    parser.add_argument('--only_dashboard', action='store_true',
                        help='Only launch the dashboard without running forecasts')
    
    # Parse arguments
    args = parser.parse_args()
    return args

def load_dataset(dataset_name, data_path=None, date_col=None, target_col=None):
    """
    Load and preprocess dataset
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset (amazon, car_prices, or custom)
    data_path : str, optional
        Path to custom dataset
    date_col : str, optional
        Name of date column for custom dataset
    target_col : str, optional
        Name of target column for custom dataset
    
    Returns:
    --------
    tuple
        (processed_df, date_col, target_col)
    """
    analyzer = DatasetAnalyzer()
    processed_df = None
    
    output_dir = os.path.join('results', dataset_name)
    create_directory(output_dir)
    
    if dataset_name == 'amazon':
        # Load and preprocess Amazon dataset
        try:
            amazon_df = analyzer.load_dataset('amazon.csv')
            processed_df = analyzer.preprocess_amazon_data(amazon_df)
            date_col = 'date'
            target_col = 'daily_sales'
        except FileNotFoundError:
            print("Error: Amazon dataset not found. Please download the dataset and place it in the 'data' directory.")
            sys.exit(1)
        except Exception as e:
            print(f"Error processing Amazon dataset: {str(e)}")
            sys.exit(1)
    
    elif dataset_name == 'car_prices':
        # Load and preprocess Car Prices dataset
        try:
            car_prices_df = analyzer.load_dataset('car_prices.csv')
            processed_df = analyzer.preprocess_car_prices_data(car_prices_df)
            date_col = 'saledate'
            target_col = 'sellingprice'
        except FileNotFoundError:
            print("Error: Car Prices dataset not found. Please download the dataset and place it in the 'data' directory.")
            sys.exit(1)
        except Exception as e:
            print(f"Error processing Car Prices dataset: {str(e)}")
            sys.exit(1)
    
    elif dataset_name == 'custom':
        # Load and preprocess custom dataset
        if not data_path:
            print("Error: For custom dataset, --data_path is required.")
            sys.exit(1)
            
        try:
            # Check if the file exists
            if not os.path.exists(data_path):
                print(f"Error: Custom dataset file not found at '{data_path}'.")
                sys.exit(1)
                
            # Load the dataset
            print(f"Loading custom dataset from {data_path}...")
            df = pd.read_csv(data_path)
            
            # Auto-detect date and target columns if not provided
            if date_col is None or target_col is None:
                # Validate dataset and auto-detect columns
                is_valid, validation_results = validate_and_report(df, date_col, target_col, output_dir)
                
                if not is_valid:
                    print("Error: Invalid dataset. Please check the validation report and provide valid date and target columns.")
                    sys.exit(1)
                
                # Get auto-detected columns from validation results
                if date_col is None and 'suggestions' in validation_results:
                    for suggestion in validation_results['suggestions']:
                        if 'Auto-detected date column' in suggestion:
                            date_col = suggestion.split("'")[1]
                            print(f"Auto-detected date column: {date_col}")
                            break
                
                if target_col is None and 'suggestions' in validation_results:
                    for suggestion in validation_results['suggestions']:
                        if 'Auto-detected target column' in suggestion:
                            target_col = suggestion.split("'")[1]
                            print(f"Auto-detected target column: {target_col}")
                            break
            
            # Process the dataset
            print(f"Processing custom dataset with date column '{date_col}' and target column '{target_col}'...")
            processed_df = analyzer.preprocess_generic_data(df, date_col, target_col)
            
            # Save column info for future use
            column_info = {'date_col': date_col, 'target_col': target_col}
            column_info_path = os.path.join(output_dir, 'column_info.json')
            import json
            with open(column_info_path, 'w') as f:
                json.dump(column_info, f)
            
        except Exception as e:
            print(f"Error processing custom dataset: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    else:
        print(f"Error: Unknown dataset '{dataset_name}'. Supported datasets: amazon, car_prices, custom.")
        sys.exit(1)
    
    if processed_df is not None:
        # Save processed dataframe
        processed_path = os.path.join(output_dir, f'{dataset_name}_processed.csv')
        processed_df.to_csv(processed_path)
        print(f"Processed dataset saved to {processed_path}")
    
    return processed_df, date_col, target_col

def run_models(dataset_df, dataset_name, date_col, target_col, forecast_horizon, test_size, models):
    """
    Run forecasting models
    
    Parameters:
    -----------
    dataset_df : pd.DataFrame
        Preprocessed dataset
    dataset_name : str
        Name of the dataset
    date_col : str
        Name of date column
    target_col : str
        Name of target column
    forecast_horizon : int
        Number of periods to forecast
    test_size : float
        Proportion of data to use for testing
    models : list
        List of model names to run
    """
    # Create results directory
    output_dir = os.path.join('results', dataset_name)
    create_directory(output_dir)
    
    # Create visualizations directory
    viz_dir = os.path.join(output_dir, 'visualizations')
    create_directory(viz_dir)
    
    # Create forecasts directory
    forecasts_dir = os.path.join(output_dir, 'forecasts')
    create_directory(forecasts_dir)
    
    # Create models directory
    models_dir = os.path.join(output_dir, 'models')
    create_directory(models_dir)
    
    # Map model names to model classes
    model_map = {
        'arima': ARIMAModel,
        'sarima': SARIMAModel,
        'ets': ExponentialSmoothingModel,
        'prophet': ProphetModel,
        'rf': RandomForestModel,
        'gbm': GradientBoostingModel,
        'lstm': LSTMModel,
        'ensemble': EnsembleModel
    }
    
    # Add XGBoost if available
    if 'XGBOOST_AVAILABLE' in globals() and XGBOOST_AVAILABLE:
        model_map['xgboost'] = XGBoostModel
    
    # Run selected models
    all_models = []
    all_forecasts = []
    
    for model_name in models:
        if model_name.lower() not in model_map:
            # Special case for xgboost when not available
            if model_name.lower() == 'xgboost' and ('XGBOOST_AVAILABLE' not in globals() or not XGBOOST_AVAILABLE):
                print(f"Warning: XGBoost is not available. Skipping XGBoost model.")
            else:
                print(f"Warning: Unknown model '{model_name}'. Skipping.")
            continue
        
        print(f"\n{'='*50}")
        print(f"Running {model_name.upper()} model")
        print(f"{'='*50}")
        
        model_class = model_map[model_name.lower()]
        
        try:
            # Initialize and fit model
            model = model_class(output_dir=output_dir)
            model.fit(
                dataset_df, 
                date_col=date_col,
                target_col=target_col,
                test_size=test_size
            )
            
            # Generate and save forecast
            forecast = model.forecast(forecast_horizon)
            model.save_forecast(forecast, f"{model_name}_{dataset_name}")
            
            # Generate and save visualizations
            model.plot_forecast(f"{model_name}_{dataset_name}")
            
            # Evaluate model
            metrics = model.evaluate()
            print(f"\n{model_name.upper()} Model Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
            
            # Save model
            model.save_model(os.path.join(models_dir, f"{model_name}_{dataset_name}.pkl"))
            
            all_models.append(model)
            all_forecasts.append((model_name, forecast))
            
        except Exception as e:
            print(f"Error running {model_name} model: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Train ensemble model if specified and we have at least 2 other models
    if 'ensemble' in [m.lower() for m in models] and len(all_models) > 1 and len(all_forecasts) > 1:
        try:
            print(f"\n{'='*50}")
            print(f"Running ENSEMBLE model")
            print(f"{'='*50}")
            
            # Get component models (excluding Ensemble itself)
            component_models = [model for model in all_models if not isinstance(model, EnsembleModel)]
            
            # Initialize and fit ensemble model
            ensemble = EnsembleModel(output_dir=output_dir)
            ensemble.fit_with_models(
                component_models,
                dataset_df,
                date_col=date_col,
                target_col=target_col,
                test_size=test_size
            )
            
            # Generate and save forecast
            forecast = ensemble.forecast(forecast_horizon)
            ensemble.save_forecast(forecast, f"ensemble_{dataset_name}")
            
            # Generate and save visualizations
            ensemble.plot_forecast(f"ensemble_{dataset_name}")
            
            # Evaluate model
            metrics = ensemble.evaluate()
            print(f"\nENSEMBLE Model Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
            
            # Save model
            ensemble.save_model(os.path.join(models_dir, f"ensemble_{dataset_name}.pkl"))
            
        except Exception as e:
            print(f"Error running ensemble model: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\nForecasting completed!")

def run_dashboard():
    """
    Launch the Streamlit dashboard
    """
    try:
        # Check if streamlit is installed
        import streamlit
        print("Starting the dashboard...")
        
        # Important: Make sure we're using the dashboard.py in the current directory
        dashboard_path = 'dashboard.py'
        
        # Debug information
        print(f"Debug - Dashboard path: {os.path.abspath(dashboard_path)}")
        print(f"Debug - Path exists: {os.path.exists(dashboard_path)}")
        
        # Ensure the dashboard file exists
        if not os.path.exists(dashboard_path):
            print(f"Error: Dashboard file not found at '{dashboard_path}'")
            print("Current directory:", os.getcwd())
            print("Files in current directory:", os.listdir('.'))
            return
        
        # Run the dashboard directly - simplest approach
        command = f"streamlit run {dashboard_path}"
        print(f"Running command: {command}")
        
        # Use direct system call without capturing output to allow streamlit to open browser
        os.system(command)
        
    except ImportError:
        print("Error: Streamlit is not installed. Please install it using 'pip install streamlit'.")
        sys.exit(1)
    except Exception as e:
        print(f"Error launching dashboard: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def ensure_directories_exist():
    """
    Create necessary directories if they don't exist
    
    This function checks for and creates the essential project directories:
    - data: for storing raw datasets
    - models: for saving trained model objects
    - results: for storing forecasting results
    - archive: for deprecated files
    """
    required_dirs = ['data', 'models', 'results', 'archive']
    
    # Check if data directory has necessary datasets
    data_dir = 'data'
    if not os.path.exists(data_dir):
        try:
            os.makedirs(data_dir)
            print(f"Created directory: {data_dir}")
            print(f"Warning: The data directory was created but contains no datasets.")
            print(f"         Please add datasets to the 'data' directory or use a custom dataset with --data_path.")
        except Exception as e:
            print(f"Error: Could not create data directory: {str(e)}")
            print(f"       Please create the 'data' directory manually and add datasets.")
    else:
        # Check if standard datasets exist
        standard_datasets = ['amazon.csv', 'car_prices.csv']
        missing_datasets = [ds for ds in standard_datasets if not os.path.exists(os.path.join(data_dir, ds))]
        
        if missing_datasets:
            print(f"Warning: The following standard datasets are missing from the 'data' directory:")
            for ds in missing_datasets:
                print(f"         - {ds}")
            print(f"         You can still use custom datasets with --dataset custom --data_path YOUR_FILE.csv")
    
    # Create other required directories
    for directory in required_dirs[1:]:  # Skip data directory as it's handled above
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"Created directory: {directory}")
            except Exception as e:
                print(f"Error: Could not create {directory} directory: {str(e)}")
                print(f"       This may cause issues when saving results.")
    
    # Check write permissions on directories
    for directory in required_dirs:
        if os.path.exists(directory):
            try:
                # Try to create a test file to verify write permissions
                test_file = os.path.join(directory, '.write_test')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)  # Clean up test file
            except Exception as e:
                print(f"Warning: The '{directory}' directory exists but may not be writable: {str(e)}")
                print(f"         This may cause issues when saving files.")

def main():
    """
    Main function to run forecasting models
    """
    # Parse arguments
    args = parse_arguments()
    
    # Print information about available features
    if 'XGBOOST_AVAILABLE' not in globals() or not XGBOOST_AVAILABLE:
        print("\nNote: XGBoost is not available. Install it with 'pip install xgboost' to use XGBoost models.")
    
    # Ensure required directories exist
    ensure_directories_exist()
    
    # If only dashboard is requested, launch it and exit
    if args.only_dashboard:
        run_dashboard()
        return
    
    # Use quick models if specified
    if args.quick:
        quick_models = ['arima', 'prophet', 'rf']
        if 'XGBOOST_AVAILABLE' in globals() and XGBOOST_AVAILABLE:
            quick_models.append('xgboost')
        args.models = quick_models
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    dataset_df, date_col, target_col = load_dataset(
        args.dataset, 
        args.data_path, 
        args.date_col, 
        args.target_col
    )
    
    # Run forecasting models
    print(f"\nRunning forecasting models for {args.dataset} dataset...")
    print(f"Models: {', '.join(args.models)}")
    print(f"Forecast horizon: {args.forecast_horizon} periods")
    print(f"Test size: {args.test_size}")
    
    run_models(
        dataset_df, 
        args.dataset, 
        date_col, 
        target_col, 
        args.forecast_horizon, 
        args.test_size, 
        args.models
    )
    
    # Launch dashboard if requested
    if args.run_dashboard:
        run_dashboard()

if __name__ == "__main__":
    main() 