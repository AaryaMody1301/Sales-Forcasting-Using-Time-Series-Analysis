"""
Simple wrapper script to run the Amazon dataset forecasting.
"""
import os
import sys
from src.analyze_datasets import DatasetAnalyzer
from src.forecasting import TimeSeriesForecasting

def main():
    """Main function to run the Amazon forecasting example"""
    print("\n" + "="*60)
    print("Time Series Forecasting Example: AMAZON")
    print("="*60)
    
    # Create output directory
    output_dir = "results/amazon_forecast"
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Analyze and preprocess dataset
    print("\n" + "="*60)
    print("Step 1: Analyze and preprocess dataset")
    print("="*60)
    
    analyzer = DatasetAnalyzer(output_path=os.path.join(output_dir, 'analysis'))
    
    try:
        # Load and process Amazon dataset
        data_path = 'data/amazon.csv'
        df = analyzer.load_dataset(data_path)
        processed_df = analyzer.preprocess_amazon_data(df)
        target_column = 'daily_sales'
        
        # Analyze dataset
        analyzer.analyze_dataset(processed_df, 'amazon')
        
        # Save processed dataset
        processed_path = os.path.join(output_dir, 'amazon_processed.csv')
        processed_df.to_csv(processed_path)
        print(f"Processed data saved to {processed_path}")
    
    except Exception as e:
        print(f"Error in dataset analysis: {str(e)}")
        return
    
    # Step 2: Train forecasting models
    print("\n" + "="*60)
    print("Step 2: Train forecasting models")
    print("="*60)
    
    forecaster = TimeSeriesForecasting(output_path=os.path.join(output_dir, 'forecasts'))
    
    try:
        # Load data for forecasting
        forecaster.load_data(processed_path, target_col=target_column)
        
        # Split data
        forecaster.split_data(test_size=0.2)
        
        # Train models - avoiding LSTM since we don't have TensorFlow
        models_to_train = ['arima', 'ets', 'rf', 'lr']
        for model in models_to_train:
            try:
                print(f"\nTraining {model} model...")
                if model.lower() == 'arima':
                    forecaster.train_arima(order=(1, 1, 1))
                elif model.lower() == 'ets':
                    forecaster.train_exponential_smoothing(trend='add', seasonal=None)
                elif model.lower() == 'rf':
                    forecaster.train_ml_model(model_type='rf', n_lags=12)
                elif model.lower() == 'lr':
                    forecaster.train_ml_model(model_type='lr', n_lags=12)
            except Exception as e:
                print(f"Error training {model} model: {str(e)}")
        
        # Step 3: Generate forecasts and evaluate models
        print("\n" + "="*60)
        print("Step 3: Generate forecasts and evaluate models")
        print("="*60)
        
        # Evaluate models
        results = forecaster.evaluate_models(steps=30)
        
        # Print evaluation results
        print("\nModel Evaluation Results:")
        print(results)
        
        # Create ensemble forecast
        ensemble = forecaster.ensemble_forecast(steps=30)
        
        print(f"\nForecasting completed successfully!")
        print(f"Results saved to {output_dir}")
        
    except Exception as e:
        print(f"Error in forecasting: {str(e)}")

if __name__ == "__main__":
    main() 