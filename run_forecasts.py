"""
Script to run forecasts for different datasets and model combinations.
"""
import argparse
import os
import sys

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Time Series Forecasting Runner')
    
    parser.add_argument('--dataset', type=str, required=True, choices=['amazon', 'car_prices', 'all'],
                      help='Dataset to use (amazon, car_prices, or all)')
    parser.add_argument('--mode', type=str, choices=['sample', 'full'], default='sample',
                      help='Mode to run (sample generates mock data, full runs actual forecasts)')
    
    return parser.parse_args()

def main():
    """Main function to run forecasts"""
    args = parse_arguments()
    
    if args.mode == 'sample':
        print(f"Generating sample forecasts for {args.dataset} dataset(s)...")
        
        if args.dataset in ['amazon', 'all']:
            print("\nRunning Amazon sample forecasts...")
            os.system('python generate_forecasts.py')
            
        if args.dataset in ['car_prices', 'all']:
            print("\nRunning Car Prices sample forecasts...")
            os.system('python generate_car_prices_forecasts.py')
            
    else:  # Full mode
        print("Full mode is not yet implemented.")
        print("This would run the actual forecasting pipeline on real data.")
    
    # Launch the dashboard
    print("\nLaunching the dashboard...")
    os.system('streamlit run dashboard.py')

if __name__ == "__main__":
    main() 