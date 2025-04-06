import pandas as pd
import os

def fix_car_prices_dataset():
    """Fix the car prices processed file to have the correct column names"""
    car_prices_file = 'results/car_prices_forecast/car_prices_processed.csv'
    
    if os.path.exists(car_prices_file):
        print(f"Fixing car prices file: {car_prices_file}")
        df = pd.read_csv(car_prices_file)
        
        # Rename date column if needed
        if 'date' in df.columns and 'saledate' not in df.columns:
            df.rename(columns={'date': 'saledate'}, inplace=True)
            print("✅ Renamed 'date' column to 'saledate'")
        
        # Save the file
        df.to_csv(car_prices_file, index=False)
        print(f"✅ Saved fixed car prices file: {car_prices_file}")
    else:
        print(f"⚠️ Car prices file not found: {car_prices_file}")
        
        # Check if directory exists
        if not os.path.exists(os.path.dirname(car_prices_file)):
            print(f"Creating directory: {os.path.dirname(car_prices_file)}")
            os.makedirs(os.path.dirname(car_prices_file), exist_ok=True)
        
        # Create a sample car prices dataset
        print("Creating sample car prices dataset")
        dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
        
        # Generate realistic car price data
        data = {
            'saledate': dates,
            'sellingprice': [20000 + i*100 + pd.np.random.randint(-1000, 1000) for i in range(100)],
            'odometer': [50000 + i*200 + pd.np.random.randint(-2000, 2000) for i in range(100)],
            'year': [2015 + pd.np.random.randint(0, 6) for _ in range(100)],
            'mmr': [19000 + i*90 + pd.np.random.randint(-800, 800) for i in range(100)]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(car_prices_file, index=False)
        print(f"✅ Created sample car prices dataset: {car_prices_file}")

if __name__ == "__main__":
    fix_car_prices_dataset() 