# Package Upgrade Notes

This project has been updated to work with the latest versions of all dependencies. Below are the changes made to ensure compatibility:

## Requirements

The project now uses the following package versions:

```
numpy>=1.26.0
pandas>=2.2.0
matplotlib>=3.8.0
seaborn>=0.13.0
scikit-learn>=1.4.0
statsmodels>=0.14.0
tensorflow>=2.15.0
scipy>=1.12.0
prophet>=1.1.5
pmdarima>=2.0.4
jupyterlab>=4.0.11
ipywidgets>=8.1.1
```

## Code Changes

### Pandas API Changes

1. **isocalendar() Method**: Updated to handle the change in pandas 2.0+ where `isocalendar()` returns a DataFrame instead of a Series with attributes.

   ```python
   # Before
   df['weekofyear'] = df.index.isocalendar().week
   
   # After
   try:
       isocal = df.index.isocalendar()
       if hasattr(isocal, 'week'):  # Older pandas versions
           df['weekofyear'] = isocal.week
       else:  # Newer pandas versions (2.0+)
           df['weekofyear'] = isocal['week']
   except Exception as e:
       # Fallback implementation
       df['weekofyear'] = (df['dayofyear'] - 1) // 7 + 1
   ```

2. **infer_freq Method**: Added error handling for cases where frequency cannot be inferred, which is more common in newer pandas versions.

   ```python
   # Before
   index = pd.date_range(start=start_date, periods=steps, freq=pd.infer_freq(series.index))
   
   # After
   freq = pd.infer_freq(series.index)
   if freq is None:
       print("Warning: Could not infer frequency. Using daily frequency.")
       freq = 'D'
   index = pd.date_range(start=start_date, periods=steps, freq=freq)
   ```

### TensorFlow API Changes

1. **Optimizer Changes**: Updated to handle changes in TensorFlow 2.15+ optimizer API.

   ```python
   # Before
   model.compile(optimizer='adam', loss='mse')
   
   # After
   try:
       from tensorflow.keras.optimizers.legacy import Adam
       model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
   except ImportError:
       model.compile(optimizer='adam', loss='mse')
   ```

2. **predict() Method**: Updated to suppress warnings in TensorFlow 2.15+.

   ```python
   # Before
   pred = model.predict(input_data)[0]
   
   # After
   pred = model.predict(input_data, verbose=0)[0]
   ```

### Seaborn API Changes

1. **barplot Function**: Updated to use the new DataFrame-based API in seaborn 0.13+.

   ```python
   # Before
   sns.barplot(x=models, y=rmse_values)
   
   # After
   try:
       # New syntax (seaborn 0.13+)
       model_comparison_df = pd.DataFrame({'Model': models, 'RMSE': rmse_values})
       sns.barplot(data=model_comparison_df, x='Model', y='RMSE')
   except TypeError:
       # Fall back to old syntax for older seaborn versions
       sns.barplot(x=models, y=rmse_values)
   ```

## Benefits of the Upgrade

1. **Security Improvements**: Latest versions include security patches and bug fixes.
2. **Performance Enhancements**: Newer algorithms and optimizations in the latest versions.
3. **New Features**: Access to new functionality in the updated libraries.
4. **Better Error Handling**: More robust code that handles edge cases better.
5. **Future-Proofing**: The code is now ready for forward compatibility.

## Testing

After upgrading, it's recommended to test the forecasting pipeline with both datasets:

```bash
python src/run_forecasting.py --dataset amazon
python src/run_forecasting.py --dataset car_prices
``` 