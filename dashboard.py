import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
import glob
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure the page
st.set_page_config(
    page_title="Advanced Time Series Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .title {
        font-size: 46px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 20px;
        text-align: center;
        padding: 20px 0;
        border-bottom: 2px solid #ecf0f1;
    }
    .subtitle {
        font-size: 28px;
        font-weight: 500;
        color: #3498db;
        margin-bottom: 15px;
        padding-left: 10px;
        border-left: 5px solid #3498db;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f3f9;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding: 10px 16px;
        font-size: 16px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #d9e2ef;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding: 20px;
        border: 1px solid #e6e9ef;
        border-radius: 0px 4px 4px 4px;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .insight-card {
        background-color: #fff;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
        border-left: 4px solid #3498db;
    }
    .info-box {
        padding: 12px;
        border-radius: 6px;
        background-color: #e8f4fd;
        border-left: 4px solid #3498db;
        margin-bottom: 15px;
    }
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e6e9ef;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_dataset_info(selected_dir):
    """
    Extract dataset information from the results directory
    
    Parameters:
    -----------
    selected_dir : str
        Path to the selected results directory
        
    Returns:
    --------
    tuple
        (dataset_name, dataset_type, date_col, target_col)
    """
    dataset_name = None
    dataset_type = None
    date_col = None
    target_col = None
    
    # Print debug info
    print(f"Getting dataset info for directory: {selected_dir}")
    
    # Check for known dataset types
    for name in ["amazon", "car_prices"]:
        if name in selected_dir.lower():  # Make case-insensitive
            dataset_name = name.upper()
            dataset_type = name
            
            # Set default columns for known datasets
            if name == "amazon":
                date_col = "date"
                target_col = "daily_sales"
                print(f"Detected Amazon dataset - date_col: {date_col}, target_col: {target_col}")
            elif name == "car_prices":
                date_col = "saledate"
                target_col = "sellingprice"
                print(f"Detected Car Prices dataset - date_col: {date_col}, target_col: {target_col}")
            
            break
    
    # If not a known dataset, extract name from directory
    if not dataset_name:
        # Extract dataset name from directory (format: dataset_forecast_timestamp)
        dir_parts = os.path.basename(selected_dir).split('_')
        if len(dir_parts) > 0:
            dataset_name = dir_parts[0].upper()
            dataset_type = "custom"
        else:
            dataset_name = "DATASET"
            dataset_type = "custom"
        print(f"Using custom dataset name: {dataset_name}")
    
    # Try to get column info from config file if it exists
    config_file = os.path.join(selected_dir, 'config.json')
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                # Only override default columns if specified in config
                if config.get('date_col'):
                    date_col = config.get('date_col')
                if config.get('target_col'):
                    target_col = config.get('target_col')
                print(f"Found config file: date_col={date_col}, target_col={target_col}")
        except Exception as e:
            print(f"Error reading config file: {str(e)}")
    
    # Also check column_info.json which is newer format
    column_info_file = os.path.join(selected_dir, 'column_info.json')
    if os.path.exists(column_info_file):
        try:
            with open(column_info_file, 'r') as f:
                column_info = json.load(f)
                # Only override if not already set
                if column_info.get('date_col') and not date_col:
                    date_col = column_info.get('date_col')
                if column_info.get('target_col') and not target_col:
                    target_col = column_info.get('target_col')
                print(f"Found column_info file: date_col={date_col}, target_col={target_col}")
        except Exception as e:
            print(f"Error reading column_info file: {str(e)}")
    
    print(f"Final column selection: date_col={date_col}, target_col={target_col}")
    return dataset_name, dataset_type, date_col, target_col 

@st.cache_data
def detect_columns(df):
    """
    Auto-detect date and target columns in a DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
        
    Returns:
    --------
    tuple
        (date_col, target_col)
    """
    date_col = None
    target_col = None
    
    # Try to detect date column
    # First check if the first column looks like a date
    if len(df.columns) > 0:
        first_col = df.columns[0]
        try:
            pd.to_datetime(df[first_col])
            date_col = first_col
        except:
            # Look for columns that might be dates
            date_candidates = [col for col in df.columns if any(date_term in col.lower() 
                            for date_term in ['date', 'time', 'day', 'month', 'year'])]
            if date_candidates:
                date_col = date_candidates[0]
    
    # Try to detect target column
    numeric_cols = df.select_dtypes(include=['number']).columns
    target_candidates = [col for col in numeric_cols if any(target_term in col.lower() 
                        for target_term in ['price', 'sale', 'value', 'target', 'amount', 'quantity'])]
    
    if target_candidates:
        target_col = target_candidates[0]
    elif len(numeric_cols) > 0:
        # Use first numeric column as fallback
        target_col = numeric_cols[0]
        
    return date_col, target_col

@st.cache_data
def load_and_process_dataframe(file_path, date_col=None):
    """
    Load and do initial processing of the dataframe with caching
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    date_col : str, optional
        Name of the date column
        
    Returns:
    --------
    pd.DataFrame
        Processed dataframe
    """
    try:
        print(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        # If date_col exists, convert it to datetime for processing
        if date_col and date_col in df.columns:
            # Convert to datetime for processing
            try:
                print(f"Converting {date_col} to datetime")
                df[date_col] = pd.to_datetime(df[date_col])
            except Exception as e:
                print(f"Warning: Could not convert {date_col} to datetime: {str(e)}")
                # Try to infer date column if conversion fails
                date_candidates = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                for candidate in date_candidates:
                    if candidate != date_col:
                        try:
                            df[candidate] = pd.to_datetime(df[candidate])
                            print(f"Successfully converted alternative date column {candidate}")
                            # We don't change date_col here as that should be handled at higher level
                            break
                        except:
                            continue
        
        return df
    except Exception as e:
        print(f"Error loading dataframe: {str(e)}")
        # Return empty dataframe on error
        return pd.DataFrame()

@st.cache_data
def get_date_range_str(df, date_col):
    """Get formatted date range string"""
    if date_col in df.columns and len(df) > 0:
        try:
            # First check if values are already datetime objects
            if pd.api.types.is_datetime64_any_dtype(df[date_col]):
                min_date = df[date_col].min()
                max_date = df[date_col].max()
                return f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
            else:
                # Try to convert strings to datetime for the range calculation only
                try:
                    date_series = pd.to_datetime(df[date_col])
                    min_date = date_series.min()
                    max_date = date_series.max()
                    return f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
                except:
                    # If conversion fails, just return min and max as strings
                    min_date = str(df[date_col].min())
                    max_date = str(df[date_col].max())
                    return f"{min_date} to {max_date}"
        except Exception as e:
            print(f"Error getting date range: {str(e)}")
            return "Date range unavailable"
    return "No data"

@st.cache_data
def load_forecast_results(eval_file):
    """Load model evaluation results with caching"""
    try:
        # Try to load as a standard CSV with metrics
        results_df = pd.read_csv(eval_file)
        
        # Check if this looks like a model evaluation file (should have Model and metrics columns)
        if "Model" in results_df.columns and any(metric in results_df.columns for metric in ["RMSE", "MAE", "MAPE"]):
            return results_df
        
        # If the file doesn't have the expected structure, it might be a forecast CSV or other file
        # Try to determine content type and convert to a usable format
        print(f"File {eval_file} doesn't have the expected evaluation metrics structure")
        print(f"Columns found: {results_df.columns.tolist()}")
        
        # If it's a forecast file with model names as the first column, try to convert it
        if results_df.shape[1] > 1:
            first_col = results_df.columns[0]
            # Check if first values look like model names
            first_values = results_df[first_col].head(3).tolist()
            if any(model_name in str(val).lower() for val in first_values for model_name in 
                  ['arima', 'sarima', 'prophet', 'xgboost', 'lstm']):
                # This is likely a summary table with model names rather than dates
                print("File appears to contain model names rather than dates - trying to convert to metrics format")
                
                # Try to extract metrics from other columns
                numeric_cols = results_df.select_dtypes(include=['number']).columns.tolist()
                
                if numeric_cols:
                    # Create a new DataFrame in the expected format
                    metrics_data = {
                        "Model": results_df[first_col].tolist()
                    }
                    
                    # Add any numeric columns as metrics
                    for col in numeric_cols:
                        metrics_data[col.upper()] = results_df[col].tolist()
                    
                    return pd.DataFrame(metrics_data)
        
        # If none of the above worked, return as-is and let the UI handle it
        return results_df
            
    except Exception as e:
        print(f"Error loading forecast results: {str(e)}")
        # Return empty DataFrame instead of failing
        return pd.DataFrame({"Model": ["Error loading data"], 
                            "RMSE": [999.99], 
                            "MAE": [999.99],
                            "MAPE": [999.99],
                            "Error": [str(e)]}) 

@st.cache_data
def get_result_directories():
    """Get sorted list of result directories"""
    results_dirs = glob.glob("results/*")
    if results_dirs:
        results_dirs.sort(key=os.path.getmtime, reverse=True)
    return results_dirs

def render_model_comparison_tab(tab, selected_dir, display_options):
    """Render content for model comparison tab"""
    with tab:
        st.markdown("<div class='subtitle'>Model Performance Comparison</div>", unsafe_allow_html=True)
        
        # Check for model evaluation results
        eval_file = os.path.join(selected_dir, "forecasts", "model_evaluation.csv")
        
        if os.path.exists(eval_file):
            try:
                # Safely load evaluation results
                results_df = load_forecast_results(eval_file)
                
                # Verify this is a proper model evaluation file
                required_cols = ["Model", "RMSE", "MAE"]
                missing_cols = [col for col in required_cols if col not in results_df.columns]
                
                if missing_cols:
                    st.error(f"Model evaluation file is missing required columns: {', '.join(missing_cols)}")
                    with st.expander("Data Preview"):
                        st.dataframe(results_df.head())
                    st.warning("Please run model evaluation again to generate proper metrics.")
                    return
                
                # Display metrics in columns with modern cards
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.subheader("🏆 Best Model by RMSE")
                    best_rmse = results_df.loc[results_df["RMSE"].idxmin()]
                    st.metric("Model", best_rmse["Model"], help="Model with lowest Root Mean Squared Error")
                    st.metric("RMSE", f"{best_rmse['RMSE']:.4f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                with col2:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.subheader("🥇 Best Model by MAE")
                    best_mae = results_df.loc[results_df["MAE"].idxmin()]
                    st.metric("Model", best_mae["Model"], help="Model with lowest Mean Absolute Error")
                    st.metric("MAE", f"{best_mae['MAE']:.4f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                with col3:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.subheader("🎯 Best Model by MAPE")
                    if "MAPE" in results_df.columns:
                        best_mape = results_df.loc[results_df["MAPE"].idxmin()]
                        st.metric("Model", best_mape["Model"], help="Model with lowest Mean Absolute Percentage Error")
                        st.metric("MAPE", f"{best_mape['MAPE']:.2f}%")
                    else:
                        st.metric("Model", "N/A", help="MAPE metrics not available")
                        st.metric("MAPE", "N/A")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Comparison table with a modern look
                st.subheader("📋 All Models Performance")
                
                # Select only numeric columns for styling
                numeric_cols = results_df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    styled_results = results_df.style.background_gradient(subset=numeric_cols, cmap='RdYlGn_r')
                    st.dataframe(styled_results, use_container_width=True)
                else:
                    st.dataframe(results_df, use_container_width=True)
                
                # Use Plotly for interactive visualizations instead of static images
                if "Forecast Metrics" in display_options and "RMSE" in results_df.columns:
                    st.subheader("📉 RMSE Comparison")
                    
                    # Create interactive bar chart with Plotly
                    fig = px.bar(
                        results_df, 
                        x="Model", 
                        y="RMSE", 
                        color="RMSE",
                        color_continuous_scale="Viridis",
                        title="Model Comparison - RMSE (lower is better)"
                    )
                    fig.update_layout(
                        xaxis_title="Model",
                        yaxis_title="Root Mean Squared Error (RMSE)",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading model evaluation results: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.warning("⚠️ No model evaluation results found. Please run the model evaluation first.") 

def render_forecast_tab(tab, selected_dir, display_options, dataset_type, dataset_name, target_col):
    """Render content for forecast visualization tab"""
    with tab:
        st.markdown("<div class='subtitle'>Forecast Visualization</div>", unsafe_allow_html=True)
        
        # Find all forecast CSV files (prefer CSVs over images for interactive plots)
        forecast_csvs = glob.glob(f"{selected_dir}/forecasts/*forecast*.csv")
        forecast_images = glob.glob(f"{selected_dir}/visualizations/*forecast*.png")  # Check in visualizations directory too
        
        # If no forecast files found in standard locations, try broader search
        if not forecast_csvs and not forecast_images:
            st.warning("No forecast files found in standard location. Searching in other directories...")
            
            # Try alternative directories and patterns
            forecast_csvs = (
                glob.glob(f"{selected_dir}/*forecast*.csv") +
                glob.glob(f"{selected_dir}/**/*forecast*.csv") +
                glob.glob(f"{selected_dir}/*predict*.csv") +
                glob.glob(f"{selected_dir}/**/*predict*.csv") +
                glob.glob(f"{selected_dir}/**/*.csv")  # Last resort: any CSV file
            )
            
            forecast_images = (
                glob.glob(f"{selected_dir}/*forecast*.png") +
                glob.glob(f"{selected_dir}/**/*forecast*.png") +
                glob.glob(f"{selected_dir}/**/*.png")  # Last resort: any PNG file
            )
            
            if forecast_csvs:
                st.info(f"Found {len(forecast_csvs)} CSV files in alternative locations.")
            if forecast_images:
                st.info(f"Found {len(forecast_images)} image files in alternative locations.")
        
        # Debug info about directories and files
        st.sidebar.expander("Debug Info").markdown(f"""
        - Selected directory: `{selected_dir}`
        - Forecast CSVs found: {len(forecast_csvs)}
        - Forecast Images found: {len(forecast_images)}
        - Current working directory: `{os.getcwd()}`
        """)
        
        if forecast_csvs or forecast_images:
            # Get model names from files (prefer CSVs)
            if forecast_csvs:
                model_options = [os.path.basename(csv).replace("_forecast.csv", "") for csv in forecast_csvs]
            else:
                model_options = [os.path.basename(img).replace("_forecast.png", "") for img in forecast_images]
            
            # Add ensemble if available
            if "ensemble" in [os.path.basename(f).split("_")[0] for f in (forecast_csvs or forecast_images)]:
                models_without_ensemble = [m for m in model_options if m != "ensemble"]
                model_options = ["ensemble"] + models_without_ensemble
            
            # Column layout for filters
            col1, col2 = st.columns([1, 2])
            
            with col1:
                selected_model = st.selectbox("Select Model", model_options)
                
                # Additional options based on selected elements
                if "Confidence Intervals" in display_options:
                    show_ci = st.checkbox("Show Confidence Intervals", value=True)
                if "Trend Decomposition" in display_options:
                    show_decomposition = st.checkbox("Show Trend Decomposition", value=False)
            
            with col2:
                st.markdown("""
                <div class="info-box">
                    <p><strong>About the forecast:</strong> This visualization shows the predicted values and actual values (if available) for the time series.</p>
                    <p>The model performance can be evaluated by how closely the predictions match the actual values in the test period.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Try to find CSV file for interactive plot
            csv_files = [csv for csv in forecast_csvs if selected_model in os.path.basename(csv)]
            if csv_files:
                forecast_df = pd.read_csv(csv_files[0])
                
                # Create interactive plot
                # First determine the date column - it's typically the first column but check to be sure
                date_col_forecast = forecast_df.columns[0]
                
                # Check if the supposed date column actually contains date-like values
                try:
                    # Print first few values to debug
                    first_values = forecast_df[date_col_forecast].head(3).tolist()
                    print(f"First values in date column '{date_col_forecast}': {first_values}")
                    
                    # Check if values look like model names (common error pattern)
                    if any(model_name in str(val).lower() for val in first_values for model_name in 
                          ['arima', 'sarima', 'prophet', 'xgboost', 'lstm']):
                        st.error(f"First column '{date_col_forecast}' contains model names, not dates. This is likely not a forecast file but a summary file.")
                        st.info("Please select a different file or run forecasts again to generate valid forecast files.")
                        
                        # Show raw data for debugging
                        with st.expander("Raw Data Preview"):
                            st.dataframe(forecast_df.head())
                        return
                    
                    # Try to convert to datetime
                    forecast_df[date_col_forecast] = pd.to_datetime(forecast_df[date_col_forecast])
                except Exception as e:
                    st.warning(f"Could not convert column '{date_col_forecast}' to dates: {str(e)}")
                    # Try to find a different date column
                    date_candidates = [col for col in forecast_df.columns if any(date_term in col.lower() 
                                      for date_term in ['date', 'time', 'day', 'month', 'year'])]
                    
                    if date_candidates:
                        date_col_forecast = date_candidates[0]
                        st.info(f"Using alternative date column: {date_col_forecast}")
                        try:
                            forecast_df[date_col_forecast] = pd.to_datetime(forecast_df[date_col_forecast])
                        except:
                            st.error(f"Could not convert alternative column '{date_col_forecast}' to dates either.")
                            # Show raw data to help diagnose
                            with st.expander("Raw Data Preview"):
                                st.dataframe(forecast_df.head())
                            # Continue anyway and use the original column as-is
                
                fig = go.Figure()
                
                # Add actual values
                if 'actual' in forecast_df.columns:
                    fig.add_trace(go.Scatter(
                        x=forecast_df[date_col_forecast], y=forecast_df['actual'],
                        mode='lines', name='Actual', line=dict(color='blue', width=2)
                    ))
                
                # Add training data
                if 'train' in forecast_df.columns:
                    fig.add_trace(go.Scatter(
                        x=forecast_df[date_col_forecast], y=forecast_df['train'],
                        mode='lines', name='Training Data', line=dict(color='green', width=2)
                    ))
                
                # Add test data
                if 'test' in forecast_df.columns:
                    fig.add_trace(go.Scatter(
                        x=forecast_df[date_col_forecast], y=forecast_df['test'],
                        mode='lines', name='Test Data', line=dict(color='orange', width=2)
                    ))
                
                # Add forecast
                if 'forecast' in forecast_df.columns:
                    fig.add_trace(go.Scatter(
                        x=forecast_df[date_col_forecast], y=forecast_df['forecast'],
                        mode='lines', name='Forecast', line=dict(color='red', width=2, dash='dash')
                    ))
                elif 'prediction' in forecast_df.columns:
                    # Alternative column name for forecast
                    fig.add_trace(go.Scatter(
                        x=forecast_df[date_col_forecast], y=forecast_df['prediction'],
                        mode='lines', name='Forecast', line=dict(color='red', width=2, dash='dash')
                    ))
                elif 'pred' in forecast_df.columns:
                    # Another alternative column name
                    fig.add_trace(go.Scatter(
                        x=forecast_df[date_col_forecast], y=forecast_df['pred'],
                        mode='lines', name='Forecast', line=dict(color='red', width=2, dash='dash')
                    ))
                else:
                    # Handle case when no forecast column exists
                    st.warning("Forecast data column not found in the CSV file. Expected 'forecast', 'prediction', or 'pred' column.")
                    
                    # Try to find the forecast column based on patterns
                    forecast_cols = [col for col in forecast_df.columns if 'forecast' in col.lower() 
                                    or 'predict' in col.lower() or 'pred' in col.lower()]
                    
                    if forecast_cols:
                        # Use the first matching column
                        forecast_col = forecast_cols[0]
                        st.info(f"Using '{forecast_col}' as the forecast column.")
                        fig.add_trace(go.Scatter(
                            x=forecast_df[date_col_forecast], y=forecast_df[forecast_col],
                            mode='lines', name='Forecast', line=dict(color='red', width=2, dash='dash')
                        ))
                    else:
                        # Display the columns we found for debugging
                        st.error(f"Available columns: {', '.join(forecast_df.columns)}")
                
                # Add confidence intervals if available and requested
                if "show_ci" in locals() and show_ci:
                    if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
                        fig.add_trace(go.Scatter(
                            x=forecast_df[date_col_forecast], y=forecast_df['upper_bound'],
                            mode='lines', line=dict(width=0), showlegend=False
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=forecast_df[date_col_forecast], y=forecast_df['lower_bound'],
                            mode='lines', line=dict(width=0), fill='tonexty',
                            fillcolor='rgba(255, 0, 0, 0.1)', name='95% Confidence Interval'
                        ))
                    elif 'lower_ci' in forecast_df.columns and 'upper_ci' in forecast_df.columns:
                        # Alternative column names for confidence intervals
                        fig.add_trace(go.Scatter(
                            x=forecast_df[date_col_forecast], y=forecast_df['upper_ci'],
                            mode='lines', line=dict(width=0), showlegend=False
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=forecast_df[date_col_forecast], y=forecast_df['lower_ci'],
                            mode='lines', line=dict(width=0), fill='tonexty',
                            fillcolor='rgba(255, 0, 0, 0.1)', name='95% Confidence Interval'
                        ))
                    elif any('lower' in col.lower() for col in forecast_df.columns) and any('upper' in col.lower() for col in forecast_df.columns):
                        # Try to find any columns with lower/upper in their names
                        lower_cols = [col for col in forecast_df.columns if 'lower' in col.lower()]
                        upper_cols = [col for col in forecast_df.columns if 'upper' in col.lower()]
                        
                        if lower_cols and upper_cols:
                            lower_col = lower_cols[0]
                            upper_col = upper_cols[0]
                            
                            st.info(f"Using '{lower_col}' and '{upper_col}' for confidence intervals.")
                            
                            fig.add_trace(go.Scatter(
                                x=forecast_df[date_col_forecast], y=forecast_df[upper_col],
                                mode='lines', line=dict(width=0), showlegend=False
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=forecast_df[date_col_forecast], y=forecast_df[lower_col],
                                mode='lines', line=dict(width=0), fill='tonexty',
                                fillcolor='rgba(255, 0, 0, 0.1)', name='95% Confidence Interval'
                            ))
                    else:
                        st.info("Confidence intervals not found in the forecast data.")
                
                fig.update_layout(
                    title=f"{selected_model.upper()} - Time Series Forecast",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    hovermode="x unified",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Fallback to image if CSV not found
            elif forecast_images:
                selected_images = [img for img in forecast_images if selected_model in os.path.basename(img)]
                if selected_images:
                    st.image(selected_images[0], use_container_width=True)
            
            # Show forecast period and target information based on dataset
            if dataset_type == 'amazon':
                st.info("📈 The forecast shows predicted **daily sales** for Amazon products.")
            elif dataset_type == 'car_prices':
                st.info("📈 The forecast shows predicted **selling prices** for cars over time.")
            else:
                if target_col:
                    st.info(f"📈 The forecast shows predicted **{target_col}** for {dataset_name.lower()}.")
                else:
                    st.info(f"📈 The forecast shows predicted values for {dataset_name.lower()}.")
                    
            # Add additional information about forecast files
            with st.expander("Forecast Files Details"):
                if forecast_csvs:
                    st.markdown("### Available CSV Files")
                    for csv in forecast_csvs:
                        st.code(f"- {os.path.basename(csv)}")
                        
                        # Try to provide column info for the selected file
                        if selected_model in os.path.basename(csv):
                            try:
                                df_sample = pd.read_csv(csv)
                                st.markdown(f"**Columns in {os.path.basename(csv)}:** {', '.join(df_sample.columns)}")
                            except:
                                pass
                
                if forecast_images:
                    st.markdown("### Available Image Files")
                    for img in forecast_images:
                        st.code(f"- {os.path.basename(img)}")
        else:
            st.warning("⚠️ No forecast data found. Please run the forecasting models first.")
            st.markdown("""
            #### How to generate forecasts:
            
            Run the following command to generate forecasts:
            ```
            python run_forecasting.py --dataset amazon --forecast_horizon 30
            ```
            
            Or for a custom dataset:
            ```
            python run_forecasting.py --dataset custom --data_path your_data.csv --forecast_horizon 30
            ```
            """)
            
            # Check if results directory exists at all
            if not os.path.exists(selected_dir):
                st.error(f"The results directory '{selected_dir}' does not exist yet. Run forecasting models to create it.")
            else:
                st.info(f"Selected directory '{selected_dir}' exists but no forecast files were found.")
                # List contents of the directory
                dir_contents = os.listdir(selected_dir)
                if dir_contents:
                    st.markdown("#### Directory contents:")
                    st.code("\n".join(dir_contents)) 

def render_advanced_analysis_tab(tab, selected_dir, processed_df, date_col, target_col):
    """Render content for advanced analysis tab"""
    with tab:
        st.markdown("<div class='subtitle'>Time Series Analysis</div>", unsafe_allow_html=True)
        
        if date_col is None or target_col is None or processed_df.empty:
            st.warning("⚠️ Cannot perform advanced analysis without proper date and target columns.")
            return
            
        # Debug info to help troubleshoot column issues
        st.sidebar.expander("Column Debug Info").markdown(f"""
        - Date column specified: `{date_col}`
        - Target column specified: `{target_col}`
        - Available columns: {list(processed_df.columns)}
        """)
        
        # Check if specified columns exist in dataframe
        if date_col not in processed_df.columns:
            st.error(f"Error: Date column '{date_col}' not found in dataframe. Available columns: {list(processed_df.columns)}")
            # Try to find a suitable date column as fallback
            date_candidates = [col for col in processed_df.columns if 'date' in col.lower() or 'time' in col.lower() or 'day' in col.lower()]
            if date_candidates:
                date_col = date_candidates[0]
                st.info(f"Using '{date_col}' as date column instead.")
            else:
                st.warning("No suitable date column found. Analysis cannot continue.")
                return
                
        if target_col not in processed_df.columns:
            st.error(f"Error: Target column '{target_col}' not found in dataframe. Available columns: {list(processed_df.columns)}")
            # Try to find a suitable numeric column as fallback
            numeric_cols = processed_df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                target_col = numeric_cols[0]
                st.info(f"Using '{target_col}' as target column instead.")
            else:
                st.warning("No suitable target column found. Analysis cannot continue.")
                return
        
        # Create analysis sections
        analysis_tabs = st.tabs(["📊 Overview", "📈 Decomposition", "🔄 Seasonality", "📉 Statistics"])
        
        with analysis_tabs[0]:
            st.subheader("Time Series Overview")
            
            # Basic time series visualization
            fig = px.line(
                processed_df, 
                x=date_col, 
                y=target_col,
                title=f"{target_col} Over Time",
                template="plotly_white"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.subheader("Summary Statistics")
            stats_df = processed_df[target_col].describe().reset_index()
            stats_df.columns = ['Statistic', 'Value']
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(stats_df, hide_index=True)
                
            with col2:
                # Distribution plot
                fig = px.histogram(
                    processed_df, 
                    x=target_col,
                    title=f"Distribution of {target_col}",
                    template="plotly_white",
                    nbins=30
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        with analysis_tabs[1]:
            st.subheader("Trend-Seasonality Decomposition")
            
            st.info("Decomposition breaks the time series into trend, seasonal, and residual components.")
            
            # Check if we have enough data for decomposition
            if len(processed_df) < 14:
                st.warning("Not enough data for decomposition (need at least 14 data points).")
            else:
                try:
                    # Calculate moving averages for trend
                    window_size = min(30, max(7, len(processed_df) // 10))
                    processed_df['trend_ma'] = processed_df[target_col].rolling(window=window_size, center=True).mean()
                    
                    # Detrend the series
                    processed_df['detrended'] = processed_df[target_col] - processed_df['trend_ma']
                    
                    # Create interactive decomposition plot
                    fig = make_subplots(
                        rows=3, 
                        cols=1,
                        subplot_titles=["Original Series", "Trend Component (Moving Average)", "Detrended Series"],
                        shared_xaxes=True,
                        vertical_spacing=0.1
                    )
                    
                    # Original series
                    fig.add_trace(
                        go.Scatter(x=processed_df[date_col], y=processed_df[target_col], name="Original"),
                        row=1, col=1
                    )
                    
                    # Trend component
                    fig.add_trace(
                        go.Scatter(x=processed_df[date_col], y=processed_df['trend_ma'], name="Trend", line=dict(color="red")),
                        row=2, col=1
                    )
                    
                    # Detrended series
                    fig.add_trace(
                        go.Scatter(x=processed_df[date_col], y=processed_df['detrended'], name="Detrended", line=dict(color="green")),
                        row=3, col=1
                    )
                    
                    fig.update_layout(height=600, title_text="Time Series Decomposition")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add explanation
                    st.markdown("""
                    <div class="info-box">
                        <h4>Interpreting the Decomposition</h4>
                        <ul>
                            <li><strong>Trend Component:</strong> Shows the long-term progression of the series.</li>
                            <li><strong>Detrended Series:</strong> Shows the seasonal and residual components after removing the trend.</li>
                        </ul>
                        <p>A strong upward or downward trend indicates a consistent direction in the data. Repeating patterns in the detrended series suggest seasonality.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error performing decomposition: {str(e)}")
        
        with analysis_tabs[2]:
            st.subheader("Seasonality Analysis")
            
            # Try to detect seasonality patterns
            if len(processed_df) < 14:
                st.warning("Not enough data for seasonality analysis (need at least 14 data points).")
            else:
                try:
                    # Check if date column exists and is datetime
                    if date_col not in processed_df.columns:
                        st.error(f"Date column '{date_col}' not found in dataframe.")
                        # Try to find a date column
                        date_candidates = [col for col in processed_df.columns if 'date' in col.lower() or 'time' in col.lower()]
                        if date_candidates:
                            date_col = date_candidates[0]
                            st.info(f"Using '{date_col}' as date column instead.")
                        else:
                            st.warning("No suitable date column found. Seasonality analysis cannot continue.")
                            return
                        
                    # Ensure date column is datetime type
                    if not pd.api.types.is_datetime64_any_dtype(processed_df[date_col]):
                        try:
                            st.info(f"Converting {date_col} to datetime format...")
                            processed_df[date_col] = pd.to_datetime(processed_df[date_col])
                        except Exception as e:
                            st.error(f"Could not convert '{date_col}' to datetime: {str(e)}")
                            st.warning("Seasonality analysis requires a proper date column. Skipping analysis.")
                            return
                
                    # Add date components if they don't exist
                    st.info("Extracting date components for seasonality analysis...")
                    processed_df['year'] = processed_df[date_col].dt.year
                    processed_df['month'] = processed_df[date_col].dt.month
                    processed_df['day_of_week'] = processed_df[date_col].dt.dayofweek
                    
                    # Create temporal aggregations
                    # Monthly pattern
                    monthly_avg = processed_df.groupby('month')[target_col].mean().reset_index()
                    monthly_avg['month_name'] = monthly_avg['month'].apply(lambda x: datetime(2000, int(x), 1).strftime('%B'))
                    
                    fig = px.bar(
                        monthly_avg, 
                        x='month_name', 
                        y=target_col,
                        title=f"Monthly Pattern - {target_col}",
                        color=target_col,
                        color_continuous_scale="Viridis"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                    # Day of week pattern
                    dow_avg = processed_df.groupby('day_of_week')[target_col].mean().reset_index()
                    dow_avg['day_name'] = dow_avg['day_of_week'].apply(lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][int(x)])
                    
                    fig = px.bar(
                        dow_avg, 
                        x='day_name', 
                        y=target_col,
                        title=f"Day of Week Pattern - {target_col}",
                        color=target_col,
                        color_continuous_scale="Viridis"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Check for weekly seasonality using autocorrelation
                    try:
                        from statsmodels.graphics.tsaplots import plot_acf
                        import matplotlib.pyplot as plt
                        import io
                        
                        # Create autocorrelation plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        plot_acf(processed_df[target_col].dropna(), ax=ax, lags=min(50, len(processed_df)//2))
                        ax.set_title(f"Autocorrelation Function (ACF) for {target_col}")
                        
                        # Save to buffer
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png")
                        buf.seek(0)
                        
                        # Show the ACF plot
                        st.subheader("Autocorrelation Analysis")
                        st.image(buf, use_container_width=True)
                        
                        # Add interpretation
                        st.markdown("""
                        <div class="info-box">
                            <h4>Interpreting the ACF Plot</h4>
                            <p>The Autocorrelation Function (ACF) shows the correlation between a time series and its lagged values.</p>
                            <ul>
                                <li>Significant spikes at regular intervals (e.g., at lags 7, 14, 21) suggest weekly seasonality.</li>
                                <li>Significant spikes at lags 12, 24, 36 suggest monthly or yearly seasonality.</li>
                                <li>Bars beyond the blue lines are statistically significant.</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Could not generate autocorrelation plot: {str(e)}")
                        
                except Exception as e:
                    st.error(f"Error analyzing seasonality: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        with analysis_tabs[3]:
            st.subheader("Statistical Tests")
            
            try:
                # Check if target column exists
                if target_col not in processed_df.columns:
                    st.error(f"Target column '{target_col}' not found in dataframe.")
                    # Try to find a numeric column
                    numeric_cols = processed_df.select_dtypes(include=['number']).columns.tolist()
                    if numeric_cols:
                        target_col = numeric_cols[0]
                        st.info(f"Using '{target_col}' as target column instead.")
                    else:
                        st.warning("No suitable numeric column found. Analysis cannot continue.")
                        return
                
                # Calculate additional statistics
                mean = processed_df[target_col].mean()
                median = processed_df[target_col].median()
                variance = processed_df[target_col].var()
                skewness = processed_df[target_col].skew()
                kurtosis = processed_df[target_col].kurtosis()
                
                # Create statistics table
                stats = [
                    {"Metric": "Mean", "Value": f"{mean:.4f}"},
                    {"Metric": "Median", "Value": f"{median:.4f}"},
                    {"Metric": "Variance", "Value": f"{variance:.4f}"},
                    {"Metric": "Skewness", "Value": f"{skewness:.4f}"},
                    {"Metric": "Kurtosis", "Value": f"{kurtosis:.4f}"}
                ]
                
                stats_df = pd.DataFrame(stats)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                # Interpret distribution
                st.markdown("### Distribution Analysis")
                
                interpretation = []
                
                # Interpret skewness
                if skewness < -1:
                    interpretation.append("The distribution is heavily left-skewed (negative skew).")
                elif skewness < -0.5:
                    interpretation.append("The distribution is moderately left-skewed.")
                elif skewness < 0.5:
                    interpretation.append("The distribution is approximately symmetric.")
                elif skewness < 1:
                    interpretation.append("The distribution is moderately right-skewed.")
                else:
                    interpretation.append("The distribution is heavily right-skewed (positive skew).")
                
                # Interpret kurtosis
                if kurtosis < -1:
                    interpretation.append("The distribution has light tails (platykurtic).")
                elif kurtosis < 1:
                    interpretation.append("The distribution has close to normal tails (mesokurtic).")
                else:
                    interpretation.append("The distribution has heavy tails with potential outliers (leptokurtic).")
                
                for line in interpretation:
                    st.info(line)
                
                # Stationarity test
                try:
                    from statsmodels.tsa.stattools import adfuller
                    
                    # Run Augmented Dickey-Fuller test
                    # Clean data for the test - remove NaN values
                    clean_data = processed_df[target_col].dropna()
                    
                    if len(clean_data) < 5:
                        st.warning("Not enough data points for stationarity test after removing NaN values.")
                    else:
                        adf_result = adfuller(clean_data)
                        
                        adf_stats = [
                            {"Metric": "ADF Test Statistic", "Value": f"{adf_result[0]:.4f}"},
                            {"Metric": "p-value", "Value": f"{adf_result[1]:.4f}"},
                            {"Metric": "# Lags Used", "Value": f"{adf_result[2]}"},
                            {"Metric": "# Observations", "Value": f"{adf_result[3]}"}
                        ]
                        
                        # Add critical values
                        for key, value in adf_result[4].items():
                            adf_stats.append({"Metric": f"Critical Value ({key})", "Value": f"{value:.4f}"})
                        
                        st.markdown("### Stationarity Test (Augmented Dickey-Fuller)")
                        st.dataframe(pd.DataFrame(adf_stats), use_container_width=True, hide_index=True)
                        
                        # Interpret stationarity
                        is_stationary = adf_result[1] < 0.05
                        if is_stationary:
                            st.success("The time series is stationary (p-value < 0.05).")
                        else:
                            st.warning("The time series is not stationary (p-value > 0.05).")
                            
                        st.markdown("""
                        <div class="info-box">
                            <h4>What is Stationarity?</h4>
                            <p>A stationary time series has statistical properties that do not change over time, such as mean, variance, and autocorrelation.</p>
                            <p>Many time series models (like ARIMA) assume stationarity, so non-stationary data may need differencing or transformation.</p>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Could not perform stationarity test: {str(e)}")
            except Exception as e:
                st.error(f"Error calculating statistics: {str(e)}")
                import traceback
                st.code(traceback.format_exc()) 

def render_sidebar(dataset_name, processed_df, date_col, date_range_str, dataset_type, target_col):
    """Render sidebar with dataset information and options"""
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2329/2329080.png", width=100)
    st.sidebar.markdown("## Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Show dataset info
    st.sidebar.markdown(f"### {dataset_name} Dataset")
    
    # Create a styled info box in the sidebar
    st.sidebar.markdown("""
    <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; margin-top:10px;">
    """, unsafe_allow_html=True)
    
    # Basic stats with icons
    st.sidebar.markdown(f"📊 **Rows:** {processed_df.shape[0]}")
    st.sidebar.markdown(f"🧮 **Columns:** {processed_df.shape[1]}")
    
    # Date range
    if date_col in processed_df.columns and len(processed_df) > 0:
        st.sidebar.markdown(f"📅 **Date Range:** {date_range_str}")
        
    # Dataset-specific metrics
    if dataset_type == 'amazon':
        if all(col in processed_df.columns for col in ['discounted_price', 'actual_price']):
            avg_discount = (processed_df['actual_price'] - processed_df['discounted_price']).mean()
            st.sidebar.markdown(f"💰 **Avg. Discount:** ${avg_discount:.2f}")
        
        if 'rating' in processed_df.columns:
            avg_rating = processed_df['rating'].mean()
            st.sidebar.markdown(f"⭐ **Avg. Rating:** {avg_rating:.1f}")
            
        if 'category' in processed_df.columns:
            top_categories = processed_df['category'].value_counts().head(3).index.tolist()
            st.sidebar.markdown(f"📦 **Top Categories:** {', '.join(top_categories)}")
            
    elif dataset_type == 'car_prices':
        if 'odometer' in processed_df.columns:
            avg_odometer = processed_df['odometer'].mean()
            st.sidebar.markdown(f"🚗 **Avg. Odometer:** {avg_odometer:.0f} miles")
            
        if 'year' in processed_df.columns:
            avg_year = processed_df['year'].mean()
            st.sidebar.markdown(f"📆 **Avg. Year:** {avg_year:.0f}")
            
        if 'make' in processed_df.columns:
            top_makes = processed_df['make'].value_counts().head(3).index.tolist()
            st.sidebar.markdown(f"🏭 **Top Makes:** {', '.join(top_makes)}")
    else:
        # For custom datasets, show some basic statistics for target column
        if target_col and target_col in processed_df.columns:
            if pd.api.types.is_numeric_dtype(processed_df[target_col]):
                mean_val = processed_df[target_col].mean()
                min_val = processed_df[target_col].min()
                max_val = processed_df[target_col].max()
                st.sidebar.markdown(f"📊 **Target Column:** {target_col}")
                st.sidebar.markdown(f"📈 **Mean Value:** {mean_val:.2f}")
                st.sidebar.markdown(f"📉 **Range:** {min_val:.2f} - {max_val:.2f}")
    
    st.sidebar.markdown("</div>", unsafe_allow_html=True)
    
    # Advanced options
    st.sidebar.markdown("### ⚙️ Advanced Options")
    display_options = st.sidebar.multiselect(
        "Display Elements",
        options=["Forecast Metrics", "Confidence Intervals", "Trend Decomposition", "Cross Validation", "Feature Importance"],
        default=["Forecast Metrics", "Confidence Intervals"]
    )
    
    return display_options

def main():
    # Header with professional logo and title
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("<div class='title'>Advanced Time Series Forecasting Dashboard</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #7f8c8d;'>Interactive visualization and analysis of time series forecasting models</p>", unsafe_allow_html=True)
    
    # Get all result directories
    results_dirs = get_result_directories()
    if not results_dirs:
        st.error("⚠️ No results found. Please run the forecasting models first.")
        return
    
    # Select dataset
    selected_dir = st.sidebar.selectbox(
        "Select Results Directory",
        results_dirs,
        format_func=lambda x: os.path.basename(x)
    )
    
    # Get dataset information
    dataset_name, dataset_type, date_col, target_col = get_dataset_info(selected_dir)
    
    # Check if processed data exists
    processed_files = glob.glob(f"{selected_dir}/*_processed.csv")
    
    date_range_str = "No data"
    processed_df = pd.DataFrame()
    
    if processed_files:
        try:
            # Load and process the dataframe
            processed_df = load_and_process_dataframe(processed_files[0], date_col)
            print(f"Loaded processed dataframe from {processed_files[0]}, shape: {processed_df.shape}")
            
            # If columns weren't set in config, try to auto-detect them
            if date_col is None or target_col is None:
                auto_date_col, auto_target_col = detect_columns(processed_df)
                if date_col is None:
                    date_col = auto_date_col
                    print(f"Auto-detected date column: {date_col}")
                if target_col is None:
                    target_col = auto_target_col
                    print(f"Auto-detected target column: {target_col}")
            
            # Get date range string for display
            if date_col in processed_df.columns:
                date_range_str = get_date_range_str(processed_df, date_col)
                print(f"Date range: {date_range_str}")
            else:
                st.warning(f"Date column '{date_col}' not found in dataframe. Available columns: {list(processed_df.columns)}")
                # Try to find a date column
                date_candidates = [col for col in processed_df.columns if 'date' in col.lower() or 'time' in col.lower()]
                if date_candidates:
                    date_col = date_candidates[0]
                    print(f"Using alternative date column: {date_col}")
                    date_range_str = get_date_range_str(processed_df, date_col)
        except Exception as e:
            st.error(f"Error loading processed data: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        st.warning(f"No processed data found in '{selected_dir}'. Run forecasting models first.")
        # Try to find any CSV to display as example
        any_csvs = glob.glob(f"{selected_dir}/**/*.csv", recursive=True)
        if any_csvs:
            try:
                example_df = pd.read_csv(any_csvs[0])
                st.info(f"Found a CSV file that could be used: {os.path.basename(any_csvs[0])}")
                processed_df = example_df
                if date_col is None or target_col is None:
                    auto_date_col, auto_target_col = detect_columns(processed_df)
                    date_col = auto_date_col
                    target_col = auto_target_col
            except Exception as e:
                st.error(f"Error loading alternative CSV: {str(e)}")
    
    # Render sidebar elements and get display options
    display_options = render_sidebar(dataset_name, processed_df, date_col, date_range_str, dataset_type, target_col)
    
    # Main tabs with icons
    tab1, tab2, tab3 = st.tabs([
        "📊 Model Comparison", 
        "🔮 Forecasts", 
        "📈 Advanced Analysis"
    ])
    
    # Render each tab content
    render_model_comparison_tab(tab1, selected_dir, display_options)
    render_forecast_tab(tab2, selected_dir, display_options, dataset_type, dataset_name, target_col)
    render_advanced_analysis_tab(tab3, selected_dir, processed_df, date_col, target_col)

if __name__ == "__main__":
    main() 