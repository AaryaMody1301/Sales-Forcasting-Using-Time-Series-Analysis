import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import glob

# Configure the page
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .title {
        font-size: 42px;
        font-weight: bold;
        color: #3366ff;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 26px;
        font-weight: 500;
        color: #0047ab;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
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
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown("<div class='title'>Sales Forecasting Dashboard</div>", unsafe_allow_html=True)
    st.write("Interactive visualization of time series forecasting results")
    
    # Sidebar
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2329/2329080.png", width=100)
    st.sidebar.markdown("## Dashboard Controls")
    
    # Find all result directories
    results_dirs = glob.glob("results/*")
    if not results_dirs:
        st.error("No results found. Please run the forecasting models first.")
        return
    
    # Sort by most recent
    results_dirs.sort(key=os.path.getmtime, reverse=True)
    
    # Select dataset
    selected_dir = st.sidebar.selectbox(
        "Select Results Directory",
        results_dirs,
        format_func=lambda x: os.path.basename(x)
    )
    
    # Try to determine which dataset this is
    dataset_name = None
    dataset_type = None
    for name in ["amazon", "car_prices"]:
        if name in selected_dir:
            dataset_name = name.upper()
            dataset_type = name
            break
    
    if not dataset_name:
        dataset_name = "DATASET"
    
    # Show dataset info
    st.sidebar.markdown(f"### {dataset_name} Dataset")
    
    # Check if processed data exists
    processed_files = glob.glob(f"{selected_dir}/*_processed.csv")
    
    if processed_files:
        processed_df = pd.read_csv(processed_files[0])
        
        # Determine the date column based on dataset type
        date_col = None
        target_col = None
        
        if dataset_type == 'amazon':
            date_col = 'date'  # The synthetic column created during preprocessing
            target_col = 'daily_sales'
        elif dataset_type == 'car_prices':
            date_col = 'saledate'
            target_col = 'sellingprice'
        
        # If date_col exists, convert it to datetime for processing
        if date_col in processed_df.columns:
            # Convert to datetime temporarily for display purposes
            processed_df[date_col] = pd.to_datetime(processed_df[date_col])
            
            # Calculate min and max dates before converting to string
            if len(processed_df) > 0:
                min_date = processed_df[date_col].min()
                max_date = processed_df[date_col].max()
                date_range_str = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
            else:
                date_range_str = "No data"
                
            # Convert back to string for display in Streamlit
            processed_df[date_col] = processed_df[date_col].dt.strftime('%Y-%m-%d')
        
        # Basic stats
        st.sidebar.markdown(f"**Rows:** {processed_df.shape[0]}")
        st.sidebar.markdown(f"**Columns:** {processed_df.shape[1]}")
        
        # Date range
        if date_col in processed_df.columns and len(processed_df) > 0:
            st.sidebar.markdown(f"**Date Range:** {date_range_str}")
            
        # Dataset-specific metrics
        if dataset_type == 'amazon':
            if all(col in processed_df.columns for col in ['discounted_price', 'actual_price']):
                avg_discount = (processed_df['actual_price'] - processed_df['discounted_price']).mean()
                st.sidebar.markdown(f"**Avg. Discount:** ${avg_discount:.2f}")
            
            if 'rating' in processed_df.columns:
                avg_rating = processed_df['rating'].mean()
                st.sidebar.markdown(f"**Avg. Rating:** {avg_rating:.1f}")
                
            if 'category' in processed_df.columns:
                top_categories = processed_df['category'].value_counts().head(3).index.tolist()
                st.sidebar.markdown(f"**Top Categories:** {', '.join(top_categories)}")
                
        elif dataset_type == 'car_prices':
            if 'odometer' in processed_df.columns:
                avg_odometer = processed_df['odometer'].mean()
                st.sidebar.markdown(f"**Avg. Odometer:** {avg_odometer:.0f} miles")
                
            if 'year' in processed_df.columns:
                avg_year = processed_df['year'].mean()
                st.sidebar.markdown(f"**Avg. Year:** {avg_year:.0f}")
                
            if 'make' in processed_df.columns:
                top_makes = processed_df['make'].value_counts().head(3).index.tolist()
                st.sidebar.markdown(f"**Top Makes:** {', '.join(top_makes)}")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Model Comparison", "🔮 Forecasts", "📈 Time Series Analysis", "📋 Data", "🔍 Data Insights"])
    
    with tab1:
        st.markdown("<div class='subtitle'>Model Performance Comparison</div>", unsafe_allow_html=True)
        
        # Check for model evaluation results
        eval_file = os.path.join(selected_dir, "forecasts", "model_evaluation.csv")
        comparison_img = os.path.join(selected_dir, "forecasts", "model_comparison_rmse.png")
        
        if os.path.exists(eval_file):
            results_df = pd.read_csv(eval_file)
            
            # Display metrics in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.subheader("Best Model by RMSE")
                best_rmse = results_df.loc[results_df["RMSE"].idxmin()]
                st.metric("Model", best_rmse["Model"], help="Model with lowest Root Mean Squared Error")
                st.metric("RMSE", f"{best_rmse['RMSE']:.4f}")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col2:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.subheader("Best Model by MAE")
                best_mae = results_df.loc[results_df["MAE"].idxmin()]
                st.metric("Model", best_mae["Model"], help="Model with lowest Mean Absolute Error")
                st.metric("MAE", f"{best_mae['MAE']:.4f}")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col3:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.subheader("Best Model by MAPE")
                best_mape = results_df.loc[results_df["MAPE"].idxmin()]
                st.metric("Model", best_mape["Model"], help="Model with lowest Mean Absolute Percentage Error")
                st.metric("MAPE", f"{best_mape['MAPE']:.2f}%")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Comparison table
            st.subheader("All Models Performance")
            styled_results = results_df.style.background_gradient(subset=['RMSE', 'MAE', 'MAPE'], cmap='RdYlGn_r')
            st.dataframe(styled_results, use_container_width=True)
            
            # Visualization
            if os.path.exists(comparison_img):
                st.subheader("RMSE Comparison")
                st.image(comparison_img)
            else:
                # Create visualization
                fig, ax = plt.figure(figsize=(12, 6))
                sns.barplot(x="Model", y="RMSE", data=results_df)
                plt.title("Model Comparison - RMSE")
                plt.xticks(rotation=45)
                st.pyplot(fig)
        else:
            st.warning("No model evaluation results found. Please run the model evaluation first.")
    
    with tab2:
        st.markdown("<div class='subtitle'>Forecast Visualization</div>", unsafe_allow_html=True)
        
        # Find all forecast images
        forecast_images = glob.glob(f"{selected_dir}/forecasts/*forecast*.png")
        
        if forecast_images:
            # Create a dropdown for model selection
            model_options = [os.path.basename(img).replace("_forecast.png", "") for img in forecast_images]
            
            # Add ensemble if available
            if "ensemble_forecast.png" in [os.path.basename(img) for img in forecast_images]:
                model_options = ["ensemble"] + [m for m in model_options if m != "ensemble"]
            
            selected_model = st.selectbox("Select Model", model_options)
            
            # Show the forecast image for the selected model
            selected_image = [img for img in forecast_images if selected_model in os.path.basename(img)][0]
            st.image(selected_image, use_container_width=True)
            
            # Show forecast period and target information based on dataset
            if dataset_type == 'amazon':
                st.info("📈 The forecast shows predicted **daily sales** for Amazon products.")
            elif dataset_type == 'car_prices':
                st.info("📈 The forecast shows predicted **selling prices** for cars over time.")
                
            # Link to other visualizations
            st.write("**Related Visualizations:**")
            for model in model_options:
                if model != selected_model:
                    st.markdown(f"- [{model.upper()}](#{model})")
        else:
            st.warning("No forecast visualizations found. Please run the forecasting models first.")
    
    with tab3:
        st.markdown("<div class='subtitle'>Time Series Analysis</div>", unsafe_allow_html=True)
        
        # Find analysis images
        analysis_dir = os.path.join(selected_dir, "analysis")
        if os.path.exists(analysis_dir):
            analysis_images = glob.glob(f"{analysis_dir}/*.png")
            
            if analysis_images:
                # Group images by type
                acf_images = [img for img in analysis_images if "acf" in img.lower()]
                stationarity_images = [img for img in analysis_images if "stationarity" in img.lower()]
                distribution_images = [img for img in analysis_images if "distribution" in img.lower()]
                other_images = [img for img in analysis_images if img not in acf_images + stationarity_images + distribution_images]
                
                # Show images by category
                if acf_images:
                    st.subheader("Autocorrelation Analysis")
                    st.write("These plots show how values are correlated with their past values, helping identify patterns and seasonality.")
                    for img in acf_images:
                        st.image(img, caption=os.path.basename(img), use_container_width=True)
                
                if stationarity_images:
                    st.subheader("Stationarity Tests")
                    st.write("Stationarity tests check if the time series has constant statistical properties over time, which is important for many forecasting models.")
                    for img in stationarity_images:
                        st.image(img, caption=os.path.basename(img), use_container_width=True)
                
                if distribution_images:
                    st.subheader("Distribution Analysis")
                    if dataset_type == 'amazon':
                        st.write("Shows the distribution of daily sales values.")
                    elif dataset_type == 'car_prices':
                        st.write("Shows the distribution of car selling prices.")
                    for img in distribution_images:
                        st.image(img, caption=os.path.basename(img), use_container_width=True)
                
                if other_images:
                    st.subheader("Other Analysis")
                    for img in other_images:
                        st.image(img, caption=os.path.basename(img), use_container_width=True)
            else:
                st.warning("No analysis visualizations found.")
                
            # Add explanations based on dataset type
            if dataset_type == 'amazon':
                st.markdown("""
                ### Understanding Amazon Sales Time Series
                
                The time series analysis for Amazon data focuses on the `daily_sales` metric, which is derived from 
                product prices and rating counts. Key aspects to look for:
                
                - **Seasonality**: Look for weekly patterns or seasonal trends
                - **Price Sensitivity**: How changes in discounted prices affect sales
                - **Rating Impact**: How product ratings correlate with sales
                """)
            elif dataset_type == 'car_prices':
                st.markdown("""
                ### Understanding Car Prices Time Series
                
                The time series analysis for car prices focuses on the `sellingprice` over time. Key aspects to look for:
                
                - **Depreciation Trends**: How prices decrease as cars age
                - **Seasonal Patterns**: When prices tend to be higher or lower
                - **Mileage Impact**: How odometer readings correlate with price
                """)
        else:
            st.warning("No analysis directory found.")
    
    with tab4:
        st.markdown("<div class='subtitle'>Dataset Information</div>", unsafe_allow_html=True)
        
        if processed_files:
            # Show dataset
            st.subheader("Processed Dataset")
            st.dataframe(processed_df.head(100), use_container_width=True)
            
            # Show summary statistics
            st.subheader("Summary Statistics")
            # Only include numeric columns in the describe to avoid PyArrow issues
            numeric_df = processed_df.select_dtypes(include=[np.number])
            st.dataframe(numeric_df.describe(), use_container_width=True)
            
            # Column information
            st.subheader("Column Information")
            
            # Create a safer column info dataframe
            col_info_data = {
                'Column': processed_df.columns.tolist(),
                'Type': [str(dtype) for dtype in processed_df.dtypes.tolist()],
                'Non-Null Count': processed_df.count().tolist(),
                'Null Count': processed_df.isna().sum().tolist()
            }
            
            # Calculate null percentage as strings to avoid conversion issues
            null_percentages = []
            for col in processed_df.columns:
                null_count = processed_df[col].isna().sum()
                total_count = len(processed_df)
                if total_count > 0:
                    percentage = (null_count / total_count) * 100
                    null_percentages.append(f"{percentage:.2f}%")
                else:
                    null_percentages.append("0.00%")
            
            col_info_data['Null %'] = null_percentages
            
            col_info = pd.DataFrame(col_info_data)
            st.dataframe(col_info, use_container_width=True)
            
            # Data dictionary based on dataset type
            st.subheader("Data Dictionary")
            
            if dataset_type == 'amazon':
                st.markdown("""
                | Column | Description |
                | ------ | ----------- |
                | product_id | Unique identifier for each product |
                | product_name | Name of the product |
                | category | Product category |
                | discounted_price | Discounted price of the product |
                | actual_price | Original price before discount |
                | discount_percentage | Percentage of discount |
                | rating | Product rating (usually 1-5) |
                | rating_count | Number of ratings received |
                | about_product | Product description |
                | user_id | Identifier for the user |
                | user_name | Name of the user |
                | review_id | Unique identifier for the review |
                | review_title | Title of the review |
                | review_content | Content of the review |
                | img_link | Link to product image |
                | product_link | Link to product page |
                | daily_sales | Derived metric for daily sales (calculated) |
                """)
            elif dataset_type == 'car_prices':
                st.markdown("""
                | Column | Description |
                | ------ | ----------- |
                | year | Manufacturing year of the car |
                | make | Car manufacturer (brand) |
                | model | Car model |
                | trim | Specific trim level of the model |
                | body | Body type (sedan, SUV, etc.) |
                | transmission | Transmission type |
                | vin | Vehicle Identification Number |
                | state | State where the car was sold |
                | condition | Condition of the car |
                | odometer | Odometer reading (mileage) |
                | color | Exterior color |
                | interior | Interior color/material |
                | seller | Seller information |
                | mmr | Manheim Market Report value (wholesale price indicator) |
                | sellingprice | Final selling price of the car |
                | saledate | Date when the car was sold |
                """)
        else:
            st.warning("No processed dataset found.")
    
    with tab5:
        st.markdown("<div class='subtitle'>Data Insights</div>", unsafe_allow_html=True)
        
        if processed_files and processed_df is not None:
            if dataset_type == 'amazon':
                # Amazon-specific visualizations
                st.subheader("Amazon Product Analysis")
                
                # Visualizations for Amazon data
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create a visualization for discounted vs. actual price
                    if all(col in processed_df.columns for col in ['discounted_price', 'actual_price']):
                        st.write("**Discount Analysis**")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        plt.scatter(processed_df['actual_price'], processed_df['discounted_price'], alpha=0.5)
                        
                        # Add a reference line for no discount
                        max_price = max(processed_df['actual_price'].max(), processed_df['discounted_price'].max())
                        plt.plot([0, max_price], [0, max_price], 'r--')
                        
                        plt.xlabel('Original Price ($)')
                        plt.ylabel('Discounted Price ($)')
                        plt.title('Original vs. Discounted Prices')
                        plt.grid(True)
                        st.pyplot(fig)
                
                with col2:
                    # Create a visualization for rating vs. price
                    if all(col in processed_df.columns for col in ['rating', 'discounted_price']):
                        st.write("**Rating vs. Price Analysis**")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        plt.scatter(processed_df['rating'], processed_df['discounted_price'], alpha=0.5)
                        plt.xlabel('Rating')
                        plt.ylabel('Discounted Price ($)')
                        plt.title('Product Rating vs. Price')
                        plt.grid(True)
                        st.pyplot(fig)
                
                # Category analysis
                if 'category' in processed_df.columns:
                    st.subheader("Category Analysis")
                    
                    # Get top categories
                    top_cats = processed_df['category'].value_counts().head(10)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    plt.bar(top_cats.index, top_cats.values, color='skyblue')
                    plt.xlabel('Category')
                    plt.ylabel('Count')
                    plt.title('Top 10 Product Categories')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # If we also have price data, show average price by category
                    if 'discounted_price' in processed_df.columns:
                        avg_prices = processed_df.groupby('category')['discounted_price'].mean().sort_values(ascending=False).head(10)
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        plt.bar(avg_prices.index, avg_prices.values, color='salmon')
                        plt.xlabel('Category')
                        plt.ylabel('Average Price ($)')
                        plt.title('Average Price by Category (Top 10)')
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)
                    
            elif dataset_type == 'car_prices':
                # Car prices-specific visualizations
                st.subheader("Car Price Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create a visualization for year vs. price
                    if all(col in processed_df.columns for col in ['year', 'sellingprice']):
                        st.write("**Car Age vs. Price Analysis**")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        plt.scatter(processed_df['year'], processed_df['sellingprice'], alpha=0.5)
                        plt.xlabel('Year')
                        plt.ylabel('Selling Price ($)')
                        plt.title('Car Year vs. Selling Price')
                        plt.grid(True)
                        st.pyplot(fig)
                
                with col2:
                    # Create a visualization for odometer vs. price
                    if all(col in processed_df.columns for col in ['odometer', 'sellingprice']):
                        st.write("**Mileage vs. Price Analysis**")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        plt.scatter(processed_df['odometer'], processed_df['sellingprice'], alpha=0.5)
                        plt.xlabel('Odometer (miles)')
                        plt.ylabel('Selling Price ($)')
                        plt.title('Mileage vs. Selling Price')
                        plt.grid(True)
                        st.pyplot(fig)
                
                # Make and model analysis
                if all(col in processed_df.columns for col in ['make', 'sellingprice']):
                    st.subheader("Make Analysis")
                    
                    # Get average price by make
                    avg_by_make = processed_df.groupby('make')['sellingprice'].agg(['mean', 'count']).sort_values('mean', ascending=False)
                    # Filter to makes with at least 5 entries
                    avg_by_make = avg_by_make[avg_by_make['count'] >= 5].head(10)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    plt.bar(avg_by_make.index, avg_by_make['mean'], color='skyblue')
                    plt.xlabel('Make')
                    plt.ylabel('Average Selling Price ($)')
                    plt.title('Average Selling Price by Make (Top 10)')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                # Condition analysis
                if all(col in processed_df.columns for col in ['condition', 'sellingprice']):
                    st.subheader("Condition Analysis")
                    
                    avg_by_condition = processed_df.groupby('condition')['sellingprice'].mean().sort_values(ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plt.bar(avg_by_condition.index, avg_by_condition.values, color='lightgreen')
                    plt.xlabel('Condition')
                    plt.ylabel('Average Selling Price ($)')
                    plt.title('Average Selling Price by Condition')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
            
            # Common insights for both datasets
            st.subheader("Time Series Patterns")
            
            # Find the target column based on dataset
            if dataset_type == 'amazon':
                target_col = 'daily_sales'
            elif dataset_type == 'car_prices':
                target_col = 'sellingprice'
                
            if target_col in processed_df.columns and date_col in processed_df.columns:
                # Convert date column to datetime for this visualization
                processed_df[date_col] = pd.to_datetime(processed_df[date_col])
                
                # Resample data for trend visualization
                try:
                    # First sort by date
                    temp_df = processed_df.sort_values(by=date_col)
                    # Set date as index
                    temp_df = temp_df.set_index(date_col)
                    
                    # Create monthly averages
                    monthly_avg = temp_df[target_col].resample('ME').mean()  # Using 'ME' (month end) instead of deprecated 'M'
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    monthly_avg.plot(ax=ax)
                    plt.xlabel('Date')
                    plt.ylabel(target_col)
                    plt.title(f'Monthly Average {target_col}')
                    plt.grid(True)
                    st.pyplot(fig)
                    
                    # Add weekly seasonality visualization
                    try:
                        # Group by day of week to show weekly patterns
                        temp_df['day_of_week'] = temp_df.index.day_name()
                        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        weekly_avg = temp_df.groupby('day_of_week')[target_col].mean()
                        weekly_avg = weekly_avg.reindex(day_order)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        weekly_avg.plot(kind='bar', ax=ax, color='skyblue')
                        plt.xlabel('Day of Week')
                        plt.ylabel(f'Average {target_col}')
                        plt.title(f'Weekly Seasonality Pattern')
                        plt.grid(True, axis='y')
                        plt.tight_layout()
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"Could not create weekly seasonality visualization: {str(e)}")
                    
                    # Add decomposition plot if statsmodels is available
                    try:
                        from statsmodels.tsa.seasonal import seasonal_decompose
                        
                        # For decomposition, need regular time series
                        # First check if we have enough data and regular intervals
                        if len(temp_df) > 14:  # Need a reasonable amount of data
                            # Try to determine frequency
                            decomposition_period = 7  # Default to weekly
                            
                            # For car prices, might want monthly
                            if dataset_type == 'car_prices':
                                decomposition_period = 30
                                
                            # Perform decomposition
                            decomposition = seasonal_decompose(temp_df[target_col], 
                                                             model='additive', 
                                                             period=decomposition_period,
                                                             extrapolate_trend='freq')
                            
                            # Plot decomposition components
                            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12))
                            
                            decomposition.observed.plot(ax=ax1)
                            ax1.set_title('Observed')
                            ax1.grid(True)
                            
                            decomposition.trend.plot(ax=ax2)
                            ax2.set_title('Trend')
                            ax2.grid(True)
                            
                            decomposition.seasonal.plot(ax=ax3)
                            ax3.set_title('Seasonality')
                            ax3.grid(True)
                            
                            decomposition.resid.plot(ax=ax4)
                            ax4.set_title('Residuals')
                            ax4.grid(True)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Add analysis of the decomposition
                            st.subheader("Time Series Decomposition Analysis")
                            if dataset_type == 'amazon':
                                st.markdown("""
                                The time series decomposition breaks down the daily sales into:
                                
                                - **Trend**: The long-term progression of the sales (increasing or decreasing)
                                - **Seasonality**: Regular patterns that repeat at fixed intervals (weekly patterns for products)
                                - **Residual**: The random variation left after removing trend and seasonality
                                
                                Understanding these components helps select appropriate forecasting models. Strong seasonality 
                                suggests using models like SARIMA or models with seasonal features.
                                """)
                            elif dataset_type == 'car_prices':
                                st.markdown("""
                                The time series decomposition breaks down the car selling prices into:
                                
                                - **Trend**: The long-term progression of prices (usually declining due to depreciation)
                                - **Seasonality**: Regular patterns that repeat at fixed intervals (monthly or seasonal pricing)
                                - **Residual**: The random variation left after removing trend and seasonality
                                
                                This decomposition helps understand market patterns and select appropriate forecasting models.
                                """)
                    except Exception as e:
                        st.warning(f"Could not create time series decomposition: {str(e)}")
                except Exception as e:
                    st.error(f"Could not create time series visualization: {str(e)}")
        else:
            st.warning("No processed dataset found.")

if __name__ == "__main__":
    main() 