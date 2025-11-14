import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.tseries.offsets import DateOffset
import warnings

# Suppress common convergence and date warnings for a cleaner report
warnings.filterwarnings("ignore")

def load_and_prepare_data(csv_path: str) -> pd.Series:
    """
    Loads the natural gas CSV, parses dates, and sets a proper
    DatetimeIndex.
    
    Args:
        csv_path: The file path to 'Nat_Gas.csv'.

    Returns:
        A pandas Series with a DatetimeIndex and monthly prices.
    """
    try:
        data = pd.read_csv(
            csv_path,
            parse_dates=['Dates'],
            dayfirst=False,  # Assumes MM/DD/YY format from CSV
            date_format='%m/%d/%y'
        )
    except Exception as e:
        print(f"Error reading or parsing dates: {e}")
        print("Please ensure the CSV file is in the correct path and 'Dates' column is in MM/DD/YY format.")
        return pd.Series(dtype=float)
        
    data.set_index('Dates', inplace=True)
    data.columns = ['Price']
    
    # Ensure the data is sorted by date
    data.sort_index(inplace=True)
    
    # Convert the index to month-end frequency, which is standard
    # for financial time series modeling.
    data.index = data.index.to_period('M').to_timestamp('M')
    
    return data['Price']

def create_unified_price_curve(monthly_prices: pd.Series, forecast_steps: int = 12) -> pd.DataFrame:
    """
    Creates a unified daily price curve by:
    1. Fitting a SARIMA model to the monthly historical data.
    2. Forecasting 'forecast_steps' months into the future.
    3. Appending the forecast to the historical data.
    4. Upsampling the combined monthly data to a daily frequency.
    5. Interpolating all missing daily values using a cubic spline.

    Args:
        monthly_prices: Series of historical monthly prices.
        forecast_steps: Number of months to forecast (default 12).

    Returns:
        A pandas DataFrame with a daily index and columns for
        'Price', 'Is_Forecast', 'CI_Lower', and 'CI_Upper'.
    """
    
    # --- 1. Define and Fit SARIMA Model ---
    # We choose a robust, common SARIMA(p,d,q)(P,D,Q,s) model.
    # (p,d,q) = (1,1,1) - A standard non-seasonal ARIMA(1,1,1)
    #   p=1: Autoregressive component (price depends on last month's price)
    #   d=1: Integrated component (data is non-stationary, use 1st difference)
    #   q=1: Moving Average component (price depends on last month's shock)
    # (P,D,Q,s) = (1,1,1,12) - A standard seasonal component
    #   P=1: Seasonal AR (depends on price 12 months ago)
    #   D=1: Seasonal I (use 12-month difference to remove seasonality)
    #   Q=1: Seasonal MA (depends on shock 12 months ago)
    #   s=12: The seasonal period is 12 months.
    
    sarima_model = SARIMAX(
        monthly_prices,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        trend='c',  # 'c' adds a constant drift (accounts for the trend)
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    # Fit the model. 
    # `disp=False` turns off convergence logging.
    try:
        model_fit = sarima_model.fit(disp=False)
    except Exception as e:
        print(f"Error fitting SARIMA model: {e}")
        return pd.DataFrame()

    # --- 2. Generate Forecast ---
    forecast_obj = model_fit.get_forecast(steps=forecast_steps)
    
    # Extract mean forecast and 95% confidence interval
    forecast_mean = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int(alpha=0.05)
    conf_int.columns = ['CI_Lower', 'CI_Upper']
    
    # Combine forecast mean and CI into a single DataFrame
    forecast_df = pd.concat([forecast_mean.to_frame('Price'), conf_int], axis=1)

    # --- 3. Combine Historical and Forecast Data ---
    
    # Create a DataFrame for historical prices
    historical_df = monthly_prices.to_frame('Price')
    
    # Add 'Is_Forecast', 'CI_Lower', 'CI_Upper' to historical data
    historical_df['Is_Forecast'] = False
    # For historical data, the "confidence interval" is the data itself
    historical_df['CI_Lower'] = monthly_prices
    historical_df['CI_Upper'] = monthly_prices

    # Add 'Is_Forecast' to forecast data
    forecast_df['Is_Forecast'] = True
    
    # Combine the historical and forecast monthly data
    combined_monthly_df = pd.concat([historical_df, forecast_df])

    # --- 4. Upsample and Interpolate to Daily Curve ---
    
    # Define the full daily date range for the unified curve
    start_date = combined_monthly_df.index.min()
    end_date = combined_monthly_df.index.max()
    daily_index = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Reindex the monthly data to the new daily index (creates NaNs)
    daily_df = combined_monthly_df.reindex(daily_index)
    
    # Interpolate the 'Price' column using a cubic spline.
    # This creates the smooth, continuous curve.
    daily_df['Price'] = daily_df['Price'].interpolate(method='cubic', order=3)
    
    # Interpolate the CI boundaries. Linear is fine here as they are
    # just boundaries, and splines can overshoot, which is undesirable
    # for a confidence interval.
    daily_df['CI_Lower'] = daily_df['CI_Lower'].interpolate(method='linear')
    daily_df['CI_Upper'] = daily_df['CI_Upper'].interpolate(method='linear')
    
    # Forward-fill the 'Is_Forecast' flag
    daily_df['Is_Forecast'] = daily_df['Is_Forecast'].ffill().astype(bool)
    
    # Handle the first few days which might be NaN after spline
    daily_df.bfill(inplace=True) 

    return daily_df

def get_price_estimate(input_date: str, unified_curve: pd.DataFrame) -> dict:
    """
    Takes a date string and the unified daily curve and returns
    the estimated price for that date.

    Args:
        input_date: A date string (e.g., "YYYY-MM-DD", "MM/DD/YYYY").
        unified_curve: The daily DataFrame from create_unified_price_curve.

    Returns:
        A dictionary with price information.
    """
    try:
        # Parse the input date
        query_date = pd.to_datetime(input_date)
    except Exception as e:
        return {'status': 'error', 'message': f"Invalid date format: {e}"}

    # Ensure the query date is within the model's range
    if not (unified_curve.index.min() <= query_date <= unified_curve.index.max()):
        return {
            'status': 'error',
            'message': f"Date {query_date.date()} is outside the model's range "
                       f"({unified_curve.index.min().date()} to {unified_curve.index.max().date()})."
        }
        
    # --- Retrieve the estimated price ---
    # Use.loc to find the exact date. We round the price for clarity.
    try:
        price_data = unified_curve.loc[query_date]
        
        result = {
            'status': 'success',
            'date': query_date.date(),
            'estimated_price': round(price_data['Price'], 2),
            'is_forecast': price_data['Is_Forecast']
        }
        
        if price_data['Is_Forecast']:
            result['confidence_interval'] = (
                f"${round(price_data['CI_Lower'], 2)} - "
                f"${round(price_data['CI_Upper'], 2)}"
            )
            
        return result
        
    except KeyError:
        # This can happen if the parsed date is not exactly in the daily index
        # (e.g., issues with timezones). A robust lookup finds the closest.
        closest_date_index = unified_curve.index.get_indexer([query_date], method='nearest')
        price_data = unified_curve.iloc[closest_date_index]
        
        result = {
            'status': 'success',
            'date': price_data.name.date(),
            'estimated_price': round(price_data['Price'], 2),
            'is_forecast': price_data['Is_Forecast']
        }
        
        if price_data['Is_Forecast']:
            result['confidence_interval'] = (
                f"${round(price_data['CI_Lower'], 2)} - "
                f"${round(price_data['CI_Upper'], 2)}"
            )
            
        return result
    except Exception as e:
        return {'status': 'error', 'message': f"An unexpected error occurred: {e}"}

def plot_unified_curve(daily_curve: pd.DataFrame, original_monthly: pd.Series):
    """
    Generates a high-quality visualization of the historical data,
    the cubic-spline interpolation, and the SARIMA forecast.
    """
    plt.figure(figsize=(15, 8))
    
    # Split historical and forecast data for plotting
    historical_curve = daily_curve[daily_curve['Is_Forecast'] == False]
    forecast_curve = daily_curve[daily_curve['Is_Forecast'] == True]
    
    # 1. Plot the historical interpolated curve
    plt.plot(
        historical_curve.index,
        historical_curve['Price'],
        label='Historical (Cubic Spline Interpolation)',
        color='blue',
        linewidth=2
    )
    
    # 2. Plot the original monthly data points
    plt.plot(
        original_monthly.index,
        original_monthly,
        'o',
        color='black',
        markersize=5,
        label='Original Monthly Data (EOM)'
    )
    
    # 3. Plot the mean forecast curve
    # We must include the last historical point to make the line continuous
    last_hist_point = historical_curve.iloc[-1:]
    continuous_forecast_curve = pd.concat([last_hist_point, forecast_curve])
    
    plt.plot(
        continuous_forecast_curve.index,
        continuous_forecast_curve['Price'],
        label='Forecast (SARIMA + Spline)',
        color='red',
        linestyle='--',
        linewidth=2
    )
    
    # 4. Plot the 95% confidence interval
    plt.fill_between(
        forecast_curve.index,
        forecast_curve['CI_Lower'],
        forecast_curve['CI_Upper'],
        color='red',
        alpha=0.15,
        label='95% Confidence Interval'
    )
    
    # --- Formatting ---
    plt.title('Natural Gas Price: Historical Estimation and 1-Year Forecast', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (per Unit)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axvline(
        historical_curve.index[-1],
        color='gray',
        linestyle='-',
        linewidth=1.5,
        label='Forecast Start'
    )
    plt.tight_layout()
    
    # Save the plot
    plt.savefig("natural_gas_price_forecast.png", dpi=300)
    print("\nForecast plot saved as 'natural_gas_price_forecast.png'")

# --- Main Execution ---
if __name__ == "__main__":
    
    CSV_FILE = 'Nat_Gas.csv'
    
    # 1. Load and prepare the data
    monthly_price_data = load_and_prepare_data(CSV_FILE)
    
    if not monthly_price_data.empty:
        # 2. Decompose the series to show components (as discussed in Part 2)
        decomposition = seasonal_decompose(monthly_price_data, model='multiplicative', period=12)
        
        # Plot decomposition
        fig = decomposition.plot()
        fig.set_size_inches(10, 8)
        fig.suptitle('Time Series Decomposition (Multiplicative)', y=1.02)
        plt.tight_layout()
        plt.savefig("price_decomposition.png", dpi=300)
        print("Decomposition plot saved as 'price_decomposition.png'")

        # 3. Create the unified historical and forecast curve
        print("Fitting SARIMA model and generating 12-month forecast... This may take a moment.")
        unified_daily_curve = create_unified_price_curve(monthly_price_data, forecast_steps=12)
        
        if not unified_daily_curve.empty:
            # 4. Generate the main plot
            plot_unified_curve(unified_daily_curve, monthly_price_data)

            # 5. Demonstrate the get_price_estimate function
            print("\n--- Price Estimation Examples ---")
            
            # Example 1: A historical, interpolated date
            past_date = "2023-01-15"
            print(f"Querying for: {past_date}")
            print(get_price_estimate(past_date, unified_daily_curve))
            
            # Example 2: A future, forecasted date
            future_date = "2025-01-15"
            print(f"\nQuerying for: {future_date}")
            print(get_price_estimate(future_date, unified_daily_curve))
            
            # Example 3: A date at the exact EOM snapshot
            eom_date = "2023-12-31"
            print(f"\nQuerying for: {eom_date}")
            print(get_price_estimate(eom_date, unified_daily_curve))

            # Example 4: An out-of-bounds date
            oob_date = "2028-01-01"
            print(f"\nQuerying for: {oob_date}")
            print(get_price_estimate(oob_date, unified_daily_curve))