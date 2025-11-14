# Natural-Gas-Price-Data-model

## 1. Project Overview:
   
This repository contains a Python-based quantitative model designed to analyze, interpolate, and forecast natural gas prices. Given a time series of historical end-of-month (EOM) prices, this tool provides a complete, continuous daily price curve by:

**Interpolating** historical daily prices from monthly data points using cubic spline interpolation.

Forecasting future daily prices with a 95% confidence interval using a **Seasonal Autoregressive Integrated Moving Average (SARIMA)** model.

This project was developed to provide a robust method for estimating historical prices and creating indicative forecasts for applications like valuing energy storage contracts.

## 2. Features:
   
**Time Series Decomposition:** Automatically separates the price data into its constituent trend, seasonal, and residual components.

**Cubic Spline Interpolation:** Creates a smooth, continuous, and more realistic daily price curve from sparse monthly data, superior to simple linear interpolation.

**SARIMA Forecasting:** Implements a SARIMA(1,1,1)(1,1,1,12) model, which is well-suited for data with strong seasonality, to extrapolate prices for the next 12 months.

**Price Estimation Function:** Includes a simple utility function (get_price_estimate) to retrieve the estimated or forecasted price for any given date.

## 3. Sample Output
Running the script will generate the following plots in your project directory.

**Time Series Decomposition**

<img width="3000" height="2400" alt="price_decomposition" src="https://github.com/user-attachments/assets/fa8b0aec-f765-4026-9b61-a64dccebdcaa" />

**Price Forecast and Interpolation**

<img width="4500" height="2400" alt="natural_gas_price_forecast" src="https://github.com/user-attachments/assets/64a42c20-75c2-4a70-86cf-0db239a3cb29" />


price_decomposition.png: A 4-panel plot of the decomposition.

natural_gas_price_forecast.png: A complete plot of the historical data, interpolated curve, and 12-month fore
