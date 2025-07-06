# ARIMA Parameter Optimization with d=1 fixed
# Find optimal (p,1,q) with lowest AIC and forecast using 80/20 split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Grid search for optimal ARIMA(p,1,q) parameters
def find_optimal_arima(data, max_p=5, max_q=5, d=1):
    """
    Find optimal ARIMA parameters with fixed d=1
    Returns best parameters and AIC comparison table
    """
    best_aic = float('inf')
    best_params = None
    results = []
    
    print("ARIMA Parameter Search (d=1 fixed)")
    print("-" * 50)
    print(f"{'p':<3} {'q':<3} {'AIC':<12} {'BIC':<12} {'Status'}")
    print("-" * 50)
    
    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            try:
                # Fit ARIMA model
                model = ARIMA(data, order=(p, d, q))
                fitted_model = model.fit()
                
                aic = fitted_model.aic
                bic = fitted_model.bic
                
                # Store results
                results.append({
                    'p': p,
                    'q': q,
                    'aic': aic,
                    'bic': bic,
                    'model': fitted_model
                })
                
                # Check if this is the best model
                status = ""
                if aic < best_aic:
                    best_aic = aic
                    best_params = (p, d, q)
                    status = "← BEST"
                
                print(f"{p:<3} {q:<3} {aic:<12.2f} {bic:<12.2f} {status}")
                
            except Exception as e:
                print(f"{p:<3} {q:<3} {'Failed':<12} {'Failed':<12} ✗")
                continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('aic').reset_index(drop=True)
    
    return best_params, results_df

# Find optimal parameters
# Replace 'your_column_name' with your actual column name (e.g., 'value' or 'price')
print("Finding optimal ARIMA parameters...")
best_params, results_df = find_optimal_arima(df['your_column_name'])  # Change column name here

print(f"\nBest ARIMA model: ARIMA{best_params}")
print(f"Best AIC: {results_df.iloc[0]['aic']:.2f}")

# Display top 5 models
print(f"\nTop 5 Models by AIC:")
print(results_df.head()[['p', 'q', 'aic', 'bic']])

# 80/20 Train-Test Split
train_size = int(len(df) * 0.8)
train_data = df['your_column_name'][:train_size]  # Change column name here
test_data = df['your_column_name'][train_size:]   # Change column name here

print(f"\nData Split:")
print(f"Training data: {len(train_data)} observations")
print(f"Test data: {len(test_data)} observations")
print(f"Train period: {train_data.index[0]} to {train_data.index[-1]}")
print(f"Test period: {test_data.index[0]} to {test_data.index[-1]}")

# Fit the best model on training data
print(f"\nFitting ARIMA{best_params} on training data...")
train_model = ARIMA(train_data, order=best_params)
fitted_model = train_model.fit()

# Model summary
print(f"\nModel Summary:")
print(fitted_model.summary())

# Generate forecasts for test period
forecast_steps = len(test_data)
forecast = fitted_model.forecast(steps=forecast_steps)
forecast_ci = fitted_model.get_forecast(steps=forecast_steps).conf_int()

# Calculate forecast accuracy metrics
mse = mean_squared_error(test_data, forecast)
mae = mean_absolute_error(test_data, forecast)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100

print(f"\nForecast Accuracy Metrics:")
print(f"MSE:  {mse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")

# Create forecast results DataFrame
forecast_df = pd.DataFrame({
    'actual': test_data,
    'forecast': forecast,
    'lower_ci': forecast_ci.iloc[:, 0],
    'upper_ci': forecast_ci.iloc[:, 1],
    'error': test_data - forecast,
    'abs_error': np.abs(test_data - forecast),
    'pct_error': ((test_data - forecast) / test_data) * 100
}, index=test_data.index)

print(f"\nForecast Results (First 10 observations):")
print(forecast_df.head(10))

# Plot results
fig, axes = plt.subplots(2, 1, figsize=(15, 12))

# Plot 1: Full series with train/test split and forecasts
axes[0].plot(train_data.index, train_data, label='Training Data', color='blue', alpha=0.7)
axes[0].plot(test_data.index, test_data, label='Actual (Test)', color='green', linewidth=2)
axes[0].plot(test_data.index, forecast, label='Forecast', color='red', linestyle='--', linewidth=2)
axes[0].fill_between(test_data.index, 
                     forecast_ci.iloc[:, 0], 
                     forecast_ci.iloc[:, 1], 
                     color='red', alpha=0.2, label='95% Confidence Interval')
axes[0].axvline(x=train_data.index[-1], color='black', linestyle=':', alpha=0.7, label='Train/Test Split')
axes[0].set_title(f'ARIMA{best_params} Forecast vs Actual (AIC: {results_df.iloc[0]["aic"]:.2f})')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Value')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Forecast errors
axes[1].plot(test_data.index, forecast_df['error'], color='red', marker='o', markersize=3)
axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
axes[1].fill_between(test_data.index, forecast_df['error'], 0, 
                     color='red', alpha=0.3)
axes[1].set_title('Forecast Errors (Actual - Forecast)')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Error')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional diagnostic plot: Forecast vs Actual scatter
plt.figure(figsize=(10, 6))
plt.scatter(test_data, forecast, alpha=0.7, color='blue')
plt.plot([test_data.min(), test_data.max()], [test_data.min(), test_data.max()], 
         'r--', lw=2, label='Perfect Forecast Line')
plt.xlabel('Actual Values')
plt.ylabel('Forecasted Values')
plt.title('Forecast vs Actual Values')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Save results to CSV (optional)
# forecast_df.to_csv('arima_forecast_results.csv')
# results_df.to_csv('arima_parameter_comparison.csv')

print(f"\nFinal Results Summary:")
print(f"Best Model: ARIMA{best_params}")
print(f"AIC: {results_df.iloc[0]['aic']:.2f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")
