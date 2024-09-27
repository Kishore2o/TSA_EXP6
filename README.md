## NAME   : Kishore S
## REG NO : 212222240050

# Ex.No: 6               HOLT WINTERS METHOD
### Date: 



### AIM: 

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load the dataset
data = pd.read_csv('/content/NFLX.csv')

# Convert the 'Date' column to datetime format and set it as the index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Convert 'Close' column to numeric (removing invalid values)
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')

# Drop rows with missing values in 'Close' column
clean_data = data.dropna(subset=['Close'])

# Extract 'Close' column for time series forecasting
close_data_clean = clean_data['Close']

# Perform Holt-Winters exponential smoothing
model = ExponentialSmoothing(close_data_clean, trend="add", seasonal="add", seasonal_periods=12)
fit = model.fit()

# Forecast for the next 12 steps (business days)
n_steps = 200
forecast = fit.forecast(steps=n_steps)

# Plot the original data and the forecast
plt.figure(figsize=(10, 6))
plt.plot(close_data_clean.index, close_data_clean, label='Original Data')
plt.plot(pd.date_range(start=close_data_clean.index[-1], periods=n_steps+1, freq='B')[1:], forecast, label='Forecast')  # 'B' for business days
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Holt-Winters Forecast for Google Stock Prices')
plt.legend()
plt.show()
```

### OUTPUT:
![image](https://github.com/user-attachments/assets/f50afedb-c49d-4891-9509-542706d47c08)


### RESULT:
The program run successfully based on the Holt Winters Method model.
