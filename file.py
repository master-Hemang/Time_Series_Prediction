import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load data
file_path = 'stock_price.csv'  # Replace with your file path
df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')

# Convert 'volume' to numeric by replacing 'B', 'M', and 'K'
def convert_volume(volume_str):
    if 'B' in volume_str:
        return float(volume_str.replace('B', '')) * 1_000_000_000
    elif 'M' in volume_str:
        return float(volume_str.replace('M', '')) * 1_000_000
    elif 'K' in volume_str:
        return float(volume_str.replace('K', '')) * 1_000
    else:
        return float(volume_str)

df['volume'] = df['volume'].apply(convert_volume)

# Handle missing values (if any) using forward fill
df.ffill(inplace=True)

# Feature engineering: Create moving averages, lag features
df['MA_50'] = df['closing price'].rolling(window=50).mean()
df['MA_200'] = df['closing price'].rolling(window=200).mean()
df['lag_1'] = df['closing price'].shift(1)
df['lag_3'] = df['closing price'].shift(3)

# Remove rows with NaN values caused by lag features
df.dropna(inplace=True)

# Feature Scaling
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[['closing price', 'open price', 'high price', 'low price', 'volume', 'MA_50', 'MA_200', 'lag_1', 'lag_3']])

# Train-test split
train_size = int(len(df_scaled) * 0.8)
train_data, test_data = df_scaled[:train_size], df_scaled[train_size:]

from statsmodels.tsa.arima.model import ARIMA

# ARIMA requires unscaled closing price data
train_closing = df['closing price'][:train_size]
test_closing = df['closing price'][train_size:]

# Train ARIMA model
model = ARIMA(train_closing, order=(5,1,0))
arima_model = model.fit()

# Forecast
arima_predictions = arima_model.forecast(steps=len(test_closing))

# Visualize ARIMA predictions
plt.figure(figsize=(10,6))
plt.plot(test_closing.index, test_closing, label='Actual')
plt.plot(test_closing.index, arima_predictions, label='ARIMA Predictions', color='red')
plt.title('ARIMA Model Predictions vs Actual')
plt.legend()
plt.show()

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Reshape data for LSTM (3D: [samples, timesteps, features])
X_train = []
y_train = []

for i in range(60, train_size):
    X_train.append(train_data[i-60:i])
    y_train.append(train_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape test data
X_test = []
y_test = []

for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i])
    y_test.append(test_data[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)

# Build LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Predictions
lstm_predictions = model.predict(X_test)

# Rescale predictions back to original scale
lstm_predictions = scaler.inverse_transform(np.concatenate([lstm_predictions, np.zeros((lstm_predictions.shape[0], df_scaled.shape[1] - 1))], axis=1))[:, 0]

# Visualize LSTM predictions
plt.figure(figsize=(10,6))
plt.plot(df.index[-len(y_test):], scaler.inverse_transform(test_data[60:])[:,0], label='Actual')
plt.plot(df.index[-len(y_test):], lstm_predictions, label='LSTM Predictions', color='red')
plt.title('LSTM Model Predictions vs Actual')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Evaluate ARIMA model
arima_rmse = np.sqrt(mean_squared_error(test_closing, arima_predictions))
arima_mae = mean_absolute_error(test_closing, arima_predictions)
print(f'ARIMA RMSE: {arima_rmse}, MAE: {arima_mae}')

# Evaluate LSTM model
actual_test_data = scaler.inverse_transform(test_data)[:, 0]
lstm_rmse = np.sqrt(mean_squared_error(actual_test_data[60:], lstm_predictions))
lstm_mae = mean_absolute_error(actual_test_data[60:], lstm_predictions)
print(f'LSTM RMSE: {lstm_rmse}, MAE: {lstm_mae}')
