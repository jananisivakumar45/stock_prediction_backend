import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2  # L2 Regularization
import joblib  # For saving the scaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.losses import Huber


# Download stock data
stock_symbol = 'AAPL'  
start_date = '2020-01-01'
end_date = '2024-10-01'
data = yf.download(stock_symbol, start=start_date, end=end_date)

# Ensure data is available
if data.empty:
    raise ValueError(f"No stock data found for {stock_symbol} from {start_date} to {end_date}.")

# Preprocess data
close_prices = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
close_prices_scaled = scaler.fit_transform(close_prices)

# Save the scaler
joblib.dump(scaler, "models/scaler.pkl")

# Prepare LSTM data
def create_lstm_data(data, time_steps=1):
    x, y = [], []
    for i in range(len(data) - time_steps):
        x.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(x), np.array(y)

time_steps = 10
x, y = create_lstm_data(close_prices_scaled, time_steps)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Build and train the LSTM model
model = Sequential()
# Add LSTM layers with increased units but reduced complexity
model.add(LSTM(units=50, return_sequences=True, input_shape=(x.shape[1], x.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))


# Add Dense output layer
model.add(Dense(units=1))

# Compile the model with Adam optimizer and Huber Loss (robust to outliers)
optimizer = Adam(learning_rate=0.0001)  # Reduced learning rate
model.compile(optimizer=optimizer, loss=Huber())

# Train the model
model.fit(x, y, epochs=50, batch_size=32)

# Save the model as .h5 to match loading in app.py
model.save("models/lstm_model.h5")
print("Model and scaler saved successfully!")

# Define the split ratio for train and test data (e.g., 80% train, 20% test)
train_size = int(len(close_prices_scaled) * 0.8)
train_data, test_data = close_prices_scaled[:train_size], close_prices_scaled[train_size:]

# Create LSTM data for train and test
x_train, y_train = create_lstm_data(train_data, time_steps)
x_test, y_test = create_lstm_data(test_data, time_steps)

# Reshape x_train and x_test to be 3D as expected by the LSTM model
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Make predictions on the test set
y_pred = model.predict(x_test)

# Inverse scale the predictions and the actual values back to the original scale
y_pred = y_pred.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_pred_original = scaler.inverse_transform(y_pred)
y_test_original = scaler.inverse_transform(y_test)

# Calculate performance metrics
mse = mean_squared_error(y_test_original, y_pred_original)
rmse = np.sqrt(mse)
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Plot actual vs predicted prices
plt.figure(figsize=(10, 5))
plt.plot(y_test_original, label='Actual Prices', color='b')
plt.plot(y_pred_original, label='Predicted Prices', color='r')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()
plt.show()