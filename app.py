from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import joblib  # For saving/loading scaler
from datetime import timedelta
import os  # Imported as per your requirement

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model and scaler
try:
    model = load_model("models/lstm_model.h5")  # Updated to .h5 extension
    scaler = joblib.load("models/scaler.pkl")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model, scaler = None, None  # Set to None if loading fails

# Helper function to create data for LSTM prediction
def create_lstm_data(data, time_steps=1):
    x = []
    for i in range(len(data) - time_steps):
        x.append(data[i:(i + time_steps), 0])
    return np.array(x)

# Endpoint to render the home page
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint for stock prediction
@app.route('/https://stock-prediction-frontend.onrender.com/predict', methods=['POST'])
def predict():
    # Ensure model and scaler are loaded
    if model is None or scaler is None:
        return jsonify({"error": "Model or scaler could not be loaded"}), 500

    # Get user inputs from JSON data
    data = request.get_json()
    stock_symbol = data.get('ticker')
    num_of_days = data.get('days')

    if not stock_symbol or not num_of_days:
        return jsonify({"error": "Invalid input: Please provide a stock ticker and the number of days"}), 400

    # Convert number of days to integer
    try:
        num_of_days = int(num_of_days)
    except ValueError:
        return jsonify({"error": "Number of days should be an integer"}), 400

    # Fetch the latest stock data
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    start_date = (pd.Timestamp.now() - timedelta(days=1500)).strftime('%Y-%m-%d')
    
    try:
        data = yf.download(stock_symbol, start=start_date, end=end_date)
        if data.empty:
            return jsonify({"error": f"No data found for {stock_symbol}"}), 404
        close_prices = data['Close'].values.reshape(-1, 1)
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve stock data: {e}"}), 500

    # Scale and prepare data for LSTM
    try:
        close_prices_scaled = scaler.transform(close_prices)
        if np.isnan(close_prices_scaled).any():
            return jsonify({"error": "Stock data contains invalid values (NaN/Inf)"}), 500
    except Exception as e:
        return jsonify({"error": f"Scaling error: {e}"}), 500

    # Set up for predictions
    time_steps = 10
    last_prices_scaled = close_prices_scaled[-time_steps:]

    # Predict future prices
    predicted_prices = []
    for _ in range(num_of_days):
        x_pred = np.array([last_prices_scaled[-time_steps:, 0]])
        x_pred = np.reshape(x_pred, (x_pred.shape[0], x_pred.shape[1], 1))
        
        predicted_price_scaled = model.predict(x_pred)
        predicted_price = scaler.inverse_transform(predicted_price_scaled)
        
        # Convert the predicted price to a standard Python float for JSON serialization
        predicted_prices.append(float(predicted_price[0, 0]))
        
        # Update last prices for next prediction
        last_prices_scaled = np.append(last_prices_scaled, predicted_price_scaled, axis=0)
        last_prices_scaled = last_prices_scaled[-time_steps:]

    # Create future dates
    future_dates = pd.date_range(start=end_date, periods=num_of_days + 1)[1:]

    # Prepare response data
    predicted_data = [{"date": date.strftime("%Y-%m-%d"), "price": round(price, 2)} 
                      for date, price in zip(future_dates, predicted_prices)]

    return jsonify(predicted_data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Added per your request
    app.run(host='0.0.0.0', port=port)  # Updated app.run to use specified host and port



































# from flask import Flask, render_template, request, jsonify
# from flask_cors import CORS
# import numpy as np
# import pandas as pd
# from keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler
# import yfinance as yf
# import joblib  # For saving/loading scaler
# from datetime import timedelta

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # Load the trained model and scaler
# try:
#     model = load_model("models/lstm_model.h5")  # Updated to .h5 extension
#     scaler = joblib.load("models/scaler.pkl")
# except Exception as e:
#     print(f"Error loading model or scaler: {e}")
#     model, scaler = None, None  # Set to None if loading fails

# # Helper function to create data for LSTM prediction
# def create_lstm_data(data, time_steps=1):
#     x = []
#     for i in range(len(data) - time_steps):
#         x.append(data[i:(i + time_steps), 0])
#     return np.array(x)

# # Endpoint to render the home page
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Endpoint for stock prediction
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Ensure model and scaler are loaded
#     if model is None or scaler is None:
#         return jsonify({"error": "Model or scaler could not be loaded"}), 500

#     # Get user inputs from JSON data
#     data = request.get_json()
#     stock_symbol = data.get('ticker')
#     num_of_days = data.get('days')

#     if not stock_symbol or not num_of_days:
#         return jsonify({"error": "Invalid input: Please provide a stock ticker and the number of days"}), 400

#     # Convert number of days to integer
#     try:
#         num_of_days = int(num_of_days)
#     except ValueError:
#         return jsonify({"error": "Number of days should be an integer"}), 400

#     # Fetch the latest stock data
#     end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
#     start_date = (pd.Timestamp.now() - timedelta(days=1500)).strftime('%Y-%m-%d')
    
#     try:
#         data = yf.download(stock_symbol, start=start_date, end=end_date)
#         if data.empty:
#             return jsonify({"error": f"No data found for {stock_symbol}"}), 404
#         close_prices = data['Close'].values.reshape(-1, 1)
#     except Exception as e:
#         return jsonify({"error": f"Failed to retrieve stock data: {e}"}), 500

#     # Scale and prepare data for LSTM
#     try:
#         close_prices_scaled = scaler.transform(close_prices)
#         if np.isnan(close_prices_scaled).any():
#             return jsonify({"error": "Stock data contains invalid values (NaN/Inf)"}), 500
#     except Exception as e:
#         return jsonify({"error": f"Scaling error: {e}"}), 500

#     # Set up for predictions
#     time_steps = 10
#     last_prices_scaled = close_prices_scaled[-time_steps:]

#     # Predict future prices
#     predicted_prices = []
#     for _ in range(num_of_days):
#         x_pred = np.array([last_prices_scaled[-time_steps:, 0]])
#         x_pred = np.reshape(x_pred, (x_pred.shape[0], x_pred.shape[1], 1))
        
#         predicted_price_scaled = model.predict(x_pred)
#         predicted_price = scaler.inverse_transform(predicted_price_scaled)
        
#         # Convert the predicted price to a standard Python float for JSON serialization
#         predicted_prices.append(float(predicted_price[0, 0]))
        
#         # Update last prices for next prediction
#         last_prices_scaled = np.append(last_prices_scaled, predicted_price_scaled, axis=0)
#         last_prices_scaled = last_prices_scaled[-time_steps:]

#     # Create future dates
#     future_dates = pd.date_range(start=end_date, periods=num_of_days + 1)[1:]

#     # Prepare response data
#     predicted_data = [{"date": date.strftime("%Y-%m-%d"), "price": round(price, 2)} 
#                       for date, price in zip(future_dates, predicted_prices)]

#     return jsonify(predicted_data)

# if __name__ == '__main__':
#     app.run(debug=True)
