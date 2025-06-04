# app.py
import pandas as pd
from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Load the Prophet model
# Ensure the 'model' directory exists in the same location as app.py
# and prophet_model.pkl is inside it.
model_path = os.path.join('model', 'prophet_model.pkl')

# Check if the model file exists
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}. Please run train_model.py first.")
    # Exit or handle error appropriately in a production environment
    exit()

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Prophet model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit() # Exit if model cannot be loaded

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # You can add logic here to get the number of days to predict from the user
    # For now, let's predict for 30 days as a default
    periods = 30
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    # Prepare data for display
    # Convert 'ds' to string for easier display in HTML
    forecast['ds'] = forecast['ds'].dt.strftime('%Y-%m-%d')
    # Select relevant columns and get only the new predictions
    forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)

    return render_template('results.html', forecast=forecast_data.to_dict(orient='records'))

if __name__ == '__main__':
    # Use debug=True only for development. Set to False in production.
    app.run(debug=True)