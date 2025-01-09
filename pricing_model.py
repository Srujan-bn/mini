# pricing_model.py
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Function to train the dynamic pricing model
def train_dynamic_pricing_model(data):
    features = ['price', 'remaining_stock', 'quantity_sold', 'num_customers_visited']
    target = 'optimal_price'
    
    if target not in data.columns:
        # Create a dummy 'optimal_price' column if it doesn't exist
        # This is just a placeholder for training purposes
        data[target] = data['price'] * (1 + np.random.uniform(-0.2, 0.2, len(data)))

    X = data[features]
    y = data[target]
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # Save the model and scaler
    with open("models/dynamic_pricing_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return model, scaler

# Function to predict the optimal price based on the trained model
def predict_optimal_price(model, scaler, input_data):
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    # Predict the optimal price
    return model.predict(input_data_scaled)

# Function to predict demand based on price difference and price sensitivity factor
def predict_demand(competitor_price, current_price, price_sensitivity_factor):
    # Calculate the price difference
    price_diff = current_price - competitor_price

    # Assume a base demand of 100 (this can be adjusted or modeled based on actual data)
    base_demand = 100

    # Adjust demand based on the price difference and sensitivity factor
    demand_adjustment = price_diff * price_sensitivity_factor
    predicted_demand = base_demand - demand_adjustment

    # Ensure that demand does not go below zero
    predicted_demand = max(predicted_demand, 0)

    return predicted_demand
