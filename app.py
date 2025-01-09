import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from pricing_model import train_dynamic_pricing_model, predict_optimal_price
from authentication import authenticate_user, register_user
from complaints import ensure_database, store_complaint, view_complaints

import smtplib
from email.mime.text import MIMEText
import matplotlib.pyplot as plt  # Importing matplotlib for plotting
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler

# Ensure required directories and database exist
def ensure_directory_structure():
    if not os.path.exists("models"):
        os.makedirs("models")
    # ensure_database()

ensure_directory_structure()

def login_page():
    st.title("Login")
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        login_button = st.button("Login")

        if login_button:
            if authenticate_user(username, password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.success(f"Welcome, {username}!")
            else:
                st.error("Invalid username or password.")

    with tab2:
        username = st.text_input("New Username", key="register_username")
        password = st.text_input("New Password", type="password", key="register_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm_password")
        register_button = st.button("Register")

        if register_button:
            if password != confirm_password:
                st.error("Passwords do not match!")
            elif register_user(username, password):
                st.success("Registration successful! You can now log in.")
            else:
                st.error("Username already exists.")

def train_forecasting_model(data):
    # Remove duplicates if any
    if data['date'].duplicated().any():
        st.warning("The 'date' column has duplicate values. Removing duplicates.")
        data = data.drop_duplicates(subset=['date'])

    # Optional: Aggregate data by date
    # data = data.groupby('date').agg({'quantity_sold': 'sum'}).reset_index()

    data = data.sort_values('date')
    ts_data = data.set_index('date')['quantity_sold']
    ts_data = ts_data.asfreq('D').fillna(0)

    model = SARIMAX(ts_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)

    return model_fit

def predict_demand(model, future_days=30):
    future = model.forecast(steps=future_days)
    return future

def predict_optimal_price(model, scaler, current_price, avg_competitor_price, current_inventory, num_customers_visited):
    # Here, we will use the model to predict the demand for different prices and select the optimal one
    price_range = np.linspace(current_price * 0.8, current_price * 1.2, 20)  # Try 20 different price points in range
    predicted_demands = []
    for price in price_range:
        # Predict demand using the dynamic pricing model
        demand_input = np.array([price, avg_competitor_price, current_inventory, num_customers_visited]).reshape(1, -1)
        scaled_input = scaler.transform(demand_input)
        demand = model.predict(scaled_input)
        predicted_demands.append(demand[0])

    # Find the price with the maximum predicted demand
    optimal_price = price_range[np.argmax(predicted_demands)]
    return optimal_price, max(predicted_demands)

def main_app():
    st.title("Dynamic Pricing Optimization for Retail Shops")

    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
        if "username" in st.session_state:
            del st.session_state["username"]
        st.success("Logged out successfully! ")
        st.stop()

    if st.session_state["authenticated"]:
        pass

    # Step 1: Dataset upload
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file with the required columns", type="csv")

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            data['date'] = pd.to_datetime(data['date'])

            required_columns = ['item_name', 'date', 'price', 'remaining_stock', 'quantity_sold', 'num_customers_visited']
            if not all(col in data.columns for col in required_columns):
                st.error(f"Dataset must contain the following columns: {', '.join(required_columns)}")
                return

            st.success("Dataset uploaded successfully!")

            if st.button("Train Demand Forecasting Model"):
                forecasting_model = train_forecasting_model(data)
                st.session_state["forecasting_model"] = forecasting_model
                st.success("Demand forecasting model trained successfully!")

            if "forecasting_model" in st.session_state:
                st.header("Demand Forecasting")
                future_days = st.number_input("Enter number of days to forecast", min_value=1, value=30, step=1)
                forecast = predict_demand(st.session_state["forecasting_model"], future_days)
                st.write(f"Forecast for next {future_days} days:")
                st.line_chart(forecast)

            # Train dynamic pricing model
            if 'model' not in st.session_state:
                if st.button("Train Dynamic Pricing Model"):
                    st.info("Training the model, please wait...")
                    model, scaler = train_dynamic_pricing_model(data)
                    st.session_state.model = model
                    st.session_state.scaler = scaler
                    st.success("Dynamic Pricing Model trained successfully!")

            # Input fields for predictions
            st.header("Provide Current Inputs")
            item_name = st.selectbox("Select Item", data['item_name'].unique(), key="item_name")
            current_price = st.number_input("Current Price", min_value=0.0, format="%.2f", key="current_price")
            avg_competitor_price = st.number_input("Average Competitor Price", min_value=0.0, format="%.2f", key="avg_competitor_price")
            current_inventory = st.number_input("Current Inventory Quantity", min_value=0, key="current_inventory")
            quantity_sold = st.number_input("Quantity Sold (Optional)", min_value=0, value=12, key="quantity_sold")
            num_customers_visited = st.number_input("Number of Customers Visited (Optional)", min_value=0, value=20, key="num_customers_visited")

            if st.button("Generate Recommendations"):
                if 'model' in st.session_state:
                    model = st.session_state.model
                    scaler = st.session_state.scaler

                    # Get optimal price and predicted demand
                    optimal_price, predicted_demand = predict_optimal_price(
                        model, scaler, current_price, avg_competitor_price, current_inventory, num_customers_visited
                    )

                    profit = (optimal_price - 80) * predicted_demand  # Assume a cost of ₹80 per item for profit calculation

                    st.write(f"Optimal Price: ₹{optimal_price:.2f}")
                    st.write(f"Predicted Demand: {predicted_demand:.2f} units")
                    st.write(f"Estimated Profit: ₹{profit:.2f}")

            # Graph: Past 6 Months Sales
            st.header("Sales Trends (Past 6 Months)")
            six_months_ago = datetime.now() - timedelta(days=180)
            past_sales_data = data[data['date'] > six_months_ago]

            if not past_sales_data.empty:
                monthly_sales = past_sales_data.groupby(past_sales_data['date'].dt.to_period('M'))['quantity_sold'].sum()
                fig, ax = plt.subplots(figsize=(10, 5))
                monthly_sales.plot(kind='bar', ax=ax, color="blue")
                ax.set_title("Sales Over the Past 6 Months")
                ax.set_xlabel("Month")
                ax.set_ylabel("Quantity Sold")
                st.pyplot(fig)
            else:
                st.write("No sales data available for the past 6 months.")

        except Exception as e:
            st.error(f"Error processing the file: {e}")

    # Complaint Submission
    st.header("Submit a Complaint")
    ensure_database()

    complaint_body = st.text_area('Your Complaint')

    if st.button("Submit Complaint"):
        if complaint_body:
            try:
                store_complaint(complaint_body)
                st.success('Complaint submitted successfully! 🚀')
            except Exception as e:
                st.error(f"Error submitting the complaint: {e}")
        else:
            st.error('Please fill in the complaint body.')

    # Admin view for complaints
    if st.sidebar.checkbox("Admin: View Complaints"):
        if st.session_state.get("username") == "admin":
            view_complaints()
        else:
            st.error("Access denied. Admins only.")

def main():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if st.session_state["authenticated"]:
        main_app()
    else:
        login_page()

if __name__ == "__main__":
    main()
