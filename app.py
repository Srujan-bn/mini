import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from authentication import authenticate_user, register_user
from complaints import ensure_database, store_complaint, view_complaints
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")

#checking github



import streamlit as st
import requests

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Chatbot",
    page_icon="🤖",
    layout="centered",
)

# Sidebar configuration
st.sidebar.title("Navigation")
selected_option = st.sidebar.radio("Choose an option:", ["Home", "Chatbot"])

# Initialize session state for the chatbot
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Chatbot interface
if selected_option == "Chatbot":
    # Chatbot title
    st.title("🤖 Chatbot")

    # Chat input box
    question = st.text_input("Ask a question:", key="input", placeholder="Type your question here...")

    # Submit button
    if st.button("Get Answer") and question.strip():
        # Display user question in chat
        st.session_state.conversation.append({"type": "user", "text": question})

        # Add a loading message temporarily
        st.session_state.conversation.append({"type": "bot", "text": "Loading..."})

        # Send the question to the API
        try:
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
            api_key = "${api_key}"  # Replace with your API key
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": question}]}],
                "generationConfig": {
                    "maxOutputTokens": 600,  # Limits the output to approximately 500-600 words
                    "temperature": 0.7,
                    "topK": 1,
                    "topP": 1,
                },
            }
            response = requests.post(f"{url}?key={api_key}", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

            # Parse the response
            bot_answer = (
                data["candidates"][0]["content"]["parts"][0]["text"]
                .replace("## ", "")  # Remove markdown headers
                .replace("**", "")   # Remove bold markers
                .replace("* ", "")   # Remove list markers
            )
            # Update the conversation with the bot's response
            st.session_state.conversation[-1] = {"type": "bot", "text": bot_answer}
        except Exception as e:
            # Handle errors
            st.session_state.conversation[-1] = {"type": "bot", "text": "Error generating answer"}

    # Display the chat conversation in a frame
    st.subheader("Chat History")
    with st.container():
        for msg in st.session_state.conversation:
            if msg["type"] == "user":
                st.markdown(f"<div style='background-color:#e8f4fa; padding:8px; border-radius:5px; margin-bottom:5px;'>"
                            f"<b>You:</b> {msg['text']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background-color:#f4e8fa; padding:8px; border-radius:5px; margin-bottom:5px;'>"
                            f"<b>Bot:</b> {msg['text']}</div>", unsafe_allow_html=True)

    # Reset conversation button
    if st.button("Reset Conversation"):
        st.session_state.conversation = []

# Home interface
if selected_option == "Home":
    st.title("Welcome!")
    st.write("Select 'Chatbot' from the sidebar to start interacting with the chatbot.")





def ensure_directory_structure():
    if not os.path.exists("models"):
        os.makedirs("models")

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

def train_forecasting_model(data, item_name):
    """ Train a forecasting model for a specific item's demand. """
    item_data = data[data['item_name'] == item_name]
    item_data = item_data.sort_values('date').drop_duplicates(subset=['date'])
    ts_data = item_data.set_index('date')['quantity_sold'].asfreq('D').fillna(0)

    model = SARIMAX(ts_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)

    return model_fit

def predict_demand(model, future_days=30):
    future = model.forecast(steps=future_days)
    return future

def train_dynamic_pricing_model(data):
    """ Train a dynamic pricing model using the provided data. """
    X = data[['price', 'remaining_stock', 'num_customers_visited']]
    y = data['quantity_sold']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler

def predict_optimal_price(model, scaler, current_price, avg_competitor_price, remaining_stock, num_customers_visited, cost_price_per_unit):
    price_range = np.linspace(current_price * 0.8, current_price * 1.2, 20)
    predicted_demands = []

    for price in price_range:
        demand_input = np.array([price, remaining_stock, num_customers_visited]).reshape(1, -1)
        scaled_input = scaler.transform(demand_input)
        demand = model.predict(scaled_input)
        predicted_demands.append(demand[0])

    # Find optimal price
    optimal_price = price_range[np.argmax(predicted_demands)]

    # Ensure the optimal price is at least 10% higher than the cost price
    optimal_price = max(optimal_price, cost_price_per_unit * 1.1)

    predicted_demand = max(predicted_demands)
    return optimal_price, predicted_demand

def main_app():
    st.title("Dynamic Pricing Optimization for Retail Shops")

    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
        if "username" in st.session_state:
            del st.session_state["username"]
        st.success("Logged out successfully!")
        st.stop()

    if st.session_state["authenticated"]:
        pass

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

            if st.button("Train Dynamic Pricing Model"):
                model, scaler = train_dynamic_pricing_model(data)
                st.session_state["model"] = model
                st.session_state["scaler"] = scaler
                st.success("Dynamic pricing model trained successfully!")

            st.header("Demand Forecasting")
            item_name = st.selectbox("Select Item for Forecasting", data['item_name'].unique(), key="forecast_item_name")

            if st.button("Train Demand Forecasting Model for Selected Item"):
                forecasting_model = train_forecasting_model(data, item_name)
                st.session_state[f"forecasting_model_{item_name}"] = forecasting_model
                st.success(f"Demand forecasting model for {item_name} trained successfully!")

            if f"forecasting_model_{item_name}" in st.session_state:
                forecasting_model = st.session_state[f"forecasting_model_{item_name}"]
                future_days = st.number_input("Enter number of days to forecast", min_value=1, value=30, step=1)
                forecast = predict_demand(forecasting_model, future_days)

                st.write(f"Forecast for next {future_days} days for {item_name}:")
                forecast_df = pd.DataFrame({
                    'Date': [datetime.today() + timedelta(days=i) for i in range(future_days)],
                    'Predicted Demand': forecast
                })
                st.dataframe(forecast_df)
                st.line_chart(forecast)

            st.header("Sales Trends (Past 6 Months)")
            data = data.sort_values(by='date')

            # Filter the first 6 months of data
            past_sales_data = data[(data['date'].dt.month >= 7) & (data['date'].dt.month <= 12)]
            if not past_sales_data.empty:
                # Group by month and sum the quantity sold
                past_sales_data = past_sales_data.copy()
                past_sales_data['month'] = past_sales_data['date'].dt.to_period('M')
                monthly_sales = past_sales_data.groupby('month')['quantity_sold'].sum()
                monthly_sales_scaled = monthly_sales / 100

                # Plotting
                fig, ax = plt.subplots(figsize=(10, 5))
                monthly_sales_scaled.plot(kind='bar', ax=ax, color="blue")
                ax.set_title("Sales Over the Past 6 Months")
                ax.set_xlabel("Month")
                ax.set_ylabel("Quantity Sold")
                st.pyplot(fig)
            else:
                st.write("No sales data available.")
                

            st.header("Provide Current Inputs")
            item_name = st.selectbox("Select Item", data['item_name'].unique(), key="item_name")
            current_price = st.number_input("Current Price", min_value=0.0, format="%.2f", key="current_price")
            avg_competitor_price = st.number_input("Average Competitor Price", min_value=0.0, format="%.2f", key="avg_competitor_price")
            remaining_stock = st.number_input("Remaining Stock", min_value=0, key="remaining_stock")
            num_customers_visited = st.number_input("Number of Customers Visited", min_value=0, value=20, key="num_customers_visited")

            cost_price_per_unit = 80  # Set this value appropriately

            if st.button("Generate Recommendations"):
                if 'model' in st.session_state:
                    model = st.session_state.model
                    scaler = st.session_state.scaler

                    try:
                        optimal_price, predicted_demand = predict_optimal_price(
                            model, scaler, current_price, avg_competitor_price, remaining_stock, num_customers_visited, cost_price_per_unit
                        )

                        revenue = optimal_price * predicted_demand
                        cost = cost_price_per_unit * predicted_demand
                        profit = revenue - cost

                        st.write(f"Optimal Price: ₹{optimal_price:.2f}")
                        st.write(f"Predicted Demand: {predicted_demand:.2f} units")
                        st.write(f"Estimated Revenue: ₹{revenue:.2f}")
                        st.write(f"Estimated Profit: ₹{profit:.2f}")

                    except Exception as e:
                        st.error(f"Error generating recommendations: {e}")
                else:
                    st.error("Dynamic Pricing Model is not trained yet. Train the model first.")

        except Exception as e:
            st.error(f"Error processing the file: {e}")

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