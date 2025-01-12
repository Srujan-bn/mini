import streamlit as st
import subprocess
import os
from datetime import datetime

# Set page configuration
st.set_page_config(layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #007BFF;  
        font-family: 'Arial', sans-serif;
    }
    .glass-effect {
        background: rgba(255, 255, 255, 0.7);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease;
    }
    .glass-effect:hover {
        transform: scale(1.02);  /* Slight zoom effect on hover */
    }
    .main-title {
        color: #oo7BFF;
        font-size: 3.5rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        padding: 40px 0;
        text-align: center;
    }
    .section-title {
        color: #1A237E;
        font-size: 2rem;
        margin-bottom: 20px;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
        text-align: center;
    }
    .info-box {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease;
        margin: 10px;  /* Space between boxes */
    }
    .info-box:hover {
        transform: translateY(-5px);  /* Lift effect on hover */
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 20px;
        border: none;
        margin: 5px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Create a container for buttons in the top right
with st.container():
    col_right = st.columns([6, 1])  # Create space on the left and one column for buttons
    
    with col_right[1]:
        if st.button("Login/Register"):
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current directory
                app_path = os.path.join(current_dir, 'app.py')  # Path to app.py
                subprocess.Popen(['streamlit', 'run', app_path])  # Run app.py
                st.success("Opening login page...")
            except Exception as e:
                st.error(f"Error running app.py: {str(e)}")

# Add the header section
st.markdown("<h1 class='main-title'>Dynamic Pricing Optimization for Retail</h1>", unsafe_allow_html=True)

# Instructions for users
st.markdown("""
    <div class="glass-effect">
        <h2 style='color: #1A237E;'>Important Instructions</h2>
        <p style='color: #555;'>
            The dataset you are going to upload will be safe and cannot be seen by anyone except you. Not even the admin can access the dataset you upload. 
            Additionally, your upload history will not be saved.
        </p>
    </div>
""", unsafe_allow_html=True)

# Benefits Section
st.markdown("<h1 class='section-title'>What do you get</h1>", unsafe_allow_html=True)

# Create two rows with two columns each
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# First row
with col1:
    st.markdown("""
        <div class="info-box">
            <h3 class="text-xl text-blue-800">🏷️ Price Optimization</h3>
            <p>Optimize the way you handle your inventory, product demand, supply, and price elasticity.</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="info-box">
            <h3 class="text-xl text-blue-800">📈 Sales Forecasting</h3>
            <p>Predict future sales trends using advanced analytics and historical data to make informed business decisions.</p>
        </div>
    """, unsafe_allow_html=True)

# Second row
with col3:
    st.markdown("""
        <div class="info-box">
            <h3 class="text-xl text-blue-800">💳 Capture your customers' willingness-to-pay</h3>
            <p>For any product, at any time, without any market analysis in days instead of weeks.</p>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
        <div class="info-box">
            <h3 class="text-xl text-blue-800">📄 Simplify the manual task of pricing products</h3>
            <p>Across your entire catalog, done in minutes instead of hours.</p>
        </div>
    """, unsafe_allow_html=True)

# Add spacing after benefits section
st.markdown("<br><br>", unsafe_allow_html=True)