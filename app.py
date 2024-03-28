import streamlit as st
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Fetch live stock data from Alpha Vantage
@st.cache
def load_data(symbol):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey=YOUR_API_KEY"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame.from_dict(data['Time Series (5min)'], orient='index')
    df.index = pd.to_datetime(df.index)
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Convert data types to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Drop rows with missing values
    df = df.dropna()
    
    return df

# Train a linear regression model
def train_model(data):
    # Extract features and target variable
    X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = data['Close']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model

# Main function to run the Streamlit app
def main():
    st.title('Stock Price Prediction')

    # Sidebar for user input
    st.sidebar.title('User Input')
    symbol = st.sidebar.text_input('Enter Stock Symbol (e.g., AAPL)', 'AAPL')

    # Fetch and display live stock data
    data = load_data(symbol)
    st.write('Live Stock Data:')
    st.write(data)

    # Train and display prediction
    if not data.empty:
        model = train_model(data)
        st.write('Model Training Complete')

        # Make prediction
        latest_data = data.iloc[-1][['Open', 'High', 'Low', 'Close', 'Volume']]
        prediction = model.predict([latest_data])[0]
        st.write('Predicted Closing Price:', prediction)

if __name__ == '__main__':
    main()
