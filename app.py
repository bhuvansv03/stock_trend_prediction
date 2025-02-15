import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from flask import Flask, render_template, request, send_file
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os
plt.style.use("fivethirtyeight")

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('stock_model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock = request.form.get('stock')
        if not stock:
            stock = 'AAPL'  # Default stock
        
        # Define the start and end dates for stock data
        start = dt.datetime(2000, 1, 1)
        end = dt.datetime(2024, 12, 1)
        
        # Download stock data
        df = yf.download(stock, start=start, end=end)
        
        # Descriptive Data
        data_desc = df.describe()
        
        # Exponential Moving Averages
        ema_20 = df.Close.ewm(span=20, adjust=False).mean()
        ema_50 = df.Close.ewm(span=50, adjust=False).mean()
        ema_100 = df.Close.ewm(span=100, adjust=False).mean()
        ema_200 = df.Close.ewm(span=200, adjust=False).mean()
        
        #Relative Strength Index (RSI)
        window_length = 14
        delta = df['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=window_length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window_length).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        #Moving Average Convergence Divergence (MACD)
        short_ema = df['Close'].ewm(span=12, adjust=False).mean()
        long_ema = df['Close'].ewm(span=26, adjust=False).mean()
        macd = short_ema - long_ema
        signal_line = macd.ewm(span=9, adjust=False).mean()

        #Bollinger Bands
        window = 20
        sma = df['Close'].rolling(window=window).mean()
        std_dev = df['Close'].rolling(window=window).std()
        upper_band = sma + (std_dev * 2)
        lower_band = sma - (std_dev * 2)
        lower_band = np.array(lower_band).flatten()
        upper_band = np.array(upper_band).flatten()

        #Rate of Change
        n = 14  # 14-day ROC
        roc = ((df['Close'] - df['Close'].shift(n)) / df['Close'].shift(n)) * 100

        #On-balance Volume
        obv = (df['Volume'] * np.sign(df['Close'].diff())).fillna(0).cumsum()

        # Data splitting
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])
        
        # Scaling data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)
        
        # Prepare data for prediction
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.fit_transform(final_df)
        
        x_test, y_test = [], []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i - 100:i])
            y_test.append(input_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)

        # Make predictions
        y_predicted = model.predict(x_test)

        # Inverse scaling for predictions
        y_predicted = scaler.inverse_transform(y_predicted.reshape(-1, 1))
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Plot 1: Closing Price vs Time Chart with 20 & 50 Days EMA
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df.Close, 'y', label='Closing Price', linewidth = 1)
        ax1.plot(ema_20, 'g', label='EMA 20', linewidth = 1)
        ax1.plot(ema_50, 'r', label='EMA 50', linewidth = 1)
        ax1.set_title("Closing Price vs Time (20 & 50 Days EMA)")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Price")
        ax1.legend()
        ema_chart_path = "static/ema_20_50.png"
        fig1.savefig(ema_chart_path)
        plt.close(fig1)
        
        # Plot 2: Closing Price vs Time Chart with 100 & 200 Days EMA
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(df.Close, 'y', label='Closing Price', linewidth = 1)
        ax2.plot(ema_100, 'g', label='EMA 100', linewidth = 1)
        ax2.plot(ema_200, 'r', label='EMA 200', linewidth = 1)
        ax2.set_title("Closing Price vs Time (100 & 200 Days EMA)")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Price")
        ax2.legend()
        ema_chart_path_100_200 = "static/ema_100_200.png"
        fig2.savefig(ema_chart_path_100_200)
        plt.close(fig2)
        
        # Plot 3: Relative Strength Index
        fig_rsi, ax_rsi = plt.subplots(figsize=(12, 6))
        ax_rsi.plot(df.index, rsi, label="RSI", color='purple', linewidth = 1)
        ax_rsi.axhline(70, linestyle='--', color='red', label="Overbought (70)")
        ax_rsi.axhline(30, linestyle='--', color='green', label="Oversold (30)")
        ax_rsi.set_title("Relative Strength Index (RSI)")
        ax_rsi.set_xlabel("Date")
        ax_rsi.set_ylabel("RSI Value")
        ax_rsi.legend()
        rsi_chart_path = "static/rsi.png"
        fig_rsi.savefig(rsi_chart_path)
        plt.close(fig_rsi)

        # Plot 4: Moving Average Convergence Divergence
        fig_macd, ax_macd = plt.subplots(figsize=(12, 6))
        ax_macd.plot(df.index, macd, label="MACD", color='blue', linewidth = 1)
        ax_macd.plot(df.index, signal_line, label="Signal Line", color='orange', linewidth = 1)
        ax_macd.set_title("MACD vs Signal Line")
        ax_macd.set_xlabel("Date")
        ax_macd.set_ylabel("Value")
        ax_macd.legend()
        macd_chart_path = "static/macd.png"
        fig_macd.savefig(macd_chart_path)
        plt.close(fig_macd)

        # Plot 5: Bollinger Bands
        fig_bb, ax_bb = plt.subplots(figsize=(12, 6))
        ax_bb.plot(df.index, df['Close'], label="Closing Price", color='black', linewidth = 1)
        ax_bb.plot(df.index, upper_band, label="Upper Band", color='red', linewidth = 1)
        ax_bb.plot(df.index, lower_band, label="Lower Band", color='green', linewidth = 1)
        ax_bb.fill_between(df.index, lower_band, upper_band, color='gray', alpha=0.3)
        ax_bb.set_title("Bollinger Bands")
        ax_bb.set_xlabel("Date")
        ax_bb.set_ylabel("Price")
        ax_bb.legend()
        bb_chart_path = "static/bollinger_bands.png"
        fig_bb.savefig(bb_chart_path)
        plt.close(fig_bb)

        # Plot 6: Rate Of Change
        fig_roc, ax_roc = plt.subplots(figsize=(12, 6))
        ax_roc.plot(df.index, roc, label="Rate of Change", color='brown', linewidth = 1)
        ax_roc.axhline(0, linestyle='--', color='black')  # Reference line at 0
        ax_roc.set_title("Rate of Change (ROC)")
        ax_roc.set_xlabel("Date")
        ax_roc.set_ylabel("ROC (%)")
        ax_roc.legend()
        roc_chart_path = "static/roc.png"
        fig_roc.savefig(roc_chart_path)
        plt.close(fig_roc)

        # Plot 7: On Balance Volume
        fig_obv, ax_obv = plt.subplots(figsize=(12, 6))
        ax_obv.plot(df.index, obv, label="OBV", color='blue', linewidth = 1)
        ax_obv.set_title("On-Balance Volume (OBV)")
        ax_obv.set_xlabel("Date")
        ax_obv.set_ylabel("OBV Value")
        ax_obv.legend()
        obv_chart_path = "static/obv.png"
        fig_obv.savefig(obv_chart_path)
        plt.close(fig_obv)

        # Plot 8: Prediction vs Original Trend
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(y_test, 'g', label="Original Price", linewidth = 1)
        ax3.plot(y_predicted, 'r', label="Predicted Price", linewidth = 1)
        ax3.set_title("Prediction vs Original Trend")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Price")
        ax3.legend()
        prediction_chart_path = "static/stock_prediction.png"
        fig3.savefig(prediction_chart_path)
        plt.close(fig3)

        # Save dataset as CSV
        csv_file_path = f"static/{stock}_dataset.csv"
        df.to_csv(csv_file_path)

        # Return the rendered template with charts and dataset
        return render_template('index.html', 
                               plot_path_ema_20_50=ema_chart_path, 
                               plot_path_ema_100_200=ema_chart_path_100_200, 
                               plot_path_rsi=rsi_chart_path,
                               plot_path_macd=macd_chart_path,
                               plot_path_bb=bb_chart_path,
                               plot_path_roc=roc_chart_path,
                               plot_path_obv=obv_chart_path,
                               plot_path_prediction=prediction_chart_path,
                               data_desc=data_desc.to_html(classes='table table-bordered'),
                               dataset_link=csv_file_path)

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(f"static/{filename}", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)