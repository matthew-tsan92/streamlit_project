import streamlit as st
from streamlit_option_menu import option_menu

import pandas as pd
import yfinance as yf
from yfinance.utils import auto_adjust

from plotly import graph_objs as go
import cufflinks as cf
from plotly.subplots import make_subplots

from datetime import date

from prophet import Prophet
from prophet.plot import plot_plotly

import datetime
import requests

import streamlit as st
import requests
import os
import sys
import subprocess

# check if the library folder already exists, to avoid building everytime you load the pahe
if not os.path.isdir("/tmp/ta-lib"):

    # Download ta-lib to disk
    with open("/tmp/ta-lib-0.4.0-src.tar.gz", "wb") as file:
        response = requests.get(
            "http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz"
        )
        file.write(response.content)
    # get our current dir, to configure it back again. Just house keeping
    default_cwd = os.getcwd()
    os.chdir("/tmp")
    # untar
    os.system("tar -zxvf ta-lib-0.4.0-src.tar.gz")
    os.chdir("/tmp/ta-lib")
    os.system("ls -la /app/equity/")
    # build
    os.system("./configure --prefix=/home/appuser")
    os.system("make")
    # install
    os.system("make install")
    # back to the cwd
    os.chdir(default_cwd)
    sys.stdout.flush()

# add the library to our current environment
from ctypes import *

lib = CDLL("/home/appuser/lib/libta_lib.so.0.0.0")
# import library
try:
    import talib
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--global-option=build_ext", "--global-option=-L/home/appuser/lib/", "--global-option=-I/home/appuser/include/", "ta-lib"])
finally:
    import talib
import ta
yf.pdr_override()

#webpage title
st.set_page_config(page_title="Oil Stock",
layout="wide"
)

#dashboard
start_date = "2020-01-01"
today = date.today().strftime("%Y-%m-%d")
stocks = ("AAPL", "GOOG", "MSFT", "GME", "0883.HK")


#Load dataset
def load_data(stock):
    stock_data=yf.download(stock, start_date , today, interval="1wk")
    stock_data.reset_index(inplace=True)
    return stock_data

#Load dataset 2nd stock
def load_2nddata(stock):
    stock_data2=yf.download(stock, start_date , today, interval="1wk")
    stock_data2.reset_index(inplace=True)
    return stock_data2


#Sidebar
with st.sidebar:
    selected = option_menu(
        menu_title="Select a function to use",
        options=["Homepage","Company Info","Company indicators","Company stock price comparison","Stock Price Prediction", "Next"],
        icons=["house"]
    )


#page 1st title
if selected == "Homepage":
    st.title("Our first web app")
    header = st.container()
    with header:
        st.title("Welcome!!")
    st.markdown("Select an item from left panel to view")


if selected == "Company Info":
    with st.sidebar:
        select_stocks = st.selectbox("Please select stock to view", stocks)
    df_stock = load_data(select_stocks)

    ##Comapny details
    def pv(fv,requiredRateOfReturn,years):
        return fv / ((1 + requiredRateOfReturn / 100) ** years)

    def fv(pv,growth,years):
        return pv * (1 + growth)  ** years  

    # Get the data
    link  = f"""https://query1.finance.yahoo.com/v10/finance/quoteSummary/{select_stocks}?"""
    modules = f"""modules=assetProfile%2Cprice%2CfinancialData%2CearningsTrend%2CdefaultKeyStatistics"""
    requestString = link + modules

    request = requests.get(f"{requestString}", headers={"USER-AGENT": "Mozilla/5.0"})
    json = request.json()
    data = json["quoteSummary"]["result"][0]

    st.session_state.data = data

    if 'data' in st.session_state:

        data = st.session_state.data
        # Print company profile
        st.header("Company Profile")

        closeprice_today =  df_stock.iloc[-1,4]
        closeprice_yesterday =  df_stock.iloc[-2,4]
        price_delta = closeprice_today - closeprice_yesterday

        st.subheader(select_stocks)
        with st.expander("Company Description"):
            st.write(data["assetProfile"]["longBusinessSummary"])
        #kpi cards
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi4, kpi5 = st.columns([1,2])
        kpi1.metric("Stock symbol", select_stocks)
        kpi2.metric("Sector", data["assetProfile"]["sector"])
        kpi3.metric("Industry", data["assetProfile"]["industry"])

        kpi4.metric(label=select_stocks,value=closeprice_today,delta=price_delta)
        kpi5.metric("Website", data["assetProfile"]["website"])

        st.header("Valuation")
        # Add space between sections
        st.markdown("", unsafe_allow_html=True)
        st.markdown("", unsafe_allow_html=True)  

        # Get the metrics needed for valuation
        currentPrice = data["financialData"]["currentPrice"]["raw"]
        growth = data["earningsTrend"]["trend"][4][ "growth" ][ "raw" ] * 100
        peFWD = data["defaultKeyStatistics"]["forwardPE"]["raw"]
        epsFWD = data["defaultKeyStatistics"]["forwardEps"]["raw"]
        requiredRateOfReturn = 10.0
        yearsToProject = 5
  
        # Add controls
        with st.sidebar:
            growth = st.number_input("Growth", value=growth, step = 1.0)
            peFWD = st.number_input("P/E", value=peFWD, step = 1.0)
            requiredRateOfReturn = st.number_input("Required Rate Of Return", value=requiredRateOfReturn, step = 1.0)

        # Fair value calculation
        futureEPS = fv(epsFWD,growth/100,yearsToProject)
        futurePrice = futureEPS * peFWD 
        stickerPrice = pv(futurePrice, requiredRateOfReturn, yearsToProject)
        upside = (stickerPrice - currentPrice)/stickerPrice * 100

        # Show result
        kpi7, kpi8, kpi9 = st.columns(3)
        kpi10, kpi11, kpi12 = st.columns(3)
        kpi7.metric("Market Cap", data["price"]["marketCap"]["fmt"])
        kpi8.metric("EPS", "{:.2f}".format(futureEPS))
        kpi9.metric("Future Price", "{:.2f}".format(futurePrice))
        kpi10.metric("Sticker Price", "{:.2f}".format(stickerPrice))
        kpi11.metric("Current Price", "{:.2f}".format(currentPrice))
        kpi12.metric("Upside", "{:.2f}".format(upside))

if selected == "Company indicators":
    with st.sidebar:
        select_stocks = st.selectbox("Please select stock to view", stocks)
    df_stock = load_data(select_stocks)
    
    days_to_plot = st.sidebar.slider(
    'Choose number of days to plot', 
    min_value = 1,
    max_value = 300,
    value = 120,
    )
    ma1 = st.sidebar.number_input(
    'Moving Average #1 Length',
    value = 10,
    min_value = 1,
    max_value = 120,
    step = 1,    
    )
    ma2 = st.sidebar.number_input(
    'Moving Average #2 Length',
    value = 20,
    min_value = 1,
    max_value = 120,
    step = 1,    
    )

    #Candlestick with indicators
    df_candle = df_stock.copy()
    def get_candlestick_plot(
        df: pd.DataFrame,
        ma1: int,
        ma2: int,
        ticker: str
        ):
    
        fig = make_subplots(
            rows = 2,
            cols = 1,
            shared_xaxes = True,
            vertical_spacing = 0.1,
            subplot_titles = (f'{ticker} Stock Price', 'Volume Chart'),
            row_width = [0.3, 0.7]
        )
    
        fig.add_trace(
            go.Candlestick(
                x = df_candle['Date'],
                open = df_candle['Open'], 
                high = df_candle['High'],
                low = df_candle['Low'],
                close = df_candle['Close'],
                name = 'Candlestick'
            ),
            row = 1,
            col = 1,
        )
    
        fig.add_trace(
            go.Line(x = df_candle['Date'], y = df_candle[f'{ma1}_ma'], name = f'{ma1} SMA'),
            row = 1,
            col = 1,
        )
    
        fig.add_trace(
            go.Line(x = df_candle['Date'], y = df_candle[f'{ma2}_ma'], name = f'{ma2} SMA'),
            row = 1,
            col = 1,
        )
    
        fig.add_trace(
            go.Bar(x = df_candle['Date'], y = df_candle['Volume'], name = 'Volume'),
            row = 2,
            col = 1,
        )
    
        fig['layout']['xaxis2']['title'] = 'Date'
        fig['layout']['yaxis']['title'] = 'Price'
        fig['layout']['yaxis2']['title'] = 'Volume'
    
        fig.update_xaxes(
            rangebreaks = [{'bounds': ['sat', 'mon']}],
            rangeslider_visible = False,
        )
    
        return fig

    df_candle[f'{ma1}_ma'] = df_candle['Close'].rolling(ma1).mean()
    df_candle[f'{ma2}_ma'] = df_candle['Close'].rolling(ma2).mean()
    df_candle = df_candle[-days_to_plot:]
    fig = get_candlestick_plot(df_candle, ma1, ma2, select_stocks)

    st.plotly_chart(fig,  use_container_width = True)

    ##Indicators (separate)
    # Adjusted Close Price
    df_stock.set_index(['Date'], inplace=True)
    st.header(f"Adjusted Close Price of {select_stocks}")
    st.line_chart(df_stock['Adj Close'])

    # SMA (Simple Moving Average)
    df_stock['SMA'] = talib.SMA(df_stock['Adj Close'], timeperiod = 20)
    # EMA (Exponential Moving Average)
    df_stock['EMA'] = talib.EMA(df_stock['Adj Close'], timeperiod = 20)

    st.header(f"Simple Moving Average vs. Exponential Moving Average of {select_stocks}")
    st.line_chart(df_stock[['Adj Close','SMA','EMA']])

    # Bollinger Bands
    df_stock['upper_band'], df_stock['middle_band'], df_stock['lower_band'] = talib.BBANDS(df_stock['Adj Close'], timeperiod =20)

    st.header(f"Bollinger Bands of {select_stocks}")
    st.line_chart(df_stock[['Adj Close','upper_band','middle_band','lower_band']])

    # MACD (Moving Average Convergence Divergence)
    df_stock['macd'], df_stock['macdsignal'], df_stock['macdhist'] = talib.MACD(df_stock['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)

    st.header(f"Moving Average Convergence Divergence of {select_stocks}")
    st.line_chart(df_stock[['macd','macdsignal']])

    # CCI (Commodity Channel Index)
    cci = ta.trend.cci(df_stock['High'], df_stock['Low'], df_stock['Close'])

    st.header(f"Commodity Channel Index of {select_stocks}")
    st.line_chart(cci)

    # RSI (Relative Strength Index)
    df_stock['RSI'] = talib.RSI(df_stock['Adj Close'], timeperiod=14)

    st.header(f"Relative Strength Index of {select_stocks}")
    st.line_chart(df_stock['RSI'])

    # OBV (On Balance Volume)
    df_stock['OBV'] = talib.OBV(df_stock['Adj Close'], df_stock['Volume'])/10**6

    st.header(f"On Balance Volume of {select_stocks}")
    st.line_chart(df_stock['OBV'])



if selected == "Company stock price comparison":
    with st.sidebar:
        stocks_compare = st.multiselect("Pick stocks to compare", stocks)
    start_date_menu = st.date_input("Select a start date", value=pd.to_datetime(start_date))
    end_date_menu = st.date_input("Select an end date", value=pd.to_datetime(today))

    def relativeret(df):
            rel = df.pct_change()
            cumuret = (1+rel).cumprod() - 1
            cumuret = cumuret.fillna(0)
            return cumuret
    
    if len(stocks_compare) > 0:
        df_stock_compare = relativeret(yf.download(stocks_compare, start_date, today)["Adj Close"])
        st.line_chart(df_stock_compare)



if selected == "Stock Price Prediction":
    with st.sidebar:
        select_stocks = st.selectbox("Please select stock to view", stocks)
    df_stock = load_data(select_stocks)
    
    #Forecast stock price
    nyears = st.slider("Predited years:", 1,10)
    period = nyears*365
    df_train = df_stock[["Date", "Close"]]
    df_train = df_train.rename(columns={"Date":"ds", "Close":"y"})
    m = Prophet(daily_seasonality=True)
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period,freq='D')
    future['day'] = future['ds'].dt.weekday
    future = future[future['day']<=4]
    forecast = m.predict(future)

    st.subheader("Forecast stock")
    with st.expander("Show raw data"):
        st.subheader(f"Raw data of {select_stocks}")
        st.write(forecast.tail())

    fore_fig = plot_plotly(m, forecast)
    st.plotly_chart(fore_fig, use_container_width = True)
    fore_fig2 = m.plot_components(forecast, figsize=(8,4))
    st.write(fore_fig2)
