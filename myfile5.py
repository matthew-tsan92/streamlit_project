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

import ta
from ta.volatility import BollingerBands
import pandas_ta as pta

yf.pdr_override()

#webpage title
st.set_page_config(page_title="HK Oil Stocks APP",
layout="wide"
)

#dashboard
start_date = "2020-01-01"
today = date.today().strftime("%Y-%m-%d")
stocks = ("0857.HK", "0386.HK", "0467.HK", "0650.HK", "2386.HK", "2686.HK", "^HSI")


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
        menu_title="Select a page",
        options=["Homepage","Company Info","Company Indicators","Company Stock Return Comparison","Stock Price Prediction"],
        icons=["house","book","graph-up","arrows-angle-contract","clock-history"]
    )


#page 1st title
if selected == "Homepage":
    st.title("Our first web app")
    header = st.container()
    with header:
        st.title("Welcome!!")
    st.markdown("This web app aims to include several functions for users to have a look at financial data of local crude oil companies.")
    st.markdown("Select an item from the left panel to view.")


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

        st.header("Company Valuation")
        # Add space between sections
        st.markdown("", unsafe_allow_html=True)
        st.markdown("", unsafe_allow_html=True)  

        # Get the metrics needed for valuation
        currentPrice = data["financialData"]["currentPrice"]["raw"]
        growth = data["earningsTrend"]["trend"][4]["growth"]["raw"] * 100
        peFWD = data["defaultKeyStatistics"]["forwardPE"]["raw"]
        epsFWD = data["defaultKeyStatistics"]["forwardEps"]["raw"]
        requiredRateOfReturn = 10.0
        yearsToProject = 5
  
        # User's input
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
        kpi9.metric("Current Price", "{:.2f}".format(currentPrice))
        kpi10.metric("Sticker Price", "{:.2f}".format(stickerPrice))
        kpi11.metric("Future Price", "{:.2f}".format(futurePrice))
        kpi12.metric("Upside", "{:.2f}".format(upside))

if selected == "Company Indicators":
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
    'Moving Average No.1 Length',
    value = 10,
    min_value = 1,
    max_value = 120,
    step = 1,    
    )
    ma2 = st.sidebar.number_input(
    'Moving Average No.2 Length',
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
    df_stock.set_index(['Date'], inplace=True)
    # Adjusted Close Price
    st.header(f"Adjusted Close Price of {select_stocks}")
    st.line_chart(df_stock['Adj Close'])

    # Crude Oil Price
    df_clf = yf.download("CL=F", start_date , today, interval="1wk")
    st.header(f"Adjusted Close Price of Crude Oil Feb 23")
    st.line_chart(df_clf['Adj Close'])

    # SMA (Simple Moving Average)
    df_stock['SMA'] = pta.sma(df_stock['Adj Close'], timeperiod = 20)
    # EMA (Exponential Moving Average)
    df_stock['EMA'] = pta.ema(df_stock['Adj Close'], timeperiod = 20)

    st.header(f"Simple Moving Average vs. Exponential Moving Average of {select_stocks}")
    st.line_chart(df_stock[['Adj Close','SMA','EMA']])

    # Bollinger Bands
    indicator_bb = BollingerBands(close=df_stock["Adj Close"], window=20, window_dev=2)
    df_stock['middle_band'] = indicator_bb.bollinger_mavg()
    df_stock['upper_band'] = indicator_bb.bollinger_hband()
    df_stock['lower_band'] = indicator_bb.bollinger_lband()
    st.header(f"Bollinger Bands of {select_stocks}")
    st.line_chart(df_stock[['Adj Close','middle_band','upper_band','lower_band']])

    # MACD (Moving Average Convergence Divergence)
    df_stock['macd'] = ta.trend.macd(df_stock['Adj Close'], window_fast=12, window_slow=26)
    df_stock['macdsignal'] = ta.trend.macd_signal(df_stock['Adj Close'], window_fast=12, window_slow=26, window_sign=9)

    st.header(f"Moving Average Convergence Divergence of {select_stocks}")
    st.line_chart(df_stock[['macd','macdsignal']])

    # RSI (Relative Strength Index)
    df_stock['RSI'] = pta.rsi(df_stock['Adj Close'], length=14)

    st.header(f"Relative Strength Index of {select_stocks}")
    st.line_chart(df_stock['RSI'])

    # OBV (On Balance Volume)
    df_stock['OBV'] = pta.obv(df_stock['Adj Close'], df_stock['Volume'])/10**6

    st.header(f"On Balance Volume of {select_stocks}")
    st.line_chart(df_stock['OBV'])



if selected == "Company Stock Return Comparison":
    with st.sidebar:
        stocks_compare = st.multiselect("Pick stocks to compare", stocks)
    start_date_menu = st.date_input("Select a start date", value=pd.to_datetime(start_date))
    end_date_menu = st.date_input("Select an end date", value=pd.to_datetime(today))

    def relative_return(df):
            rel = df.pct_change()
            cumul_return = (1+rel).cumprod() - 1
            cumul_return = cumul_return.fillna(0)
            return cumul_return
    
    if len(stocks_compare) > 0:
        st.header(f"Return of {stocks_compare}")
        df_stock_compare = relative_return(yf.download(stocks_compare, start_date, today)["Adj Close"])
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
