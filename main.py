# # import pandas as pd
# # import numpy as np
# # import pandas_ta as ta
# # from scipy.signal import argrelextrema
# # from concurrent.futures import ThreadPoolExecutor, as_completed
# # import random
# # import warnings
# # from datetime import datetime, timedelta
# # import json
# # from sklearn.linear_model import LinearRegression
# # from flask import Flask, jsonify, request
# # from flask_cors import CORS
# # import logging
# # import time
# # import requests
# # import math
# # import yfinance as yf
# # import os

# # warnings.filterwarnings('ignore')

# # # ================= GLOBAL CONFIGURATION =================
# # TWELVE_DATA_API_KEY = "73adc6cc7e43476e851dcf54c705aeeb"

# # RISK_FREE_RATE = 0.02
# # MAX_WORKERS = 1
# # MIN_MARKET_CAP = 500e6
# # MIN_PRICE = 5.0
# # PATTERN_SENSITIVITY = 0.05
# # FIBONACCI_TOLERANCE = 0.05
# # CHANNEL_CONFIRMATION_BARS = 3
# # PATTERN_LOOKBACK = 20
# # ZIGZAG_LENGTH = 5
# # ZIGZAG_DEPTH = 10
# # ZIGZAG_NUM_PIVOTS = 10
# # CYCLE_MIN_DURATION = 30
# # PATTERN_ANGLE_THRESHOLD = 1.5
# # PATTERN_EXPANSION_RATIO = 1.2
# # PATTERN_CONTRACTION_RATIO = 0.8
# # MIN_TRENDLINE_R2 = 0.75
# # CONFIRMATION_VOL_RATIO = 1.2
# # MIN_TRENDLINE_ANGLE = 0.5
# # MAX_TRENDLINE_ANGLE = 85
# # HARMONIC_ERROR_TOLERANCE = 0.05
# # PRZ_LEFT_RANGE = 20
# # PRZ_RIGHT_RANGE = 20
# # FIBONACCI_LINE_LENGTH = 30
# # FUNDAMENTAL_WEIGHT = 0.3
# # SENTIMENT_WEIGHT = 0.2
# # TECHNICAL_WEIGHT = 0.5

# # # Rate limiting configuration
# # TWELVE_DATA_RATE_LIMIT_PER_MIN = 6
# # TWELVE_DATA_BATCH_SIZE = 3
# # TWELVE_DATA_BATCH_SLEEP = 65
# # YFINANCE_DELAY = 0.5

# # # Setup logging for production
# # logging.basicConfig(
# #     level=logging.INFO,
# #     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
# #     handlers=[
# #         logging.StreamHandler()
# #     ]
# # )
# # logger = logging.getLogger(__name__)

# # # ================= FLASK APP SETUP =================
# # app = Flask(__name__)

# # # Simple CORS configuration - avoid conflicts
# # CORS(app, resources={
# #     r"/*": {
# #         "origins": ["*", "http://localhost:5177", "https://my-stocks-s2at.onrender.com"]  # Add your frontend's deployed URL
# #     }
# # })
# # # ================= DATA FUNCTIONS =================
# # def get_filtered_stocks(num_stocks=20):
# #     """Get list of stocks to analyze - split between Twelve Data and yfinance"""
# #     stock_list = [
# #         "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN",
# #         "NVDA", "META", "NFLX", "BABA", "AMD",
# #         "INTC", "UBER", "SHOP", "PYPL", "PEP",
# #         "KO", "DIS", "NKE", "WMT", "CRM"
# #     ]
    
# #     # Split stocks: first 15 for Twelve Data, last 5 for yfinance
# #     twelve_data_stocks = stock_list[:15]
# #     yfinance_stocks = stock_list[15:20]
    
# #     return {
# #         'twelve_data': twelve_data_stocks,
# #         'yfinance': yfinance_stocks,
# #         'all': stock_list[:num_stocks]
# #     }

# # def fetch_stock_data_twelve(symbol, interval="1day", outputsize=100):
# #     """Fetch stock data from Twelve Data API with proper error handling"""
# #     try:
# #         url = f"https://api.twelvedata.com/time_series"
# #         params = {
# #             'symbol': symbol,
# #             'interval': interval,
# #             'outputsize': outputsize,
# #             'apikey': TWELVE_DATA_API_KEY,
# #             'format': 'JSON'
# #         }
        
# #         response = requests.get(url, params=params, timeout=30)
# #         response.raise_for_status()
        
# #         data = response.json()
        
# #         if 'values' not in data:
# #             logger.error(f"No values in response for {symbol}: {data}")
# #             return pd.DataFrame()
        
# #         df = pd.DataFrame(data['values'])
        
# #         if df.empty:
# #             logger.error(f"Empty DataFrame for {symbol}")
# #             return pd.DataFrame()
        
# #         numeric_columns = ['open', 'high', 'low', 'close', 'volume']
# #         for col in numeric_columns:
# #             if col in df.columns:
# #                 df[col] = pd.to_numeric(df[col], errors='coerce')
        
# #         df['datetime'] = pd.to_datetime(df['datetime'])
# #         df.set_index('datetime', inplace=True)
# #         df.sort_index(inplace=True)
# #         df.dropna(inplace=True)
        
# #         logger.info(f"Successfully fetched {len(df)} rows for {symbol} from Twelve Data")
# #         return df
        
# #     except Exception as e:
# #         logger.error(f"Error fetching data from Twelve Data for {symbol}: {str(e)}")
# #         return pd.DataFrame()

# # def fetch_stock_data_yfinance(symbol, period="100d", interval="1d"):
# #     """Fetch stock data from yfinance with proper error handling"""
# #     try:
# #         time.sleep(YFINANCE_DELAY)
        
# #         ticker = yf.Ticker(symbol)
# #         df = ticker.history(period=period, interval=interval)
        
# #         if df.empty:
# #             logger.error(f"Empty DataFrame for {symbol} from yfinance")
# #             return pd.DataFrame()
        
# #         df.columns = df.columns.str.lower()
# #         df.reset_index(inplace=True)
# #         df.rename(columns={'date': 'datetime'}, inplace=True)
# #         df.set_index('datetime', inplace=True)
# #         df.dropna(inplace=True)
        
# #         logger.info(f"Successfully fetched {len(df)} rows for {symbol} from yfinance")
# #         return df
        
# #     except Exception as e:
# #         logger.error(f"Error fetching data from yfinance for {symbol}: {str(e)}")
# #         return pd.DataFrame()

# # def fetch_stock_data(symbol, interval="1day", outputsize=100, source="twelve_data"):
# #     """Unified function to fetch stock data from either source"""
# #     if source == "twelve_data":
# #         return fetch_stock_data_twelve(symbol, interval, outputsize)
# #     elif source == "yfinance":
# #         yf_interval = "1d" if interval == "1day" else "1wk" if interval == "1week" else "1d"
# #         yf_period = f"{outputsize}d" if interval == "1day" else f"{outputsize//7}wk"
# #         return fetch_stock_data_yfinance(symbol, period=yf_period, interval=yf_interval)
# #     else:
# #         logger.error(f"Unknown data source: {source}")
# #         return pd.DataFrame()

# # def heikin_ashi(df):
# #     """Convert dataframe to Heikin-Ashi candles with proper error handling"""
# #     if df.empty:
# #         return pd.DataFrame()
    
# #     try:
# #         df = df.copy()
        
# #         required_cols = ['open', 'high', 'low', 'close']
# #         if not all(col in df.columns for col in required_cols):
# #             logger.error(f"Missing required columns. Available: {df.columns.tolist()}")
# #             return pd.DataFrame()
        
# #         df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
# #         ha_open = [(df['open'].iloc[0] + df['close'].iloc[0]) / 2]
# #         for i in range(1, len(df)):
# #             ha_open.append((ha_open[i-1] + df['HA_Close'].iloc[i-1]) / 2)
        
# #         df['HA_Open'] = ha_open
# #         df['HA_High'] = df[['high', 'HA_Open', 'HA_Close']].max(axis=1)
# #         df['HA_Low'] = df[['low', 'HA_Open', 'HA_Close']].min(axis=1)
        
# #         df.dropna(subset=['HA_Close', 'HA_Open', 'HA_High', 'HA_Low'], inplace=True)
        
# #         return df
        
# #     except Exception as e:
# #         logger.error(f"Error in heikin_ashi calculation: {str(e)}")
# #         return pd.DataFrame()

# # def detect_zigzag_pivots(data):
# #     """Detect significant pivot points using zigzag algorithm"""
# #     try:
# #         if len(data) < 10 or 'HA_Close' not in data.columns:
# #             return []
        
# #         prices = data['HA_Close'].values
        
# #         highs = argrelextrema(prices, np.greater, order=ZIGZAG_LENGTH)[0]
# #         lows = argrelextrema(prices, np.less, order=ZIGZAG_LENGTH)[0]
        
# #         pivot_indices = np.concatenate([highs, lows])
# #         pivot_indices.sort()
        
# #         filtered_pivots = []
# #         for i in pivot_indices:
# #             if len(filtered_pivots) < 2:
# #                 filtered_pivots.append(i)
# #             else:
# #                 last_price = prices[filtered_pivots[-1]]
# #                 current_price = prices[i]
# #                 change = abs(current_price - last_price) / last_price
# #                 if change > PATTERN_SENSITIVITY:
# #                     filtered_pivots.append(i)
        
# #         pivot_data = []
# #         for i in filtered_pivots:
# #             start_idx = max(0, i - ZIGZAG_DEPTH)
# #             end_idx = min(len(prices), i + ZIGZAG_DEPTH)
# #             local_max = np.max(prices[start_idx:end_idx])
# #             local_min = np.min(prices[start_idx:end_idx])
            
# #             if prices[i] == local_max:
# #                 pivot_type = 'high'
# #             else:
# #                 pivot_type = 'low'
            
# #             pivot_data.append((i, prices[i], pivot_type))
        
# #         return pivot_data[-ZIGZAG_NUM_PIVOTS:]
        
# #     except Exception as e:
# #         logger.error(f"Error in detect_zigzag_pivots: {str(e)}")
# #         return []

# # def calculate_ha_indicators(df):
# #     """Calculate technical indicators on Heikin-Ashi data"""
# #     try:
# #         if df.empty or len(df) < 20:
# #             return None
        
# #         df = df.copy()
        
# #         df['ATR'] = ta.atr(df['HA_High'], df['HA_Low'], df['HA_Close'], length=14)
# #         df['RSI'] = ta.rsi(df['HA_Close'], length=14)
        
# #         adx_data = ta.adx(df['HA_High'], df['HA_Low'], df['HA_Close'], length=14)
# #         if isinstance(adx_data, pd.DataFrame) and 'ADX_14' in adx_data.columns:
# #             df['ADX'] = adx_data['ADX_14']
# #         else:
# #             df['ADX'] = 25.0
        
# #         df['Cycle_Phase'] = 'Bull'
# #         df['Cycle_Duration'] = 30
# #         df['Cycle_Momentum'] = (df['HA_Close'] - df['HA_Close'].shift(10)) / df['HA_Close'].shift(10)
        
# #         df['ATR'] = df['ATR'].fillna(df['ATR'].mean())
# #         df['RSI'] = df['RSI'].fillna(50.0)
# #         df['ADX'] = df['ADX'].fillna(25.0)
# #         df['Cycle_Momentum'] = df['Cycle_Momentum'].fillna(0.0)
        
# #         return df
        
# #     except Exception as e:
# #         logger.error(f"Error calculating indicators: {str(e)}")
# #         return None

# # def detect_geometric_patterns(df, pivots):
# #     """Detect geometric patterns with simplified logic"""
# #     patterns = {
# #         'rising_wedge': False,
# #         'falling_wedge': False,
# #         'ascending_triangle': False,
# #         'descending_triangle': False,
# #         'channel': False,
# #         'head_shoulders': False,
# #         'pennant': False
# #     }
    
# #     try:
# #         if len(pivots) < 5:
# #             return patterns, {}
        
# #         recent_pivots = pivots[-5:]
# #         prices = [p[1] for p in recent_pivots]
# #         types = [p[2] for p in recent_pivots]
        
# #         if len([p for p in types if p == 'high']) >= 2 and len([p for p in types if p == 'low']) >= 2:
# #             highs = [p[1] for p in recent_pivots if p[2] == 'high']
# #             lows = [p[1] for p in recent_pivots if p[2] == 'low']
            
# #             if len(highs) >= 2 and len(lows) >= 2:
# #                 if highs[-1] > highs[0] and lows[-1] > lows[0]:
# #                     patterns['rising_wedge'] = True
# #                 elif highs[-1] < highs[0] and lows[-1] < lows[0]:
# #                     patterns['falling_wedge'] = True
        
# #         return patterns, {}
        
# #     except Exception as e:
# #         logger.error(f"Error in pattern detection: {str(e)}")
# #         return patterns, {}

# # def detect_elliott_waves(pivots, prices):
# #     """Simplified Elliott Wave detection"""
# #     waves = {
# #         'impulse': {'detected': False, 'wave1': False, 'wave2': False, 'wave3': False, 'wave4': False, 'wave5': False},
# #         'diagonal': {'detected': False, 'leading': False, 'ending': False},
# #         'zigzag': {'detected': False, 'waveA': False, 'waveB': False, 'waveC': False},
# #         'flat': {'detected': False, 'waveA': False, 'waveB': False, 'waveC': False}
# #     }
    
# #     try:
# #         if len(pivots) >= 5:
# #             waves['impulse']['detected'] = True
# #             waves['impulse']['wave1'] = True
# #             waves['impulse']['wave3'] = True
# #     except Exception as e:
# #         logger.error(f"Error in Elliott Wave detection: {str(e)}")
    
# #     return waves

# # def detect_confluence(df, pivots):
# #     """Detect Smart Money Concepts confluence"""
# #     confluence = {
# #         'bullish_confluence': False,
# #         'bearish_confluence': False,
# #         'factors': []
# #     }
    
# #     try:
# #         if df.empty or len(df) < 10:
# #             return confluence
        
# #         last_close = df['HA_Close'].iloc[-1]
# #         prev_close = df['HA_Close'].iloc[-5]
        
# #         if last_close > prev_close:
# #             confluence['factors'].append('Bullish Trend')
# #             confluence['bullish_confluence'] = True
# #         else:
# #             confluence['factors'].append('Bearish Trend')
# #             confluence['bearish_confluence'] = True
        
# #         return confluence
        
# #     except Exception as e:
# #         logger.error(f"Error in confluence detection: {str(e)}")
# #         return confluence

# # def generate_cycle_analysis(df, symbol):
# #     """Generate simplified cycle analysis"""
# #     try:
# #         if df.empty or len(df) < 10:
# #             return {
# #                 'current_phase': 'Unknown',
# #                 'stage': 'Unknown',
# #                 'duration_days': 0,
# #                 'momentum': 0.0,
# #                 'momentum_visual': '----------',
# #                 'bull_continuation_probability': 50,
# #                 'bear_transition_probability': 50,
# #                 'expected_continuation': 'Unknown',
# #                 'risk_level': 'Medium'
# #             }
        
# #         last_close = df['HA_Close'].iloc[-1]
# #         prev_close = df['HA_Close'].iloc[-10] if len(df) >= 10 else df['HA_Close'].iloc[0]
        
# #         current_phase = 'Bull' if last_close > prev_close else 'Bear'
# #         momentum = (last_close - prev_close) / prev_close if prev_close != 0 else 0
        
# #         return {
# #             'current_phase': current_phase,
# #             'stage': f"Mid {current_phase}",
# #             'duration_days': 30,
# #             'momentum': round(momentum, 3),
# #             'momentum_visual': '▲' * 5 + '△' * 5 if momentum > 0 else '▼' * 5 + '▽' * 5,
# #             'bull_continuation_probability': 70 if current_phase == 'Bull' else 30,
# #             'bear_transition_probability': 30 if current_phase == 'Bull' else 70,
# #             'expected_continuation': '30-60 days',
# #             'risk_level': 'Medium'
# #         }
        
# #     except Exception as e:
# #         logger.error(f"Error in cycle analysis for {symbol}: {str(e)}")
# #         return {
# #             'current_phase': 'Unknown',
# #             'stage': 'Unknown',
# #             'duration_days': 0,
# #             'momentum': 0.0,
# #             'momentum_visual': '----------',
# #             'bull_continuation_probability': 50,
# #             'bear_transition_probability': 50,
# #             'expected_continuation': 'Unknown',
# #             'risk_level': 'Medium'
# #         }

# # def get_fundamental_data(symbol):
# #     """Get fundamental data (simulated with realistic values)"""
# #     pe_ratios = {
# #         'AAPL': 28.5, 'MSFT': 32.1, 'TSLA': 45.2, 'GOOGL': 24.8, 'AMZN': 38.9,
# #         'META': 22.7, 'NVDA': 55.3, 'JPM': 12.4, 'JNJ': 16.8, 'V': 34.2,
# #         'INTC': 15.2, 'UBER': 28.9, 'SHOP': 42.1, 'PYPL': 18.7, 'PEP': 25.3,
# #         'KO': 24.8, 'DIS': 35.6, 'NKE': 31.2, 'WMT': 26.4, 'CRM': 48.9
# #     }
    
# #     return {
# #         'PE_Ratio': pe_ratios.get(symbol, 20.0),
# #         'EPS': random.uniform(5.0, 15.0),
# #         'Revenue_Growth': random.uniform(0.05, 0.25),
# #         'Net_Income_Growth': random.uniform(0.03, 0.20)
# #     }

# # def get_market_sentiment(symbol):
# #     """Get market sentiment (simulated)"""
# #     sentiment_scores = {
# #         'AAPL': 0.75, 'MSFT': 0.80, 'TSLA': 0.60, 'GOOGL': 0.70, 'AMZN': 0.65,
# #         'META': 0.55, 'NVDA': 0.85, 'JPM': 0.60, 'JNJ': 0.70, 'V': 0.75,
# #         'INTC': 0.45, 'UBER': 0.58, 'SHOP': 0.62, 'PYPL': 0.52, 'PEP': 0.68,
# #         'KO': 0.72, 'DIS': 0.63, 'NKE': 0.69, 'WMT': 0.66, 'CRM': 0.71
# #     }
# #     return sentiment_scores.get(symbol, 0.5)

# # def generate_smc_signals(chart_patterns, indicators, confluence, waves, fundamentals, sentiment):
# #     """Generate trading signals with simplified logic"""
# #     try:
# #         signal_score = 0.0
        
# #         if chart_patterns.get('rising_wedge', False):
# #             signal_score += 1.0
# #         if chart_patterns.get('falling_wedge', False):
# #             signal_score -= 1.0
        
# #         if waves['impulse']['detected']:
# #             signal_score += 1.5
        
# #         if confluence['bullish_confluence']:
# #             signal_score += 1.0
# #         if confluence['bearish_confluence']:
# #             signal_score -= 1.0
        
# #         if 'RSI' in indicators and indicators['RSI'] < 30:
# #             signal_score += 0.5
# #         elif 'RSI' in indicators and indicators['RSI'] > 70:
# #             signal_score -= 0.5
        
# #         pe_ratio = fundamentals['PE_Ratio']
# #         if pe_ratio < 20:
# #             signal_score += 0.5
# #         elif pe_ratio > 40:
# #             signal_score -= 0.5
        
# #         signal_score += sentiment * 0.5
        
# #         if signal_score >= 2.0:
# #             return 'Strong Buy', round(signal_score, 2)
# #         elif signal_score >= 1.0:
# #             return 'Buy', round(signal_score, 2)
# #         elif signal_score <= -2.0:
# #             return 'Strong Sell', round(signal_score, 2)
# #         elif signal_score <= -1.0:
# #             return 'Sell', round(signal_score, 2)
# #         else:
# #             return 'Neutral', round(signal_score, 2)
        
# #     except Exception as e:
# #         logger.error(f"Error in signal generation: {str(e)}")
# #         return 'Neutral', 0.0

# # def analyze_stock(symbol, data_source="twelve_data"):
# #     """Analyze a single stock and return structured JSON data"""
# #     try:
# #         logger.info(f"Starting analysis for {symbol} using {data_source}")
        
# #         daily_data = fetch_stock_data(symbol, "1day", 100, data_source)
# #         weekly_data = fetch_stock_data(symbol, "1week", 50, data_source)
        
# #         if daily_data.empty and weekly_data.empty:
# #             logger.error(f"No data available for {symbol}")
# #             return None
        
# #         daily_analysis = None
# #         if not daily_data.empty:
# #             daily_analysis = analyze_timeframe(daily_data, symbol, "DAILY")
        
# #         weekly_analysis = None
# #         if not weekly_data.empty:
# #             weekly_analysis = analyze_timeframe(weekly_data, symbol, "WEEKLY")
        
# #         result = {
# #             symbol: {
# #                 'data_source': data_source
# #             }
# #         }
        
# #         if daily_analysis:
# #             result[symbol]["DAILY_TIMEFRAME"] = daily_analysis
        
# #         if weekly_analysis:
# #             result[symbol]["WEEKLY_TIMEFRAME"] = weekly_analysis
        
# #         logger.info(f"Successfully analyzed {symbol}")
# #         return result
        
# #     except Exception as e:
# #         logger.error(f"Error analyzing {symbol}: {str(e)}")
# #         return None

# # def analyze_timeframe(data, symbol, timeframe):
# #     """Analyze a specific timeframe and return structured data"""
# #     try:
# #         ha_data = heikin_ashi(data)
# #         if ha_data.empty:
# #             logger.error(f"Failed to convert to HA for {symbol} {timeframe}")
# #             return None
        
# #         indicators_data = calculate_ha_indicators(ha_data)
# #         if indicators_data is None:
# #             logger.error(f"Failed to calculate indicators for {symbol} {timeframe}")
# #             return None
        
# #         pivots = detect_zigzag_pivots(ha_data)
# #         patterns, _ = detect_geometric_patterns(ha_data, pivots)
# #         waves = detect_elliott_waves(pivots, ha_data['HA_Close'])
# #         confluence = detect_confluence(ha_data, pivots)
        
# #         cycle_analysis = generate_cycle_analysis(indicators_data, symbol)
        
# #         fundamentals = get_fundamental_data(symbol)
# #         sentiment = get_market_sentiment(symbol)
        
# #         last_indicators = indicators_data.iloc[-1].to_dict()
# #         signal, score = generate_smc_signals(patterns, last_indicators, confluence, waves, fundamentals, sentiment)
        
# #         current_price = round(ha_data['HA_Close'].iloc[-1], 2)
        
# #         if 'Buy' in signal:
# #             entry = round(current_price * 0.99, 2)
# #             targets = [round(current_price * 1.05, 2), round(current_price * 1.10, 2)]
# #             stop_loss = round(current_price * 0.95, 2)
# #         else:
# #             entry = round(current_price * 1.01, 2)
# #             targets = [round(current_price * 0.95, 2), round(current_price * 0.90, 2)]
# #             stop_loss = round(current_price * 1.05, 2)
        
# #         change_1d = 0.0
# #         change_1w = 0.0
        
# #         if len(ha_data) >= 2:
# #             change_1d = round((ha_data['HA_Close'].iloc[-1] / ha_data['HA_Close'].iloc[-2] - 1) * 100, 2)
        
# #         if len(ha_data) >= 5:
# #             change_1w = round((ha_data['HA_Close'].iloc[-1] / ha_data['HA_Close'].iloc[-5] - 1) * 100, 2)
        
# #         rsi_verdict = "Overbought" if last_indicators.get('RSI', 50) > 70 else "Oversold" if last_indicators.get('RSI', 50) < 30 else "Neutral"
# #         adx_verdict = "Strong Trend" if last_indicators.get('ADX', 25) > 25 else "Weak Trend"
# #         momentum_verdict = "Bullish" if last_indicators.get('Cycle_Momentum', 0) > 0.02 else "Bearish" if last_indicators.get('Cycle_Momentum', 0) < -0.02 else "Neutral"
# #         pattern_verdict = "Bullish Patterns" if any(patterns.values()) and signal in ['Buy', 'Strong Buy'] else "Bearish Patterns" if any(patterns.values()) and signal in ['Sell', 'Strong Sell'] else "No Clear Patterns"
# #         pe_ratio = fundamentals['PE_Ratio']
# #         fundamental_verdict = "Undervalued" if pe_ratio < 20 else "Overvalued" if pe_ratio > 30 else "Fair Value"
# #         sentiment_verdict = "Positive" if sentiment > 0.6 else "Negative" if sentiment < 0.4 else "Neutral"
        
# #         timeframe_analysis = {
# #             'PRICE': current_price,
# #             'ACCURACY': min(95, max(60, abs(score) * 20 + 60)),
# #             'CONFIDENCE_SCORE': round(score, 2),
# #             'VERDICT': signal,
# #             'DETAILS': {
# #                 'individual_verdicts': {
# #                     'rsi_verdict': rsi_verdict,
# #                     'adx_verdict': adx_verdict,
# #                     'momentum_verdict': momentum_verdict,
# #                     'pattern_verdict': pattern_verdict,
# #                     'fundamental_verdict': fundamental_verdict,
# #                     'sentiment_verdict': sentiment_verdict,
# #                     'cycle_verdict': cycle_analysis['current_phase']
# #                 },
# #                 'price_data': {
# #                     'current_price': current_price,
# #                     'entry_price': entry,
# #                     'target_prices': targets,
# #                     'stop_loss': stop_loss,
# #                     'change_1d': change_1d,
# #                     'change_1w': change_1w
# #                 },
# #                 'technical_indicators': {
# #                     'rsi': round(last_indicators.get('RSI', 50.0), 1),
# #                     'adx': round(last_indicators.get('ADX', 25.0), 1),
# #                     'atr': round(last_indicators.get('ATR', 1.0), 2),
# #                     'cycle_phase': last_indicators.get('Cycle_Phase', 'Unknown'),
# #                     'cycle_momentum': round(last_indicators.get('Cycle_Momentum', 0.0), 3)
# #                 },
# #                 'patterns': {
# #                     'geometric': [k for k, v in patterns.items() if v] or ['None'],
# #                     'elliott_wave': [k for k, v in waves.items() if v.get('detected', False)] or ['None'],
# #                     'confluence_factors': confluence['factors'] or ['None']
# #                 },
# #                 'fundamentals': {
# #                     'pe_ratio': fundamentals['PE_Ratio'],
# #                     'eps': round(fundamentals['EPS'], 2),
# #                     'revenue_growth': round(fundamentals['Revenue_Growth'] * 100, 2),
# #                     'net_income_growth': round(fundamentals['Net_Income_Growth'] * 100, 2)
# #                 },
# #                 'sentiment_analysis': {
# #                     'score': round(sentiment, 2),
# #                     'interpretation': sentiment_verdict,
# #                     'market_mood': "Optimistic" if sentiment > 0.7 else "Pessimistic" if sentiment < 0.3 else "Cautious"
# #                 },
# #                 'cycle_analysis': cycle_analysis,
# #                 'trading_parameters': {
# #                     'position_size': '5% of portfolio' if 'Strong' in signal else '3% of portfolio',
# #                     'timeframe': '2-4 weeks' if 'Buy' in signal else '1-2 weeks',
# #                     'risk_level': 'Medium' if 'Buy' in signal else 'High' if 'Sell' in signal else 'Low'
# #                 }
# #             }
# #         }
        
# #         return timeframe_analysis
        
# #     except Exception as e:
# #         logger.error(f"Error analyzing {timeframe} timeframe for {symbol}: {str(e)}")
# #         return None

# # def analyze_all_stocks():
# #     """Analyze stocks and return JSON response"""
# #     try:
# #         stock_config = get_filtered_stocks(20)
# #         twelve_data_stocks = stock_config['twelve_data']
# #         yfinance_stocks = stock_config['yfinance']
        
# #         results = {}
        
# #         logger.info(f"Starting analysis of 20 stocks")
# #         logger.info(f"Twelve Data stocks (15): {twelve_data_stocks}")
# #         logger.info(f"yfinance stocks (5): {yfinance_stocks}")
        
# #         # Process Twelve Data stocks in batches
# #         if twelve_data_stocks:
# #             batch_size = TWELVE_DATA_BATCH_SIZE
# #             num_batches = math.ceil(len(twelve_data_stocks) / batch_size)
            
# #             for batch_idx in range(num_batches):
# #                 batch_start = batch_idx * batch_size
# #                 batch_end = min((batch_idx + 1) * batch_size, len(twelve_data_stocks))
# #                 batch_symbols = twelve_data_stocks[batch_start:batch_end]
                
# #                 logger.info(f"Processing Twelve Data batch {batch_idx+1}/{num_batches}: {batch_symbols}")
                
# #                 for symbol in batch_symbols:
# #                     try:
# #                         result = analyze_stock(symbol, "twelve_data")
# #                         if result:
# #                             results.update(result)
# #                             logger.info(f"Successfully processed {symbol} (Twelve Data)")
# #                         else:
# #                             logger.warning(f"Failed to process {symbol} (Twelve Data)")
# #                     except Exception as e:
# #                         logger.error(f"Error processing {symbol} (Twelve Data): {str(e)}")
                
# #                 if batch_idx < num_batches - 1:
# #                     logger.info(f"Sleeping {TWELVE_DATA_BATCH_SLEEP}s to respect Twelve Data rate limit...")
# #                     time.sleep(TWELVE_DATA_BATCH_SLEEP)
        
# #         # Process yfinance stocks
# #         if yfinance_stocks:
# #             logger.info(f"Processing yfinance stocks: {yfinance_stocks}")
# #             for symbol in yfinance_stocks:
# #                 try:
# #                     result = analyze_stock(symbol, "yfinance")
# #                     if result:
# #                         results.update(result)
# #                         logger.info(f"Successfully processed {symbol} (yfinance)")
# #                     else:
# #                         logger.warning(f"Failed to process {symbol} (yfinance)")
# #                 except Exception as e:
# #                     logger.error(f"Error processing {symbol} (yfinance): {str(e)}")
        
# #         response = {
# #             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
# #             'stocks_analyzed': len(results),
# #             'status': 'success' if results else 'no_data',
# #             'data_sources': {
# #                 'twelve_data_count': len([k for k, v in results.items() if v.get('data_source') == 'twelve_data']),
# #                 'yfinance_count': len([k for k, v in results.items() if v.get('data_source') == 'yfinance'])
# #             },
# #             **results
# #         }
        
# #         logger.info(f"Analysis complete. Processed {len(results)} stocks successfully.")
# #         return response
        
# #     except Exception as e:
# #         logger.error(f"Error in analyze_all_stocks: {str(e)}")
# #         return {
# #             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
# #             'stocks_analyzed': 0,
# #             'status': 'error',
# #             'error': str(e)
# #         }

# # # ================= FLASK ROUTES =================

# # @app.route('/', methods=['GET'])
# # def home():
# #     """Home endpoint"""
# #     try:
# #         return jsonify({
# #             'message': 'Stock Analysis API is running',
# #             'endpoints': {
# #                 '/analyze': 'GET - Analyze stocks and return trading signals',
# #                 '/health': 'GET - Health check',
# #                 '/': 'GET - This help message'
# #             },
# #             'data_sources': {
# #                 'twelve_data': 'First 15 stocks',
# #                 'yfinance': 'Last 5 stocks'
# #             },
# #             'status': 'online',
# #             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
# #         })
# #     except Exception as e:
# #         logger.error(f"Error in home endpoint: {str(e)}")
# #         return jsonify({'error': str(e)}), 500

# # @app.route('/health', methods=['GET'])
# # def health():
# #     """Health check endpoint"""
# #     try:
# #         return jsonify({
# #             'status': 'healthy',
# #             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
# #             'service': 'Stock Analysis API'
# #         })
# #     except Exception as e:
# #         logger.error(f"Error in health endpoint: {str(e)}")
# #         return jsonify({'error': str(e)}), 500

# # @app.route('/analyze', methods=['GET'])
# # def analyze():
# #     """API endpoint to analyze stocks and return JSON response"""
# #     try:
# #         logger.info("Starting stock analysis...")
# #         json_response = analyze_all_stocks()
# #         logger.info(f"Analysis completed. Status: {json_response.get('status')}")
# #         return jsonify(json_response)
# #     except Exception as e:
# #         logger.error(f"Error in /analyze endpoint: {str(e)}")
# #         return jsonify({
# #             'error': f"Failed to analyze stocks: {str(e)}",
# #             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
# #             'stocks_analyzed': 0,
# #             'status': 'error'
# #         }), 500

# # @app.errorhandler(404)
# # def not_found(error):
# #     return jsonify({
# #         'error': 'Endpoint not found',
# #         'message': 'The requested URL was not found on the server',
# #         'available_endpoints': ['/analyze', '/health', '/'],
# #         'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
# #     }), 404

# # @app.errorhandler(500)
# # def internal_error(error):
# #     return jsonify({
# #         'error': 'Internal server error',
# #         'message': 'An internal error occurred',
# #         'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
# #     }), 500

# # if __name__ == "__main__":
# #     port = int(os.environ.get("PORT", 5000))
# #     debug_mode = os.environ.get("FLASK_ENV") == "development"
    
# #     logger.info(f"Starting Flask server on port {port}")
# #     logger.info(f"Debug mode: {debug_mode}")
    
# #     app.run(host='0.0.0.0', port=port, debug=debug_mode, threaded=True)




# import pandas as pd
# import numpy as np
# import pandas_ta as ta
# from scipy.signal import argrelextrema
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import random
# import warnings
# from datetime import datetime, timedelta
# import json
# from sklearn.linear_model import LinearRegression
# from flask import Flask, jsonify, request
# from flask_cors import CORS
# import logging
# import time
# import requests
# import math
# import yfinance as yf
# import os
# from threading import Lock
# import queue

# warnings.filterwarnings('ignore')

# # ================= GLOBAL CONFIGURATION =================
# TWELVE_DATA_API_KEY = "73adc6cc7e43476e851dcf54c705aeeb"

# RISK_FREE_RATE = 0.02
# MAX_WORKERS = 1
# MIN_MARKET_CAP = 500e6
# MIN_PRICE = 5.0
# PATTERN_SENSITIVITY = 0.05
# FIBONACCI_TOLERANCE = 0.05
# CHANNEL_CONFIRMATION_BARS = 3
# PATTERN_LOOKBACK = 20
# ZIGZAG_LENGTH = 5
# ZIGZAG_DEPTH = 10
# ZIGZAG_NUM_PIVOTS = 10
# CYCLE_MIN_DURATION = 30
# PATTERN_ANGLE_THRESHOLD = 1.5
# PATTERN_EXPANSION_RATIO = 1.2
# PATTERN_CONTRACTION_RATIO = 0.8
# MIN_TRENDLINE_R2 = 0.75
# CONFIRMATION_VOL_RATIO = 1.2
# MIN_TRENDLINE_ANGLE = 0.5
# MAX_TRENDLINE_ANGLE = 85
# HARMONIC_ERROR_TOLERANCE = 0.05
# PRZ_LEFT_RANGE = 20
# PRZ_RIGHT_RANGE = 20
# FIBONACCI_LINE_LENGTH = 30
# FUNDAMENTAL_WEIGHT = 0.3
# SENTIMENT_WEIGHT = 0.2
# TECHNICAL_WEIGHT = 0.5

# # Enhanced Rate limiting configuration
# TWELVE_DATA_RATE_LIMIT_PER_MIN = 6
# TWELVE_DATA_BATCH_SIZE = 2  # Reduced for safety
# TWELVE_DATA_BATCH_SLEEP = 70  # Increased sleep time
# TWELVE_DATA_RETRY_ATTEMPTS = 3
# TWELVE_DATA_RETRY_DELAY = 30

# YFINANCE_BATCH_SIZE = 5
# YFINANCE_DELAY = 1.0  # Increased delay
# YFINANCE_BATCH_SLEEP = 10
# YFINANCE_RETRY_ATTEMPTS = 2
# YFINANCE_RETRY_DELAY = 5

# # Global rate limiting
# rate_limit_lock = Lock()
# last_twelve_data_request = 0
# last_yfinance_request = 0
# request_count_twelve_data = 0
# request_count_yfinance = 0

# # Setup logging for production
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # ================= FLASK APP SETUP =================
# app = Flask(__name__)

# # Simple CORS configuration - avoid conflicts
# CORS(app, resources={
#     r"/*": {
#         "origins": ["*", "http://localhost:5177", "https://my-stocks-s2at.onrender.com"],
#         "methods": ["GET", "POST", "OPTIONS"],
#         "allow_headers": ["Content-Type", "Authorization"]
#     }
# })

# # ================= ENHANCED STOCK CONFIGURATION =================
# def get_filtered_stocks(num_stocks=175):
#     """Get list of stocks to analyze - US stocks + Nigerian stocks with proper distribution"""
    
#     # US Stocks (20)
#     us_stocks = [
#         "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN",
#         "NVDA", "META", "NFLX", "BABA", "AMD",
#         "INTC", "UBER", "SHOP", "PYPL", "PEP",
#         "KO", "DIS", "NKE", "WMT", "CRM"
#     ]
    
#     # Nigerian Stocks (155)
#     nigerian_stocks = [
#         # Banks
#         "ACCESS.NG", "FBNH.NG", "FCMB.NG", "FIDELITYBK.NG", "GTCO.NG",
#         "JAIZBANK.NG", "STERLNBANK.NG", "UBA.NG", "UNIONBANK.NG", "WEMABANK.NG",
#         "ZENITHBANK.NG",
        
#         # Oil & Gas
#         "ARDOVA.NG", "CONOIL.NG", "ETERNA.NG", "MOBIL.NG", "OANDO.NG",
#         "SEPLAT.NG", "TOTAL.NG",
        
#         # Consumer Goods
#         "CADBURY.NG", "DANGSUGAR.NG", "FLOURMILL.NG", "GUINNESS.NG", "HONYFLOUR.NG",
#         "NASCON.NG", "NESTLE.NG", "PZ.NG", "UNILEVER.NG",
        
#         # Industrial Goods
#         "BUACEMENT.NG", "CUTIX.NG", "DANGCEM.NG", "WAPCO.NG",
        
#         # Healthcare
#         "FIDSON.NG", "GLAXOSMITH.NG", "MAYBAKER.NG", "NEIMETH.NG",
        
#         # Agriculture
#         "OKOMUOIL.NG", "PRESCO.NG", "PRESTIGE.NG", "FTNCOCOA.NG",
        
#         # Conglomerates
#         "SCOA.NG", "TRANSCORP.NG", "UACN.NG",
        
#         # ICT
#         "CHAMS.NG", "CORNERST.NG", "ETRANZACT.NG", "NCR.NG",
        
#         # Insurance
#         "AIICO.NG", "CUSTODIAN.NG", "LASACO.NG", "LINKASSURE.NG", "MANSARD.NG",
#         "NEM.NG", "REGALINS.NG", "ROYALEX.NG", "STACO.NG", "VERITASKAP.NG",
        
#         # Others (Real Estate, Services, etc.)
#         "ABCTRANS.NG", "ACADEMY.NG", "CAVERTON.NG", "CHELLARAM.NG",
#         "ELLAHLAKES.NG", "GOLDBREW.NG", "JBERGER.NG", "LEARNAFRCA.NG", "MCNICHOLS.NG",
#         "MRS.NG", "RTBRISCOE.NG", "TANTALIZERS.NG", "TRIPPLEG.NG", "UPDC.NG",
#         "VITAFOAM.NG",
        
#         # Additional Nigerian Stocks
#         "AFRIPRUD.NG", "ASOSAVINGS.NG", "BETAGLAS.NG", "BUAFOODS.NG", "CHAMPION.NG",
#         "COURTVILLE.NG", "DEAPCAP.NG", "EUNISELL.NG", "GOLDINSURE.NG", "GUINEAINS.NG",
#         "IKEJAHOTEL.NG", "JOHNHOLT.NG", "JULI.NG", "LAWUNION.NG", "LIVESTOCK.NG",
#         "MBENEFIT.NG", "MEDVIEWAIR.NG", "MEYER.NG", "MORISON.NG", "NAHCO.NG",
#         "NASCO.NG", "NGXGROUP.NG", "NNFM.NG", "NOTORE.NG", "NPFMCRFBK.NG",
#         "OASISINS.NG", "OMATEK.NG", "PHARMDEKO.NG", "PORTPAINT.NG", "PREMPAINTS.NG",
#         "REDSTAREX.NG", "RESORTSAL.NG", "SKYAVN.NG", "SMURFIT.NG", "SOVERINS.NG",
#         "THOMASWY.NG", "TIGERBRAND.NG", "TRANSEXPR.NG", "UCAP.NG", "UNHOMES.NG",
#         "UNIONDICON.NG", "UNIVINSURE.NG", "VFDGROUP.NG", "WAPIC.NG"
#     ]
    
#     # Distribution strategy:
#     # US stocks: Use mix of Twelve Data and yfinance
#     # Nigerian stocks: Primarily yfinance (better support for international markets)
    
#     twelve_data_stocks = us_stocks[:10]  # First 10 US stocks
#     yfinance_stocks = us_stocks[10:] + nigerian_stocks  # Rest of US + all Nigerian
    
#     all_stocks = us_stocks + nigerian_stocks
    
#     return {
#         'twelve_data': twelve_data_stocks,
#         'yfinance': yfinance_stocks,
#         'all': all_stocks[:num_stocks],
#         'us_stocks': us_stocks,
#         'nigerian_stocks': nigerian_stocks,
#         'total_count': len(all_stocks)
#     }

# # ================= ENHANCED RATE LIMITING FUNCTIONS =================
# def wait_for_rate_limit_twelve_data():
#     """Implement rate limiting for Twelve Data API"""
#     global last_twelve_data_request, request_count_twelve_data
    
#     with rate_limit_lock:
#         current_time = time.time()
        
#         # Reset counter every minute
#         if current_time - last_twelve_data_request > 60:
#             request_count_twelve_data = 0
#             last_twelve_data_request = current_time
        
#         # If we've hit the rate limit, wait
#         if request_count_twelve_data >= TWELVE_DATA_RATE_LIMIT_PER_MIN:
#             sleep_time = 60 - (current_time - last_twelve_data_request)
#             if sleep_time > 0:
#                 logger.info(f"Rate limit reached for Twelve Data. Sleeping for {sleep_time:.1f} seconds...")
#                 time.sleep(sleep_time)
#                 request_count_twelve_data = 0
#                 last_twelve_data_request = time.time()
        
#         request_count_twelve_data += 1

# def wait_for_rate_limit_yfinance():
#     """Implement rate limiting for yfinance"""
#     global last_yfinance_request, request_count_yfinance
    
#     with rate_limit_lock:
#         current_time = time.time()
        
#         # Ensure minimum delay between requests
#         time_since_last = current_time - last_yfinance_request
#         if time_since_last < YFINANCE_DELAY:
#             sleep_time = YFINANCE_DELAY - time_since_last
#             time.sleep(sleep_time)
        
#         last_yfinance_request = time.time()
#         request_count_yfinance += 1

# # ================= ENHANCED DATA FETCHING WITH RETRY LOGIC =================
# def fetch_stock_data_twelve_with_retry(symbol, interval="1day", outputsize=100, max_retries=TWELVE_DATA_RETRY_ATTEMPTS):
#     """Fetch stock data from Twelve Data API with retry logic"""
#     for attempt in range(max_retries):
#         try:
#             wait_for_rate_limit_twelve_data()
            
#             url = f"https://api.twelvedata.com/time_series"
#             params = {
#                 'symbol': symbol,
#                 'interval': interval,
#                 'outputsize': outputsize,
#                 'apikey': TWELVE_DATA_API_KEY,
#                 'format': 'JSON'
#             }
            
#             logger.info(f"Fetching {symbol} from Twelve Data (attempt {attempt + 1}/{max_retries})")
            
#             response = requests.get(url, params=params, timeout=30)
#             response.raise_for_status()
            
#             data = response.json()
            
#             # Check for API errors
#             if 'code' in data and data['code'] != 200:
#                 logger.warning(f"API error for {symbol}: {data.get('message', 'Unknown error')}")
#                 if attempt < max_retries - 1:
#                     time.sleep(TWELVE_DATA_RETRY_DELAY)
#                     continue
#                 return pd.DataFrame()
            
#             if 'values' not in data:
#                 logger.error(f"No values in response for {symbol}: {data}")
#                 if attempt < max_retries - 1:
#                     time.sleep(TWELVE_DATA_RETRY_DELAY)
#                     continue
#                 return pd.DataFrame()
            
#             df = pd.DataFrame(data['values'])
            
#             if df.empty:
#                 logger.error(f"Empty DataFrame for {symbol}")
#                 if attempt < max_retries - 1:
#                     time.sleep(TWELVE_DATA_RETRY_DELAY)
#                     continue
#                 return pd.DataFrame()
            
#             numeric_columns = ['open', 'high', 'low', 'close', 'volume']
#             for col in numeric_columns:
#                 if col in df.columns:
#                     df[col] = pd.to_numeric(df[col], errors='coerce')
            
#             df['datetime'] = pd.to_datetime(df['datetime'])
#             df.set_index('datetime', inplace=True)
#             df.sort_index(inplace=True)
#             df.dropna(inplace=True)
            
#             logger.info(f"Successfully fetched {len(df)} rows for {symbol} from Twelve Data")
#             return df
            
#         except Exception as e:
#             logger.error(f"Error fetching data from Twelve Data for {symbol} (attempt {attempt + 1}): {str(e)}")
#             if attempt < max_retries - 1:
#                 time.sleep(TWELVE_DATA_RETRY_DELAY)
#             else:
#                 return pd.DataFrame()
    
#     return pd.DataFrame()

# def fetch_stock_data_yfinance_with_retry(symbol, period="100d", interval="1d", max_retries=YFINANCE_RETRY_ATTEMPTS):
#     """Fetch stock data from yfinance with retry logic"""
#     for attempt in range(max_retries):
#         try:
#             wait_for_rate_limit_yfinance()
            
#             logger.info(f"Fetching {symbol} from yfinance (attempt {attempt + 1}/{max_retries})")
            
#             ticker = yf.Ticker(symbol)
#             df = ticker.history(period=period, interval=interval)
            
#             if df.empty:
#                 logger.warning(f"Empty DataFrame for {symbol} from yfinance")
#                 if attempt < max_retries - 1:
#                     time.sleep(YFINANCE_RETRY_DELAY)
#                     continue
#                 return pd.DataFrame()
            
#             df.columns = df.columns.str.lower()
#             df.reset_index(inplace=True)
#             df.rename(columns={'date': 'datetime'}, inplace=True)
#             df.set_index('datetime', inplace=True)
#             df.dropna(inplace=True)
            
#             logger.info(f"Successfully fetched {len(df)} rows for {symbol} from yfinance")
#             return df
            
#         except Exception as e:
#             logger.error(f"Error fetching data from yfinance for {symbol} (attempt {attempt + 1}): {str(e)}")
#             if attempt < max_retries - 1:
#                 time.sleep(YFINANCE_RETRY_DELAY)
#             else:
#                 return pd.DataFrame()
    
#     return pd.DataFrame()

# def fetch_stock_data(symbol, interval="1day", outputsize=100, source="twelve_data"):
#     """Unified function to fetch stock data from either source with retry logic"""
#     if source == "twelve_data":
#         return fetch_stock_data_twelve_with_retry(symbol, interval, outputsize)
#     elif source == "yfinance":
#         yf_interval = "1d" if interval == "1day" else "1wk" if interval == "1week" else "1d"
#         yf_period = f"{outputsize}d" if interval == "1day" else f"{outputsize//7}wk"
#         return fetch_stock_data_yfinance_with_retry(symbol, period=yf_period, interval=yf_interval)
#     else:
#         logger.error(f"Unknown data source: {source}")
#         return pd.DataFrame()

# # ================= EXISTING ANALYSIS FUNCTIONS (UNCHANGED) =================
# def heikin_ashi(df):
#     """Convert dataframe to Heikin-Ashi candles with proper error handling"""
#     if df.empty:
#         return pd.DataFrame()
    
#     try:
#         df = df.copy()
        
#         required_cols = ['open', 'high', 'low', 'close']
#         if not all(col in df.columns for col in required_cols):
#             logger.error(f"Missing required columns. Available: {df.columns.tolist()}")
#             return pd.DataFrame()
        
#         df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
#         ha_open = [(df['open'].iloc[0] + df['close'].iloc[0]) / 2]
#         for i in range(1, len(df)):
#             ha_open.append((ha_open[i-1] + df['HA_Close'].iloc[i-1]) / 2)
        
#         df['HA_Open'] = ha_open
#         df['HA_High'] = df[['high', 'HA_Open', 'HA_Close']].max(axis=1)
#         df['HA_Low'] = df[['low', 'HA_Open', 'HA_Close']].min(axis=1)
        
#         df.dropna(subset=['HA_Close', 'HA_Open', 'HA_High', 'HA_Low'], inplace=True)
        
#         return df
        
#     except Exception as e:
#         logger.error(f"Error in heikin_ashi calculation: {str(e)}")
#         return pd.DataFrame()

# def detect_zigzag_pivots(data):
#     """Detect significant pivot points using zigzag algorithm"""
#     try:
#         if len(data) < 10 or 'HA_Close' not in data.columns:
#             return []
        
#         prices = data['HA_Close'].values
        
#         highs = argrelextrema(prices, np.greater, order=ZIGZAG_LENGTH)[0]
#         lows = argrelextrema(prices, np.less, order=ZIGZAG_LENGTH)[0]
        
#         pivot_indices = np.concatenate([highs, lows])
#         pivot_indices.sort()
        
#         filtered_pivots = []
#         for i in pivot_indices:
#             if len(filtered_pivots) < 2:
#                 filtered_pivots.append(i)
#             else:
#                 last_price = prices[filtered_pivots[-1]]
#                 current_price = prices[i]
#                 change = abs(current_price - last_price) / last_price
#                 if change > PATTERN_SENSITIVITY:
#                     filtered_pivots.append(i)
        
#         pivot_data = []
#         for i in filtered_pivots:
#             start_idx = max(0, i - ZIGZAG_DEPTH)
#             end_idx = min(len(prices), i + ZIGZAG_DEPTH)
#             local_max = np.max(prices[start_idx:end_idx])
#             local_min = np.min(prices[start_idx:end_idx])
            
#             if prices[i] == local_max:
#                 pivot_type = 'high'
#             else:
#                 pivot_type = 'low'
            
#             pivot_data.append((i, prices[i], pivot_type))
        
#         return pivot_data[-ZIGZAG_NUM_PIVOTS:]
        
#     except Exception as e:
#         logger.error(f"Error in detect_zigzag_pivots: {str(e)}")
#         return []

# def calculate_ha_indicators(df):
#     """Calculate technical indicators on Heikin-Ashi data"""
#     try:
#         if df.empty or len(df) < 20:
#             return None
        
#         df = df.copy()
        
#         df['ATR'] = ta.atr(df['HA_High'], df['HA_Low'], df['HA_Close'], length=14)
#         df['RSI'] = ta.rsi(df['HA_Close'], length=14)
        
#         adx_data = ta.adx(df['HA_High'], df['HA_Low'], df['HA_Close'], length=14)
#         if isinstance(adx_data, pd.DataFrame) and 'ADX_14' in adx_data.columns:
#             df['ADX'] = adx_data['ADX_14']
#         else:
#             df['ADX'] = 25.0
        
#         df['Cycle_Phase'] = 'Bull'
#         df['Cycle_Duration'] = 30
#         df['Cycle_Momentum'] = (df['HA_Close'] - df['HA_Close'].shift(10)) / df['HA_Close'].shift(10)
        
#         df['ATR'] = df['ATR'].fillna(df['ATR'].mean())
#         df['RSI'] = df['RSI'].fillna(50.0)
#         df['ADX'] = df['ADX'].fillna(25.0)
#         df['Cycle_Momentum'] = df['Cycle_Momentum'].fillna(0.0)
        
#         return df
        
#     except Exception as e:
#         logger.error(f"Error calculating indicators: {str(e)}")
#         return None

# def detect_geometric_patterns(df, pivots):
#     """Detect geometric patterns with simplified logic"""
#     patterns = {
#         'rising_wedge': False,
#         'falling_wedge': False,
#         'ascending_triangle': False,
#         'descending_triangle': False,
#         'channel': False,
#         'head_shoulders': False,
#         'pennant': False
#     }
    
#     try:
#         if len(pivots) < 5:
#             return patterns, {}
        
#         recent_pivots = pivots[-5:]
#         prices = [p[1] for p in recent_pivots]
#         types = [p[2] for p in recent_pivots]
        
#         if len([p for p in types if p == 'high']) >= 2 and len([p for p in types if p == 'low']) >= 2:
#             highs = [p[1] for p in recent_pivots if p[2] == 'high']
#             lows = [p[1] for p in recent_pivots if p[2] == 'low']
            
#             if len(highs) >= 2 and len(lows) >= 2:
#                 if highs[-1] > highs[0] and lows[-1] > lows[0]:
#                     patterns['rising_wedge'] = True
#                 elif highs[-1] < highs[0] and lows[-1] < lows[0]:
#                     patterns['falling_wedge'] = True
        
#         return patterns, {}
        
#     except Exception as e:
#         logger.error(f"Error in pattern detection: {str(e)}")
#         return patterns, {}

# def detect_elliott_waves(pivots, prices):
#     """Simplified Elliott Wave detection"""
#     waves = {
#         'impulse': {'detected': False, 'wave1': False, 'wave2': False, 'wave3': False, 'wave4': False, 'wave5': False},
#         'diagonal': {'detected': False, 'leading': False, 'ending': False},
#         'zigzag': {'detected': False, 'waveA': False, 'waveB': False, 'waveC': False},
#         'flat': {'detected': False, 'waveA': False, 'waveB': False, 'waveC': False}
#     }
    
#     try:
#         if len(pivots) >= 5:
#             waves['impulse']['detected'] = True
#             waves['impulse']['wave1'] = True
#             waves['impulse']['wave3'] = True
#     except Exception as e:
#         logger.error(f"Error in Elliott Wave detection: {str(e)}")
    
#     return waves

# def detect_confluence(df, pivots):
#     """Detect Smart Money Concepts confluence"""
#     confluence = {
#         'bullish_confluence': False,
#         'bearish_confluence': False,
#         'factors': []
#     }
    
#     try:
#         if df.empty or len(df) < 10:
#             return confluence
        
#         last_close = df['HA_Close'].iloc[-1]
#         prev_close = df['HA_Close'].iloc[-5]
        
#         if last_close > prev_close:
#             confluence['factors'].append('Bullish Trend')
#             confluence['bullish_confluence'] = True
#         else:
#             confluence['factors'].append('Bearish Trend')
#             confluence['bearish_confluence'] = True
        
#         return confluence
        
#     except Exception as e:
#         logger.error(f"Error in confluence detection: {str(e)}")
#         return confluence

# def generate_cycle_analysis(df, symbol):
#     """Generate simplified cycle analysis"""
#     try:
#         if df.empty or len(df) < 10:
#             return {
#                 'current_phase': 'Unknown',
#                 'stage': 'Unknown',
#                 'duration_days': 0,
#                 'momentum': 0.0,
#                 'momentum_visual': '----------',
#                 'bull_continuation_probability': 50,
#                 'bear_transition_probability': 50,
#                 'expected_continuation': 'Unknown',
#                 'risk_level': 'Medium'
#             }
        
#         last_close = df['HA_Close'].iloc[-1]
#         prev_close = df['HA_Close'].iloc[-10] if len(df) >= 10 else df['HA_Close'].iloc[0]
        
#         current_phase = 'Bull' if last_close > prev_close else 'Bear'
#         momentum = (last_close - prev_close) / prev_close if prev_close != 0 else 0
        
#         return {
#             'current_phase': current_phase,
#             'stage': f"Mid {current_phase}",
#             'duration_days': 30,
#             'momentum': round(momentum, 3),
#             'momentum_visual': '▲' * 5 + '△' * 5 if momentum > 0 else '▼' * 5 + '▽' * 5,
#             'bull_continuation_probability': 70 if current_phase == 'Bull' else 30,
#             'bear_transition_probability': 30 if current_phase == 'Bull' else 70,
#             'expected_continuation': '30-60 days',
#             'risk_level': 'Medium'
#         }
        
#     except Exception as e:
#         logger.error(f"Error in cycle analysis for {symbol}: {str(e)}")
#         return {
#             'current_phase': 'Unknown',
#             'stage': 'Unknown',
#             'duration_days': 0,
#             'momentum': 0.0,
#             'momentum_visual': '----------',
#             'bull_continuation_probability': 50,
#             'bear_transition_probability': 50,
#             'expected_continuation': 'Unknown',
#             'risk_level': 'Medium'
#         }

# def get_fundamental_data(symbol):
#     """Get fundamental data (simulated with realistic values)"""
#     # Enhanced with Nigerian stock PE ratios
#     pe_ratios = {
#         # US Stocks
#         'AAPL': 28.5, 'MSFT': 32.1, 'TSLA': 45.2, 'GOOGL': 24.8, 'AMZN': 38.9,
#         'META': 22.7, 'NVDA': 55.3, 'JPM': 12.4, 'JNJ': 16.8, 'V': 34.2,
#         'INTC': 15.2, 'UBER': 28.9, 'SHOP': 42.1, 'PYPL': 18.7, 'PEP': 25.3,
#         'KO': 24.8, 'DIS': 35.6, 'NKE': 31.2, 'WMT': 26.4, 'CRM': 48.9,
        
#         # Nigerian Banks
#         'ACCESS.NG': 8.5, 'FBNH.NG': 6.2, 'FCMB.NG': 7.8, 'FIDELITYBK.NG': 9.1, 'GTCO.NG': 12.3,
#         'JAIZBANK.NG': 15.2, 'STERLNBANK.NG': 8.9, 'UBA.NG': 7.4, 'UNIONBANK.NG': 6.8, 'WEMABANK.NG': 9.5,
#         'ZENITHBANK.NG': 11.2,
        
#         # Nigerian Consumer Goods
#         'DANGSUGAR.NG': 18.5, 'FLOURMILL.NG': 14.2, 'GUINNESS.NG': 22.1, 'NESTLE.NG': 35.8, 'UNILEVER.NG': 28.4,
        
#         # Nigerian Industrial
#         'BUACEMENT.NG': 16.8, 'DANGCEM.NG': 19.2, 'WAPCO.NG': 15.5,
        
#         # Default for others
#     }
    
#     # Determine if it's a Nigerian stock
#     is_nigerian = symbol.endswith('.NG')
#     base_pe = pe_ratios.get(symbol, 12.0 if is_nigerian else 20.0)
    
#     return {
#         'PE_Ratio': base_pe,
#         'EPS': random.uniform(2.0 if is_nigerian else 5.0, 8.0 if is_nigerian else 15.0),
#         'Revenue_Growth': random.uniform(0.03 if is_nigerian else 0.05, 0.15 if is_nigerian else 0.25),
#         'Net_Income_Growth': random.uniform(0.02 if is_nigerian else 0.03, 0.12 if is_nigerian else 0.20)
#     }

# def get_market_sentiment(symbol):
#     """Get market sentiment (simulated with regional adjustments)"""
#     sentiment_scores = {
#         # US Stocks
#         'AAPL': 0.75, 'MSFT': 0.80, 'TSLA': 0.60, 'GOOGL': 0.70, 'AMZN': 0.65,
#         'META': 0.55, 'NVDA': 0.85, 'JPM': 0.60, 'JNJ': 0.70, 'V': 0.75,
#         'INTC': 0.45, 'UBER': 0.58, 'SHOP': 0.62, 'PYPL': 0.52, 'PEP': 0.68,
#         'KO': 0.72, 'DIS': 0.63, 'NKE': 0.69, 'WMT': 0.66, 'CRM': 0.71,
        
#         # Nigerian top stocks
#         'DANGCEM.NG': 0.68, 'GTCO.NG': 0.72, 'ZENITHBANK.NG': 0.65, 'UBA.NG': 0.63,
#         'ACCESS.NG': 0.61, 'NESTLE.NG': 0.70, 'UNILEVER.NG': 0.66
#     }
    
#     # Default sentiment based on market
#     is_nigerian = symbol.endswith('.NG')
#     default_sentiment = 0.45 if is_nigerian else 0.5
    
#     return sentiment_scores.get(symbol, default_sentiment)

# def generate_smc_signals(chart_patterns, indicators, confluence, waves, fundamentals, sentiment):
#     """Generate trading signals with simplified logic"""
#     try:
#         signal_score = 0.0
        
#         if chart_patterns.get('rising_wedge', False):
#             signal_score += 1.0
#         if chart_patterns.get('falling_wedge', False):
#             signal_score -= 1.0
        
#         if waves['impulse']['detected']:
#             signal_score += 1.5
        
#         if confluence['bullish_confluence']:
#             signal_score += 1.0
#         if confluence['bearish_confluence']:
#             signal_score -= 1.0
        
#         if 'RSI' in indicators and indicators['RSI'] < 30:
#             signal_score += 0.5
#         elif 'RSI' in indicators and indicators['RSI'] > 70:
#             signal_score -= 0.5
        
#         pe_ratio = fundamentals['PE_Ratio']
#         if pe_ratio < 15:  # Adjusted for Nigerian stocks
#             signal_score += 0.5
#         elif pe_ratio > 30:
#             signal_score -= 0.5
        
#         signal_score += sentiment * 0.5
        
#         if signal_score >= 2.0:
#             return 'Strong Buy', round(signal_score, 2)
#         elif signal_score >= 1.0:
#             return 'Buy', round(signal_score, 2)
#         elif signal_score <= -2.0:
#             return 'Strong Sell', round(signal_score, 2)
#         elif signal_score <= -1.0:
#             return 'Sell', round(signal_score, 2)
#         else:
#             return 'Neutral', round(signal_score, 2)
        
#     except Exception as e:
#         logger.error(f"Error in signal generation: {str(e)}")
#         return 'Neutral', 0.0

# def analyze_stock(symbol, data_source="twelve_data"):
#     """Analyze a single stock and return structured JSON data"""
#     try:
#         logger.info(f"Starting analysis for {symbol} using {data_source}")
        
#         daily_data = fetch_stock_data(symbol, "1day", 100, data_source)
#         weekly_data = fetch_stock_data(symbol, "1week", 50, data_source)
        
#         if daily_data.empty and weekly_data.empty:
#             logger.error(f"No data available for {symbol}")
#             return None
        
#         daily_analysis = None
#         if not daily_data.empty:
#             daily_analysis = analyze_timeframe(daily_data, symbol, "DAILY")
        
#         weekly_analysis = None
#         if not weekly_data.empty:
#             weekly_analysis = analyze_timeframe(weekly_data, symbol, "WEEKLY")
        
#         result = {
#             symbol: {
#                 'data_source': data_source,
#                 'market': 'Nigerian' if symbol.endswith('.NG') else 'US'
#             }
#         }
        
#         if daily_analysis:
#             result[symbol]["DAILY_TIMEFRAME"] = daily_analysis
        
#         if weekly_analysis:
#             result[symbol]["WEEKLY_TIMEFRAME"] = weekly_analysis
        
#         logger.info(f"Successfully analyzed {symbol}")
#         return result
        
#     except Exception as e:
#         logger.error(f"Error analyzing {symbol}: {str(e)}")
#         return None

# def analyze_timeframe(data, symbol, timeframe):
#     """Analyze a specific timeframe and return structured data"""
#     try:
#         ha_data = heikin_ashi(data)
#         if ha_data.empty:
#             logger.error(f"Failed to convert to HA for {symbol} {timeframe}")
#             return None
        
#         indicators_data = calculate_ha_indicators(ha_data)
#         if indicators_data is None:
#             logger.error(f"Failed to calculate indicators for {symbol} {timeframe}")
#             return None
        
#         pivots = detect_zigzag_pivots(ha_data)
#         patterns, _ = detect_geometric_patterns(ha_data, pivots)
#         waves = detect_elliott_waves(pivots, ha_data['HA_Close'])
#         confluence = detect_confluence(ha_data, pivots)
        
#         cycle_analysis = generate_cycle_analysis(indicators_data, symbol)
        
#         fundamentals = get_fundamental_data(symbol)
#         sentiment = get_market_sentiment(symbol)
        
#         last_indicators = indicators_data.iloc[-1].to_dict()
#         signal, score = generate_smc_signals(patterns, last_indicators, confluence, waves, fundamentals, sentiment)
        
#         current_price = round(ha_data['HA_Close'].iloc[-1], 2)
        
#         if 'Buy' in signal:
#             entry = round(current_price * 0.99, 2)
#             targets = [round(current_price * 1.05, 2), round(current_price * 1.10, 2)]
#             stop_loss = round(current_price * 0.95, 2)
#         else:
#             entry = round(current_price * 1.01, 2)
#             targets = [round(current_price * 0.95, 2), round(current_price * 0.90, 2)]
#             stop_loss = round(current_price * 1.05, 2)
        
#         change_1d = 0.0
#         change_1w = 0.0
        
#         if len(ha_data) >= 2:
#             change_1d = round((ha_data['HA_Close'].iloc[-1] / ha_data['HA_Close'].iloc[-2] - 1) * 100, 2)
        
#         if len(ha_data) >= 5:
#             change_1w = round((ha_data['HA_Close'].iloc[-1] / ha_data['HA_Close'].iloc[-5] - 1) * 100, 2)
        
#         rsi_verdict = "Overbought" if last_indicators.get('RSI', 50) > 70 else "Oversold" if last_indicators.get('RSI', 50) < 30 else "Neutral"
#         adx_verdict = "Strong Trend" if last_indicators.get('ADX', 25) > 25 else "Weak Trend"
#         momentum_verdict = "Bullish" if last_indicators.get('Cycle_Momentum', 0) > 0.02 else "Bearish" if last_indicators.get('Cycle_Momentum', 0) < -0.02 else "Neutral"
#         pattern_verdict = "Bullish Patterns" if any(patterns.values()) and signal in ['Buy', 'Strong Buy'] else "Bearish Patterns" if any(patterns.values()) and signal in ['Sell', 'Strong Sell'] else "No Clear Patterns"
#         pe_ratio = fundamentals['PE_Ratio']
#         fundamental_verdict = "Undervalued" if pe_ratio < 15 else "Overvalued" if pe_ratio > 25 else "Fair Value"
#         sentiment_verdict = "Positive" if sentiment > 0.6 else "Negative" if sentiment < 0.4 else "Neutral"
        
#         timeframe_analysis = {
#             'PRICE': current_price,
#             'ACCURACY': min(95, max(60, abs(score) * 20 + 60)),
#             'CONFIDENCE_SCORE': round(score, 2),
#             'VERDICT': signal,
#             'DETAILS': {
#                 'individual_verdicts': {
#                     'rsi_verdict': rsi_verdict,
#                     'adx_verdict': adx_verdict,
#                     'momentum_verdict': momentum_verdict,
#                     'pattern_verdict': pattern_verdict,
#                     'fundamental_verdict': fundamental_verdict,
#                     'sentiment_verdict': sentiment_verdict,
#                     'cycle_verdict': cycle_analysis['current_phase']
#                 },
#                 'price_data': {
#                     'current_price': current_price,
#                     'entry_price': entry,
#                     'target_prices': targets,
#                     'stop_loss': stop_loss,
#                     'change_1d': change_1d,
#                     'change_1w': change_1w
#                 },
#                 'technical_indicators': {
#                     'rsi': round(last_indicators.get('RSI', 50.0), 1),
#                     'adx': round(last_indicators.get('ADX', 25.0), 1),
#                     'atr': round(last_indicators.get('ATR', 1.0), 2),
#                     'cycle_phase': last_indicators.get('Cycle_Phase', 'Unknown'),
#                     'cycle_momentum': round(last_indicators.get('Cycle_Momentum', 0.0), 3)
#                 },
#                 'patterns': {
#                     'geometric': [k for k, v in patterns.items() if v] or ['None'],
#                     'elliott_wave': [k for k, v in waves.items() if v.get('detected', False)] or ['None'],
#                     'confluence_factors': confluence['factors'] or ['None']
#                 },
#                 'fundamentals': {
#                     'pe_ratio': fundamentals['PE_Ratio'],
#                     'eps': round(fundamentals['EPS'], 2),
#                     'revenue_growth': round(fundamentals['Revenue_Growth'] * 100, 2),
#                     'net_income_growth': round(fundamentals['Net_Income_Growth'] * 100, 2)
#                 },
#                 'sentiment_analysis': {
#                     'score': round(sentiment, 2),
#                     'interpretation': sentiment_verdict,
#                     'market_mood': "Optimistic" if sentiment > 0.7 else "Pessimistic" if sentiment < 0.3 else "Cautious"
#                 },
#                 'cycle_analysis': cycle_analysis,
#                 'trading_parameters': {
#                     'position_size': '5% of portfolio' if 'Strong' in signal else '3% of portfolio',
#                     'timeframe': '2-4 weeks' if 'Buy' in signal else '1-2 weeks',
#                     'risk_level': 'Medium' if 'Buy' in signal else 'High' if 'Sell' in signal else 'Low'
#                 }
#             }
#         }
        
#         return timeframe_analysis
        
#     except Exception as e:
#         logger.error(f"Error analyzing {timeframe} timeframe for {symbol}: {str(e)}")
#         return None

# # ================= ENHANCED BATCH PROCESSING =================
# def analyze_all_stocks():
#     """Analyze stocks with enhanced batch processing and progress tracking"""
#     try:
#         stock_config = get_filtered_stocks(175)
#         twelve_data_stocks = stock_config['twelve_data']
#         yfinance_stocks = stock_config['yfinance']
        
#         results = {}
#         total_stocks = len(twelve_data_stocks) + len(yfinance_stocks)
#         processed_count = 0
        
#         logger.info(f"Starting analysis of {total_stocks} stocks")
#         logger.info(f"Twelve Data stocks ({len(twelve_data_stocks)}): {twelve_data_stocks}")
#         logger.info(f"yfinance stocks ({len(yfinance_stocks)}): First 10: {yfinance_stocks[:10]}...")
        
#         # Process Twelve Data stocks in smaller batches
#         if twelve_data_stocks:
#             batch_size = TWELVE_DATA_BATCH_SIZE
#             num_batches = math.ceil(len(twelve_data_stocks) / batch_size)
            
#             for batch_idx in range(num_batches):
#                 batch_start = batch_idx * batch_size
#                 batch_end = min((batch_idx + 1) * batch_size, len(twelve_data_stocks))
#                 batch_symbols = twelve_data_stocks[batch_start:batch_end]
                
#                 logger.info(f"Processing Twelve Data batch {batch_idx+1}/{num_batches}: {batch_symbols}")
                
#                 for symbol in batch_symbols:
#                     try:
#                         result = analyze_stock(symbol, "twelve_data")
#                         if result:
#                             results.update(result)
#                             processed_count += 1
#                             logger.info(f"✓ {symbol} ({processed_count}/{total_stocks}) - Twelve Data")
#                         else:
#                             logger.warning(f"✗ Failed to process {symbol} (Twelve Data)")
#                     except Exception as e:
#                         logger.error(f"✗ Error processing {symbol} (Twelve Data): {str(e)}")
                
#                 # Sleep between batches to respect rate limits
#                 if batch_idx < num_batches - 1:
#                     logger.info(f"Sleeping {TWELVE_DATA_BATCH_SLEEP}s to respect Twelve Data rate limit...")
#                     time.sleep(TWELVE_DATA_BATCH_SLEEP)
        
#         # Process yfinance stocks in batches
#         if yfinance_stocks:
#             batch_size = YFINANCE_BATCH_SIZE
#             num_batches = math.ceil(len(yfinance_stocks) / batch_size)
            
#             for batch_idx in range(num_batches):
#                 batch_start = batch_idx * batch_size
#                 batch_end = min((batch_idx + 1) * batch_size, len(yfinance_stocks))
#                 batch_symbols = yfinance_stocks[batch_start:batch_end]
                
#                 logger.info(f"Processing yfinance batch {batch_idx+1}/{num_batches}: {batch_symbols}")
                
#                 for symbol in batch_symbols:
#                     try:
#                         result = analyze_stock(symbol, "yfinance")
#                         if result:
#                             results.update(result)
#                             processed_count += 1
#                             market = "Nigerian" if symbol.endswith('.NG') else "US"
#                             logger.info(f"✓ {symbol} ({processed_count}/{total_stocks}) - yfinance ({market})")
#                         else:
#                             logger.warning(f"✗ Failed to process {symbol} (yfinance)")
#                     except Exception as e:
#                         logger.error(f"✗ Error processing {symbol} (yfinance): {str(e)}")
                
#                 # Sleep between batches
#                 if batch_idx < num_batches - 1:
#                     logger.info(f"Sleeping {YFINANCE_BATCH_SLEEP}s between yfinance batches...")
#                     time.sleep(YFINANCE_BATCH_SLEEP)
        
#         # Calculate statistics
#         us_stocks_count = len([k for k, v in results.items() if not k.endswith('.NG')])
#         nigerian_stocks_count = len([k for k, v in results.items() if k.endswith('.NG')])
        
#         response = {
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#             'stocks_analyzed': len(results),
#             'total_requested': total_stocks,
#             'success_rate': round((len(results) / total_stocks) * 100, 1) if total_stocks > 0 else 0,
#             'status': 'success' if results else 'no_data',
#             'data_sources': {
#                 'twelve_data_count': len([k for k, v in results.items() if v.get('data_source') == 'twelve_data']),
#                 'yfinance_count': len([k for k, v in results.items() if v.get('data_source') == 'yfinance'])
#             },
#             'markets': {
#                 'us_stocks': us_stocks_count,
#                 'nigerian_stocks': nigerian_stocks_count
#             },
#             'processing_info': {
#                 'total_batches_twelve_data': math.ceil(len(twelve_data_stocks) / TWELVE_DATA_BATCH_SIZE) if twelve_data_stocks else 0,
#                 'total_batches_yfinance': math.ceil(len(yfinance_stocks) / YFINANCE_BATCH_SIZE) if yfinance_stocks else 0,
#                 'estimated_time_minutes': round((len(twelve_data_stocks) * TWELVE_DATA_BATCH_SLEEP / 60) + (len(yfinance_stocks) * YFINANCE_DELAY / 60), 1)
#             },
#             **results
#         }
        
#         logger.info(f"Analysis complete. Processed {len(results)}/{total_stocks} stocks successfully.")
#         logger.info(f"US stocks: {us_stocks_count}, Nigerian stocks: {nigerian_stocks_count}")
#         return response
        
#     except Exception as e:
#         logger.error(f"Error in analyze_all_stocks: {str(e)}")
#         return {
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#             'stocks_analyzed': 0,
#             'status': 'error',
#             'error': str(e)
#         }

# # ================= FLASK ROUTES =================
# @app.route('/', methods=['GET'])
# def home():
#     """Home endpoint with enhanced info"""
#     try:
#         stock_config = get_filtered_stocks()
#         return jsonify({
#             'message': 'Enhanced Stock Analysis API is running',
#             'version': '2.0 - Multi-Market Support',
#             'endpoints': {
#                 '/analyze': 'GET - Analyze stocks and return trading signals',
#                 '/health': 'GET - Health check',
#                 '/stocks': 'GET - List all available stocks',
#                 '/': 'GET - This help message'
#             },
#             'markets': {
#                 'us_stocks': len(stock_config['us_stocks']),
#                 'nigerian_stocks': len(stock_config['nigerian_stocks']),
#                 'total_stocks': stock_config['total_count']
#             },
#             'data_sources': {
#                 'twelve_data': f"{len(stock_config['twelve_data'])} stocks (US only)",
#                 'yfinance': f"{len(stock_config['yfinance'])} stocks (US + Nigerian)"
#             },
#             'rate_limits': {
#                 'twelve_data_per_minute': TWELVE_DATA_RATE_LIMIT_PER_MIN,
#                 'twelve_data_batch_size': TWELVE_DATA_BATCH_SIZE,
#                 'yfinance_batch_size': YFINANCE_BATCH_SIZE,
#                 'estimated_total_time_minutes': round((len(stock_config['twelve_data']) * TWELVE_DATA_BATCH_SLEEP / 60) + (len(stock_config['yfinance']) * YFINANCE_DELAY / 60), 1)
#             },
#             'status': 'online',
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         })
#     except Exception as e:
#         logger.error(f"Error in home endpoint: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/stocks', methods=['GET'])
# def list_stocks():
#     """List all available stocks"""
#     try:
#         stock_config = get_filtered_stocks()
#         return jsonify({
#             'us_stocks': stock_config['us_stocks'],
#             'nigerian_stocks': stock_config['nigerian_stocks'],
#             'data_source_distribution': {
#                 'twelve_data': stock_config['twelve_data'],
#                 'yfinance': stock_config['yfinance']
#             },
#             'total_count': stock_config['total_count'],
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         })
#     except Exception as e:
#         logger.error(f"Error in stocks endpoint: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/health', methods=['GET'])
# def health():
#     """Health check endpoint"""
#     try:
#         return jsonify({
#             'status': 'healthy',
#             'version': '2.0',
#             'markets': ['US', 'Nigerian'],
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#             'service': 'Multi-Market Stock Analysis API'
#         })
#     except Exception as e:
#         logger.error(f"Error in health endpoint: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/analyze', methods=['GET'])
# def analyze():
#     """API endpoint to analyze stocks and return JSON response"""
#     try:
#         logger.info("Starting comprehensive stock analysis...")
#         json_response = analyze_all_stocks()
#         logger.info(f"Analysis completed. Status: {json_response.get('status')}")
#         return jsonify(json_response)
#     except Exception as e:
#         logger.error(f"Error in /analyze endpoint: {str(e)}")
#         return jsonify({
#             'error': f"Failed to analyze stocks: {str(e)}",
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#             'stocks_analyzed': 0,
#             'status': 'error'
#         }), 500

# @app.errorhandler(404)
# def not_found(error):
#     return jsonify({
#         'error': 'Endpoint not found',
#         'message': 'The requested URL was not found on the server',
#         'available_endpoints': ['/analyze', '/health', '/stocks', '/'],
#         'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     }), 404

# @app.errorhandler(500)
# def internal_error(error):
#     return jsonify({
#         'error': 'Internal server error',
#         'message': 'An internal error occurred',
#         'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     }), 500

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))
#     debug_mode = os.environ.get("FLASK_ENV") == "development"
    
#     logger.info(f"Starting Enhanced Multi-Market Stock Analysis API on port {port}")
#     logger.info(f"Debug mode: {debug_mode}")
#     logger.info(f"Total stocks configured: {get_filtered_stocks()['total_count']}")
    
#     app.run(host='0.0.0.0', port=port, debug=debug_mode, threaded=True)



import pandas as pd
import numpy as np
import pandas_ta as ta
from scipy.signal import argrelextrema
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import warnings
from datetime import datetime, timedelta
import json
from sklearn.linear_model import LinearRegression
from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
import time
import requests
import math
import os
from threading import Lock, Thread
import queue
import anthropic
import sqlite3
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pickle
import threading

warnings.filterwarnings('ignore')

# ================= ENHANCED GLOBAL CONFIGURATION =================
TWELVE_DATA_API_KEY = "73adc6cc7e43476e851dcf54c705aeeb"
ALPHA_VANTAGE_API_KEY = "AK656KG03APJM5ZC"  # Add your Alpha Vantage key
CLAUDE_API_KEY = "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"  # Add your Claude API key
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

# Database configuration
DATABASE_PATH = "stock_analysis.db"
ANALYSIS_CACHE_FILE = "latest_analysis.json"

RISK_FREE_RATE = 0.02
MAX_WORKERS = 2
MIN_MARKET_CAP = 500e6
MIN_PRICE = 5.0
PATTERN_SENSITIVITY = 0.05
FIBONACCI_TOLERANCE = 0.05
CHANNEL_CONFIRMATION_BARS = 3
PATTERN_LOOKBACK = 20
ZIGZAG_LENGTH = 5
ZIGZAG_DEPTH = 10
ZIGZAG_NUM_PIVOTS = 10
CYCLE_MIN_DURATION = 30
PATTERN_ANGLE_THRESHOLD = 1.5
PATTERN_EXPANSION_RATIO = 1.2
PATTERN_CONTRACTION_RATIO = 0.8
MIN_TRENDLINE_R2 = 0.75
CONFIRMATION_VOL_RATIO = 1.2
MIN_TRENDLINE_ANGLE = 0.5
MAX_TRENDLINE_ANGLE = 85
HARMONIC_ERROR_TOLERANCE = 0.05
PRZ_LEFT_RANGE = 20
PRZ_RIGHT_RANGE = 20
FIBONACCI_LINE_LENGTH = 30
FUNDAMENTAL_WEIGHT = 0.3
SENTIMENT_WEIGHT = 0.2
TECHNICAL_WEIGHT = 0.5

# Enhanced Rate limiting configuration - Fixed for CoinGecko
TWELVE_DATA_RATE_LIMIT_PER_MIN = 8
TWELVE_DATA_BATCH_SIZE = 4
TWELVE_DATA_BATCH_SLEEP = 45
TWELVE_DATA_RETRY_ATTEMPTS = 2
TWELVE_DATA_RETRY_DELAY = 15
ALPHA_VANTAGE_BATCH_SIZE = 2  # Alpha Vantage: 5 calls per minute
ALPHA_VANTAGE_DELAY = 15  # 15 seconds between calls
COINGECKO_BATCH_SIZE = 3  # Reduced batch size
COINGECKO_DELAY = 10.0  # Increased delay to avoid rate limits
COINGECKO_RETRY_DELAY = 30

# Global rate limiting
rate_limit_lock = Lock()
last_twelve_data_request = 0
last_alpha_vantage_request = 0
last_coingecko_request = 0
request_count_twelve_data = 0
request_count_alpha_vantage = 0

# Background processing
analysis_in_progress = False
analysis_lock = threading.Lock()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ================= FLASK APP SETUP =================
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["*", "http://localhost:5177", "https://my-stocks-s2at.onrender.com"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Initialize Claude client
claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY) if CLAUDE_API_KEY != "YOUR_CLAUDE_API_KEY" else None

# Initialize scheduler
scheduler = BackgroundScheduler()

# ================= DATABASE SETUP =================
def init_database():
    """Initialize SQLite database for persistent storage"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            market TEXT NOT NULL,
            data_source TEXT NOT NULL,
            analysis_data TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol) ON CONFLICT REPLACE
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_metadata (
            id INTEGER PRIMARY KEY,
            total_analyzed INTEGER,
            success_rate REAL,
            last_update DATETIME,
            status TEXT,
            processing_time_minutes REAL
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

def save_analysis_to_db(results):
    """Save analysis results to database"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        # Save individual stock results
        for symbol, data in results.items():
            if symbol not in ['timestamp', 'stocks_analyzed', 'status', 'data_sources', 'markets', 'processing_info']:
                cursor.execute('''
                    INSERT OR REPLACE INTO analysis_results 
                    (symbol, market, data_source, analysis_data) 
                    VALUES (?, ?, ?, ?)
                ''', (
                    symbol,
                    data.get('market', 'Unknown'),
                    data.get('data_source', 'Unknown'),
                    json.dumps(data)
                ))
        
        # Save metadata
        cursor.execute('''
            INSERT OR REPLACE INTO analysis_metadata 
            (id, total_analyzed, success_rate, last_update, status, processing_time_minutes) 
            VALUES (1, ?, ?, ?, ?, ?)
        ''', (
            results.get('stocks_analyzed', 0),
            results.get('success_rate', 0),
            datetime.now().isoformat(),
            results.get('status', 'unknown'),
            results.get('processing_time_minutes', 0)
        ))
        
        conn.commit()
        logger.info(f"Saved {len([k for k in results.keys() if k not in ['timestamp', 'stocks_analyzed', 'status', 'data_sources', 'markets', 'processing_info']])} analysis results to database")
        
    except Exception as e:
        logger.error(f"Error saving to database: {str(e)}")
        conn.rollback()
    finally:
        conn.close()

def load_analysis_from_db():
    """Load latest analysis results from database"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        # Get metadata
        cursor.execute('SELECT * FROM analysis_metadata WHERE id = 1')
        metadata = cursor.fetchone()
        
        if not metadata:
            return None
        
        # Get all stock results
        cursor.execute('SELECT symbol, analysis_data FROM analysis_results ORDER BY timestamp DESC')
        stock_results = cursor.fetchall()
        
        if not stock_results:
            return None
        
        # Reconstruct response format
        response = {
            'timestamp': metadata[3],  # last_update
            'stocks_analyzed': metadata[1],  # total_analyzed
            'success_rate': metadata[2],  # success_rate
            'status': metadata[4],  # status
            'processing_time_minutes': metadata[5],  # processing_time_minutes
            'data_source': 'database_cache',
            'markets': {'us_stocks': 0, 'nigerian_stocks': 0, 'crypto_assets': 0},
            'data_sources': {'twelve_data_count': 0, 'alpha_vantage_count': 0, 'coingecko_count': 0}
        }
        
        # Add stock data
        for symbol, analysis_json in stock_results:
            try:
                analysis_data = json.loads(analysis_json)
                response[symbol] = analysis_data
                
                # Count by market
                market = analysis_data.get('market', 'Unknown')
                if market == 'US':
                    response['markets']['us_stocks'] += 1
                elif market == 'Nigerian':
                    response['markets']['nigerian_stocks'] += 1
                elif market == 'Crypto':
                    response['markets']['crypto_assets'] += 1
                
                # Count by data source
                data_source = analysis_data.get('data_source', 'Unknown')
                if data_source == 'twelve_data':
                    response['data_sources']['twelve_data_count'] += 1
                elif data_source == 'alpha_vantage':
                    response['data_sources']['alpha_vantage_count'] += 1
                elif data_source == 'coingecko':
                    response['data_sources']['coingecko_count'] += 1
                    
            except json.JSONDecodeError:
                logger.error(f"Error parsing analysis data for {symbol}")
                continue
        
        logger.info(f"Loaded {len(stock_results)} analysis results from database")
        return response
        
    except Exception as e:
        logger.error(f"Error loading from database: {str(e)}")
        return None
    finally:
        conn.close()

# ================= ENHANCED STOCK CONFIGURATION =================
def get_filtered_stocks(num_stocks=70):
    """Get optimized list of stocks with Alpha Vantage for Nigerian stocks"""
    
    # US Stocks (25) - Use Twelve Data
    us_stocks = [
        "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN",
        "NVDA", "META", "NFLX", "AMD", "INTC",
        "UBER", "SHOP", "PYPL", "PEP", "KO",
        "DIS", "NKE", "WMT", "CRM", "JPM",
        "JNJ", "V", "MA", "BABA", "ORCL"
    ]
    
    # Major Nigerian Stocks (30) - Use Alpha Vantage or simulated data
    major_nigerian_stocks = [
        # Top Banks
        "ACCESS.NG", "FBNH.NG", "GTCO.NG", "UBA.NG", "ZENITHBANK.NG",
        "STERLNBANK.NG", "FIDELITYBK.NG", "WEMABANK.NG",
        
        # Major Industrial/Cement
        "DANGCEM.NG", "BUACEMENT.NG", "WAPCO.NG",
        
        # Top Consumer Goods
        "DANGSUGAR.NG", "FLOURMILL.NG", "GUINNESS.NG", "NESTLE.NG", 
        "UNILEVER.NG", "PZ.NG", "CADBURY.NG", "NASCON.NG",
        
        # Oil & Gas Leaders
        "SEPLAT.NG", "TOTAL.NG", "OANDO.NG", "ARDOVA.NG",
        
        # Major Conglomerates
        "TRANSCORP.NG", "UACN.NG", "SCOA.NG",
        
        # Top Insurance & Others
        "AIICO.NG", "NEM.NG", "MTNN.NG"
    ]
    
    # Major Cryptocurrencies (15) - Reduced for better rate limiting
    major_cryptos = [
        "bitcoin", "ethereum", "binancecoin", "solana", "cardano",
        "avalanche-2", "polkadot", "chainlink", "polygon", "litecoin",
        "near", "uniswap", "cosmos", "algorand", "stellar"
    ]
    
    # Data source distribution
    twelve_data_stocks = us_stocks  # US stocks via Twelve Data
    alpha_vantage_stocks = major_nigerian_stocks  # Nigerian stocks via Alpha Vantage
    coingecko_cryptos = major_cryptos  # Cryptos via CoinGecko
    
    all_stocks = us_stocks + major_nigerian_stocks
    
    return {
        'twelve_data_us': twelve_data_stocks,
        'alpha_vantage_nigerian': alpha_vantage_stocks,
        'coingecko_cryptos': coingecko_cryptos,
        'all_traditional': all_stocks,
        'us_stocks': us_stocks,
        'nigerian_stocks': major_nigerian_stocks,
        'crypto_stocks': major_cryptos,
        'total_count': len(all_stocks) + len(major_cryptos)
    }

# ================= ENHANCED RATE LIMITING FUNCTIONS =================
def wait_for_rate_limit_twelve_data():
    """Optimized rate limiting for Twelve Data API"""
    global last_twelve_data_request, request_count_twelve_data
    
    with rate_limit_lock:
        current_time = time.time()
        
        if current_time - last_twelve_data_request > 60:
            request_count_twelve_data = 0
            last_twelve_data_request = current_time
        
        if request_count_twelve_data >= TWELVE_DATA_RATE_LIMIT_PER_MIN:
            sleep_time = 60 - (current_time - last_twelve_data_request)
            if sleep_time > 0:
                logger.info(f"Rate limit reached for Twelve Data. Sleeping for {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
                request_count_twelve_data = 0
                last_twelve_data_request = time.time()
        
        request_count_twelve_data += 1

def wait_for_rate_limit_yfinance():
    """Rate limiting for yfinance (Nigerian stocks)"""
    global last_yfinance_request
    
    with rate_limit_lock:
        current_time = time.time()
        time_since_last = current_time - last_yfinance_request
        
        if time_since_last < YFINANCE_DELAY:
            sleep_time = YFINANCE_DELAY - time_since_last
            time.sleep(sleep_time)
        
        last_yfinance_request = time.time()

def wait_for_rate_limit_alpha_vantage():
    """Rate limiting for Alpha Vantage API"""
    global last_alpha_vantage_request, request_count_alpha_vantage
    
    with rate_limit_lock:
        current_time = time.time()
        time_since_last = current_time - last_alpha_vantage_request
        
        if time_since_last < ALPHA_VANTAGE_DELAY:
            sleep_time = ALPHA_VANTAGE_DELAY - time_since_last
            time.sleep(sleep_time)
        
        last_alpha_vantage_request = time.time()
        request_count_alpha_vantage += 1

def wait_for_rate_limit_coingecko():
    """Enhanced rate limiting for CoinGecko API"""
    global last_coingecko_request
    
    with rate_limit_lock:
        current_time = time.time()
        time_since_last = current_time - last_coingecko_request
        
        if time_since_last < COINGECKO_DELAY:
            sleep_time = COINGECKO_DELAY - time_since_last
            logger.info(f"CoinGecko rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
        
        last_coingecko_request = time.time()

# ================= ENHANCED DATA FETCHING =================
def fetch_stock_data_twelve_with_retry(symbol, interval="1day", outputsize=100, max_retries=TWELVE_DATA_RETRY_ATTEMPTS):
    """Enhanced Twelve Data fetching"""
    for attempt in range(max_retries):
        try:
            wait_for_rate_limit_twelve_data()
            
            url = f"https://api.twelvedata.com/time_series"
            params = {
                'symbol': symbol,
                'interval': interval,
                'outputsize': outputsize,
                'apikey': TWELVE_DATA_API_KEY,
                'format': 'JSON'
            }
            
            logger.info(f"Fetching {symbol} ({interval}) from Twelve Data (attempt {attempt + 1}/{max_retries})")
            
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            
            data = response.json()
            
            if 'code' in data and data['code'] != 200:
                logger.warning(f"API error for {symbol}: {data.get('message', 'Unknown error')}")
                if attempt < max_retries - 1:
                    time.sleep(TWELVE_DATA_RETRY_DELAY)
                    continue
                return pd.DataFrame()
            
            if 'values' not in data:
                logger.error(f"No values in response for {symbol}: {data}")
                if attempt < max_retries - 1:
                    time.sleep(TWELVE_DATA_RETRY_DELAY)
                    continue
                return pd.DataFrame()
            
            df = pd.DataFrame(data['values'])
            
            if df.empty:
                logger.error(f"Empty DataFrame for {symbol}")
                if attempt < max_retries - 1:
                    time.sleep(TWELVE_DATA_RETRY_DELAY)
                    continue
                return pd.DataFrame()
            
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            df.dropna(inplace=True)
            
            logger.info(f"Successfully fetched {len(df)} rows for {symbol} ({interval}) from Twelve Data")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from Twelve Data for {symbol} (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(TWELVE_DATA_RETRY_DELAY)
            else:
                return pd.DataFrame()
    
    return pd.DataFrame()

def fetch_nigerian_stock_data_alpha_vantage(symbol, function="TIME_SERIES_DAILY", outputsize="compact"):
    """Fetch Nigerian stock data from Alpha Vantage or generate simulated data"""
    try:
        wait_for_rate_limit_alpha_vantage()
        
        # For now, generate simulated data for Nigerian stocks since Alpha Vantage may not support them
        # In production, you would try Alpha Vantage first, then fall back to simulation
        logger.info(f"Generating simulated data for Nigerian stock {symbol}")
        
        # Generate realistic Nigerian stock data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        dates = dates[dates.weekday < 5]  # Remove weekends
        
        # Base price for different Nigerian stocks
        base_prices = {
            'ACCESS.NG': 12.50, 'FBNH.NG': 8.30, 'GTCO.NG': 25.40, 'UBA.NG': 7.85, 'ZENITHBANK.NG': 22.10,
            'DANGCEM.NG': 245.50, 'BUACEMENT.NG': 85.20, 'NESTLE.NG': 1250.00, 'UNILEVER.NG': 14.50,
            'SEPLAT.NG': 850.00, 'TOTAL.NG': 165.30, 'TRANSCORP.NG': 4.25, 'MTNN.NG': 185.50
        }
        
        base_price = base_prices.get(symbol, 50.0)
        
        # Generate price data with realistic volatility
        np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
        returns = np.random.normal(0.001, 0.025, len(dates))  # Nigerian market volatility
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 0.01))  # Prevent negative prices
        
        # Create OHLCV data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            high = close * random.uniform(1.001, 1.03)
            low = close * random.uniform(0.97, 0.999)
            open_price = prices[i-1] * random.uniform(0.99, 1.01) if i > 0 else close
            volume = random.randint(100000, 5000000)
            
            data.append({
                'datetime': date,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        
        logger.info(f"Generated {len(df)} rows of simulated data for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error generating data for Nigerian stock {symbol}: {str(e)}")
        return pd.DataFrame()

def fetch_crypto_data_coingecko_with_retry(crypto_id, days=100, max_retries=3):
    """Fetch crypto data from CoinGecko with enhanced retry logic"""
    for attempt in range(max_retries):
        try:
            wait_for_rate_limit_coingecko()
            
            url = f"{COINGECKO_BASE_URL}/coins/{crypto_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'daily'
            }
            
            logger.info(f"Fetching {crypto_id} from CoinGecko (attempt {attempt + 1}/{max_retries})")
            
            response = requests.get(url, params=params, timeout=20)
            
            if response.status_code == 429:  # Rate limited
                logger.warning(f"Rate limited for {crypto_id}, waiting {COINGECKO_RETRY_DELAY}s...")
                time.sleep(COINGECKO_RETRY_DELAY)
                continue
            
            response.raise_for_status()
            
            data = response.json()
            
            if 'prices' not in data:
                logger.error(f"No price data for {crypto_id}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return pd.DataFrame()
            
            prices = data['prices']
            volumes = data.get('total_volumes', [])
            
            df_data = []
            for i, price_point in enumerate(prices):
                timestamp = pd.to_datetime(price_point[0], unit='ms')
                price = price_point[1]
                volume = volumes[i][1] if i < len(volumes) else 0
                
                df_data.append({
                    'datetime': timestamp,
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': volume
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            
            logger.info(f"Successfully fetched {len(df)} rows for {crypto_id} from CoinGecko")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching crypto data for {crypto_id} (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(COINGECKO_RETRY_DELAY)
            else:
                return pd.DataFrame()
    
    return pd.DataFrame()

def fetch_stock_data(symbol, interval="1day", outputsize=100, source="twelve_data"):
    """Unified function to fetch data from multiple sources"""
    if source == "twelve_data":
        return fetch_stock_data_twelve_with_retry(symbol, interval, outputsize)
    elif source == "alpha_vantage":
        # For Nigerian stocks, use simulated data for now
        return fetch_nigerian_stock_data_alpha_vantage(symbol)
    elif source == "coingecko":
        return fetch_crypto_data_coingecko_with_retry(symbol, outputsize)
    else:
        logger.error(f"Unknown data source: {source}")
        return pd.DataFrame()

# ================= EXISTING ANALYSIS FUNCTIONS (KEEP ALL UNCHANGED) =================
def heikin_ashi(df):
    """Convert dataframe to Heikin-Ashi candles with proper error handling"""
    if df.empty:
        return pd.DataFrame()
    
    try:
        df = df.copy()
        
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing required columns. Available: {df.columns.tolist()}")
            return pd.DataFrame()
        
        df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        ha_open = [(df['open'].iloc[0] + df['close'].iloc[0]) / 2]
        for i in range(1, len(df)):
            ha_open.append((ha_open[i-1] + df['HA_Close'].iloc[i-1]) / 2)
        
        df['HA_Open'] = ha_open
        df['HA_High'] = df[['high', 'HA_Open', 'HA_Close']].max(axis=1)
        df['HA_Low'] = df[['low', 'HA_Open', 'HA_Close']].min(axis=1)
        
        df.dropna(subset=['HA_Close', 'HA_Open', 'HA_High', 'HA_Low'], inplace=True)
        
        return df
        
    except Exception as e:
        logger.error(f"Error in heikin_ashi calculation: {str(e)}")
        return pd.DataFrame()

def detect_zigzag_pivots(data):
    """Detect significant pivot points using zigzag algorithm"""
    try:
        if len(data) < 10 or 'HA_Close' not in data.columns:
            return []
        
        prices = data['HA_Close'].values
        
        highs = argrelextrema(prices, np.greater, order=ZIGZAG_LENGTH)[0]
        lows = argrelextrema(prices, np.less, order=ZIGZAG_LENGTH)[0]
        
        pivot_indices = np.concatenate([highs, lows])
        pivot_indices.sort()
        
        filtered_pivots = []
        for i in pivot_indices:
            if len(filtered_pivots) < 2:
                filtered_pivots.append(i)
            else:
                last_price = prices[filtered_pivots[-1]]
                current_price = prices[i]
                change = abs(current_price - last_price) / last_price
                if change > PATTERN_SENSITIVITY:
                    filtered_pivots.append(i)
        
        pivot_data = []
        for i in filtered_pivots:
            start_idx = max(0, i - ZIGZAG_DEPTH)
            end_idx = min(len(prices), i + ZIGZAG_DEPTH)
            local_max = np.max(prices[start_idx:end_idx])
            local_min = np.min(prices[start_idx:end_idx])
            
            if prices[i] == local_max:
                pivot_type = 'high'
            else:
                pivot_type = 'low'
            
            pivot_data.append((i, prices[i], pivot_type))
        
        return pivot_data[-ZIGZAG_NUM_PIVOTS:]
        
    except Exception as e:
        logger.error(f"Error in detect_zigzag_pivots: {str(e)}")
        return []

def calculate_ha_indicators(df):
    """Calculate technical indicators on Heikin-Ashi data"""
    try:
        if df.empty or len(df) < 20:
            return None
        
        df = df.copy()
        
        df['ATR'] = ta.atr(df['HA_High'], df['HA_Low'], df['HA_Close'], length=14)
        df['RSI'] = ta.rsi(df['HA_Close'], length=14)
        
        adx_data = ta.adx(df['HA_High'], df['HA_Low'], df['HA_Close'], length=14)
        if isinstance(adx_data, pd.DataFrame) and 'ADX_14' in adx_data.columns:
            df['ADX'] = adx_data['ADX_14']
        else:
            df['ADX'] = 25.0
        
        df['Cycle_Phase'] = 'Bull'
        df['Cycle_Duration'] = 30
        df['Cycle_Momentum'] = (df['HA_Close'] - df['HA_Close'].shift(10)) / df['HA_Close'].shift(10)
        
        df['ATR'] = df['ATR'].fillna(df['ATR'].mean())
        df['RSI'] = df['RSI'].fillna(50.0)
        df['ADX'] = df['ADX'].fillna(25.0)
        df['Cycle_Momentum'] = df['Cycle_Momentum'].fillna(0.0)
        
        return df
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        return None

def detect_geometric_patterns(df, pivots):
    """Detect geometric patterns with simplified logic"""
    patterns = {
        'rising_wedge': False,
        'falling_wedge': False,
        'ascending_triangle': False,
        'descending_triangle': False,
        'channel': False,
        'head_shoulders': False,
        'pennant': False
    }
    
    try:
        if len(pivots) < 5:
            return patterns, {}
        
        recent_pivots = pivots[-5:]
        prices = [p[1] for p in recent_pivots]
        types = [p[2] for p in recent_pivots]
        
        if len([p for p in types if p == 'high']) >= 2 and len([p for p in types if p == 'low']) >= 2:
            highs = [p[1] for p in recent_pivots if p[2] == 'high']
            lows = [p[1] for p in recent_pivots if p[2] == 'low']
            
            if len(highs) >= 2 and len(lows) >= 2:
                if highs[-1] > highs[0] and lows[-1] > lows[0]:
                    patterns['rising_wedge'] = True
                elif highs[-1] < highs[0] and lows[-1] < lows[0]:
                    patterns['falling_wedge'] = True
        
        return patterns, {}
        
    except Exception as e:
        logger.error(f"Error in pattern detection: {str(e)}")
        return patterns, {}

def detect_elliott_waves(pivots, prices):
    """Simplified Elliott Wave detection"""
    waves = {
        'impulse': {'detected': False, 'wave1': False, 'wave2': False, 'wave3': False, 'wave4': False, 'wave5': False},
        'diagonal': {'detected': False, 'leading': False, 'ending': False},
        'zigzag': {'detected': False, 'waveA': False, 'waveB': False, 'waveC': False},
        'flat': {'detected': False, 'waveA': False, 'waveB': False, 'waveC': False}
    }
    
    try:
        if len(pivots) >= 5:
            waves['impulse']['detected'] = True
            waves['impulse']['wave1'] = True
            waves['impulse']['wave3'] = True
    except Exception as e:
        logger.error(f"Error in Elliott Wave detection: {str(e)}")
    
    return waves

def detect_confluence(df, pivots):
    """Detect Smart Money Concepts confluence"""
    confluence = {
        'bullish_confluence': False,
        'bearish_confluence': False,
        'factors': []
    }
    
    try:
        if df.empty or len(df) < 10:
            return confluence
        
        last_close = df['HA_Close'].iloc[-1]
        prev_close = df['HA_Close'].iloc[-5]
        
        if last_close > prev_close:
            confluence['factors'].append('Bullish Trend')
            confluence['bullish_confluence'] = True
        else:
            confluence['factors'].append('Bearish Trend')
            confluence['bearish_confluence'] = True
        
        return confluence
        
    except Exception as e:
        logger.error(f"Error in confluence detection: {str(e)}")
        return confluence

def generate_cycle_analysis(df, symbol):
    """Generate simplified cycle analysis"""
    try:
        if df.empty or len(df) < 10:
            return {
                'current_phase': 'Unknown',
                'stage': 'Unknown',
                'duration_days': 0,
                'momentum': 0.0,
                'momentum_visual': '----------',
                'bull_continuation_probability': 50,
                'bear_transition_probability': 50,
                'expected_continuation': 'Unknown',
                'risk_level': 'Medium'
            }
        
        last_close = df['HA_Close'].iloc[-1]
        prev_close = df['HA_Close'].iloc[-10] if len(df) >= 10 else df['HA_Close'].iloc[0]
        
        current_phase = 'Bull' if last_close > prev_close else 'Bear'
        momentum = (last_close - prev_close) / prev_close if prev_close != 0 else 0
        
        return {
            'current_phase': current_phase,
            'stage': f"Mid {current_phase}",
            'duration_days': 30,
            'momentum': round(momentum, 3),
            'momentum_visual': '▲' * 5 + '△' * 5 if momentum > 0 else '▼' * 5 + '▽' * 5,
            'bull_continuation_probability': 70 if current_phase == 'Bull' else 30,
            'bear_transition_probability': 30 if current_phase == 'Bull' else 70,
            'expected_continuation': '30-60 days',
            'risk_level': 'Medium'
        }
        
    except Exception as e:
        logger.error(f"Error in cycle analysis for {symbol}: {str(e)}")
        return {
            'current_phase': 'Unknown',
            'stage': 'Unknown',
            'duration_days': 0,
            'momentum': 0.0,
            'momentum_visual': '----------',
            'bull_continuation_probability': 50,
            'bear_transition_probability': 50,
            'expected_continuation': 'Unknown',
            'risk_level': 'Medium'
        }

def get_fundamental_data(symbol):
    """Get fundamental data with crypto support"""
    pe_ratios = {
        # US Stocks
        'AAPL': 28.5, 'MSFT': 32.1, 'TSLA': 45.2, 'GOOGL': 24.8, 'AMZN': 38.9,
        'META': 22.7, 'NVDA': 55.3, 'JPM': 12.4, 'JNJ': 16.8, 'V': 34.2,
        'INTC': 15.2, 'UBER': 28.9, 'SHOP': 42.1, 'PYPL': 18.7, 'PEP': 25.3,
        'KO': 24.8, 'DIS': 35.6, 'NKE': 31.2, 'WMT': 26.4, 'CRM': 48.9,
        
        # Nigerian Banks
        'ACCESS.NG': 8.5, 'FBNH.NG': 6.2, 'GTCO.NG': 12.3, 'UBA.NG': 7.4, 'ZENITHBANK.NG': 11.2,
        'STERLNBANK.NG': 8.9, 'FIDELITYBK.NG': 9.1, 'WEMABANK.NG': 9.5,
        
        # Nigerian Consumer Goods
        'DANGSUGAR.NG': 18.5, 'FLOURMILL.NG': 14.2, 'GUINNESS.NG': 22.1, 'NESTLE.NG': 35.8, 'UNILEVER.NG': 28.4,
        
        # Nigerian Industrial
        'DANGCEM.NG': 19.2, 'BUACEMENT.NG': 16.8, 'WAPCO.NG': 15.5,
        
        # Cryptos
        'bitcoin': 0, 'ethereum': 0, 'binancecoin': 0, 'solana': 0, 'cardano': 0
    }
    
    is_nigerian = symbol.endswith('.NG')
    is_crypto = symbol in ['bitcoin', 'ethereum', 'binancecoin', 'solana', 'cardano', 'avalanche-2', 'polkadot', 'chainlink', 'polygon', 'litecoin', 'near', 'uniswap', 'cosmos', 'algorand', 'stellar']
    
    if is_crypto:
        return {
            'PE_Ratio': 0,
            'Market_Cap_Rank': random.randint(1, 100),
            'Adoption_Score': random.uniform(0.6, 0.95),
            'Technology_Score': random.uniform(0.7, 0.98)
        }
    else:
        base_pe = pe_ratios.get(symbol, 12.0 if is_nigerian else 20.0)
        return {
            'PE_Ratio': base_pe,
            'EPS': random.uniform(2.0 if is_nigerian else 5.0, 8.0 if is_nigerian else 15.0),
            'Revenue_Growth': random.uniform(0.03 if is_nigerian else 0.05, 0.15 if is_nigerian else 0.25),
            'Net_Income_Growth': random.uniform(0.02 if is_nigerian else 0.03, 0.12 if is_nigerian else 0.20)
        }

def get_market_sentiment(symbol):
    """Get market sentiment with crypto support"""
    sentiment_scores = {
        # US Stocks
        'AAPL': 0.75, 'MSFT': 0.80, 'TSLA': 0.60, 'GOOGL': 0.70, 'AMZN': 0.65,
        'META': 0.55, 'NVDA': 0.85, 'JPM': 0.60, 'JNJ': 0.70, 'V': 0.75,
        
        # Nigerian top stocks
        'DANGCEM.NG': 0.68, 'GTCO.NG': 0.72, 'ZENITHBANK.NG': 0.65, 'UBA.NG': 0.63,
        'ACCESS.NG': 0.61, 'NESTLE.NG': 0.70, 'UNILEVER.NG': 0.66,
        
        # Major Cryptos
        'bitcoin': 0.78, 'ethereum': 0.82, 'binancecoin': 0.65, 'solana': 0.75, 'cardano': 0.58
    }
    
    is_nigerian = symbol.endswith('.NG')
    is_crypto = symbol in ['bitcoin', 'ethereum', 'binancecoin', 'solana', 'cardano', 'avalanche-2', 'polkadot', 'chainlink', 'polygon', 'litecoin', 'near', 'uniswap', 'cosmos', 'algorand', 'stellar']
    
    if is_crypto:
        default_sentiment = 0.6
    elif is_nigerian:
        default_sentiment = 0.45
    else:
        default_sentiment = 0.5
    
    return sentiment_scores.get(symbol, default_sentiment)

def generate_smc_signals(chart_patterns, indicators, confluence, waves, fundamentals, sentiment):
    """Generate trading signals with enhanced logic"""
    try:
        signal_score = 0.0
        
        if chart_patterns.get('rising_wedge', False):
            signal_score += 1.0
        if chart_patterns.get('falling_wedge', False):
            signal_score -= 1.0
        
        if waves['impulse']['detected']:
            signal_score += 1.5
        
        if confluence['bullish_confluence']:
            signal_score += 1.0
        if confluence['bearish_confluence']:
            signal_score -= 1.0
        
        if 'RSI' in indicators and indicators['RSI'] < 30:
            signal_score += 0.5
        elif 'RSI' in indicators and indicators['RSI'] > 70:
            signal_score -= 0.5
        
        if 'PE_Ratio' in fundamentals:
            pe_ratio = fundamentals['PE_Ratio']
            if pe_ratio > 0:
                if pe_ratio < 15:
                    signal_score += 0.5
                elif pe_ratio > 30:
                    signal_score -= 0.5
        elif 'Market_Cap_Rank' in fundamentals:
            if fundamentals['Market_Cap_Rank'] <= 10:
                signal_score += 0.3
            if fundamentals['Adoption_Score'] > 0.8:
                signal_score += 0.2
        
        signal_score += sentiment * 0.5
        
        if signal_score >= 2.0:
            return 'Strong Buy', round(signal_score, 2)
        elif signal_score >= 1.0:
            return 'Buy', round(signal_score, 2)
        elif signal_score <= -2.0:
            return 'Strong Sell', round(signal_score, 2)
        elif signal_score <= -1.0:
            return 'Sell', round(signal_score, 2)
        else:
            return 'Neutral', round(signal_score, 2)
        
    except Exception as e:
        logger.error(f"Error in signal generation: {str(e)}")
        return 'Neutral', 0.0

# ================= HIERARCHICAL ANALYSIS SYSTEM =================
def analyze_stock_hierarchical(symbol, data_source="twelve_data"):
    """Analyze stock with hierarchical timeframe dependency"""
    try:
        logger.info(f"Starting hierarchical analysis for {symbol} using {data_source}")
        
        timeframes = {
            'monthly': ('1month', 24),
            'weekly': ('1week', 52),
            'daily': ('1day', 100),
            '4hour': ('4h', 168)
        }
        
        timeframe_data = {}
        for tf_name, (interval, size) in timeframes.items():
            if data_source == "coingecko":
                data = fetch_stock_data(symbol, "1day", size * 7, data_source)
                if not data.empty and tf_name != 'daily':
                    if tf_name == 'monthly':
                        data = data.resample('M').agg({
                            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                        }).dropna()
                    elif tf_name == 'weekly':
                        data = data.resample('W').agg({
                            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                        }).dropna()
            elif data_source == "alpha_vantage":
                # For Nigerian stocks, use daily data and resample
                data = fetch_stock_data(symbol, "1day", 100, data_source)
                if not data.empty and tf_name != 'daily':
                    if tf_name == 'monthly':
                        data = data.resample('M').agg({
                            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                        }).dropna()
                    elif tf_name == 'weekly':
                        data = data.resample('W').agg({
                            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                        }).dropna()
                    elif tf_name == '4hour':
                        # Use daily data as proxy for 4-hour
                        pass
            else:
                data = fetch_stock_data(symbol, interval, size, data_source)
            
            if not data.empty:
                timeframe_data[tf_name] = data
        
        if not timeframe_data:
            logger.error(f"No data available for {symbol}")
            return None
        
        analyses = {}
        for tf_name, data in timeframe_data.items():
            analysis = analyze_timeframe_enhanced(data, symbol, tf_name.upper())
            if analysis:
                analyses[f"{tf_name.UPPER()}_TIMEFRAME"] = analysis
        
        final_analysis = apply_hierarchical_logic(analyses, symbol)
        
        result = {
            symbol: {
                'data_source': data_source,
                'market': 'Crypto' if data_source == 'coingecko' else ('Nigerian' if symbol.endswith('.NG') else 'US'),
                **final_analysis
            }
        }
        
        logger.info(f"Successfully analyzed {symbol} with hierarchical logic")
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {str(e)}")
        return None

def apply_hierarchical_logic(analyses, symbol):
    """Apply hierarchical logic where daily depends on weekly/monthly"""
    try:
        monthly = analyses.get('MONTHLY_TIMEFRAME')
        weekly = analyses.get('WEEKLY_TIMEFRAME')
        daily = analyses.get('DAILY_TIMEFRAME')
        four_hour = analyses.get('4HOUR_TIMEFRAME')
        
        if daily and weekly and monthly:
            monthly_weight = 0.4
            weekly_weight = 0.3
            daily_weight = 0.2
            four_hour_weight = 0.1
            
            monthly_conf = monthly['CONFIDENCE_SCORE'] * monthly_weight
            weekly_conf = weekly['CONFIDENCE_SCORE'] * weekly_weight
            daily_conf = daily['CONFIDENCE_SCORE'] * daily_weight
            four_hour_conf = four_hour['CONFIDENCE_SCORE'] * four_hour_weight if four_hour else 0
            
            if monthly['VERDICT'] in ['Strong Buy', 'Buy'] and weekly['VERDICT'] in ['Strong Buy', 'Buy']:
                if daily['VERDICT'] in ['Sell', 'Strong Sell']:
                    daily['VERDICT'] = 'Buy'
                    daily['DETAILS']['individual_verdicts']['hierarchy_override'] = 'Monthly/Weekly Bullish Override'
            elif monthly['VERDICT'] in ['Strong Sell', 'Sell'] and weekly['VERDICT'] in ['Strong Sell', 'Sell']:
                if daily['VERDICT'] in ['Buy', 'Strong Buy']:
                    daily['VERDICT'] = 'Sell'
                    daily['DETAILS']['individual_verdicts']['hierarchy_override'] = 'Monthly/Weekly Bearish Override'
            
            daily['CONFIDENCE_SCORE'] = round(monthly_conf + weekly_conf + daily_conf + four_hour_conf, 2)
            daily['ACCURACY'] = min(95, max(60, abs(daily['CONFIDENCE_SCORE']) * 15 + 70))
        
        return analyses
        
    except Exception as e:
        logger.error(f"Error in hierarchical logic for {symbol}: {str(e)}")
        return analyses

def analyze_timeframe_enhanced(data, symbol, timeframe):
    """Enhanced timeframe analysis with crypto support"""
    try:
        ha_data = heikin_ashi(data)
        if ha_data.empty:
            logger.error(f"Failed to convert to HA for {symbol} {timeframe}")
            return None
        
        indicators_data = calculate_ha_indicators(ha_data)
        if indicators_data is None:
            logger.error(f"Failed to calculate indicators for {symbol} {timeframe}")
            return None
        
        pivots = detect_zigzag_pivots(ha_data)
        patterns, _ = detect_geometric_patterns(ha_data, pivots)
        waves = detect_elliott_waves(pivots, ha_data['HA_Close'])
        confluence = detect_confluence(ha_data, pivots)
        
        cycle_analysis = generate_cycle_analysis(indicators_data, symbol)
        
        fundamentals = get_fundamental_data(symbol)
        sentiment = get_market_sentiment(symbol)
        
        last_indicators = indicators_data.iloc[-1].to_dict()
        signal, score = generate_smc_signals(patterns, last_indicators, confluence, waves, fundamentals, sentiment)
        
        current_price = round(ha_data['HA_Close'].iloc[-1], 2)
        
        if 'Buy' in signal:
            entry = round(current_price * 0.99, 2)
            targets = [round(current_price * 1.05, 2), round(current_price * 1.10, 2)]
            stop_loss = round(current_price * 0.95, 2)
        else:
            entry = round(current_price * 1.01, 2)
            targets = [round(current_price * 0.95, 2), round(current_price * 0.90, 2)]
            stop_loss = round(current_price * 1.05, 2)
        
        change_1d = 0.0
        change_1w = 0.0
        
        if len(ha_data) >= 2:
            change_1d = round((ha_data['HA_Close'].iloc[-1] / ha_data['HA_Close'].iloc[-2] - 1) * 100, 2)
        
        if len(ha_data) >= 5:
            change_1w = round((ha_data['HA_Close'].iloc[-1] / ha_data['HA_Close'].iloc[-5] - 1) * 100, 2)
        
        rsi_verdict = "Overbought" if last_indicators.get('RSI', 50) > 70 else "Oversold" if last_indicators.get('RSI', 50) < 30 else "Neutral"
        adx_verdict = "Strong Trend" if last_indicators.get('ADX', 25) > 25 else "Weak Trend"
        momentum_verdict = "Bullish" if last_indicators.get('Cycle_Momentum', 0) > 0.02 else "Bearish" if last_indicators.get('Cycle_Momentum', 0) < -0.02 else "Neutral"
        pattern_verdict = "Bullish Patterns" if any(patterns.values()) and signal in ['Buy', 'Strong Buy'] else "Bearish Patterns" if any(patterns.values()) and signal in ['Sell', 'Strong Sell'] else "No Clear Patterns"
        
        if 'PE_Ratio' in fundamentals and fundamentals['PE_Ratio'] > 0:
            pe_ratio = fundamentals['PE_Ratio']
            fundamental_verdict = "Undervalued" if pe_ratio < 15 else "Overvalued" if pe_ratio > 25 else "Fair Value"
        else:
            fundamental_verdict = "Strong Fundamentals" if fundamentals.get('Adoption_Score', 0.5) > 0.8 else "Weak Fundamentals"
        
        sentiment_verdict = "Positive" if sentiment > 0.6 else "Negative" if sentiment < 0.4 else "Neutral"
        
        timeframe_analysis = {
            'PRICE': current_price,
            'ACCURACY': min(95, max(60, abs(score) * 20 + 60)),
            'CONFIDENCE_SCORE': round(score, 2),
            'VERDICT': signal,
            'DETAILS': {
                'individual_verdicts': {
                    'rsi_verdict': rsi_verdict,
                    'adx_verdict': adx_verdict,
                    'momentum_verdict': momentum_verdict,
                    'pattern_verdict': pattern_verdict,
                    'fundamental_verdict': fundamental_verdict,
                    'sentiment_verdict': sentiment_verdict,
                    'cycle_verdict': cycle_analysis['current_phase']
                },
                'price_data': {
                    'current_price': current_price,
                    'entry_price': entry,
                    'target_prices': targets,
                    'stop_loss': stop_loss,
                    'change_1d': change_1d,
                    'change_1w': change_1w
                },
                'technical_indicators': {
                    'rsi': round(last_indicators.get('RSI', 50.0), 1),
                    'adx': round(last_indicators.get('ADX', 25.0), 1),
                    'atr': round(last_indicators.get('ATR', 1.0), 2),
                    'cycle_phase': last_indicators.get('Cycle_Phase', 'Unknown'),
                    'cycle_momentum': round(last_indicators.get('Cycle_Momentum', 0.0), 3)
                },
                'patterns': {
                    'geometric': [k for k, v in patterns.items() if v] or ['None'],
                    'elliott_wave': [k for k, v in waves.items() if v.get('detected', False)] or ['None'],
                    'confluence_factors': confluence['factors'] or ['None']
                },
                'fundamentals': fundamentals,
                'sentiment_analysis': {
                    'score': round(sentiment, 2),
                    'interpretation': sentiment_verdict,
                    'market_mood': "Optimistic" if sentiment > 0.7 else "Pessimistic" if sentiment < 0.3 else "Cautious"
                },
                'cycle_analysis': cycle_analysis,
                'trading_parameters': {
                    'position_size': '5% of portfolio' if 'Strong' in signal else '3% of portfolio',
                    'timeframe': f'{timeframe} - 2-4 weeks' if 'Buy' in signal else f'{timeframe} - 1-2 weeks',
                    'risk_level': 'Medium' if 'Buy' in signal else 'High' if 'Sell' in signal else 'Low'
                }
            }
        }
        
        return timeframe_analysis
        
    except Exception as e:
        logger.error(f"Error analyzing {timeframe} timeframe for {symbol}: {str(e)}")
        return None

# ================= CLAUDE AI INTEGRATION =================
def generate_ai_analysis(symbol, stock_data):
    """Generate detailed AI analysis using Claude"""
    if not claude_client:
        return {
            'error': 'Claude API not configured',
            'message': 'Please configure CLAUDE_API_KEY to use AI analysis'
        }
    
    try:
        context = f"""
        Stock Symbol: {symbol}
        Current Analysis Data:
        - Current Price: ${stock_data.get('DAILY_TIMEFRAME', {}).get('PRICE', 'N/A')}
        - Verdict: {stock_data.get('DAILY_TIMEFRAME', {}).get('VERDICT', 'N/A')}
        - Confidence Score: {stock_data.get('DAILY_TIMEFRAME', {}).get('CONFIDENCE_SCORE', 'N/A')}
        - Market: {stock_data.get('market', 'Unknown')}
        
        Technical Indicators:
        - RSI: {stock_data.get('DAILY_TIMEFRAME', {}).get('DETAILS', {}).get('technical_indicators', {}).get('rsi', 'N/A')}
        - ADX: {stock_data.get('DAILY_TIMEFRAME', {}).get('DETAILS', {}).get('technical_indicators', {}).get('adx', 'N/A')}
        - Cycle Phase: {stock_data.get('DAILY_TIMEFRAME', {}).get('DETAILS', {}).get('technical_indicators', {}).get('cycle_phase', 'N/A')}
        
        Individual Verdicts:
        {json.dumps(stock_data.get('DAILY_TIMEFRAME', {}).get('DETAILS', {}).get('individual_verdicts', {}), indent=2)}
        
        Patterns Detected:
        {json.dumps(stock_data.get('DAILY_TIMEFRAME', {}).get('DETAILS', {}).get('patterns', {}), indent=2)}
        """
        
        prompt = f"""
        You are an expert financial analyst with deep knowledge of technical analysis, fundamental analysis, and market psychology. 
        
        Based on the following stock analysis data for {symbol}, provide a comprehensive, detailed analysis that includes:
        
        1. **Executive Summary**: A clear, concise overview of the current situation
        2. **Technical Analysis Deep Dive**: Detailed interpretation of the technical indicators and patterns
        3. **Market Context**: How this stock fits within current market conditions
        4. **Risk Assessment**: Detailed risk factors and mitigation strategies
        5. **Trading Strategy**: Specific entry/exit strategies with reasoning
        6. **Timeline Expectations**: Short, medium, and long-term outlook
        7. **Key Catalysts**: What events or factors could change the analysis
        8. **Alternative Scenarios**: Bull case, bear case, and most likely scenario
        
        Context Data:
        {context}
        
        Please provide a thorough, professional analysis that a serious trader or investor would find valuable. 
        Use clear, actionable language and explain your reasoning for each conclusion.
        """
        
        response = claude_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=2000,
            temperature=0.3,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        return {
            'analysis': response.content[0].text,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': 'claude-3-sonnet',
            'symbol': symbol
        }
        
    except Exception as e:
        logger.error(f"Error generating AI analysis for {symbol}: {str(e)}")
        return {
            'error': 'Failed to generate AI analysis',
            'message': str(e)
        }

# ================= BACKGROUND PROCESSING =================
def analyze_all_stocks_background():
    """Background analysis function that runs at 5pm daily"""
    global analysis_in_progress
    
    with analysis_lock:
        if analysis_in_progress:
            logger.info("Analysis already in progress, skipping background run")
            return
        
        analysis_in_progress = True
    
    try:
        logger.info("Starting scheduled background analysis at 5pm")
        start_time = time.time()
        
        result = analyze_all_stocks_optimized()
        
        if result and result.get('status') == 'success':
            # Save to database
            save_analysis_to_db(result)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) / 60
            result['processing_time_minutes'] = round(processing_time, 2)
            
            logger.info(f"Background analysis completed successfully in {processing_time:.2f} minutes")
            logger.info(f"Analyzed {result.get('stocks_analyzed', 0)} assets")
        else:
            logger.error("Background analysis failed")
            
    except Exception as e:
        logger.error(f"Error in background analysis: {str(e)}")
    finally:
        with analysis_lock:
            analysis_in_progress = False

def analyze_all_stocks_optimized():
    """Optimized stock analysis with correct data source mapping"""
    try:
        stock_config = get_filtered_stocks(70)
        twelve_data_us = stock_config['twelve_data_us']
        alpha_vantage_nigerian = stock_config['alpha_vantage_nigerian']
        coingecko_cryptos = stock_config['coingecko_cryptos']
        
        results = {}
        total_stocks = len(twelve_data_us) + len(alpha_vantage_nigerian) + len(coingecko_cryptos)
        processed_count = 0
        
        logger.info(f"Starting optimized analysis of {total_stocks} assets")
        logger.info(f"US stocks (Twelve Data): {len(twelve_data_us)}")
        logger.info(f"Nigerian stocks (Alpha Vantage/Simulated): {len(alpha_vantage_nigerian)}")
        logger.info(f"Crypto (CoinGecko): {len(coingecko_cryptos)}")
        
        # Process US stocks via Twelve Data
        if twelve_data_us:
            batch_size = TWELVE_DATA_BATCH_SIZE
            num_batches = math.ceil(len(twelve_data_us) / batch_size)
            
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, len(twelve_data_us))
                batch_symbols = twelve_data_us[batch_start:batch_end]
                
                logger.info(f"Processing US batch {batch_idx+1}/{num_batches}: {batch_symbols}")
                
                for symbol in batch_symbols:
                    try:
                        result = analyze_stock_hierarchical(symbol, "twelve_data")
                        if result:
                            results.update(result)
                            processed_count += 1
                            logger.info(f"✓ {symbol} ({processed_count}/{total_stocks}) - US Stock (Twelve Data)")
                        else:
                            logger.warning(f"✗ Failed to process {symbol} (US)")
                    except Exception as e:
                        logger.error(f"✗ Error processing {symbol} (US): {str(e)}")
                
                if batch_idx < num_batches - 1:
                    logger.info(f"Sleeping {TWELVE_DATA_BATCH_SLEEP}s...")
                    time.sleep(TWELVE_DATA_BATCH_SLEEP)
        
        # Process Nigerian stocks via Alpha Vantage/Simulated
        if alpha_vantage_nigerian:
            batch_size = ALPHA_VANTAGE_BATCH_SIZE
            num_batches = math.ceil(len(alpha_vantage_nigerian) / batch_size)
            
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, len(alpha_vantage_nigerian))
                batch_symbols = alpha_vantage_nigerian[batch_start:batch_end]
                
                logger.info(f"Processing Nigerian batch {batch_idx+1}/{num_batches}: {batch_symbols}")
                
                for symbol in batch_symbols:
                    try:
                        result = analyze_stock_hierarchical(symbol, "alpha_vantage")
                        if result:
                            results.update(result)
                            processed_count += 1
                            logger.info(f"✓ {symbol} ({processed_count}/{total_stocks}) - Nigerian Stock (Simulated)")
                        else:
                            logger.warning(f"✗ Failed to process {symbol} (Nigerian)")
                    except Exception as e:
                        logger.error(f"✗ Error processing {symbol} (Nigerian): {str(e)}")
                
                if batch_idx < num_batches - 1:
                    logger.info(f"Sleeping {ALPHA_VANTAGE_DELAY}s...")
                    time.sleep(ALPHA_VANTAGE_DELAY)
        
        # Process Crypto assets via CoinGecko with enhanced rate limiting
        if coingecko_cryptos:
            batch_size = COINGECKO_BATCH_SIZE
            num_batches = math.ceil(len(coingecko_cryptos) / batch_size)
            
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, len(coingecko_cryptos))
                batch_symbols = coingecko_cryptos[batch_start:batch_end]
                
                logger.info(f"Processing Crypto batch {batch_idx+1}/{num_batches}: {batch_symbols}")
                
                for symbol in batch_symbols:
                    try:
                        result = analyze_stock_hierarchical(symbol, "coingecko")
                        if result:
                            results.update(result)
                            processed_count += 1
                            logger.info(f"✓ {symbol} ({processed_count}/{total_stocks}) - Crypto (CoinGecko)")
                        else:
                            logger.warning(f"✗ Failed to process {symbol} (Crypto)")
                    except Exception as e:
                        logger.error(f"✗ Error processing {symbol} (Crypto): {str(e)}")
                
                if batch_idx < num_batches - 1:
                    logger.info(f"Sleeping {COINGECKO_DELAY * 2}s for CoinGecko rate limits...")
                    time.sleep(COINGECKO_DELAY * 2)
        
        # Calculate statistics
        us_stocks_count = len([k for k, v in results.items() if v.get('market') == 'US'])
        nigerian_stocks_count = len([k for k, v in results.items() if v.get('market') == 'Nigerian'])
        crypto_count = len([k for k, v in results.items() if v.get('market') == 'Crypto'])
        
        response = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'stocks_analyzed': len(results),
            'total_requested': total_stocks,
            'success_rate': round((len(results) / total_stocks) * 100, 1) if total_stocks > 0 else 0,
            'status': 'success' if results else 'no_data',
            'data_sources': {
                'twelve_data_count': len([k for k, v in results.items() if v.get('data_source') == 'twelve_data']),
                'alpha_vantage_count': len([k for k, v in results.items() if v.get('data_source') == 'alpha_vantage']),
                'coingecko_count': len([k for k, v in results.items() if v.get('data_source') == 'coingecko'])
            },
            'markets': {
                'us_stocks': us_stocks_count,
                'nigerian_stocks': nigerian_stocks_count,
                'crypto_assets': crypto_count
            },
            'processing_info': {
                'hierarchical_analysis': True,
                'timeframes_analyzed': ['monthly', 'weekly', 'daily', '4hour'],
                'ai_analysis_available': claude_client is not None,
                'data_source_mapping': {
                    'us_stocks': 'twelve_data',
                    'nigerian_stocks': 'alpha_vantage (simulated)',
                    'crypto_assets': 'coingecko'
                },
                'background_processing': True,
                'daily_auto_refresh': '5:00 PM'
            },
            **results
        }
        
        logger.info(f"Optimized analysis complete. Processed {len(results)}/{total_stocks} assets successfully.")
        logger.info(f"US: {us_stocks_count}, Nigerian: {nigerian_stocks_count}, Crypto: {crypto_count}")
        return response
        
    except Exception as e:
        logger.error(f"Error in analyze_all_stocks_optimized: {str(e)}")
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'stocks_analyzed': 0,
            'status': 'error',
            'error': str(e)
        }

# ================= FLASK ROUTES =================
@app.route('/', methods=['GET'])
def home():
    """Enhanced home endpoint with persistent data info"""
    try:
        stock_config = get_filtered_stocks()
        
        # Check if we have cached data
        cached_data = load_analysis_from_db()
        has_cached_data = cached_data is not None
        
        return jsonify({
            'message': 'Enhanced Multi-Asset Analysis API v4.0',
            'version': '4.0 - Persistent Data + Background Processing + Fixed Nigerian Stocks',
            'endpoints': {
                '/analyze': 'GET - Get latest analysis (from cache or trigger new)',
                '/analyze/fresh': 'GET - Force fresh analysis (manual refresh)',
                '/ai-analysis': 'POST - Get detailed AI analysis for specific symbol',
                '/health': 'GET - Health check',
                '/assets': 'GET - List all available assets',
                '/': 'GET - This help message'
            },
            'markets': {
                'us_stocks': len(stock_config['us_stocks']),
                'nigerian_stocks': len(stock_config['nigerian_stocks']),
                'crypto_assets': len(stock_config['crypto_stocks']),
                'total_assets': stock_config['total_count']
            },
            'features': {
                'hierarchical_analysis': True,
                'timeframes': ['monthly', 'weekly', 'daily', '4hour'],
                'ai_analysis': claude_client is not None,
                'persistent_storage': True,
                'background_processing': True,
                'daily_auto_refresh': '5:00 PM',
                'data_sources': {
                    'us_stocks': 'twelve_data',
                    'nigerian_stocks': 'alpha_vantage (simulated)',
                    'crypto_assets': 'coingecko'
                },
                'optimized_processing': True
            },
            'data_status': {
                'has_cached_data': has_cached_data,
                'last_update': cached_data.get('timestamp') if cached_data else None,
                'cached_assets': cached_data.get('stocks_analyzed') if cached_data else 0,
                'analysis_in_progress': analysis_in_progress
            },
            'status': 'online',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        logger.error(f"Error in home endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/assets', methods=['GET'])
def list_assets():
    """List all available assets"""
    try:
        stock_config = get_filtered_stocks()
        return jsonify({
            'us_stocks': stock_config['us_stocks'],
            'nigerian_stocks': stock_config['nigerian_stocks'],
            'crypto_assets': stock_config['crypto_stocks'],
            'data_source_distribution': {
                'twelve_data_us': stock_config['twelve_data_us'],
                'alpha_vantage_nigerian': stock_config['alpha_vantage_nigerian'],
                'coingecko_cryptos': stock_config['coingecko_cryptos']
            },
            'total_count': stock_config['total_count'],
            'note': 'Nigerian stocks use simulated data as reliable APIs are limited for NSE',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        logger.error(f"Error in assets endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        cached_data = load_analysis_from_db()
        return jsonify({
            'status': 'healthy',
            'version': '4.0',
            'markets': ['US', 'Nigerian', 'Crypto'],
            'features': {
                'hierarchical_analysis': True,
                'ai_analysis': claude_client is not None,
                'optimized_processing': True,
                'nigerian_stock_support': True,
                'persistent_storage': True,
                'background_processing': True
            },
            'data_status': {
                'has_cached_data': cached_data is not None,
                'analysis_in_progress': analysis_in_progress,
                'last_update': cached_data.get('timestamp') if cached_data else None
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'service': 'Multi-Asset Analysis API with Persistent Data & Background Processing'
        })
    except Exception as e:
        logger.error(f"Error in health endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['GET'])
def analyze():
    """Get latest analysis - from cache if available, otherwise return cached data"""
    try:
        # First, try to load from cache
        cached_data = load_analysis_from_db()
        
        if cached_data:
            logger.info(f"Returning cached analysis data from {cached_data.get('timestamp')}")
            cached_data['data_source'] = 'database_cache'
            cached_data['note'] = 'This is cached data. Use /analyze/fresh for new analysis.'
            return jsonify(cached_data)
        else:
            # No cached data, run fresh analysis
            logger.info("No cached data found, running fresh analysis...")
            json_response = analyze_all_stocks_optimized()
            
            if json_response and json_response.get('status') == 'success':
                save_analysis_to_db(json_response)
            
            logger.info(f"Fresh analysis completed. Status: {json_response.get('status')}")
            return jsonify(json_response)
            
    except Exception as e:
        logger.error(f"Error in /analyze endpoint: {str(e)}")
        return jsonify({
            'error': f"Failed to get analysis: {str(e)}",
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'stocks_analyzed': 0,
            'status': 'error'
        }), 500

@app.route('/analyze/fresh', methods=['GET'])
def analyze_fresh():
    """Force fresh analysis (manual refresh)"""
    try:
        logger.info("Starting manual fresh analysis...")
        start_time = time.time()
        
        json_response = analyze_all_stocks_optimized()
        
        if json_response and json_response.get('status') == 'success':
            # Calculate processing time
            processing_time = (time.time() - start_time) / 60
            json_response['processing_time_minutes'] = round(processing_time, 2)
            
            # Save to database
            save_analysis_to_db(json_response)
            
            logger.info(f"Fresh analysis completed in {processing_time:.2f} minutes")
        
        logger.info(f"Fresh analysis completed. Status: {json_response.get('status')}")
        return jsonify(json_response)
        
    except Exception as e:
        logger.error(f"Error in /analyze/fresh endpoint: {str(e)}")
        return jsonify({
            'error': f"Failed to run fresh analysis: {str(e)}",
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'stocks_analyzed': 0,
            'status': 'error'
        }), 500

@app.route('/ai-analysis', methods=['POST'])
def ai_analysis():
    """AI analysis endpoint using Claude"""
    try:
        data = request.get_json()
        if not data or 'symbol' not in data:
            return jsonify({
                'error': 'Missing symbol parameter',
                'message': 'Please provide a symbol in the request body'
            }), 400
        
        symbol = data['symbol'].upper()
        
        logger.info(f"Generating AI analysis for {symbol}")
        
        # Determine data source based on symbol
        stock_config = get_filtered_stocks()
        if symbol in stock_config['crypto_stocks']:
            data_source = "coingecko"
        elif symbol in stock_config['nigerian_stocks']:
            data_source = "alpha_vantage"
        else:
            data_source = "twelve_data"
        
        # Try to get from cache first
        cached_data = load_analysis_from_db()
        stock_analysis = None
        
        if cached_data and symbol in cached_data:
            stock_analysis = {symbol: cached_data[symbol]}
            logger.info(f"Using cached data for AI analysis of {symbol}")
        else:
            # Get fresh analysis for this symbol
            stock_analysis = analyze_stock_hierarchical(symbol, data_source)
        
        if not stock_analysis or symbol not in stock_analysis:
            return jsonify({
                'error': 'Symbol not found or analysis failed',
                'message': f'Could not analyze {symbol}. Please check the symbol and try again.'
            }), 404
        
        # Generate AI analysis
        ai_result = generate_ai_analysis(symbol, stock_analysis[symbol])
        
        return jsonify({
            'symbol': symbol,
            'ai_analysis': ai_result,
            'technical_analysis': stock_analysis[symbol],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        logger.error(f"Error in /ai-analysis endpoint: {str(e)}")
        return jsonify({
            'error': f"Failed to generate AI analysis: {str(e)}",
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested URL was not found on the server',
        'available_endpoints': ['/analyze', '/analyze/fresh', '/ai-analysis', '/health', '/assets', '/'],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An internal error occurred',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }), 500

# ================= STARTUP AND SCHEDULER =================
def start_scheduler():
    """Start the background scheduler for daily 5pm analysis"""
    try:
        # Schedule daily analysis at 5:00 PM
        scheduler.add_job(
            func=analyze_all_stocks_background,
            trigger=CronTrigger(hour=17, minute=0),  # 5:00 PM
            id='daily_analysis',
            name='Daily Stock Analysis at 5PM',
            replace_existing=True
        )
        
        scheduler.start()
        logger.info("Background scheduler started - Daily analysis at 5:00 PM")
        
    except Exception as e:
        logger.error(f"Error starting scheduler: {str(e)}")

if __name__ == "__main__":
    # Initialize database
    init_database()
    
    # Start background scheduler
    start_scheduler()
    
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    
    logger.info(f"Starting Enhanced Multi-Asset Analysis API v4.0 on port {port}")
    logger.info(f"Debug mode: {debug_mode}")
    logger.info(f"Total assets configured: {get_filtered_stocks()['total_count']}")
    logger.info(f"AI Analysis available: {claude_client is not None}")
    logger.info("Data source mapping: US (Twelve Data), Nigerian (Simulated), Crypto (CoinGecko)")
    logger.info("Features: Persistent Storage + Background Processing + Daily 5PM Auto-Refresh")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=debug_mode, threaded=True)
    finally:
        # Cleanup scheduler on shutdown
        if scheduler.running:
            scheduler.shutdown()
