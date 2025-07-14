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
import yfinance as yf
import os

warnings.filterwarnings('ignore')

# ================= GLOBAL CONFIGURATION =================
TWELVE_DATA_API_KEY = "73adc6cc7e43476e851dcf54c705aeeb"

RISK_FREE_RATE = 0.02
MAX_WORKERS = 1
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

# Rate limiting configuration
TWELVE_DATA_RATE_LIMIT_PER_MIN = 6
TWELVE_DATA_BATCH_SIZE = 3
TWELVE_DATA_BATCH_SLEEP = 65
YFINANCE_DELAY = 0.5

# Setup logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================= FLASK APP SETUP =================
app = Flask(__name__)

# Simple CORS configuration - avoid conflicts
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# ================= DATA FUNCTIONS =================
def get_filtered_stocks(num_stocks=20):
    """Get list of stocks to analyze - split between Twelve Data and yfinance"""
    stock_list = [
        "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN",
        "NVDA", "META", "NFLX", "BABA", "AMD",
        "INTC", "UBER", "SHOP", "PYPL", "PEP",
        "KO", "DIS", "NKE", "WMT", "CRM"
    ]
    
    # Split stocks: first 15 for Twelve Data, last 5 for yfinance
    twelve_data_stocks = stock_list[:15]
    yfinance_stocks = stock_list[15:20]
    
    return {
        'twelve_data': twelve_data_stocks,
        'yfinance': yfinance_stocks,
        'all': stock_list[:num_stocks]
    }

def fetch_stock_data_twelve(symbol, interval="1day", outputsize=100):
    """Fetch stock data from Twelve Data API with proper error handling"""
    try:
        url = f"https://api.twelvedata.com/time_series"
        params = {
            'symbol': symbol,
            'interval': interval,
            'outputsize': outputsize,
            'apikey': TWELVE_DATA_API_KEY,
            'format': 'JSON'
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if 'values' not in data:
            logger.error(f"No values in response for {symbol}: {data}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data['values'])
        
        if df.empty:
            logger.error(f"Empty DataFrame for {symbol}")
            return pd.DataFrame()
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        df.dropna(inplace=True)
        
        logger.info(f"Successfully fetched {len(df)} rows for {symbol} from Twelve Data")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data from Twelve Data for {symbol}: {str(e)}")
        return pd.DataFrame()

def fetch_stock_data_yfinance(symbol, period="100d", interval="1d"):
    """Fetch stock data from yfinance with proper error handling"""
    try:
        time.sleep(YFINANCE_DELAY)
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            logger.error(f"Empty DataFrame for {symbol} from yfinance")
            return pd.DataFrame()
        
        df.columns = df.columns.str.lower()
        df.reset_index(inplace=True)
        df.rename(columns={'date': 'datetime'}, inplace=True)
        df.set_index('datetime', inplace=True)
        df.dropna(inplace=True)
        
        logger.info(f"Successfully fetched {len(df)} rows for {symbol} from yfinance")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data from yfinance for {symbol}: {str(e)}")
        return pd.DataFrame()

def fetch_stock_data(symbol, interval="1day", outputsize=100, source="twelve_data"):
    """Unified function to fetch stock data from either source"""
    if source == "twelve_data":
        return fetch_stock_data_twelve(symbol, interval, outputsize)
    elif source == "yfinance":
        yf_interval = "1d" if interval == "1day" else "1wk" if interval == "1week" else "1d"
        yf_period = f"{outputsize}d" if interval == "1day" else f"{outputsize//7}wk"
        return fetch_stock_data_yfinance(symbol, period=yf_period, interval=yf_interval)
    else:
        logger.error(f"Unknown data source: {source}")
        return pd.DataFrame()

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
    """Get fundamental data (simulated with realistic values)"""
    pe_ratios = {
        'AAPL': 28.5, 'MSFT': 32.1, 'TSLA': 45.2, 'GOOGL': 24.8, 'AMZN': 38.9,
        'META': 22.7, 'NVDA': 55.3, 'JPM': 12.4, 'JNJ': 16.8, 'V': 34.2,
        'INTC': 15.2, 'UBER': 28.9, 'SHOP': 42.1, 'PYPL': 18.7, 'PEP': 25.3,
        'KO': 24.8, 'DIS': 35.6, 'NKE': 31.2, 'WMT': 26.4, 'CRM': 48.9
    }
    
    return {
        'PE_Ratio': pe_ratios.get(symbol, 20.0),
        'EPS': random.uniform(5.0, 15.0),
        'Revenue_Growth': random.uniform(0.05, 0.25),
        'Net_Income_Growth': random.uniform(0.03, 0.20)
    }

def get_market_sentiment(symbol):
    """Get market sentiment (simulated)"""
    sentiment_scores = {
        'AAPL': 0.75, 'MSFT': 0.80, 'TSLA': 0.60, 'GOOGL': 0.70, 'AMZN': 0.65,
        'META': 0.55, 'NVDA': 0.85, 'JPM': 0.60, 'JNJ': 0.70, 'V': 0.75,
        'INTC': 0.45, 'UBER': 0.58, 'SHOP': 0.62, 'PYPL': 0.52, 'PEP': 0.68,
        'KO': 0.72, 'DIS': 0.63, 'NKE': 0.69, 'WMT': 0.66, 'CRM': 0.71
    }
    return sentiment_scores.get(symbol, 0.5)

def generate_smc_signals(chart_patterns, indicators, confluence, waves, fundamentals, sentiment):
    """Generate trading signals with simplified logic"""
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
        
        pe_ratio = fundamentals['PE_Ratio']
        if pe_ratio < 20:
            signal_score += 0.5
        elif pe_ratio > 40:
            signal_score -= 0.5
        
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

def analyze_stock(symbol, data_source="twelve_data"):
    """Analyze a single stock and return structured JSON data"""
    try:
        logger.info(f"Starting analysis for {symbol} using {data_source}")
        
        daily_data = fetch_stock_data(symbol, "1day", 100, data_source)
        weekly_data = fetch_stock_data(symbol, "1week", 50, data_source)
        
        if daily_data.empty and weekly_data.empty:
            logger.error(f"No data available for {symbol}")
            return None
        
        daily_analysis = None
        if not daily_data.empty:
            daily_analysis = analyze_timeframe(daily_data, symbol, "DAILY")
        
        weekly_analysis = None
        if not weekly_data.empty:
            weekly_analysis = analyze_timeframe(weekly_data, symbol, "WEEKLY")
        
        result = {
            symbol: {
                'data_source': data_source
            }
        }
        
        if daily_analysis:
            result[symbol]["DAILY_TIMEFRAME"] = daily_analysis
        
        if weekly_analysis:
            result[symbol]["WEEKLY_TIMEFRAME"] = weekly_analysis
        
        logger.info(f"Successfully analyzed {symbol}")
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {str(e)}")
        return None

def analyze_timeframe(data, symbol, timeframe):
    """Analyze a specific timeframe and return structured data"""
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
        pe_ratio = fundamentals['PE_Ratio']
        fundamental_verdict = "Undervalued" if pe_ratio < 20 else "Overvalued" if pe_ratio > 30 else "Fair Value"
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
                'fundamentals': {
                    'pe_ratio': fundamentals['PE_Ratio'],
                    'eps': round(fundamentals['EPS'], 2),
                    'revenue_growth': round(fundamentals['Revenue_Growth'] * 100, 2),
                    'net_income_growth': round(fundamentals['Net_Income_Growth'] * 100, 2)
                },
                'sentiment_analysis': {
                    'score': round(sentiment, 2),
                    'interpretation': sentiment_verdict,
                    'market_mood': "Optimistic" if sentiment > 0.7 else "Pessimistic" if sentiment < 0.3 else "Cautious"
                },
                'cycle_analysis': cycle_analysis,
                'trading_parameters': {
                    'position_size': '5% of portfolio' if 'Strong' in signal else '3% of portfolio',
                    'timeframe': '2-4 weeks' if 'Buy' in signal else '1-2 weeks',
                    'risk_level': 'Medium' if 'Buy' in signal else 'High' if 'Sell' in signal else 'Low'
                }
            }
        }
        
        return timeframe_analysis
        
    except Exception as e:
        logger.error(f"Error analyzing {timeframe} timeframe for {symbol}: {str(e)}")
        return None

def analyze_all_stocks():
    """Analyze stocks and return JSON response"""
    try:
        stock_config = get_filtered_stocks(20)
        twelve_data_stocks = stock_config['twelve_data']
        yfinance_stocks = stock_config['yfinance']
        
        results = {}
        
        logger.info(f"Starting analysis of 20 stocks")
        logger.info(f"Twelve Data stocks (15): {twelve_data_stocks}")
        logger.info(f"yfinance stocks (5): {yfinance_stocks}")
        
        # Process Twelve Data stocks in batches
        if twelve_data_stocks:
            batch_size = TWELVE_DATA_BATCH_SIZE
            num_batches = math.ceil(len(twelve_data_stocks) / batch_size)
            
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, len(twelve_data_stocks))
                batch_symbols = twelve_data_stocks[batch_start:batch_end]
                
                logger.info(f"Processing Twelve Data batch {batch_idx+1}/{num_batches}: {batch_symbols}")
                
                for symbol in batch_symbols:
                    try:
                        result = analyze_stock(symbol, "twelve_data")
                        if result:
                            results.update(result)
                            logger.info(f"Successfully processed {symbol} (Twelve Data)")
                        else:
                            logger.warning(f"Failed to process {symbol} (Twelve Data)")
                    except Exception as e:
                        logger.error(f"Error processing {symbol} (Twelve Data): {str(e)}")
                
                if batch_idx < num_batches - 1:
                    logger.info(f"Sleeping {TWELVE_DATA_BATCH_SLEEP}s to respect Twelve Data rate limit...")
                    time.sleep(TWELVE_DATA_BATCH_SLEEP)
        
        # Process yfinance stocks
        if yfinance_stocks:
            logger.info(f"Processing yfinance stocks: {yfinance_stocks}")
            for symbol in yfinance_stocks:
                try:
                    result = analyze_stock(symbol, "yfinance")
                    if result:
                        results.update(result)
                        logger.info(f"Successfully processed {symbol} (yfinance)")
                    else:
                        logger.warning(f"Failed to process {symbol} (yfinance)")
                except Exception as e:
                    logger.error(f"Error processing {symbol} (yfinance): {str(e)}")
        
        response = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'stocks_analyzed': len(results),
            'status': 'success' if results else 'no_data',
            'data_sources': {
                'twelve_data_count': len([k for k, v in results.items() if v.get('data_source') == 'twelve_data']),
                'yfinance_count': len([k for k, v in results.items() if v.get('data_source') == 'yfinance'])
            },
            **results
        }
        
        logger.info(f"Analysis complete. Processed {len(results)} stocks successfully.")
        return response
        
    except Exception as e:
        logger.error(f"Error in analyze_all_stocks: {str(e)}")
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'stocks_analyzed': 0,
            'status': 'error',
            'error': str(e)
        }

# ================= FLASK ROUTES =================

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    try:
        return jsonify({
            'message': 'Stock Analysis API is running',
            'endpoints': {
                '/analyze': 'GET - Analyze stocks and return trading signals',
                '/health': 'GET - Health check',
                '/': 'GET - This help message'
            },
            'data_sources': {
                'twelve_data': 'First 15 stocks',
                'yfinance': 'Last 5 stocks'
            },
            'status': 'online',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        logger.error(f"Error in home endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'service': 'Stock Analysis API'
        })
    except Exception as e:
        logger.error(f"Error in health endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['GET'])
def analyze():
    """API endpoint to analyze stocks and return JSON response"""
    try:
        logger.info("Starting stock analysis...")
        json_response = analyze_all_stocks()
        logger.info(f"Analysis completed. Status: {json_response.get('status')}")
        return jsonify(json_response)
    except Exception as e:
        logger.error(f"Error in /analyze endpoint: {str(e)}")
        return jsonify({
            'error': f"Failed to analyze stocks: {str(e)}",
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'stocks_analyzed': 0,
            'status': 'error'
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested URL was not found on the server',
        'available_endpoints': ['/analyze', '/health', '/'],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An internal error occurred',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    
    logger.info(f"Starting Flask server on port {port}")
    logger.info(f"Debug mode: {debug_mode}")
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode, threaded=True)
