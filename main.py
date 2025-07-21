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
# import os
# from threading import Lock, Thread
# import queue
# import anthropic
# import sqlite3
# from apscheduler.schedulers.background import BackgroundScheduler
# from apscheduler.triggers.cron import CronTrigger
# import pickle
# import threading

# warnings.filterwarnings('ignore')

# # ================= ENHANCED GLOBAL CONFIGURATION =================
# TWELVE_DATA_API_KEY = "73adc6cc7e43476e851dcf54c705aeeb"
# ALPHA_VANTAGE_API_KEY = "AK656KG03APJM5ZC"
# CLAUDE_API_KEY = "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"
# COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

# # Database configuration
# DATABASE_PATH = "stock_analysis.db"
# ANALYSIS_CACHE_FILE = "latest_analysis.json"

# RISK_FREE_RATE = 0.02
# MAX_WORKERS = 2
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
# TWELVE_DATA_RATE_LIMIT_PER_MIN = 8
# TWELVE_DATA_BATCH_SIZE = 3
# TWELVE_DATA_BATCH_SLEEP = 30
# TWELVE_DATA_RETRY_ATTEMPTS = 2
# TWELVE_DATA_RETRY_DELAY = 15
# ALPHA_VANTAGE_BATCH_SIZE = 3
# ALPHA_VANTAGE_DELAY = 12
# COINGECKO_BATCH_SIZE = 5
# COINGECKO_DELAY = 1.0

# # Global rate limiting
# rate_limit_lock = Lock()
# last_twelve_data_request = 0
# last_alpha_vantage_request = 0
# last_coingecko_request = 0
# request_count_twelve_data = 0
# request_count_alpha_vantage = 0

# # Background processing
# analysis_in_progress = False
# analysis_lock = threading.Lock()

# # Progress tracking
# progress_info = {
#     'current': 0,
#     'total': 35,  # Updated total
#     'percentage': 0,
#     'currentSymbol': '',
#     'stage': 'Initializing...',
#     'estimatedTimeRemaining': 0,
#     'startTime': None
# }
# progress_lock = threading.Lock()

# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[logging.StreamHandler()]
# )
# logger = logging.getLogger(__name__)

# # ================= FLASK APP SETUP =================
# app = Flask(__name__)

# # Enhanced CORS configuration with explicit settings
# CORS(app, 
#      origins=["*"],
#      methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
#      allow_headers=["Content-Type", "Authorization", "Accept", "Origin", "X-Requested-With"],
#      supports_credentials=False,
#      send_wildcard=True,
#      automatic_options=True)

# # Add explicit CORS headers for all responses
# @app.before_request
# def before_request():
#     if request.method == 'OPTIONS':
#         response = jsonify({'status': 'ok'})
#         response.headers.add('Access-Control-Allow-Origin', '*')
#         response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept,Origin,X-Requested-With')
#         response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
#         response.headers.add('Access-Control-Max-Age', '86400')
#         return response

# @app.after_request
# def after_request(response):
#     response.headers.add('Access-Control-Allow-Origin', '*')
#     response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept,Origin,X-Requested-With')
#     response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
#     response.headers.add('Access-Control-Allow-Credentials', 'false')
#     return response

# # Initialize Claude client
# claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY) if CLAUDE_API_KEY != "YOUR_CLAUDE_API_KEY" else None

# # Initialize scheduler
# scheduler = BackgroundScheduler()

# # ================= PROGRESS TRACKING FUNCTIONS =================
# def update_progress(current, total, symbol, stage):
#     """Update progress information"""
#     global progress_info
    
#     with progress_lock:
#         progress_info['current'] = current
#         progress_info['total'] = total
#         progress_info['percentage'] = (current / total) * 100 if total > 0 else 0
#         progress_info['currentSymbol'] = symbol
#         progress_info['stage'] = stage
        
#         # Calculate estimated time remaining
#         if progress_info['startTime'] and current > 0:
#             elapsed_time = time.time() - progress_info['startTime']
#             time_per_item = elapsed_time / current
#             remaining_items = total - current
#             progress_info['estimatedTimeRemaining'] = remaining_items * time_per_item
        
#         logger.info(f"Progress: {current}/{total} ({progress_info['percentage']:.1f}%) - {symbol} - {stage}")

# def reset_progress():
#     """Reset progress tracking"""
#     global progress_info
    
#     with progress_lock:
#         progress_info.update({
#             'current': 0,
#             'total': 35,  # Updated total
#             'percentage': 0,
#             'currentSymbol': '',
#             'stage': 'Initializing...',
#             'estimatedTimeRemaining': 0,
#             'startTime': time.time()
#         })

# def get_progress():
#     """Get current progress information"""
#     with progress_lock:
#         return progress_info.copy()

# # ================= DATABASE SETUP =================
# def init_database():
#     """Initialize SQLite database for persistent storage"""
#     conn = sqlite3.connect(DATABASE_PATH)
#     cursor = conn.cursor()
    
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS analysis_results (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             symbol TEXT NOT NULL,
#             market TEXT NOT NULL,
#             data_source TEXT NOT NULL,
#             analysis_data TEXT NOT NULL,
#             timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
#             UNIQUE(symbol) ON CONFLICT REPLACE
#         )
#     ''')
    
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS analysis_metadata (
#             id INTEGER PRIMARY KEY,
#             total_analyzed INTEGER,
#             success_rate REAL,
#             last_update DATETIME,
#             status TEXT,
#             processing_time_minutes REAL
#         )
#     ''')
    
#     conn.commit()
#     conn.close()
#     logger.info("Database initialized successfully")

# def save_analysis_to_db(results):
#     """Save analysis results to database"""
#     conn = sqlite3.connect(DATABASE_PATH)
#     cursor = conn.cursor()
    
#     try:
#         # Save individual stock results
#         for symbol, data in results.items():
#             if symbol not in ['timestamp', 'stocks_analyzed', 'status', 'data_sources', 'markets', 'processing_info']:
#                 cursor.execute('''
#                     INSERT OR REPLACE INTO analysis_results 
#                     (symbol, market, data_source, analysis_data) 
#                     VALUES (?, ?, ?, ?)
#                 ''', (
#                     symbol,
#                     data.get('market', 'Unknown'),
#                     data.get('data_source', 'Unknown'),
#                     json.dumps(data)
#                 ))
        
#         # Save metadata
#         cursor.execute('''
#             INSERT OR REPLACE INTO analysis_metadata 
#             (id, total_analyzed, success_rate, last_update, status, processing_time_minutes) 
#             VALUES (1, ?, ?, ?, ?, ?)
#         ''', (
#             results.get('stocks_analyzed', 0),
#             results.get('success_rate', 0),
#             datetime.now().isoformat(),
#             results.get('status', 'unknown'),
#             results.get('processing_time_minutes', 0)
#         ))
        
#         conn.commit()
#         logger.info(f"Saved {len([k for k in results.keys() if k not in ['timestamp', 'stocks_analyzed', 'status', 'data_sources', 'markets', 'processing_info']])} analysis results to database")
        
#     except Exception as e:
#         logger.error(f"Error saving to database: {str(e)}")
#         conn.rollback()
#     finally:
#         conn.close()

# def load_analysis_from_db():
#     """Load latest analysis results from database"""
#     conn = sqlite3.connect(DATABASE_PATH)
#     cursor = conn.cursor()
    
#     try:
#         # Get metadata
#         cursor.execute('SELECT * FROM analysis_metadata WHERE id = 1')
#         metadata = cursor.fetchone()
        
#         if not metadata:
#             return None
        
#         # Get all stock results
#         cursor.execute('SELECT symbol, analysis_data FROM analysis_results ORDER BY timestamp DESC')
#         stock_results = cursor.fetchall()
        
#         if not stock_results:
#             return None
        
#         # Reconstruct response format
#         response = {
#             'timestamp': metadata[3],  # last_update
#             'stocks_analyzed': metadata[1],  # total_analyzed
#             'success_rate': metadata[2],  # success_rate
#             'status': metadata[4],  # status
#             'processing_time_minutes': metadata[5],  # processing_time_minutes
#             'data_source': 'database_cache',
#             'markets': {'us_stocks': 0, 'nigerian_stocks': 0, 'crypto_assets': 0},
#             'data_sources': {'twelve_data_count': 0, 'coingecko_count': 0}
#         }
        
#         # Add stock data
#         for symbol, analysis_json in stock_results:
#             try:
#                 analysis_data = json.loads(analysis_json)
#                 response[symbol] = analysis_data
                
#                 # Count by market
#                 market = analysis_data.get('market', 'Unknown')
#                 if market == 'US':
#                     response['markets']['us_stocks'] += 1
#                 elif market == 'Nigerian':
#                     response['markets']['nigerian_stocks'] += 1
#                 elif market == 'Crypto':
#                     response['markets']['crypto_assets'] += 1
                
#                 # Count by data source
#                 data_source = analysis_data.get('data_source', 'Unknown')
#                 if data_source == 'twelve_data':
#                     response['data_sources']['twelve_data_count'] += 1
#                 elif data_source == 'coingecko':
#                     response['data_sources']['coingecko_count'] += 1
                    
#             except json.JSONDecodeError:
#                 logger.error(f"Error parsing analysis data for {symbol}")
#                 continue
        
#         logger.info(f"Loaded {len(stock_results)} analysis results from database")
#         return response
        
#     except Exception as e:
#         logger.error(f"Error loading from database: {str(e)}")
#         return None
#     finally:
#         conn.close()

# # ================= REDUCED STOCK CONFIGURATION =================
# def get_filtered_stocks():
#     """Get reduced list of popular stocks - 10 US + 15 Nigerian + 10 Crypto = 35 total"""
    
#     # Top 10 US Stocks - Most popular and liquid
#     us_stocks = [
#         "AAPL",    # Apple
#         "MSFT",    # Microsoft  
#         "GOOGL",   # Google
#         "TSLA",    # Tesla
#         "AMZN",    # Amazon
#         "NVDA",    # NVIDIA
#         "META",    # Meta (Facebook)
#         "NFLX",    # Netflix
#         "JPM",     # JPMorgan Chase
#         "V"        # Visa
#     ]
    
#     # Top 15 Nigerian Stocks - Most popular and liquid
#     nigerian_stocks = [
#         # Top Banks (5)
#         "ACCESS.NG",      # Access Bank
#         "GTCO.NG",        # Guaranty Trust Bank
#         "UBA.NG",         # United Bank for Africa
#         "ZENITHBANK.NG",  # Zenith Bank
#         "FBNH.NG",        # FBN Holdings
        
#         # Major Industrial/Cement (3)
#         "DANGCEM.NG",     # Dangote Cement
#         "BUACEMENT.NG",   # BUA Cement
#         "WAPCO.NG",       # Lafarge Africa
        
#         # Top Consumer Goods (3)
#         "DANGSUGAR.NG",   # Dangote Sugar
#         "NESTLE.NG",      # Nestle Nigeria
#         "UNILEVER.NG",    # Unilever Nigeria
        
#         # Oil & Gas (2)
#         "SEPLAT.NG",      # Seplat Energy
#         "TOTAL.NG",       # TotalEnergies Marketing
        
#         # Telecom & Others (2)
#         "MTNN.NG",        # MTN Nigeria
#         "TRANSCORP.NG"    # Transnational Corporation
#     ]
    
#     # Top 10 Cryptocurrencies - Highest market cap and most popular
#     crypto_stocks = [
#         "bitcoin",        # Bitcoin
#         "ethereum",       # Ethereum
#         "binancecoin",    # BNB
#         "solana",         # Solana
#         "cardano",        # Cardano
#         "avalanche-2",    # Avalanche
#         "polkadot",       # Polkadot
#         "chainlink",      # Chainlink
#         "polygon",        # Polygon
#         "litecoin"        # Litecoin
#     ]
    
#     return {
#         'twelve_data_us': us_stocks,
#         'twelve_data_nigerian': nigerian_stocks,
#         'coingecko_cryptos': crypto_stocks,
#         'us_stocks': us_stocks,
#         'nigerian_stocks': nigerian_stocks,
#         'crypto_stocks': crypto_stocks,
#         'total_count': len(us_stocks) + len(nigerian_stocks) + len(crypto_stocks)
#     }

# # ================= ENHANCED RATE LIMITING FUNCTIONS =================
# def wait_for_rate_limit_twelve_data():
#     """Optimized rate limiting for Twelve Data API"""
#     global last_twelve_data_request, request_count_twelve_data
    
#     with rate_limit_lock:
#         current_time = time.time()
        
#         if current_time - last_twelve_data_request > 60:
#             request_count_twelve_data = 0
#             last_twelve_data_request = current_time
        
#         if request_count_twelve_data >= TWELVE_DATA_RATE_LIMIT_PER_MIN:
#             sleep_time = 60 - (current_time - last_twelve_data_request)
#             if sleep_time > 0:
#                 logger.info(f"Rate limit reached for Twelve Data. Sleeping for {sleep_time:.1f} seconds...")
#                 time.sleep(sleep_time)
#                 request_count_twelve_data = 0
#                 last_twelve_data_request = time.time()
        
#         request_count_twelve_data += 1

# def wait_for_rate_limit_alpha_vantage():
#     """Rate limiting for Alpha Vantage API"""
#     global last_alpha_vantage_request, request_count_alpha_vantage
    
#     with rate_limit_lock:
#         current_time = time.time()
#         time_since_last = current_time - last_alpha_vantage_request
        
#         if time_since_last < ALPHA_VANTAGE_DELAY:
#             sleep_time = ALPHA_VANTAGE_DELAY - time_since_last
#             time.sleep(sleep_time)
        
#         last_alpha_vantage_request = time.time()
#         request_count_alpha_vantage += 1

# def wait_for_rate_limit_coingecko():
#     """Rate limiting for CoinGecko API"""
#     global last_coingecko_request
    
#     with rate_limit_lock:
#         current_time = time.time()
#         time_since_last = current_time - last_coingecko_request
        
#         if time_since_last < COINGECKO_DELAY:
#             sleep_time = COINGECKO_DELAY - time_since_last
#             time.sleep(sleep_time)
        
#         last_coingecko_request = time.time()

# # ================= ENHANCED DATA FETCHING =================
# def fetch_stock_data_twelve_with_retry(symbol, interval="1day", outputsize=100, max_retries=TWELVE_DATA_RETRY_ATTEMPTS):
#     """Enhanced Twelve Data fetching with multiple timeframes"""
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
            
#             logger.info(f"Fetching {symbol} ({interval}) from Twelve Data (attempt {attempt + 1}/{max_retries})")
            
#             response = requests.get(url, params=params, timeout=20)
#             response.raise_for_status()
            
#             data = response.json()
            
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
            
#             logger.info(f"Successfully fetched {len(df)} rows for {symbol} ({interval}) from Twelve Data")
#             return df
            
#         except Exception as e:
#             logger.error(f"Error fetching data from Twelve Data for {symbol} (attempt {attempt + 1}): {str(e)}")
#             if attempt < max_retries - 1:
#                 time.sleep(TWELVE_DATA_RETRY_DELAY)
#             else:
#                 return pd.DataFrame()
    
#     return pd.DataFrame()

# def fetch_crypto_data_coingecko(crypto_id, days=100):
#     """Fetch crypto data from CoinGecko"""
#     try:
#         wait_for_rate_limit_coingecko()
        
#         url = f"{COINGECKO_BASE_URL}/coins/{crypto_id}/market_chart"
#         params = {
#             'vs_currency': 'usd',
#             'days': days,
#             'interval': 'daily'
#         }
        
#         logger.info(f"Fetching {crypto_id} from CoinGecko")
        
#         response = requests.get(url, params=params, timeout=15)
#         response.raise_for_status()
        
#         data = response.json()
        
#         if 'prices' not in data:
#             logger.error(f"No price data for {crypto_id}")
#             return pd.DataFrame()
        
#         prices = data['prices']
#         volumes = data.get('total_volumes', [])
        
#         df_data = []
#         for i, price_point in enumerate(prices):
#             timestamp = pd.to_datetime(price_point[0], unit='ms')
#             price = price_point[1]
#             volume = volumes[i][1] if i < len(volumes) else 0
            
#             df_data.append({
#                 'datetime': timestamp,
#                 'open': price,
#                 'high': price,
#                 'low': price,
#                 'close': price,
#                 'volume': volume
#             })
        
#         df = pd.DataFrame(df_data)
#         df.set_index('datetime', inplace=True)
#         df.sort_index(inplace=True)
        
#         logger.info(f"Successfully fetched {len(df)} rows for {crypto_id} from CoinGecko")
#         return df
        
#     except Exception as e:
#         logger.error(f"Error fetching crypto data for {crypto_id}: {str(e)}")
#         return pd.DataFrame()

# def fetch_stock_data(symbol, interval="1day", outputsize=100, source="twelve_data"):
#     """Unified function to fetch data from multiple sources"""
#     if source == "twelve_data":
#         return fetch_stock_data_twelve_with_retry(symbol, interval, outputsize)
#     elif source == "coingecko":
#         return fetch_crypto_data_coingecko(symbol, outputsize)
#     else:
#         logger.error(f"Unknown data source: {source}")
#         return pd.DataFrame()

# # ================= EXISTING ANALYSIS FUNCTIONS =================
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
#     """Get fundamental data with crypto support"""
#     pe_ratios = {
#         # US Stocks
#         'AAPL': 28.5, 'MSFT': 32.1, 'TSLA': 45.2, 'GOOGL': 24.8, 'AMZN': 38.9,
#         'META': 22.7, 'NVDA': 55.3, 'JPM': 12.4, 'V': 34.2, 'NFLX': 35.8,
        
#         # Nigerian Stocks
#         'ACCESS.NG': 8.5, 'GTCO.NG': 12.3, 'UBA.NG': 7.4, 'ZENITHBANK.NG': 11.2,
#         'FBNH.NG': 6.2, 'DANGCEM.NG': 19.2, 'BUACEMENT.NG': 16.8, 'WAPCO.NG': 15.5,
#         'DANGSUGAR.NG': 18.5, 'NESTLE.NG': 35.8, 'UNILEVER.NG': 28.4,
#         'SEPLAT.NG': 14.2, 'TOTAL.NG': 16.8, 'MTNN.NG': 22.1, 'TRANSCORP.NG': 12.5,
        
#         # Cryptos
#         'bitcoin': 0, 'ethereum': 0, 'binancecoin': 0, 'solana': 0, 'cardano': 0
#     }
    
#     is_nigerian = symbol.endswith('.NG')
#     is_crypto = symbol in ['bitcoin', 'ethereum', 'binancecoin', 'solana', 'cardano', 'avalanche-2', 'polkadot', 'chainlink', 'polygon', 'litecoin']
    
#     if is_crypto:
#         return {
#             'PE_Ratio': 0,
#             'Market_Cap_Rank': random.randint(1, 50),
#             'Adoption_Score': random.uniform(0.6, 0.95),
#             'Technology_Score': random.uniform(0.7, 0.98)
#         }
#     else:
#         base_pe = pe_ratios.get(symbol, 12.0 if is_nigerian else 20.0)
#         return {
#             'PE_Ratio': base_pe,
#             'EPS': random.uniform(2.0 if is_nigerian else 5.0, 8.0 if is_nigerian else 15.0),
#             'Revenue_Growth': random.uniform(0.03 if is_nigerian else 0.05, 0.15 if is_nigerian else 0.25),
#             'Net_Income_Growth': random.uniform(0.02 if is_nigerian else 0.03, 0.12 if is_nigerian else 0.20)
#         }

# def get_market_sentiment(symbol):
#     """Get market sentiment with crypto support"""
#     sentiment_scores = {
#         # US Stocks
#         'AAPL': 0.75, 'MSFT': 0.80, 'TSLA': 0.60, 'GOOGL': 0.70, 'AMZN': 0.65,
#         'META': 0.55, 'NVDA': 0.85, 'JPM': 0.60, 'V': 0.75, 'NFLX': 0.65,
        
#         # Nigerian top stocks
#         'DANGCEM.NG': 0.68, 'GTCO.NG': 0.72, 'ZENITHBANK.NG': 0.65, 'UBA.NG': 0.63,
#         'ACCESS.NG': 0.61, 'NESTLE.NG': 0.70, 'UNILEVER.NG': 0.66, 'MTNN.NG': 0.74,
        
#         # Major Cryptos
#         'bitcoin': 0.78, 'ethereum': 0.82, 'binancecoin': 0.65, 'solana': 0.75, 'cardano': 0.58
#     }
    
#     is_nigerian = symbol.endswith('.NG')
#     is_crypto = symbol in ['bitcoin', 'ethereum', 'binancecoin', 'solana', 'cardano', 'avalanche-2', 'polkadot', 'chainlink', 'polygon', 'litecoin']
    
#     if is_crypto:
#         default_sentiment = 0.6
#     elif is_nigerian:
#         default_sentiment = 0.45
#     else:
#         default_sentiment = 0.5
    
#     return sentiment_scores.get(symbol, default_sentiment)

# def generate_smc_signals(chart_patterns, indicators, confluence, waves, fundamentals, sentiment):
#     """Generate trading signals with enhanced logic"""
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
        
#         if 'PE_Ratio' in fundamentals:
#             pe_ratio = fundamentals['PE_Ratio']
#             if pe_ratio > 0:
#                 if pe_ratio < 15:
#                     signal_score += 0.5
#                 elif pe_ratio > 30:
#                     signal_score -= 0.5
#         elif 'Market_Cap_Rank' in fundamentals:
#             if fundamentals['Market_Cap_Rank'] <= 10:
#                 signal_score += 0.3
#             if fundamentals['Adoption_Score'] > 0.8:
#                 signal_score += 0.2
        
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

# # ================= HIERARCHICAL ANALYSIS SYSTEM =================
# def analyze_stock_hierarchical(symbol, data_source="twelve_data"):
#     """Analyze stock with hierarchical timeframe dependency"""
#     try:
#         logger.info(f"Starting hierarchical analysis for {symbol} using {data_source}")
        
#         timeframes = {
#             'monthly': ('1month', 24),
#             'weekly': ('1week', 52),
#             'daily': ('1day', 100),
#             '4hour': ('4h', 168)
#         }
        
#         timeframe_data = {}
#         for tf_name, (interval, size) in timeframes.items():
#             if data_source == "coingecko":
#                 data = fetch_stock_data(symbol, "1day", size * 7, data_source)
#                 if not data.empty and tf_name != 'daily':
#                     if tf_name == 'monthly':
#                         data = data.resample('M').agg({
#                             'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
#                         }).dropna()
#                     elif tf_name == 'weekly':
#                         data = data.resample('W').agg({
#                             'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
#                         }).dropna()
#                     elif tf_name == '4hour':
#                         pass
#             else:
#                 data = fetch_stock_data(symbol, interval, size, data_source)
            
#             if not data.empty:
#                 timeframe_data[tf_name] = data
        
#         if not timeframe_data:
#             logger.error(f"No data available for {symbol}")
#             return None
        
#         analyses = {}
#         for tf_name, data in timeframe_data.items():
#             analysis = analyze_timeframe_enhanced(data, symbol, tf_name.upper())
#             if analysis:
#                 analyses[f"{tf_name.upper()}_TIMEFRAME"] = analysis
        
#         final_analysis = apply_hierarchical_logic(analyses, symbol)
        
#         result = {
#             symbol: {
#                 'data_source': data_source,
#                 'market': 'Crypto' if data_source == 'coingecko' else ('Nigerian' if symbol.endswith('.NG') else 'US'),
#                 **final_analysis
#             }
#         }
        
#         logger.info(f"Successfully analyzed {symbol} with hierarchical logic")
#         return result
        
#     except Exception as e:
#         logger.error(f"Error analyzing {symbol}: {str(e)}")
#         return None

# def apply_hierarchical_logic(analyses, symbol):
#     """Apply hierarchical logic where daily depends on weekly/monthly"""
#     try:
#         monthly = analyses.get('MONTHLY_TIMEFRAME')
#         weekly = analyses.get('WEEKLY_TIMEFRAME')
#         daily = analyses.get('DAILY_TIMEFRAME')
#         four_hour = analyses.get('4HOUR_TIMEFRAME')
        
#         if daily and weekly and monthly:
#             monthly_weight = 0.4
#             weekly_weight = 0.3
#             daily_weight = 0.2
#             four_hour_weight = 0.1
            
#             monthly_conf = monthly['CONFIDENCE_SCORE'] * monthly_weight
#             weekly_conf = weekly['CONFIDENCE_SCORE'] * weekly_weight
#             daily_conf = daily['CONFIDENCE_SCORE'] * daily_weight
#             four_hour_conf = four_hour['CONFIDENCE_SCORE'] * four_hour_weight if four_hour else 0
            
#             if monthly['VERDICT'] in ['Strong Buy', 'Buy'] and weekly['VERDICT'] in ['Strong Buy', 'Buy']:
#                 if daily['VERDICT'] in ['Sell', 'Strong Sell']:
#                     daily['VERDICT'] = 'Buy'
#                     daily['DETAILS']['individual_verdicts']['hierarchy_override'] = 'Monthly/Weekly Bullish Override'
#             elif monthly['VERDICT'] in ['Strong Sell', 'Sell'] and weekly['VERDICT'] in ['Strong Sell', 'Sell']:
#                 if daily['VERDICT'] in ['Buy', 'Strong Buy']:
#                     daily['VERDICT'] = 'Sell'
#                     daily['DETAILS']['individual_verdicts']['hierarchy_override'] = 'Monthly/Weekly Bearish Override'
            
#             daily['CONFIDENCE_SCORE'] = round(monthly_conf + weekly_conf + daily_conf + four_hour_conf, 2)
#             daily['ACCURACY'] = min(95, max(60, abs(daily['CONFIDENCE_SCORE']) * 15 + 70))
        
#         return analyses
        
#     except Exception as e:
#         logger.error(f"Error in hierarchical logic for {symbol}: {str(e)}")
#         return analyses

# def analyze_timeframe_enhanced(data, symbol, timeframe):
#     """Enhanced timeframe analysis with crypto support"""
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
        
#         if 'PE_Ratio' in fundamentals and fundamentals['PE_Ratio'] > 0:
#             pe_ratio = fundamentals['PE_Ratio']
#             fundamental_verdict = "Undervalued" if pe_ratio < 15 else "Overvalued" if pe_ratio > 25 else "Fair Value"
#         else:
#             fundamental_verdict = "Strong Fundamentals" if fundamentals.get('Adoption_Score', 0.5) > 0.8 else "Weak Fundamentals"
        
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
#                 'fundamentals': fundamentals,
#                 'sentiment_analysis': {
#                     'score': round(sentiment, 2),
#                     'interpretation': sentiment_verdict,
#                     'market_mood': "Optimistic" if sentiment > 0.7 else "Pessimistic" if sentiment < 0.3 else "Cautious"
#                 },
#                 'cycle_analysis': cycle_analysis,
#                 'trading_parameters': {
#                     'position_size': '5% of portfolio' if 'Strong' in signal else '3% of portfolio',
#                     'timeframe': f'{timeframe} - 2-4 weeks' if 'Buy' in signal else f'{timeframe} - 1-2 weeks',
#                     'risk_level': 'Medium' if 'Buy' in signal else 'High' if 'Sell' in signal else 'Low'
#                 }
#             }
#         }
        
#         return timeframe_analysis
        
#     except Exception as e:
#         logger.error(f"Error analyzing {timeframe} timeframe for {symbol}: {str(e)}")
#         return None

# # ================= CLAUDE AI INTEGRATION =================
# def generate_ai_analysis(symbol, stock_data):
#     """Generate detailed AI analysis using Claude"""
#     if not claude_client:
#         return {
#             'error': 'Claude API not configured',
#             'message': 'Please configure CLAUDE_API_KEY to use AI analysis'
#         }
    
#     try:
#         context = f"""
#         Stock Symbol: {symbol}
#         Current Analysis Data:
#         - Current Price: ${stock_data.get('DAILY_TIMEFRAME', {}).get('PRICE', 'N/A')}
#         - Verdict: {stock_data.get('DAILY_TIMEFRAME', {}).get('VERDICT', 'N/A')}
#         - Confidence Score: {stock_data.get('DAILY_TIMEFRAME', {}).get('CONFIDENCE_SCORE', 'N/A')}
#         - Market: {stock_data.get('market', 'Unknown')}
        
#         Technical Indicators:
#         - RSI: {stock_data.get('DAILY_TIMEFRAME', {}).get('DETAILS', {}).get('technical_indicators', {}).get('rsi', 'N/A')}
#         - ADX: {stock_data.get('DAILY_TIMEFRAME', {}).get('DETAILS', {}).get('technical_indicators', {}).get('adx', 'N/A')}
#         - Cycle Phase: {stock_data.get('DAILY_TIMEFRAME', {}).get('DETAILS', {}).get('technical_indicators', {}).get('cycle_phase', 'N/A')}
        
#         Individual Verdicts:
#         {json.dumps(stock_data.get('DAILY_TIMEFRAME', {}).get('DETAILS', {}).get('individual_verdicts', {}), indent=2)}
        
#         Patterns Detected:
#         {json.dumps(stock_data.get('DAILY_TIMEFRAME', {}).get('DETAILS', {}).get('patterns', {}), indent=2)}
#         """
        
#         prompt = f"""
#         You are an expert financial analyst with deep knowledge of technical analysis, fundamental analysis, and market psychology. 
        
#         Based on the following stock analysis data for {symbol}, provide a comprehensive, detailed analysis that includes:
        
#         1. **Executive Summary**: A clear, concise overview of the current situation
#         2. **Technical Analysis Deep Dive**: Detailed interpretation of the technical indicators and patterns
#         3. **Market Context**: How this stock fits within current market conditions
#         4. **Risk Assessment**: Detailed risk factors and mitigation strategies
#         5. **Trading Strategy**: Specific entry/exit strategies with reasoning
#         6. **Timeline Expectations**: Short, medium, and long-term outlook
#         7. **Key Catalysts**: What events or factors could change the analysis
#         8. **Alternative Scenarios**: Bull case, bear case, and most likely scenario
        
#         Context Data:
#         {context}
        
#         Please provide a thorough, professional analysis that a serious trader or investor would find valuable. 
#         Use clear, actionable language and explain your reasoning for each conclusion.
#         """
        
#         response = claude_client.messages.create(
#             model="claude-3-sonnet-20240229",
#             max_tokens=2000,
#             temperature=0.3,
#             messages=[
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ]
#         )
        
#         # Fix: response.content[0] may not have a .text attribute; use .get("text", "") if it's a dict, or str() fallback
#         analysis_text = ""
#         if hasattr(response, "content") and isinstance(response.content, list) and len(response.content) > 0:
#             block = response.content[0]
#             # Safely extract 'text' from block if possible, else fallback to str(block)
#             analysis_text = ""
#             if isinstance(block, dict) and "text" in block:
#                 analysis_text = block["text"]
#             elif hasattr(block, "__dict__") and "text" in block.__dict__:
#                 analysis_text = block.__dict__["text"]
#             else:
#                 analysis_text = str(block)
#         else:
#             analysis_text = "No analysis returned from Claude API."

#         return {
#             'analysis': analysis_text,
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#             'model': 'claude-3-sonnet',
#             'symbol': symbol
#         }
        
#     except Exception as e:
#         logger.error(f"Error generating AI analysis for {symbol}: {str(e)}")
#         return {
#             'error': 'Failed to generate AI analysis',
#             'message': str(e)
#         }

# # ================= BACKGROUND PROCESSING =================
# def analyze_all_stocks_background():
#     """Background analysis function that runs at 5pm daily"""
#     global analysis_in_progress
    
#     with analysis_lock:
#         if analysis_in_progress:
#             logger.info("Analysis already in progress, skipping background run")
#             return
        
#         analysis_in_progress = True
    
#     try:
#         logger.info("Starting scheduled background analysis at 5pm")
#         start_time = time.time()
        
#         result = analyze_all_stocks_optimized()
        
#         if result and result.get('status') == 'success':
#             # Save to database
#             save_analysis_to_db(result)
            
#             # Calculate processing time
#             processing_time = (time.time() - start_time) / 60
#             result['processing_time_minutes'] = round(processing_time, 2)
            
#             logger.info(f"Background analysis completed successfully in {processing_time:.2f} minutes")
#             logger.info(f"Analyzed {result.get('stocks_analyzed', 0)} assets")
#         else:
#             logger.error("Background analysis failed")
            
#     except Exception as e:
#         logger.error(f"Error in background analysis: {str(e)}")
#     finally:
#         with analysis_lock:
#             analysis_in_progress = False

# def analyze_all_stocks_optimized():
#     """Optimized stock analysis with database persistence and progress tracking"""
#     try:
#         reset_progress()
        
#         stock_config = get_filtered_stocks()
#         twelve_data_us = stock_config['twelve_data_us']
#         twelve_data_nigerian = stock_config['twelve_data_nigerian']
#         coingecko_cryptos = stock_config['coingecko_cryptos']
        
#         results = {}
#         total_stocks = len(twelve_data_us) + len(twelve_data_nigerian) + len(coingecko_cryptos)
#         processed_count = 0
        
#         logger.info(f"Starting optimized analysis of {total_stocks} assets")
#         logger.info(f"US stocks: {len(twelve_data_us)}, Nigerian: {len(twelve_data_nigerian)}, Crypto: {len(coingecko_cryptos)}")
        
#         update_progress(0, total_stocks, 'Initializing...', 'Starting analysis process')
        
#         # Process US stocks
#         if twelve_data_us:
#             batch_size = TWELVE_DATA_BATCH_SIZE
#             num_batches = math.ceil(len(twelve_data_us) / batch_size)
            
#             for batch_idx in range(num_batches):
#                 batch_start = batch_idx * batch_size
#                 batch_end = min((batch_idx + 1) * batch_size, len(twelve_data_us))
#                 batch_symbols = twelve_data_us[batch_start:batch_end]
                
#                 update_progress(processed_count, total_stocks, f'US Batch {batch_idx+1}', f'Processing US stocks batch {batch_idx+1}/{num_batches}')
#                 logger.info(f"Processing US batch {batch_idx+1}/{num_batches}: {batch_symbols}")
                
#                 for symbol in batch_symbols:
#                     try:
#                         update_progress(processed_count, total_stocks, symbol, f'Analyzing US stock: {symbol}')
#                         result = analyze_stock_hierarchical(symbol, "twelve_data")
#                         if result:
#                             results.update(result)
#                             processed_count += 1
#                             logger.info(f"✓ {symbol} ({processed_count}/{total_stocks}) - US Stock")
#                         else:
#                             logger.warning(f"✗ Failed to process {symbol} (US)")
#                     except Exception as e:
#                         logger.error(f"✗ Error processing {symbol} (US): {str(e)}")
                
#                 if batch_idx < num_batches - 1:
#                     update_progress(processed_count, total_stocks, 'Rate Limiting', f'Sleeping {TWELVE_DATA_BATCH_SLEEP}s for rate limits...')
#                     logger.info(f"Sleeping {TWELVE_DATA_BATCH_SLEEP}s...")
#                     time.sleep(TWELVE_DATA_BATCH_SLEEP)
        
#         # Process Nigerian stocks
#         if twelve_data_nigerian:
#             batch_size = TWELVE_DATA_BATCH_SIZE
#             num_batches = math.ceil(len(twelve_data_nigerian) / batch_size)
            
#             for batch_idx in range(num_batches):
#                 batch_start = batch_idx * batch_size
#                 batch_end = min((batch_idx + 1) * batch_size, len(twelve_data_nigerian))
#                 batch_symbols = twelve_data_nigerian[batch_start:batch_end]
                
#                 update_progress(processed_count, total_stocks, f'Nigerian Batch {batch_idx+1}', f'Processing Nigerian stocks batch {batch_idx+1}/{num_batches}')
#                 logger.info(f"Processing Nigerian batch {batch_idx+1}/{num_batches}: {batch_symbols}")
                
#                 for symbol in batch_symbols:
#                     try:
#                         update_progress(processed_count, total_stocks, symbol, f'Analyzing Nigerian stock: {symbol}')
#                         result = analyze_stock_hierarchical(symbol, "twelve_data")
#                         if result:
#                             results.update(result)
#                             processed_count += 1
#                             logger.info(f"✓ {symbol} ({processed_count}/{total_stocks}) - Nigerian Stock")
#                         else:
#                             logger.warning(f"✗ Failed to process {symbol} (Nigerian)")
#                     except Exception as e:
#                         logger.error(f"✗ Error processing {symbol} (Nigerian): {str(e)}")
                
#                 if batch_idx < num_batches - 1:
#                     update_progress(processed_count, total_stocks, 'Rate Limiting', f'Sleeping {TWELVE_DATA_BATCH_SLEEP}s for rate limits...')
#                     logger.info(f"Sleeping {TWELVE_DATA_BATCH_SLEEP}s...")
#                     time.sleep(TWELVE_DATA_BATCH_SLEEP)
        
#         # Process Crypto assets
#         if coingecko_cryptos: 
#             batch_size = COINGECKO_BATCH_SIZE
#             num_batches = math.ceil(len(coingecko_cryptos) / batch_size)
            
#             for batch_idx in range(num_batches):
#                 batch_start = batch_idx * batch_size
#                 batch_end = min((batch_idx + 1) * batch_size, len(coingecko_cryptos))
#                 batch_symbols = coingecko_cryptos[batch_start:batch_end]
                
#                 update_progress(processed_count, total_stocks, f'Crypto Batch {batch_idx+1}', f'Processing crypto assets batch {batch_idx+1}/{num_batches}')
#                 logger.info(f"Processing Crypto batch {batch_idx+1}/{num_batches}: {batch_symbols}")
                
#                 for symbol in batch_symbols:
#                     try:
#                         update_progress(processed_count, total_stocks, symbol, f'Analyzing crypto: {symbol}')
#                         result = analyze_stock_hierarchical(symbol, "coingecko")
#                         if result:
#                             results.update(result)
#                             processed_count += 1
#                             logger.info(f"✓ {symbol} ({processed_count}/{total_stocks}) - Crypto")
#                         else:
#                             logger.warning(f"✗ Failed to process {symbol} (Crypto)")
#                     except Exception as e:
#                         logger.error(f"✗ Error processing {symbol} (Crypto): {str(e)}")
                
#                 if batch_idx < num_batches - 1:
#                     update_progress(processed_count, total_stocks, 'Rate Limiting', f'Sleeping {COINGECKO_DELAY * 2}s for rate limits...')
#                     logger.info(f"Sleeping {COINGECKO_DELAY * 2}s...")
#                     time.sleep(COINGECKO_DELAY * 2)
        
#         # Calculate statistics
#         us_stocks_count = len([k for k, v in results.items() if v.get('market') == 'US'])
#         nigerian_stocks_count = len([k for k, v in results.items() if v.get('market') == 'Nigerian'])
#         crypto_count = len([k for k, v in results.items() if v.get('market') == 'Crypto'])
        
#         update_progress(processed_count, total_stocks, 'Finalizing', 'Analysis complete - finalizing results')
        
#         response = {
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#             'stocks_analyzed': len(results),
#             'total_requested': total_stocks,
#             'success_rate': round((len(results) / total_stocks) * 100, 1) if total_stocks > 0 else 0,
#             'status': 'success' if results else 'no_data',
#             'data_sources': {
#                 'twelve_data_count': len([k for k, v in results.items() if v.get('data_source') == 'twelve_data']),
#                 'coingecko_count': len([k for k, v in results.items() if v.get('data_source') == 'coingecko'])
#             },
#             'markets': {
#                 'us_stocks': us_stocks_count,
#                 'nigerian_stocks': nigerian_stocks_count,
#                 'crypto_assets': crypto_count
#             },
#             'processing_info': {
#                 'hierarchical_analysis': True,
#                 'timeframes_analyzed': ['monthly', 'weekly', 'daily', '4hour'],
#                 'ai_analysis_available': claude_client is not None,
#                 'background_processing': True,
#                 'daily_auto_refresh': '5:00 PM'
#             },
#             **results
#         }
        
#         logger.info(f"Optimized analysis complete. Processed {len(results)}/{total_stocks} assets successfully.")
#         logger.info(f"US: {us_stocks_count}, Nigerian: {nigerian_stocks_count}, Crypto: {crypto_count}")
#         return response
        
#     except Exception as e:
#         logger.error(f"Error in analyze_all_stocks_optimized: {str(e)}")
#         return {
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#             'stocks_analyzed': 0,
#             'status': 'error',
#             'error': str(e)
#         }

# # ================= FLASK ROUTES =================
# @app.route('/', methods=['GET'])
# def home():
#     """Enhanced home endpoint with persistent data info"""
#     try:
#         stock_config = get_filtered_stocks()
        
#         # Check if we have cached data
#         cached_data = load_analysis_from_db()
#         has_cached_data = cached_data is not None
        
#         return jsonify({
#             'message': 'Enhanced Multi-Asset Analysis API v4.0 - Reduced Dataset',
#             'version': '4.0 - Persistent Data + Background Processing + Progress Tracking',
#             'endpoints': {
#                 '/analyze': 'GET - Get latest analysis (from cache or trigger new)',
#                 '/analyze/fresh': 'GET - Force fresh analysis (manual refresh)',
#                 '/progress': 'GET - Get current analysis progress',
#                 '/ai-analysis': 'POST - Get detailed AI analysis for specific symbol',
#                 '/health': 'GET - Health check',
#                 '/assets': 'GET - List all available assets',
#                 '/': 'GET - This help message'
#             },
#             'markets': {
#                 'us_stocks': len(stock_config['us_stocks']),
#                 'nigerian_stocks': len(stock_config['nigerian_stocks']),
#                 'crypto_assets': len(stock_config['crypto_stocks']),
#                 'total_assets': stock_config['total_count']
#             },
#             'features': {
#                 'hierarchical_analysis': True,
#                 'timeframes': ['monthly', 'weekly', 'daily', '4hour'],
#                 'ai_analysis': claude_client is not None,
#                 'persistent_storage': True,
#                 'background_processing': True,
#                 'progress_tracking': True,
#                 'daily_auto_refresh': '5:00 PM',
#                 'data_sources': ['twelve_data', 'coingecko'],
#                 'optimized_processing': True,
#                 'reduced_dataset': True
#             },
#             'data_status': {
#                 'has_cached_data': has_cached_data,
#                 'last_update': cached_data.get('timestamp') if cached_data else None,
#                 'cached_assets': cached_data.get('stocks_analyzed') if cached_data else 0,
#                 'analysis_in_progress': analysis_in_progress
#             },
#             'status': 'online',
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         })
#     except Exception as e:
#         logger.error(f"Error in home endpoint: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/progress', methods=['GET'])
# def progress():
#     """Get current analysis progress"""
#     try:
#         current_progress = get_progress()
#         return jsonify(current_progress)
#     except Exception as e:
#         logger.error(f"Error in progress endpoint: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/assets', methods=['GET'])
# def list_assets():
#     """List all available assets"""
#     try:
#         stock_config = get_filtered_stocks()
#         return jsonify({
#             'us_stocks': stock_config['us_stocks'],
#             'nigerian_stocks': stock_config['nigerian_stocks'],
#             'crypto_assets': stock_config['crypto_stocks'],
#             'data_source_distribution': {
#                 'twelve_data_us': stock_config['twelve_data_us'],
#                 'twelve_data_nigerian': stock_config['twelve_data_nigerian'],
#                 'coingecko_cryptos': stock_config['coingecko_cryptos']
#             },
#             'total_count': stock_config['total_count'],
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         })
#     except Exception as e:
#         logger.error(f"Error in assets endpoint: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/health', methods=['GET'])
# def health():
#     """Health check endpoint"""
#     try:
#         cached_data = load_analysis_from_db()
#         return jsonify({
#             'status': 'healthy',
#             'version': '4.0 - Reduced Dataset',
#             'markets': ['US', 'Nigerian', 'Crypto'],
#             'features': {
#                 'hierarchical_analysis': True,
#                 'ai_analysis': claude_client is not None,
#                 'optimized_processing': True,
#                 'persistent_storage': True,
#                 'background_processing': True,
#                 'progress_tracking': True,
#                 'reduced_dataset': True
#             },
#             'data_status': {
#                 'has_cached_data': cached_data is not None,
#                 'analysis_in_progress': analysis_in_progress,
#                 'last_update': cached_data.get('timestamp') if cached_data else None
#             },
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#             'service': 'Multi-Asset Analysis API with Progress Tracking - Reduced Dataset'
#         })
#     except Exception as e:
#         logger.error(f"Error in health endpoint: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/analyze', methods=['GET'])
# def analyze():
#     """Get latest analysis - from cache if available, otherwise return cached data"""
#     try:
#         # First, try to load from cache
#         cached_data = load_analysis_from_db()
        
#         if cached_data:
#             logger.info(f"Returning cached analysis data from {cached_data.get('timestamp')}")
#             cached_data['data_source'] = 'database_cache'
#             cached_data['note'] = 'This is cached data. Use /analyze/fresh for new analysis.'
#             return jsonify(cached_data)
#         else:
#             # No cached data, run fresh analysis
#             logger.info("No cached data found, running fresh analysis...")
#             json_response = analyze_all_stocks_optimized()
            
#             if json_response and json_response.get('status') ==  'success':
#                 save_analysis_to_db(json_response)
            
#             logger.info(f"Fresh analysis completed. Status: {json_response.get('status')}")
#             return jsonify(json_response)
            
#     except Exception as e:
#         logger.error(f"Error in /analyze endpoint: {str(e)}")
#         return jsonify({
#             'error': f"Failed to get analysis: {str(e)}",
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#             'stocks_analyzed': 0,
#             'status': 'error'
#         }), 500

# @app.route('/analyze/fresh', methods=['GET'])
# def analyze_fresh():
#     """Force fresh analysis (manual refresh)"""
#     try:
#         logger.info("Starting manual fresh analysis...")
#         start_time = time.time()
        
#         json_response = analyze_all_stocks_optimized()
        
#         if json_response and json_response.get('status') == 'success':
#             # Calculate processing time
#             processing_time = (time.time() - start_time) / 60
#             json_response['processing_time_minutes'] = round(processing_time, 2)
            
#             # Save to database
#             save_analysis_to_db(json_response)
            
#             logger.info(f"Fresh analysis completed in {processing_time:.2f} minutes")
        
#         logger.info(f"Fresh analysis completed. Status: {json_response.get('status')}")
#         return jsonify(json_response)
        
#     except Exception as e:
#         logger.error(f"Error in /analyze/fresh endpoint: {str(e)}")
#         return jsonify({
#             'error': f"Failed to run fresh analysis: {str(e)}",
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#             'stocks_analyzed': 0,
#             'status': 'error'
#         }), 500

# @app.route('/ai-analysis', methods=['POST'])
# def ai_analysis():
#     """AI analysis endpoint using Claude"""
#     try:
#         data = request.get_json()
#         if not data or 'symbol' not in data:
#             return jsonify({
#                 'error': 'Missing symbol parameter',
#                 'message': 'Please provide a symbol in the request body'
#             }), 400
        
#         symbol = data['symbol'].upper()
        
#         logger.info(f"Generating AI analysis for {symbol}")
        
#         # Determine data source based on symbol
#         stock_config = get_filtered_stocks()
#         if symbol in stock_config['crypto_stocks']:
#             data_source = "coingecko"
#         else:
#             data_source = "twelve_data"
        
#         # Try to get from cache first
#         cached_data = load_analysis_from_db()
#         stock_analysis = None
        
#         if cached_data and symbol in cached_data:
#             stock_analysis = {symbol: cached_data[symbol]}
#             logger.info(f"Using cached data for AI analysis of {symbol}")
#         else:
#             # Get fresh analysis for this symbol
#             stock_analysis = analyze_stock_hierarchical(symbol, data_source)
        
#         if not stock_analysis or symbol not in stock_analysis:
#             return jsonify({
#                 'error': 'Symbol not found or analysis failed',
#                 'message': f'Could not analyze {symbol}. Please check the symbol and try again.'
#             }), 404
        
#         # Generate AI analysis
#         ai_result = generate_ai_analysis(symbol, stock_analysis[symbol])
        
#         return jsonify({
#             'symbol': symbol,
#             'ai_analysis': ai_result,
#             'technical_analysis': stock_analysis[symbol],
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         })
        
#     except Exception as e:
#         logger.error(f"Error in /ai-analysis endpoint: {str(e)}")
#         return jsonify({
#             'error': f"Failed to generate AI analysis: {str(e)}",
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         }), 500

# @app.errorhandler(404)
# def not_found(error):
#     return jsonify({
#         'error': 'Endpoint not found',
#         'message': 'The requested URL was not found on the server',
#         'available_endpoints': ['/analyze', '/analyze/fresh', '/progress', '/ai-analysis', '/health', '/assets', '/'],
#         'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     }), 404

# @app.errorhandler(500)
# def internal_error(error):
#     return jsonify({
#         'error': 'Internal server error',
#         'message': 'An internal error occurred',
#         'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     }), 500

# # ================= STARTUP AND SCHEDULER =================
# def start_scheduler():
#     """Start the background scheduler for daily 5pm analysis"""
#     try:
#         # Schedule daily analysis at 5:00 PM
#         scheduler.add_job(
#             func=analyze_all_stocks_background,
#             trigger=CronTrigger(hour=17, minute=0),  # 5:00 PM
#             id='daily_analysis',
#             name='Daily Stock Analysis at 5PM',
#             replace_existing=True
#         )
        
#         scheduler.start()
#         logger.info("Background scheduler started - Daily analysis at 5:00 PM")
        
#     except Exception as e:
#         logger.error(f"Error starting scheduler: {str(e)}")

# if __name__ == "__main__":
#     # Initialize database
#     init_database()
    
#     # Start background scheduler
#     start_scheduler()
    
#     port = int(os.environ.get("PORT", 5000))
#     debug_mode = os.environ.get("FLASK_ENV") == "development"
    
#     logger.info(f"Starting Enhanced Multi-Asset Analysis API v4.0 - Reduced Dataset on port {port}")
#     logger.info(f"Debug mode: {debug_mode}")
#     logger.info(f"Total assets configured: {get_filtered_stocks()['total_count']}")
#     logger.info(f"AI Analysis available: {claude_client is not None}")
#     logger.info("Features: Persistent Storage + Background Processing + Progress Tracking + Daily 5PM Auto-Refresh + Reduced Dataset")
    
#     try:
#         app.run(host='0.0.0.0', port=port, debug=debug_mode, threaded=True)
#     finally:
#         # Cleanup scheduler on shutdown
#         if scheduler.running:
#             scheduler.shutdown()




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
from flask_cors import CORS, cross_origin
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
ALPHA_VANTAGE_API_KEY = "AK656KG03APJM5ZC"
CLAUDE_API_KEY = "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"
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

# Enhanced Rate limiting configuration
TWELVE_DATA_RATE_LIMIT_PER_MIN = 8
TWELVE_DATA_BATCH_SIZE = 3
TWELVE_DATA_BATCH_SLEEP = 30
TWELVE_DATA_RETRY_ATTEMPTS = 2
TWELVE_DATA_RETRY_DELAY = 15
ALPHA_VANTAGE_BATCH_SIZE = 3
ALPHA_VANTAGE_DELAY = 12
COINGECKO_BATCH_SIZE = 5
COINGECKO_DELAY = 1.0

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

# Progress tracking
progress_info = {
    'current': 0,
    'total': 35,
    'percentage': 0,
    'currentSymbol': '',
    'stage': 'Initializing...',
    'estimatedTimeRemaining': 0,
    'startTime': None
}
progress_lock = threading.Lock()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# ================= FLASK APP SETUP =================
app = Flask(__name__)

# FIXED CORS Configuration
CORS(app, 
     origins=["https://my-stocks-s2at.onrender.com", "http://localhost:3000", "http://localhost:5173"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization", "Accept", "Origin", "X-Requested-With"],
     supports_credentials=False,
     max_age=86400)

# Additional CORS headers for all responses
@app.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    if origin in ["https://my-stocks-s2at.onrender.com", "http://localhost:3000", "http://localhost:5173"]:
        response.headers['Access-Control-Allow-Origin'] = origin
    else:
        response.headers['Access-Control-Allow-Origin'] = "https://my-stocks-s2at.onrender.com"
    
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept, Origin, X-Requested-With'
    response.headers['Access-Control-Max-Age'] = '86400'
    response.headers['Vary'] = 'Origin'
    
    logger.debug(f"CORS headers added for origin: {origin}")
    return response

# Handle preflight requests
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({'status': 'ok'})
        origin = request.headers.get('Origin')
        if origin in ["https://my-stocks-s2at.onrender.com", "http://localhost:3000", "http://localhost:5173"]:
            response.headers['Access-Control-Allow-Origin'] = origin
        else:
            response.headers['Access-Control-Allow-Origin'] = "https://my-stocks-s2at.onrender.com"
        
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept, Origin, X-Requested-With'
        response.headers['Access-Control-Max-Age'] = '86400'
        logger.info(f"Handled preflight request from origin: {origin}")
        return response

# Initialize Claude client
claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY) if CLAUDE_API_KEY != "YOUR_CLAUDE_API_KEY" else None

# Initialize scheduler
scheduler = BackgroundScheduler()

# ================= PROGRESS TRACKING FUNCTIONS =================
def update_progress(current, total, symbol, stage):
    """Update progress information"""
    global progress_info
    
    with progress_lock:
        progress_info['current'] = current
        progress_info['total'] = total
        progress_info['percentage'] = (current / total) * 100 if total > 0 else 0
        progress_info['currentSymbol'] = symbol
        progress_info['stage'] = stage
        
        # Calculate estimated time remaining
        if progress_info['startTime'] and current > 0:
            elapsed_time = time.time() - progress_info['startTime']
            time_per_item = elapsed_time / current
            remaining_items = total - current
            progress_info['estimatedTimeRemaining'] = remaining_items * time_per_item
        
        logger.info(f"Progress: {current}/{total} ({progress_info['percentage']:.1f}%) - {symbol} - {stage}")

def reset_progress():
    """Reset progress tracking"""
    global progress_info
    
    with progress_lock:
        progress_info.update({
            'current': 0,
            'total': 35,
            'percentage': 0,
            'currentSymbol': '',
            'stage': 'Initializing...',
            'estimatedTimeRemaining': 0,
            'startTime': time.time()
        })

def get_progress():
    """Get current progress information"""
    with progress_lock:
        return progress_info.copy()

# ================= ENHANCED DATABASE SETUP =================
def init_database():
    """Initialize SQLite database for persistent storage"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Create tables with better error handling
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
        logger.info("Database initialized successfully")
        
        # Test database write/read
        cursor.execute("INSERT OR REPLACE INTO analysis_metadata (id, status) VALUES (1, 'initialized')")
        conn.commit()
        
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
    finally:
        conn.close()

def save_analysis_to_db(results):
    """Enhanced save analysis results to database"""
    if not results:
        logger.warning("No results to save to database")
        return False
        
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        # Clear old data first
        cursor.execute('DELETE FROM analysis_results')
        
        # Save individual stock results
        saved_count = 0
        for symbol, data in results.items():
            if symbol not in ['timestamp', 'stocks_analyzed', 'status', 'data_sources', 'markets', 'processing_info', 'note', 'processing_time_minutes', 'success_rate', 'total_requested']:
                try:
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
                    saved_count += 1
                except Exception as e:
                    logger.error(f"Error saving {symbol}: {str(e)}")
                    continue
        
        # Save metadata
        cursor.execute('''
            INSERT OR REPLACE INTO analysis_metadata 
            (id, total_analyzed, success_rate, last_update, status, processing_time_minutes)
            VALUES (1, ?, ?, ?, ?, ?)
        ''', (
            results.get('stocks_analyzed', saved_count),
            results.get('success_rate', 0),
            datetime.now().isoformat(),
            results.get('status', 'unknown'),
            results.get('processing_time_minutes', 0)
        ))
        
        conn.commit()
        logger.info(f"Successfully saved {saved_count} analysis results to database")
        return True
        
    except Exception as e:
        logger.error(f"Error saving to database: {str(e)}")
        conn.rollback()
        return False
    finally:
        conn.close()

def load_analysis_from_db():
    """Enhanced load latest analysis results from database"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        # Get metadata
        cursor.execute('SELECT * FROM analysis_metadata WHERE id = 1')
        metadata = cursor.fetchone()
        
        if not metadata:
            logger.info("No metadata found in database")
            return None
        
        # Get all stock results
        cursor.execute('SELECT symbol, analysis_data FROM analysis_results ORDER BY timestamp DESC')
        stock_results = cursor.fetchall()
        
        if not stock_results:
            logger.info("No stock results found in database")
            return None
        
        # Reconstruct response format
        response = {
            'timestamp': metadata[3],  # last_update
            'stocks_analyzed': metadata[1],  # total_analyzed
            'success_rate': metadata[2],  # success_rate
            'status': 'success',  # Always success if we have data
            'processing_time_minutes': metadata[5],  # processing_time_minutes
            'data_source': 'database_cache',
            'markets': {'us_stocks': 0, 'international_stocks': 0, 'crypto_assets': 0},
            'data_sources': {'twelve_data_count': 0, 'coingecko_count': 0, 'alpha_vantage_count': 0}
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
                elif market == 'International':
                    response['markets']['international_stocks'] += 1
                elif market == 'Crypto':
                    response['markets']['crypto_assets'] += 1
                
                # Count by data source
                data_source = analysis_data.get('data_source', 'Unknown')
                if data_source == 'twelve_data':
                    response['data_sources']['twelve_data_count'] += 1
                elif data_source == 'coingecko':
                    response['data_sources']['coingecko_count'] += 1
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing analysis data for {symbol}: {str(e)}")
                continue
        
        logger.info(f"Loaded {len(stock_results)} analysis results from database")
        return response
        
    except Exception as e:
        logger.error(f"Error loading from database: {str(e)}")
        return None
    finally:
        conn.close()

# ================= STOCK CONFIGURATION =================
def get_filtered_stocks():
    """Get reduced list of popular stocks - 10 US + 15 International + 10 Crypto = 35 total"""
    
    # Top 10 US Stocks - Most popular and liquid
    us_stocks = [
        "AAPL",    # Apple
        "MSFT",    # Microsoft  
        "GOOGL",   # Google
        "TSLA",    # Tesla
        "AMZN",    # Amazon
        "NVDA",    # NVIDIA
        "META",    # Meta (Facebook)
        "NFLX",    # Netflix
        "JPM",     # JPMorgan Chase
        "V"        # Visa
    ]
    
    # Top 15 International Stocks - Popular global stocks that work with Twelve Data
    international_stocks = [
        # Additional US Blue Chips
        "BABA",       # Alibaba
        "ORCL",       # Oracle
        "AMD",        # AMD
        "INTC",       # Intel
        "CRM",        # Salesforce
        
        # Major International (traded on US exchanges)
        "TSM",        # Taiwan Semiconductor
        "ASML",       # ASML Holding
        "SAP",        # SAP SE
        "TM",         # Toyota Motor
        "NVO",        # Novo Nordisk
        
        # Additional US Growth Stocks
        "UBER",       # Uber
        "SHOP",       # Shopify
        "PYPL",       # PayPal
        "DIS",        # Disney
        "KO"          # Coca-Cola
    ]
    
    # Top 10 Cryptocurrencies - Highest market cap and most popular
    crypto_stocks = [
        "bitcoin",        # Bitcoin
        "ethereum",       # Ethereum
        "binancecoin",    # BNB
        "solana",         # Solana
        "cardano",        # Cardano
        "avalanche-2",    # Avalanche
        "polkadot",       # Polkadot
        "chainlink",      # Chainlink
        "polygon",        # Polygon
        "litecoin"        # Litecoin
    ]
    
    return {
        'twelve_data_us': us_stocks,
        'twelve_data_international': international_stocks,
        'coingecko_cryptos': crypto_stocks,
        'us_stocks': us_stocks,
        'international_stocks': international_stocks,
        'crypto_stocks': crypto_stocks,
        'total_count': len(us_stocks) + len(international_stocks) + len(crypto_stocks)
    }

# ================= RATE LIMITING FUNCTIONS =================
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

def wait_for_rate_limit_coingecko():
    """Rate limiting for CoinGecko API"""
    global last_coingecko_request
    
    with rate_limit_lock:
        current_time = time.time()
        time_since_last = current_time - last_coingecko_request
        
        if time_since_last < COINGECKO_DELAY:
            sleep_time = COINGECKO_DELAY - time_since_last
            time.sleep(sleep_time)
        
        last_coingecko_request = time.time()

# ================= DATA FETCHING =================
def fetch_stock_data_twelve_with_retry(symbol, interval="1day", outputsize=100, max_retries=TWELVE_DATA_RETRY_ATTEMPTS):
    """Enhanced Twelve Data fetching with multiple timeframes"""
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

def fetch_crypto_data_coingecko(crypto_id, days=100):
    """Fetch crypto data from CoinGecko"""
    try:
        wait_for_rate_limit_coingecko()
        
        url = f"{COINGECKO_BASE_URL}/coins/{crypto_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily'
        }
        
        logger.info(f"Fetching {crypto_id} from CoinGecko")
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        
        if 'prices' not in data:
            logger.error(f"No price data for {crypto_id}")
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
        logger.error(f"Error fetching crypto data for {crypto_id}: {str(e)}")
        return pd.DataFrame()

def fetch_stock_data(symbol, interval="1day", outputsize=100, source="twelve_data"):
    """Unified function to fetch data from multiple sources"""
    if source == "twelve_data":
        return fetch_stock_data_twelve_with_retry(symbol, interval, outputsize)
    elif source == "coingecko":
        return fetch_crypto_data_coingecko(symbol, outputsize)
    else:
        logger.error(f"Unknown data source: {source}")
        return pd.DataFrame()

# ================= ANALYSIS FUNCTIONS (Simplified for brevity) =================
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

def get_fundamental_data(symbol):
    """Get fundamental data with crypto support"""
    pe_ratios = {
        # US Stocks
        'AAPL': 28.5, 'MSFT': 32.1, 'TSLA': 45.2, 'GOOGL': 24.8, 'AMZN': 38.9,
        'META': 22.7, 'NVDA': 55.3, 'JPM': 12.4, 'V': 34.2, 'NFLX': 35.8,
        
        # International Stocks
        'BABA': 15.5, 'ORCL': 18.2, 'AMD': 40.1, 'INTC': 11.8, 'CRM': 50.2,
        'TSM': 22.3, 'ASML': 42.5, 'SAP': 25.7, 'TM': 14.9, 'NVO': 29.6,
        'UBER': 0, 'SHOP': 0, 'PYPL': 33.4, 'DIS': 27.1, 'KO': 23.8,
        
        # Cryptos
        'bitcoin': 0, 'ethereum': 0, 'binancecoin': 0, 'solana': 0, 'cardano': 0
    }
    
    is_crypto = symbol in ['bitcoin', 'ethereum', 'binancecoin', 'solana', 'cardano', 'avalanche-2', 'polkadot', 'chainlink', 'polygon', 'litecoin']
    
    if is_crypto:
        return {
            'PE_Ratio': 0,
            'Market_Cap_Rank': random.randint(1, 50),
            'Adoption_Score': random.uniform(0.6, 0.95),
            'Technology_Score': random.uniform(0.7, 0.98)
        }
    else:
        base_pe = pe_ratios.get(symbol, 20.0)
        return {
            'PE_Ratio': base_pe,
            'EPS': random.uniform(5.0, 15.0),
            'Revenue_Growth': random.uniform(0.05, 0.25),
            'Net_Income_Growth': random.uniform(0.03, 0.20)
        }

def get_market_sentiment(symbol):
    """Get market sentiment with crypto support"""
    sentiment_scores = {
        # US Stocks
        'AAPL': 0.75, 'MSFT': 0.80, 'TSLA': 0.60, 'GOOGL': 0.70, 'AMZN': 0.65,
        'META': 0.55, 'NVDA': 0.85, 'JPM': 0.60, 'V': 0.75, 'NFLX': 0.65,
        
        # International Stocks
        'BABA': 0.62, 'ORCL': 0.68, 'AMD': 0.77, 'INTC': 0.65, 'CRM': 0.73,
        'TSM': 0.79, 'ASML': 0.81, 'SAP': 0.70, 'TM': 0.63, 'NVO': 0.71,
        'UBER': 0.58, 'SHOP': 0.66, 'PYPL': 0.69, 'DIS': 0.64, 'KO': 0.67,
        
        # Major Cryptos
        'bitcoin': 0.78, 'ethereum': 0.82, 'binancecoin': 0.65, 'solana': 0.75, 'cardano': 0.58
    }
    
    is_crypto = symbol in ['bitcoin', 'ethereum', 'binancecoin', 'solana', 'cardano', 'avalanche-2', 'polkadot', 'chainlink', 'polygon', 'litecoin']
    
    if is_crypto:
        default_sentiment = 0.6
    else:
        default_sentiment = 0.5
    
    return sentiment_scores.get(symbol, default_sentiment)

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
        
        fundamentals = get_fundamental_data(symbol)
        sentiment = get_market_sentiment(symbol)
        
        last_indicators = indicators_data.iloc[-1].to_dict()
        
        # Simple signal generation
        signal_score = 0.0
        
        if last_indicators.get('RSI', 50) < 30:
            signal_score += 0.5
        elif last_indicators.get('RSI', 50) > 70:
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
            signal = 'Strong Buy'
        elif signal_score >= 1.0:
            signal = 'Buy'
        elif signal_score <= -2.0:
            signal = 'Strong Sell'
        elif signal_score <= -1.0:
            signal = 'Sell'
        else:
            signal = 'Neutral'
        
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
        
        timeframe_analysis = {
            'PRICE': current_price,
            'ACCURACY': min(95, max(60, abs(signal_score) * 20 + 60)),
            'CONFIDENCE_SCORE': round(signal_score, 2),
            'VERDICT': signal,
            'DETAILS': {
                'individual_verdicts': {
                    'rsi_verdict': "Overbought" if last_indicators.get('RSI', 50) > 70 else "Oversold" if last_indicators.get('RSI', 50) < 30 else "Neutral",
                    'adx_verdict': "Strong Trend" if last_indicators.get('ADX', 25) > 25 else "Weak Trend",
                    'momentum_verdict': "Bullish" if last_indicators.get('Cycle_Momentum', 0) > 0.02 else "Bearish" if last_indicators.get('Cycle_Momentum', 0) < -0.02 else "Neutral",
                    'pattern_verdict': "No Clear Patterns",
                    'fundamental_verdict': "Strong Fundamentals" if fundamentals.get('Adoption_Score', 0.5) > 0.8 else "Weak Fundamentals" if 'PE_Ratio' not in fundamentals else "Undervalued" if fundamentals.get('PE_Ratio', 20) < 15 else "Overvalued" if fundamentals.get('PE_Ratio', 20) > 25 else "Fair Value",
                    'sentiment_verdict': "Positive" if sentiment > 0.6 else "Negative" if sentiment < 0.4 else "Neutral",
                    'cycle_verdict': 'Bull'
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
                    'geometric': ['None'],
                    'elliott_wave': ['None'],
                    'confluence_factors': ['None']
                },
                'fundamentals': fundamentals,
                'sentiment_analysis': {
                    'score': round(sentiment, 2),
                    'interpretation': "Positive" if sentiment > 0.6 else "Negative" if sentiment < 0.4 else "Neutral",
                    'market_mood': "Optimistic" if sentiment > 0.7 else "Pessimistic" if sentiment < 0.3 else "Cautious"
                },
                'cycle_analysis': {
                    'current_phase': 'Bull' if signal_score > 0 else 'Bear',
                    'stage': f"Mid {'Bull' if signal_score > 0 else 'Bear'}",
                    'duration_days': 30,
                    'momentum': round(signal_score, 3),
                    'momentum_visual': '▲' * 5 + '△' * 5 if signal_score > 0 else '▼' * 5 + '▽' * 5,
                    'bull_continuation_probability': 70 if signal_score > 0 else 30,
                    'bear_transition_probability': 30 if signal_score > 0 else 70,
                    'expected_continuation': '30-60 days',
                    'risk_level': 'Medium'
                },
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
                    elif tf_name == '4hour':
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
                analyses[f"{tf_name.upper()}_TIMEFRAME"] = analysis
        
        # Determine market type
        stock_config = get_filtered_stocks()
        if data_source == 'coingecko':
            market = 'Crypto'
        elif symbol in stock_config['us_stocks']:
            market = 'US'
        else:
            market = 'International'
        
        result = {
            symbol: {
                'data_source': data_source,
                'market': market,
                **analyses
            }
        }
        
        logger.info(f"Successfully analyzed {symbol} with hierarchical logic")
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {str(e)}")
        return None

def analyze_all_stocks_optimized():
    """Optimized stock analysis with database persistence and progress tracking"""
    try:
        reset_progress()
        
        stock_config = get_filtered_stocks()
        twelve_data_us = stock_config['twelve_data_us']
        twelve_data_international = stock_config['twelve_data_international']
        coingecko_cryptos = stock_config['coingecko_cryptos']
        
        results = {}
        total_stocks = len(twelve_data_us) + len(twelve_data_international) + len(coingecko_cryptos)
        processed_count = 0
        
        logger.info(f"Starting optimized analysis of {total_stocks} assets")
        logger.info(f"US stocks: {len(twelve_data_us)}, International: {len(twelve_data_international)}, Crypto: {len(coingecko_cryptos)}")
        
        update_progress(0, total_stocks, 'Initializing...', 'Starting analysis process')
        
        # Process US stocks
        if twelve_data_us:
            batch_size = TWELVE_DATA_BATCH_SIZE
            num_batches = math.ceil(len(twelve_data_us) / batch_size)
            
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, len(twelve_data_us))
                batch_symbols = twelve_data_us[batch_start:batch_end]
                
                update_progress(processed_count, total_stocks, f'US Batch {batch_idx+1}', f'Processing US stocks batch {batch_idx+1}/{num_batches}')
                logger.info(f"Processing US batch {batch_idx+1}/{num_batches}: {batch_symbols}")
                
                for symbol in batch_symbols:
                    try:
                        update_progress(processed_count, total_stocks, symbol, f'Analyzing US stock: {symbol}')
                        result = analyze_stock_hierarchical(symbol, "twelve_data")
                        if result:
                            results.update(result)
                            processed_count += 1
                            logger.info(f"✓ {symbol} ({processed_count}/{total_stocks}) - US Stock")
                        else:
                            logger.warning(f"✗ Failed to process {symbol} (US)")
                    except Exception as e:
                        logger.error(f"✗ Error processing {symbol} (US): {str(e)}")
                
                if batch_idx < num_batches - 1:
                    update_progress(processed_count, total_stocks, 'Rate Limiting', f'Sleeping {TWELVE_DATA_BATCH_SLEEP}s for rate limits...')
                    logger.info(f"Sleeping {TWELVE_DATA_BATCH_SLEEP}s...")
                    time.sleep(TWELVE_DATA_BATCH_SLEEP)
        
        # Process International stocks
        if twelve_data_international:
            batch_size = TWELVE_DATA_BATCH_SIZE
            num_batches = math.ceil(len(twelve_data_international) / batch_size)
            
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, len(twelve_data_international))
                batch_symbols = twelve_data_international[batch_start:batch_end]
                
                update_progress(processed_count, total_stocks, f'International Batch {batch_idx+1}', f'Processing international stocks batch {batch_idx+1}/{num_batches}')
                logger.info(f"Processing International batch {batch_idx+1}/{num_batches}: {batch_symbols}")
                
                for symbol in batch_symbols:
                    try:
                        update_progress(processed_count, total_stocks, symbol, f'Analyzing international stock: {symbol}')
                        result = analyze_stock_hierarchical(symbol, "twelve_data")
                        if result:
                            results.update(result)
                            processed_count += 1
                            logger.info(f"✓ {symbol} ({processed_count}/{total_stocks}) - International Stock")
                        else:
                            logger.warning(f"✗ Failed to process {symbol} (International)")
                    except Exception as e:
                        logger.error(f"✗ Error processing {symbol} (International): {str(e)}")
                
                if batch_idx < num_batches - 1:
                    update_progress(processed_count, total_stocks, 'Rate Limiting', f'Sleeping {TWELVE_DATA_BATCH_SLEEP}s for rate limits...')
                    logger.info(f"Sleeping {TWELVE_DATA_BATCH_SLEEP}s...")
                    time.sleep(TWELVE_DATA_BATCH_SLEEP)
        
        # Process Crypto assets
        if coingecko_cryptos: 
            batch_size = COINGECKO_BATCH_SIZE
            num_batches = math.ceil(len(coingecko_cryptos) / batch_size)
            
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, len(coingecko_cryptos))
                batch_symbols = coingecko_cryptos[batch_start:batch_end]
                
                update_progress(processed_count, total_stocks, f'Crypto Batch {batch_idx+1}', f'Processing crypto assets batch {batch_idx+1}/{num_batches}')
                logger.info(f"Processing Crypto batch {batch_idx+1}/{num_batches}: {batch_symbols}")
                
                for symbol in batch_symbols:
                    try:
                        update_progress(processed_count, total_stocks, symbol, f'Analyzing crypto: {symbol}')
                        result = analyze_stock_hierarchical(symbol, "coingecko")
                        if result:
                            results.update(result)
                            processed_count += 1
                            logger.info(f"✓ {symbol} ({processed_count}/{total_stocks}) - Crypto")
                        else:
                            logger.warning(f"✗ Failed to process {symbol} (Crypto)")
                    except Exception as e:
                        logger.error(f"✗ Error processing {symbol} (Crypto): {str(e)}")
                
                if batch_idx < num_batches - 1:
                    update_progress(processed_count, total_stocks, 'Rate Limiting', f'Sleeping {COINGECKO_DELAY * 2}s for rate limits...')
                    logger.info(f"Sleeping {COINGECKO_DELAY * 2}s...")
                    time.sleep(COINGECKO_DELAY * 2)
        
        # Calculate statistics
        us_stocks_count = len([k for k, v in results.items() if v.get('market') == 'US'])
        international_stocks_count = len([k for k, v in results.items() if v.get('market') == 'International'])
        crypto_count = len([k for k, v in results.items() if v.get('market') == 'Crypto'])
        
        update_progress(processed_count, total_stocks, 'Finalizing', 'Analysis complete - finalizing results')
        
        response = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'stocks_analyzed': len(results),
            'total_requested': total_stocks,
            'success_rate': round((len(results) / total_stocks) * 100, 1) if total_stocks > 0 else 0,
            'status': 'success' if results else 'no_data',
            'data_sources': {
                'twelve_data_count': len([k for k, v in results.items() if v.get('data_source') == 'twelve_data']),
                'coingecko_count': len([k for k, v in results.items() if v.get('data_source') == 'coingecko']),
                'alpha_vantage_count': 0
            },
            'markets': {
                'us_stocks': us_stocks_count,
                'international_stocks': international_stocks_count,
                'crypto_assets': crypto_count
            },
            'processing_info': {
                'hierarchical_analysis': True,
                'timeframes_analyzed': ['monthly', 'weekly', 'daily', '4hour'],
                'ai_analysis_available': claude_client is not None,
                'background_processing': True,
                'daily_auto_refresh': '5:00 PM'
            },
            **results
        }
        
        logger.info(f"Optimized analysis complete. Processed {len(results)}/{total_stocks} assets successfully.")
        logger.info(f"US: {us_stocks_count}, International: {international_stocks_count}, Crypto: {crypto_count}")
        return response
        
    except Exception as e:
        logger.error(f"Error in analyze_all_stocks_optimized: {str(e)}")
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'stocks_analyzed': 0,
            'status': 'error',
            'error': str(e)
        }

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
        
        # Fix: response.content[0] may not have a .text attribute; use .get("text", "") if it's a dict, or str() fallback
        analysis_text = ""
        if hasattr(response, "content") and isinstance(response.content, list) and len(response.content) > 0:
            block = response.content[0]
            # Safely extract 'text' from block if possible, else fallback to str(block)
            analysis_text = ""
            if isinstance(block, dict) and "text" in block:
                analysis_text = block["text"]
            elif hasattr(block, "text"):
                analysis_text = block.text
            else:
                analysis_text = str(block)
        else:
            analysis_text = "No analysis returned from Claude API."
        
        return {
            'analysis': analysis_text,
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

# ================= FLASK ROUTES =================
@app.route('/', methods=['GET'])
@cross_origin()
def home():
    """Enhanced home endpoint with persistent data info"""
    try:
        stock_config = get_filtered_stocks()
        
        # Check if we have cached data
        cached_data = load_analysis_from_db()
        has_cached_data = cached_data is not None
        
        return jsonify({
            'message': 'Enhanced Multi-Asset Analysis API v4.0 - International Focus',
            'version': '4.0 - Persistent Data + Background Processing + Progress Tracking',
            'endpoints': {
                '/analyze': 'GET - Get latest analysis (from cache or trigger new)',
                '/analyze/fresh': 'GET - Force fresh analysis (manual refresh)',
                '/progress': 'GET - Get current analysis progress',
                '/ai-analysis': 'POST - Get detailed AI analysis for specific symbol',
                '/health': 'GET - Health check',
                '/assets': 'GET - List all available assets',
                '/': 'GET - This help message'
            },
            'markets': {
                'us_stocks': len(stock_config['us_stocks']),
                'international_stocks': len(stock_config['international_stocks']),
                'crypto_assets': len(stock_config['crypto_stocks']),
                'total_assets': stock_config['total_count']
            },
            'features': {
                'hierarchical_analysis': True,
                'timeframes': ['monthly', 'weekly', 'daily', '4hour'],
                'ai_analysis': claude_client is not None,
                'persistent_storage': True,
                'background_processing': True,
                'progress_tracking': True,
                'daily_auto_refresh': '5:00 PM',
                'data_sources': ['twelve_data', 'coingecko'],
                'optimized_processing': True,
                'international_focus': True
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

@app.route('/progress', methods=['GET'])
@cross_origin()
def progress():
    """Get current analysis progress"""
    try:
        current_progress = get_progress()
        return jsonify(current_progress)
    except Exception as e:
        logger.error(f"Error in progress endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/assets', methods=['GET'])
@cross_origin()
def list_assets():
    """List all available assets"""
    try:
        stock_config = get_filtered_stocks()
        return jsonify({
            'us_stocks': stock_config['us_stocks'],
            'international_stocks': stock_config['international_stocks'],
            'crypto_assets': stock_config['crypto_stocks'],
            'data_source_distribution': {
                'twelve_data_us': stock_config['twelve_data_us'],
                'twelve_data_international': stock_config['twelve_data_international'],
                'coingecko_cryptos': stock_config['coingecko_cryptos']
            },
            'total_count': stock_config['total_count'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        logger.error(f"Error in assets endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
@cross_origin()
def health():
    """Health check endpoint"""
    try:
        cached_data = load_analysis_from_db()
        return jsonify({
            'status': 'healthy',
            'version': '4.0 - International Focus',
            'markets': ['US', 'International', 'Crypto'],
            'features': {
                'hierarchical_analysis': True,
                'ai_analysis': claude_client is not None,
                'optimized_processing': True,
                'persistent_storage': True,
                'background_processing': True,
                'progress_tracking': True,
                'international_focus': True
            },
            'data_status': {
                'has_cached_data': cached_data is not None,
                'analysis_in_progress': analysis_in_progress,
                'last_update': cached_data.get('timestamp') if cached_data else None
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'service': 'Multi-Asset Analysis API with Progress Tracking - International Focus'
        })
    except Exception as e:
        logger.error(f"Error in health endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['GET'])
@cross_origin()
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
@cross_origin()
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
@cross_origin()
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
        'available_endpoints': ['/analyze', '/analyze/fresh', '/progress', '/ai-analysis', '/health', '/assets', '/'],
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
    
    logger.info(f"Starting Enhanced Multi-Asset Analysis API v4.0 - International Focus on port {port}")
    logger.info(f"Debug mode: {debug_mode}")
    logger.info(f"Total assets configured: {get_filtered_stocks()['total_count']}")
    logger.info(f"AI Analysis available: {claude_client is not None}")
    logger.info("Features: Persistent Storage + Background Processing + Progress Tracking + Daily 5PM Auto-Refresh + International Focus")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=debug_mode, threaded=True)
    finally:
        # Cleanup scheduler on shutdown
        if scheduler.running:
            scheduler.shutdown()
