# import pandas as pd
# import numpy as np
# import pandas_ta as ta
# from scipy.signal import argrelextrema
# import requests
# import time
# import logging
# from datetime import datetime, timedelta
# from flask import Flask, jsonify, request
# from flask_cors import CORS
# import sqlite3
# import json
# import threading
# from threading import Lock
# import random
# import warnings
# from sklearn.linear_model import LinearRegression
# from apscheduler.schedulers.background import BackgroundScheduler
# from apscheduler.triggers.cron import CronTrigger
# import os
# import math

# warnings.filterwarnings('ignore')

# # ================= ENHANCED GLOBAL CONFIGURATION =================
# MARKETSTACK_API_KEY = "your_marketstack_api_key_here"  # Replace with your actual API key
# MARKETSTACK_BASE_URL = "http://api.marketstack.com/v1"
# CRYPTCOMPARE_API_KEY = "3ed1cb75b7ab0925fc3af1e27c4df4aaa2b77d9668824e310c7bc0e60f83e5f7"
# CRYPTCOMPARE_BASE_URL = "https://min-api.cryptocompare.com/data"
# # Replace the Claude configuration section with OpenRouter
# OPENROUTER_API_KEY = "your_openrouter_api_key_here"  # Replace with your actual OpenRouter API key
# OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# # Database configuration
# DATABASE_PATH = "stock_analysis.db"
# CACHE_EXPIRY_HOURS = 24

# # Rate limiting configuration - MarketStack Basic Plan: 10,000 requests/month
# MARKETSTACK_RATE_LIMIT_PER_MIN = 20  # Conservative: ~600/hour, ~14,400/day max
# MARKETSTACK_BATCH_SIZE = 5
# MARKETSTACK_DELAY = 3.0  # 3 seconds between requests
# MARKETSTACK_RETRY_ATTEMPTS = 2
# MARKETSTACK_RETRY_DELAY = 10

# # CryptoCompare for crypto assets
# CRYPTCOMPARE_RATE_LIMIT_PER_MIN = 30
# CRYPTCOMPARE_BATCH_SIZE = 5
# CRYPTCOMPARE_DELAY = 2.0

# # Global rate limiting
# marketstack_rate_limit_lock = Lock()
# cryptcompare_rate_limit_lock = Lock()
# last_marketstack_request = 0
# last_cryptcompare_request = 0
# request_count_marketstack = 0
# request_count_cryptcompare = 0

# # Background processing
# analysis_in_progress = False
# analysis_lock = threading.Lock()

# # Progress tracking
# progress_info = {
#     'current': 0,
#     'total': 140,  # Updated total: 50 + 70 + 20
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

# CORS(app,
#      origins=["https://my-stocks-s2at.onrender.com", "http://localhost:3000", "http://localhost:5177"],
#      methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
#      allow_headers=["Content-Type", "Authorization", "Accept", "Origin", "X-Requested-With"],
#      supports_credentials=False,
#      max_age=86400)

# @app.after_request
# def after_request(response):
#     origin = request.headers.get('Origin')
#     if origin in ["https://my-stocks-s2at.onrender.com", "http://localhost:3000", "http://localhost:5177"]:
#         response.headers['Access-Control-Allow-Origin'] = origin
#     else:
#         response.headers['Access-Control-Allow-Origin'] = "https://my-stocks-s2at.onrender.com"
#     response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
#     response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept, Origin, X-Requested-With'
#     response.headers['Access-Control-Max-Age'] = '86400'
#     response.headers['Vary'] = 'Origin'
#     return response

# @app.before_request
# def handle_preflight():
#     if request.method == "OPTIONS":
#         response = jsonify({'status': 'ok'})
#         origin = request.headers.get('Origin')
#         if origin in ["https://my-stocks-s2at.onrender.com", "http://localhost:3000", "http://localhost:5177"]:
#             response.headers['Access-Control-Allow-Origin'] = origin
#         else:
#             response.headers['Access-Control-Allow-Origin'] = "https://my-stocks-s2at.onrender.com"
        
#         response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
#         response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept, Origin, X-Requested-With'
#         response.headers['Access-Control-Max-Age'] = '86400'
#         logger.info(f"Handled preflight request from origin: {origin}")
#         return response

# # Replace with OpenRouter client check
# openrouter_client_available = OPENROUTER_API_KEY != "your_openrouter_api_key_here"

# # Initialize scheduler
# scheduler = BackgroundScheduler()

# # ================= EXPANDED STOCK CONFIGURATION =================
# def get_expanded_stocks_config():
#     """
#     Expanded stock configuration: 50 US + 70 Nigerian + 20 Crypto = 140 total
#     Using MarketStack for stocks, CryptoCompare for crypto
#     """
    
#     # 50 US Stocks - Top companies across sectors
#     us_stocks = [
#         # Technology Giants
#         {"symbol": "AAPL", "exchange": "XNAS", "name": "Apple Inc."},
#         {"symbol": "MSFT", "exchange": "XNAS", "name": "Microsoft Corporation"},
#         {"symbol": "GOOGL", "exchange": "XNAS", "name": "Alphabet Inc."},
#         {"symbol": "AMZN", "exchange": "XNAS", "name": "Amazon.com Inc."},
#         {"symbol": "META", "exchange": "XNAS", "name": "Meta Platforms Inc."},
#         {"symbol": "TSLA", "exchange": "XNAS", "name": "Tesla Inc."},
#         {"symbol": "NVDA", "exchange": "XNAS", "name": "NVIDIA Corporation"},
#         {"symbol": "NFLX", "exchange": "XNAS", "name": "Netflix Inc."},
#         {"symbol": "ADBE", "exchange": "XNAS", "name": "Adobe Inc."},
#         {"symbol": "CRM", "exchange": "XNYS", "name": "Salesforce Inc."},
        
#         # Financial Services
#         {"symbol": "JPM", "exchange": "XNYS", "name": "JPMorgan Chase & Co."},
#         {"symbol": "BAC", "exchange": "XNYS", "name": "Bank of America Corp."},
#         {"symbol": "WFC", "exchange": "XNYS", "name": "Wells Fargo & Company"},
#         {"symbol": "GS", "exchange": "XNYS", "name": "Goldman Sachs Group Inc."},
#         {"symbol": "MS", "exchange": "XNYS", "name": "Morgan Stanley"},
#         {"symbol": "V", "exchange": "XNYS", "name": "Visa Inc."},
#         {"symbol": "MA", "exchange": "XNYS", "name": "Mastercard Inc."},
#         {"symbol": "AXP", "exchange": "XNYS", "name": "American Express Company"},
        
#         # Healthcare & Pharmaceuticals
#         {"symbol": "JNJ", "exchange": "XNYS", "name": "Johnson & Johnson"},
#         {"symbol": "PFE", "exchange": "XNYS", "name": "Pfizer Inc."},
#         {"symbol": "UNH", "exchange": "XNYS", "name": "UnitedHealth Group Inc."},
#         {"symbol": "ABBV", "exchange": "XNYS", "name": "AbbVie Inc."},
#         {"symbol": "MRK", "exchange": "XNYS", "name": "Merck & Co. Inc."},
#         {"symbol": "TMO", "exchange": "XNYS", "name": "Thermo Fisher Scientific"},
        
#         # Consumer & Retail
#         {"symbol": "WMT", "exchange": "XNYS", "name": "Walmart Inc."},
#         {"symbol": "PG", "exchange": "XNYS", "name": "Procter & Gamble Company"},
#         {"symbol": "KO", "exchange": "XNYS", "name": "Coca-Cola Company"},
#         {"symbol": "PEP", "exchange": "XNAS", "name": "PepsiCo Inc."},
#         {"symbol": "NKE", "exchange": "XNYS", "name": "Nike Inc."},
#         {"symbol": "MCD", "exchange": "XNYS", "name": "McDonald's Corporation"},
        
#         # Industrial & Energy
#         {"symbol": "XOM", "exchange": "XNYS", "name": "Exxon Mobil Corporation"},
#         {"symbol": "CVX", "exchange": "XNYS", "name": "Chevron Corporation"},
#         {"symbol": "CAT", "exchange": "XNYS", "name": "Caterpillar Inc."},
#         {"symbol": "BA", "exchange": "XNYS", "name": "Boeing Company"},
#         {"symbol": "GE", "exchange": "XNYS", "name": "General Electric Company"},
#         {"symbol": "MMM", "exchange": "XNYS", "name": "3M Company"},
        
#         # Telecommunications & Media
#         {"symbol": "VZ", "exchange": "XNYS", "name": "Verizon Communications"},
#         {"symbol": "T", "exchange": "XNYS", "name": "AT&T Inc."},
#         {"symbol": "CMCSA", "exchange": "XNAS", "name": "Comcast Corporation"},
#         {"symbol": "DIS", "exchange": "XNYS", "name": "Walt Disney Company"},
        
#         # Real Estate & Utilities
#         {"symbol": "AMT", "exchange": "XNYS", "name": "American Tower Corporation"},
#         {"symbol": "NEE", "exchange": "XNYS", "name": "NextEra Energy Inc."},
#         {"symbol": "SO", "exchange": "XNYS", "name": "Southern Company"},
        
#         # Semiconductors & Hardware
#         {"symbol": "INTC", "exchange": "XNAS", "name": "Intel Corporation"},
#         {"symbol": "AMD", "exchange": "XNAS", "name": "Advanced Micro Devices"},
#         {"symbol": "QCOM", "exchange": "XNAS", "name": "Qualcomm Inc."},
#         {"symbol": "AVGO", "exchange": "XNAS", "name": "Broadcom Inc."},
        
#         # Emerging Growth
#         {"symbol": "UBER", "exchange": "XNYS", "name": "Uber Technologies Inc."},
#         {"symbol": "LYFT", "exchange": "XNAS", "name": "Lyft Inc."},
#         {"symbol": "SPOT", "exchange": "XNYS", "name": "Spotify Technology"},
#         {"symbol": "SQ", "exchange": "XNYS", "name": "Block Inc."}
#     ]
    
#     # 70 Nigerian Stocks - Comprehensive coverage across sectors
#     nigerian_stocks = [
#         # Banking Sector (20 stocks)
#         {"symbol": "ACCESS", "exchange": "XLAG", "name": "Access Holdings Plc"},
#         {"symbol": "GTCO", "exchange": "XLAG", "name": "Guaranty Trust Holding Company Plc"},
#         {"symbol": "UBA", "exchange": "XLAG", "name": "United Bank for Africa Plc"},
#         {"symbol": "ZENITHBANK", "exchange": "XLAG", "name": "Zenith Bank Plc"},
#         {"symbol": "FBNH", "exchange": "XLAG", "name": "FBN Holdings Plc"},
#         {"symbol": "STANBIC", "exchange": "XLAG", "name": "Stanbic IBTC Holdings Plc"},
#         {"symbol": "STERLINGNG", "exchange": "XLAG", "name": "Sterling Financial Holdings Company Plc"},
#         {"symbol": "FCMB", "exchange": "XLAG", "name": "FCMB Group Plc"},
#         {"symbol": "FIDELITYBK", "exchange": "XLAG", "name": "Fidelity Bank Plc"},
#         {"symbol": "WEMABANK", "exchange": "XLAG", "name": "Wema Bank Plc"},
#         {"symbol": "UNIONBANK", "exchange": "XLAG", "name": "Union Bank of Nigeria Plc"},
#         {"symbol": "UNITYBNK", "exchange": "XLAG", "name": "Unity Bank Plc"},
#         {"symbol": "JAIZBANK", "exchange": "XLAG", "name": "Jaiz Bank Plc"},
#         {"symbol": "SUNUASSUR", "exchange": "XLAG", "name": "Sunu Assurances Nigeria Plc"},
#         {"symbol": "CORNERST", "exchange": "XLAG", "name": "Cornerstone Insurance Plc"},
#         {"symbol": "PRESTIGE", "exchange": "XLAG", "name": "Prestige Assurance Company Plc"},
#         {"symbol": "LINKASSURE", "exchange": "XLAG", "name": "Linkage Assurance Plc"},
#         {"symbol": "AIICO", "exchange": "XLAG", "name": "AIICO Insurance Plc"},
#         {"symbol": "MANSARD", "exchange": "XLAG", "name": "Mansard Insurance Plc"},
#         {"symbol": "SOVRENINS", "exchange": "XLAG", "name": "Sovereign Trust Insurance Plc"},
        
#         # Industrial/Manufacturing (15 stocks)
#         {"symbol": "DANGCEM", "exchange": "XLAG", "name": "Dangote Cement Plc"},
#         {"symbol": "BUACEMENT", "exchange": "XLAG", "name": "BUA Cement Plc"},
#         {"symbol": "WAPCO", "exchange": "XLAG", "name": "Lafarge Africa Plc"},
#         {"symbol": "DANGSUGAR", "exchange": "XLAG", "name": "Dangote Sugar Refinery Plc"},
#         {"symbol": "NASCON", "exchange": "XLAG", "name": "Nascon Allied Industries Plc"},
#         {"symbol": "FLOURMILL", "exchange": "XLAG", "name": "Flour Mills of Nigeria Plc"},
#         {"symbol": "HONEYFLOUR", "exchange": "XLAG", "name": "Honeywell Flour Mill Plc"},
#         {"symbol": "CADBURY", "exchange": "XLAG", "name": "Cadbury Nigeria Plc"},
#         {"symbol": "CHAMPION", "exchange": "XLAG", "name": "Champion Brew. Plc"},
#         {"symbol": "GUINNESS", "exchange": "XLAG", "name": "Guinness Nigeria Plc"},
#         {"symbol": "NB", "exchange": "XLAG", "name": "Nigerian Breweries Plc"},
#         {"symbol": "INTBREW", "exchange": "XLAG", "name": "International Breweries Plc"},
#         {"symbol": "PZ", "exchange": "XLAG", "name": "PZ Cussons Nigeria Plc"},
#         {"symbol": "VITAFOAM", "exchange": "XLAG", "name": "Vitafoam Nigeria Plc"},
#         {"symbol": "MEYER", "exchange": "XLAG", "name": "Meyer Plc"},
        
#         # Consumer Goods (10 stocks)
#         {"symbol": "NESTLE", "exchange": "XLAG", "name": "Nestle Nigeria Plc"},
#         {"symbol": "UNILEVER", "exchange": "XLAG", "name": "Unilever Nigeria Plc"},
#         {"symbol": "HONYFLOUR", "exchange": "XLAG", "name": "Honeywell Flour Mill Plc"},
#         {"symbol": "MCNICHOLS", "exchange": "XLAG", "name": "McNichols Plc"},
#         {"symbol": "NNFM", "exchange": "XLAG", "name": "Nigeria-German Chemicals Plc"},
#         {"symbol": "PHARMDEKO", "exchange": "XLAG", "name": "Pharma-Deko Plc"},
#         {"symbol": "MORISON", "exchange": "XLAG", "name": "Morison Industries Plc"},
#         {"symbol": "BETA", "exchange": "XLAG", "name": "Beta Glass Company Plc"},
#         {"symbol": "CUTIX", "exchange": "XLAG", "name": "Cutix Plc"},
#         {"symbol": "LIVESTOCK", "exchange": "XLAG", "name": "Livestock Feeds Plc"},
        
#         # Oil & Gas (10 stocks)
#         {"symbol": "SEPLAT", "exchange": "XLAG", "name": "Seplat Petroleum Development Company Plc"},
#         {"symbol": "TOTAL", "exchange": "XLAG", "name": "TotalEnergies Marketing Nigeria Plc"},
#         {"symbol": "OANDO", "exchange": "XLAG", "name": "Oando Plc"},
#         {"symbol": "CONOIL", "exchange": "XLAG", "name": "Conoil Plc"},
#         {"symbol": "ETERNA", "exchange": "XLAG", "name": "Eterna Plc"},
#         {"symbol": "MRS", "exchange": "XLAG", "name": "MRS Oil Nigeria Plc"},
#         {"symbol": "ARDOVA", "exchange": "XLAG", "name": "Ardova Plc"},
#         {"symbol": "JAPAULGOLD", "exchange": "XLAG", "name": "Japaul Gold & Ventures Plc"},
#         {"symbol": "OMATEK", "exchange": "XLAG", "name": "Omatek Ventures Plc"},
#         {"symbol": "PETROCAM", "exchange": "XLAG", "name": "Petrocam Trading Company Ltd"},
        
#         # Telecommunications & Technology (5 stocks)
#         {"symbol": "MTNN", "exchange": "XLAG", "name": "MTN Nigeria Communications Plc"},
#         {"symbol": "AIRTELAFRI", "exchange": "XLAG", "name": "Airtel Africa Plc"},
#         {"symbol": "IHS", "exchange": "XLAG", "name": "IHS Towers"},
#         {"symbol": "CHIPLC", "exchange": "XLAG", "name": "Champion Breweries Plc"},
#         {"symbol": "CWG", "exchange": "XLAG", "name": "Computer Warehouse Group Plc"},
        
#         # Conglomerates & Others (10 stocks)
#         {"symbol": "TRANSCORP", "exchange": "XLAG", "name": "Transnational Corporation Plc"},
#         {"symbol": "DANGOTE", "exchange": "XLAG", "name": "Dangote Industries Limited"},
#         {"symbol": "BUA", "exchange": "XLAG", "name": "BUA Group"},
#         {"symbol": "UACN", "exchange": "XLAG", "name": "UACN Plc"},
#         {"symbol": "SCOA", "exchange": "XLAG", "name": "SCOA Nigeria Plc"},
#         {"symbol": "JOHNHOLT", "exchange": "XLAG", "name": "John Holt Plc"},
#         {"symbol": "CAVERTON", "exchange": "XLAG", "name": "Caverton Offshore Support Group Plc"},
#         {"symbol": "REDSTAREX", "exchange": "XLAG", "name": "Red Star Express Plc"},
#         {"symbol": "ROYALEX", "exchange": "XLAG", "name": "Royal Exchange Plc"},
#         {"symbol": "LASACO", "exchange": "XLAG", "name": "Lasaco Assurance Plc"}
#     ]
    
#     # 20 Crypto Assets - Top market cap and popular trading pairs
#     crypto_stocks = [
#         # Top 10 by Market Cap
#         {"symbol": "BTC", "name": "Bitcoin"},
#         {"symbol": "ETH", "name": "Ethereum"},
#         {"symbol": "BNB", "name": "BNB"},
#         {"symbol": "SOL", "name": "Solana"},
#         {"symbol": "ADA", "name": "Cardano"},
#         {"symbol": "AVAX", "name": "Avalanche"},
#         {"symbol": "DOT", "name": "Polkadot"},
#         {"symbol": "LINK", "name": "Chainlink"},
#         {"symbol": "MATIC", "name": "Polygon"},
#         {"symbol": "LTC", "name": "Litecoin"},
        
#         # DeFi & Popular Altcoins
#         {"symbol": "UNI", "name": "Uniswap"},
#         {"symbol": "AAVE", "name": "Aave"},
#         {"symbol": "SUSHI", "name": "SushiSwap"},
#         {"symbol": "COMP", "name": "Compound"},
#         {"symbol": "MKR", "name": "Maker"},
#         {"symbol": "SNX", "name": "Synthetix"},
#         {"symbol": "CRV", "name": "Curve DAO Token"},
#         {"symbol": "YFI", "name": "yearn.finance"},
#         {"symbol": "1INCH", "name": "1inch"},
#         {"symbol": "BAL", "name": "Balancer"}
#     ]
    
#     return {
#         'us_stocks': us_stocks,
#         'nigerian_stocks': nigerian_stocks,
#         'crypto_stocks': crypto_stocks,
#         'total_count': len(us_stocks) + len(nigerian_stocks) + len(crypto_stocks)
#     }

# # ================= RATE LIMITING FUNCTIONS =================
# def wait_for_rate_limit_marketstack():
#     """Rate limiting for MarketStack API"""
#     global last_marketstack_request, request_count_marketstack
    
#     with marketstack_rate_limit_lock:
#         current_time = time.time()
        
#         if current_time - last_marketstack_request > 60:
#             request_count_marketstack = 0
#             last_marketstack_request = current_time
        
#         if request_count_marketstack >= MARKETSTACK_RATE_LIMIT_PER_MIN:
#             sleep_time = 60 - (current_time - last_marketstack_request)
#             if sleep_time > 0:
#                 logger.info(f"Rate limit reached for MarketStack. Sleeping for {sleep_time:.1f} seconds...")
#                 time.sleep(sleep_time)
#                 request_count_marketstack = 0
#                 last_marketstack_request = time.time()
        
#         request_count_marketstack += 1

# def wait_for_rate_limit_cryptcompare():
#     """Rate limiting for CryptoCompare API"""
#     global last_cryptcompare_request, request_count_cryptcompare
    
#     with cryptcompare_rate_limit_lock:
#         current_time = time.time()
#         if current_time - last_cryptcompare_request > 60:
#             request_count_cryptcompare = 0
#             last_cryptcompare_request = current_time
        
#         if request_count_cryptcompare >= CRYPTCOMPARE_RATE_LIMIT_PER_MIN:
#             sleep_time = 60 - (current_time - last_cryptcompare_request)
#             if sleep_time > 0:
#                 logger.info(f"Rate limit reached for CryptoCompare. Sleeping for {sleep_time:.1f} seconds...")
#                 time.sleep(sleep_time)
#                 request_count_cryptcompare = 0
#                 last_cryptcompare_request = time.time()
        
#         request_count_cryptcompare += 1

# # ================= DATA FETCHING FUNCTIONS =================
# def fetch_marketstack_data(symbol, exchange=None, limit=100, max_retries=MARKETSTACK_RETRY_ATTEMPTS):
#     """Fetch stock data from MarketStack API with retry logic"""
#     for attempt in range(max_retries):
#         try:
#             wait_for_rate_limit_marketstack()
            
#             url = f"{MARKETSTACK_BASE_URL}/eod"
#             params = {
#                 'access_key': MARKETSTACK_API_KEY,
#                 'symbols': symbol,
#                 'limit': limit,
#                 'sort': 'DESC'
#             }
            
#             if exchange:
#                 params['exchange'] = exchange
            
#             logger.info(f"Fetching {symbol} from MarketStack (exchange: {exchange}, attempt {attempt + 1}/{max_retries})")
            
#             response = requests.get(url, params=params, timeout=20)
#             response.raise_for_status()
            
#             data = response.json()
            
#             if 'error' in data:
#                 logger.warning(f"MarketStack API error for {symbol}: {data['error']}")
#                 if attempt < max_retries - 1:
#                     time.sleep(MARKETSTACK_RETRY_DELAY)
#                     continue
#                 return pd.DataFrame()
            
#             if 'data' not in data or not data['data']:
#                 logger.error(f"No data returned for {symbol}")
#                 if attempt < max_retries - 1:
#                     time.sleep(MARKETSTACK_RETRY_DELAY)
#                     continue
#                 return pd.DataFrame()
            
#             # Convert to DataFrame
#             df_data = []
#             for point in data['data']:
#                 try:
#                     df_data.append({
#                         'datetime': pd.to_datetime(point['date']),
#                         'open': float(point['open']) if point['open'] else 0,
#                         'high': float(point['high']) if point['high'] else 0,
#                         'low': float(point['low']) if point['low'] else 0,
#                         'close': float(point['close']) if point['close'] else 0,
#                         'volume': float(point['volume']) if point['volume'] else 0
#                     })
#                 except (ValueError, TypeError) as e:
#                     logger.warning(f"Skipping invalid data point for {symbol}: {e}")
#                     continue
            
#             if not df_data:
#                 logger.error(f"No valid data points for {symbol}")
#                 if attempt < max_retries - 1:
#                     time.sleep(MARKETSTACK_RETRY_DELAY)
#                     continue
#                 return pd.DataFrame()
            
#             df = pd.DataFrame(df_data)
#             df.set_index('datetime', inplace=True)
#             df.sort_index(inplace=True)
#             df.dropna(inplace=True)
            
#             logger.info(f"Successfully fetched {len(df)} rows for {symbol} from MarketStack")
#             return df
            
#         except Exception as e:
#             logger.error(f"Error fetching MarketStack data for {symbol} (attempt {attempt + 1}): {str(e)}")
#             if attempt < max_retries - 1:
#                 time.sleep(MARKETSTACK_RETRY_DELAY)
#             else:
#                 return pd.DataFrame()
    
#     return pd.DataFrame()

# def fetch_crypto_data_cryptcompare(symbol, days=100):
#     """Fetch crypto data from CryptoCompare"""
#     try:
#         wait_for_rate_limit_cryptcompare()
        
#         url = f"{CRYPTCOMPARE_BASE_URL}/v2/histoday"
#         params = {
#             'fsym': symbol,
#             'tsym': 'USD',
#             'limit': days,
#             'api_key': CRYPTCOMPARE_API_KEY
#         }
        
#         logger.info(f"Fetching {symbol} from CryptoCompare")
        
#         response = requests.get(url, params=params, timeout=15)
#         response.raise_for_status()
        
#         data = response.json()
        
#         if data.get('Response') != 'Success' or not data.get('Data', {}).get('Data'):
#             logger.error(f"No price data for {symbol}: {data.get('Message', 'Unknown error')}")
#             return pd.DataFrame()
        
#         df_data = []
#         for point in data['Data']['Data']:
#             timestamp = pd.to_datetime(point['time'], unit='s')
#             if point['high'] == 0 and point['low'] == 0 and point['open'] == 0 and point['close'] == 0:
#                 continue
#             df_data.append({
#                 'datetime': timestamp,
#                 'open': point['open'],
#                 'high': point['high'],
#                 'low': point['low'],
#                 'close': point['close'],
#                 'volume': point['volumeto']
#             })
        
#         df = pd.DataFrame(df_data)
#         if df.empty:
#             logger.error(f"Empty DataFrame after processing for {symbol}")
#             return pd.DataFrame()
        
#         df.set_index('datetime', inplace=True)
#         df.sort_index(inplace=True)
#         df.dropna(inplace=True)
        
#         logger.info(f"Successfully fetched {len(df)} rows for {symbol} from CryptoCompare")
#         return df
        
#     except Exception as e:
#         logger.error(f"Error fetching crypto data for {symbol}: {str(e)}")
#         return pd.DataFrame()

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
#             'total': 140,
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

# # ================= ANALYSIS FUNCTIONS =================
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
        
#         highs = argrelextrema(prices, np.greater, order=5)[0]
#         lows = argrelextrema(prices, np.less, order=5)[0]
        
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
#                 if change > 0.05:  # 5% threshold
#                     filtered_pivots.append(i)
        
#         pivot_data = []
#         for i in filtered_pivots:
#             start_idx = max(0, i - 10)
#             end_idx = min(len(prices), i + 10)
#             local_max = np.max(prices[start_idx:end_idx])
#             local_min = np.min(prices[start_idx:end_idx])
            
#             if prices[i] == local_max:
#                 pivot_type = 'high'
#             else:
#                 pivot_type = 'low'
            
#             pivot_data.append((i, prices[i], pivot_type))
        
#         return pivot_data[-10:]  # Return last 10 pivots
        
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
#     """Get fundamental data with expanded coverage"""
#     # Enhanced PE ratios for expanded stock list
#     pe_ratios = {
#         # US Stocks
#         'AAPL': 28.5, 'MSFT': 32.1, 'GOOGL': 24.8, 'AMZN': 38.9, 'META': 22.7,
#         'TSLA': 45.2, 'NVDA': 55.3, 'NFLX': 35.8, 'ADBE': 42.1, 'CRM': 48.3,
#         'JPM': 12.4, 'BAC': 11.8, 'WFC': 10.2, 'GS': 13.5, 'MS': 12.9,
#         'V': 34.2, 'MA': 36.8, 'AXP': 15.7, 'JNJ': 16.4, 'PFE': 13.2,
#         'UNH': 24.8, 'ABBV': 14.6, 'MRK': 17.3, 'TMO': 28.9, 'WMT': 26.1,
#         'PG': 24.7, 'KO': 25.3, 'PEP': 26.8, 'NKE': 31.4, 'MCD': 33.2,
#         'XOM': 14.8, 'CVX': 15.2, 'CAT': 16.9, 'BA': 22.4, 'GE': 18.7,
#         'MMM': 19.3, 'VZ': 9.8, 'T': 8.4, 'CMCSA': 16.2, 'DIS': 28.6,
#         'AMT': 58.2, 'NEE': 23.4, 'SO': 21.7, 'INTC': 12.8, 'AMD': 48.9,
#         'QCOM': 18.4, 'AVGO': 19.7, 'UBER': 0, 'LYFT': 0, 'SPOT': 0, 'SQ': 0,
        
#         # Nigerian Stocks
#         'ACCESS': 8.5, 'GTCO': 12.3, 'UBA': 7.4, 'ZENITHBANK': 11.2, 'FBNH': 6.2,
#         'STANBIC': 9.8, 'STERLINGNG': 8.1, 'FCMB': 7.9, 'FIDELITYBK': 6.8, 'WEMABANK': 5.4,
#         'DANGCEM': 19.2, 'BUACEMENT': 16.8, 'WAPCO': 15.5, 'DANGSUGAR': 18.5, 'NASCON': 14.2,
#         'NESTLE': 35.8, 'UNILEVER': 28.4, 'FLOURMILL': 12.6, 'CADBURY': 22.1, 'GUINNESS': 16.8,
#         'SEPLAT': 14.2, 'TOTAL': 16.8, 'OANDO': 8.9, 'CONOIL': 11.4, 'ETERNA': 9.7,
#         'MTNN': 22.1, 'AIRTELAFRI': 18.9, 'TRANSCORP': 12.5, 'UACN': 15.3, 'SCOA': 10.8,
        
#         # Cryptos (0 for all)
#         'BTC': 0, 'ETH': 0, 'BNB': 0, 'SOL': 0, 'ADA': 0, 'AVAX': 0, 'DOT': 0,
#         'LINK': 0, 'MATIC': 0, 'LTC': 0, 'UNI': 0, 'AAVE': 0, 'SUSHI': 0,
#         'COMP': 0, 'MKR': 0, 'SNX': 0, 'CRV': 0, 'YFI': 0, '1INCH': 0, 'BAL': 0
#     }
    
#     is_crypto = symbol in ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'AVAX', 'DOT', 'LINK', 'MATIC', 'LTC',
#                           'UNI', 'AAVE', 'SUSHI', 'COMP', 'MKR', 'SNX', 'CRV', 'YFI', '1INCH', 'BAL']
    
#     # Nigerian stocks (expanded list)
#     nigerian_symbols = ['ACCESS', 'GTCO', 'UBA', 'ZENITHBANK', 'FBNH', 'STANBIC', 'STERLINGNG', 'FCMB',
#                        'FIDELITYBK', 'WEMABANK', 'UNIONBANK', 'UNITYBNK', 'JAIZBANK', 'DANGCEM', 'BUACEMENT',
#                        'WAPCO', 'DANGSUGAR', 'NASCON', 'FLOURMILL', 'HONEYFLOUR', 'CADBURY', 'CHAMPION',
#                        'GUINNESS', 'NB', 'INTBREW', 'PZ', 'VITAFOAM', 'MEYER', 'NESTLE', 'UNILEVER',
#                        'SEPLAT', 'TOTAL', 'OANDO', 'CONOIL', 'ETERNA', 'MRS', 'ARDOVA', 'MTNN', 'AIRTELAFRI',
#                        'IHS', 'CWG', 'TRANSCORP', 'UACN', 'SCOA', 'JOHNHOLT', 'CAVERTON', 'REDSTAREX']
    
#     is_nigerian = symbol in nigerian_symbols
    
#     if is_crypto:
#         return {
#             'PE_Ratio': 0,
#             'Market_Cap_Rank': random.randint(1, 100),
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
#     """Get market sentiment with expanded coverage"""
#     sentiment_scores = {
#         # US Stocks
#         'AAPL': 0.75, 'MSFT': 0.80, 'GOOGL': 0.70, 'AMZN': 0.65, 'META': 0.55,
#         'TSLA': 0.60, 'NVDA': 0.85, 'NFLX': 0.65, 'ADBE': 0.72, 'CRM': 0.68,
#         'JPM': 0.60, 'BAC': 0.58, 'WFC': 0.52, 'GS': 0.62, 'MS': 0.59,
#         'V': 0.75, 'MA': 0.73, 'AXP': 0.61, 'JNJ': 0.68, 'PFE': 0.55,
        
#         # Nigerian Stocks
#         'DANGCEM': 0.68, 'GTCO': 0.72, 'ZENITHBANK': 0.65, 'UBA': 0.63, 'ACCESS': 0.61,
#         'NESTLE': 0.70, 'UNILEVER': 0.66, 'MTNN': 0.74, 'SEPLAT': 0.58, 'TOTAL': 0.62,
#         'BUACEMENT': 0.64, 'WAPCO': 0.59, 'DANGSUGAR': 0.56, 'TRANSCORP': 0.54,
        
#         # Cryptos
#         'BTC': 0.78, 'ETH': 0.82, 'BNB': 0.65, 'SOL': 0.75, 'ADA': 0.58,
#         'AVAX': 0.68, 'DOT': 0.62, 'LINK': 0.70, 'MATIC': 0.66, 'LTC': 0.55,
#         'UNI': 0.72, 'AAVE': 0.69, 'SUSHI': 0.61, 'COMP': 0.64, 'MKR': 0.67
#     }
    
#     # Default sentiment based on market type
#     nigerian_symbols = ['ACCESS', 'GTCO', 'UBA', 'ZENITHBANK', 'FBNH', 'STANBIC', 'STERLINGNG', 'FCMB',
#                        'FIDELITYBK', 'WEMABANK', 'UNIONBANK', 'UNITYBNK', 'JAIZBANK', 'DANGCEM', 'BUACEMENT',
#                        'WAPCO', 'DANGSUGAR', 'NASCON', 'FLOURMILL', 'HONEYFLOUR', 'CADBURY', 'CHAMPION',
#                        'GUINNESS', 'NB', 'INTBREW', 'PZ', 'VITAFOAM', 'MEYER', 'NESTLE', 'UNILEVER',
#                        'SEPLAT', 'TOTAL', 'OANDO', 'CONOIL', 'ETERNA', 'MRS', 'ARDOVA', 'MTNN', 'AIRTELAFRI',
#                        'IHS', 'CWG', 'TRANSCORP', 'UACN', 'SCOA', 'JOHNHOLT', 'CAVERTON', 'REDSTAREX']
    
#     crypto_symbols = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'AVAX', 'DOT', 'LINK', 'MATIC', 'LTC',
#                      'UNI', 'AAVE', 'SUSHI', 'COMP', 'MKR', 'SNX', 'CRV', 'YFI', '1INCH', 'BAL']
    
#     if symbol in crypto_symbols:
#         default_sentiment = 0.6
#     elif symbol in nigerian_symbols:
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

# # ================= MAIN ANALYSIS FUNCTIONS =================
# def analyze_stock_unified(stock_info, data_source):
#     """Analyze stock using unified MarketStack/CryptoCompare approach"""
#     try:
#         symbol = stock_info['symbol']
#         exchange = stock_info.get('exchange')
        
#         logger.info(f"Starting analysis for {symbol} using {data_source}")
        
#         # Fetch data based on source
#         if data_source == "marketstack":
#             data = fetch_marketstack_data(symbol, exchange, limit=100)
#         elif data_source == "cryptcompare":
#             data = fetch_crypto_data_cryptcompare(symbol, days=100)
#         else:
#             logger.error(f"Unknown data source: {data_source}")
#             return None
        
#         if data.empty:
#             logger.error(f"No data available for {symbol}")
#             return None
        
#         # Apply analysis functions
#         ha_data = heikin_ashi(data)
#         if ha_data.empty:
#             logger.error(f"Failed to convert to HA for {symbol}")
#             return None
        
#         indicators_data = calculate_ha_indicators(ha_data)
#         if indicators_data is None:
#             logger.error(f"Failed to calculate indicators for {symbol}")
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
        
#         # Determine market type
#         if data_source == "cryptcompare":
#             market = "Crypto"
#         elif exchange == "XLAG":
#             market = "Nigerian"
#         elif exchange in ["XNAS", "XNYS"]:
#             market = "US"
#         else:
#             market = "Unknown"
        
#         # Calculate price changes
#         change_1d = 0.0
#         change_1w = 0.0
        
#         if len(ha_data) >= 2:
#             change_1d = round((ha_data['HA_Close'].iloc[-1] / ha_data['HA_Close'].iloc[-2] - 1) * 100, 2)
        
#         if len(ha_data) >= 5:
#             change_1w = round((ha_data['HA_Close'].iloc[-1] / ha_data['HA_Close'].iloc[-5] - 1) * 100, 2)
        
#         # Generate trading levels
#         if 'Buy' in signal:
#             entry = round(current_price * 0.99, 2)
#             targets = [round(current_price * 1.05, 2), round(current_price * 1.10, 2)]
#             stop_loss = round(current_price * 0.95, 2)
#         else:
#             entry = round(current_price * 1.01, 2)
#             targets = [round(current_price * 0.95, 2), round(current_price * 0.90, 2)]
#             stop_loss = round(current_price * 1.05, 2)
        
#         # Generate verdicts
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
        
#         result = {
#             symbol: {
#                 'data_source': data_source,
#                 'market': market,
#                 'exchange': exchange,
#                 'DAILY_TIMEFRAME': {
#                     'PRICE': current_price,
#                     'ACCURACY': min(95, max(60, abs(score) * 20 + 60)),
#                     'CONFIDENCE_SCORE': round(score, 2),
#                     'VERDICT': signal,
#                     'DETAILS': {
#                         'individual_verdicts': {
#                             'rsi_verdict': rsi_verdict,
#                             'adx_verdict': adx_verdict,
#                             'momentum_verdict': momentum_verdict,
#                             'pattern_verdict': pattern_verdict,
#                             'fundamental_verdict': fundamental_verdict,
#                             'sentiment_verdict': sentiment_verdict,
#                             'cycle_verdict': cycle_analysis['current_phase']
#                         },
#                         'price_data': {
#                             'current_price': current_price,
#                             'entry_price': entry,
#                             'target_prices': targets,
#                             'stop_loss': stop_loss,
#                             'change_1d': change_1d,
#                             'change_1w': change_1w
#                         },
#                         'technical_indicators': {
#                             'rsi': round(last_indicators.get('RSI', 50.0), 1),
#                             'adx': round(last_indicators.get('ADX', 25.0), 1),
#                             'atr': round(last_indicators.get('ATR', 1.0), 2),
#                             'cycle_phase': last_indicators.get('Cycle_Phase', 'Unknown'),
#                             'cycle_momentum': round(last_indicators.get('Cycle_Momentum', 0.0), 3)
#                         },
#                         'patterns': {
#                             'geometric': [k for k, v in patterns.items() if v] or ['None'],
#                             'elliott_wave': [k for k, v in waves.items() if v.get('detected', False)] or ['None'],
#                             'confluence_factors': confluence['factors'] or ['None']
#                         },
#                         'fundamentals': fundamentals,
#                         'sentiment_analysis': {
#                             'score': round(sentiment, 2),
#                             'interpretation': sentiment_verdict,
#                             'market_mood': "Optimistic" if sentiment > 0.7 else "Pessimistic" if sentiment < 0.3 else "Cautious"
#                         },
#                         'cycle_analysis': cycle_analysis,
#                         'trading_parameters': {
#                             'position_size': '5% of portfolio' if 'Strong' in signal else '3% of portfolio',
#                             'timeframe': 'DAILY - 2-4 weeks' if 'Buy' in signal else 'DAILY - 1-2 weeks',
#                             'risk_level': 'Medium' if 'Buy' in signal else 'High' if 'Sell' in signal else 'Low'
#                         }
#                     }
#                 }
#             }
#         }
        
#         logger.info(f"Successfully analyzed {symbol} with {data_source} data")
#         return result
        
#     except Exception as e:
#         logger.error(f"Error analyzing {symbol}: {str(e)}")
#         return None

# def analyze_all_stocks_unified():
#     """Analyze all stocks using unified MarketStack + CryptoCompare approach"""
#     try:
#         reset_progress()
        
#         stock_config = get_expanded_stocks_config()
#         us_stocks = stock_config['us_stocks']
#         nigerian_stocks = stock_config['nigerian_stocks']
#         crypto_stocks = stock_config['crypto_stocks']
        
#         results = {}
#         total_stocks = len(us_stocks) + len(nigerian_stocks) + len(crypto_stocks)
#         processed_count = 0
        
#         logger.info(f"Starting unified analysis of {total_stocks} assets")
#         logger.info(f"US: {len(us_stocks)}, Nigerian: {len(nigerian_stocks)}, Crypto: {len(crypto_stocks)}")
        
#         update_progress(0, total_stocks, 'Initializing...', 'Starting unified analysis process')
        
#         # Process US stocks with MarketStack
#         if us_stocks:
#             batch_size = MARKETSTACK_BATCH_SIZE
#             num_batches = math.ceil(len(us_stocks) / batch_size)
            
#             for batch_idx in range(num_batches):
#                 batch_start = batch_idx * batch_size
#                 batch_end = min((batch_idx + 1) * batch_size, len(us_stocks))
#                 batch_stocks = us_stocks[batch_start:batch_end]
                
#                 update_progress(processed_count, total_stocks, f'US Batch {batch_idx+1}', f'Processing US stocks batch {batch_idx+1}/{num_batches}')
#                 logger.info(f"Processing US batch {batch_idx+1}/{num_batches}: {[s['symbol'] for s in batch_stocks]}")
                
#                 for stock_info in batch_stocks:
#                     try:
#                         symbol = stock_info['symbol']
#                         update_progress(processed_count, total_stocks, symbol, f'Analyzing US stock: {symbol}')
#                         result = analyze_stock_unified(stock_info, "marketstack")
#                         if result:
#                             results.update(result)
#                             processed_count += 1
#                             logger.info(f"✓ {symbol} ({processed_count}/{total_stocks}) - US Stock (MarketStack)")
#                         else:
#                             logger.warning(f"✗ Failed to process {symbol} (US)")
#                     except Exception as e:
#                         logger.error(f"✗ Error processing {symbol} (US): {str(e)}")
                
#                 if batch_idx < num_batches - 1:
#                     update_progress(processed_count, total_stocks, 'Rate Limiting', f'Sleeping {MARKETSTACK_DELAY}s for rate limits...')
#                     logger.info(f"Sleeping {MARKETSTACK_DELAY}s...")
#                     time.sleep(MARKETSTACK_DELAY)
        
#         # Process Nigerian stocks with MarketStack
#         if nigerian_stocks:
#             batch_size = MARKETSTACK_BATCH_SIZE
#             num_batches = math.ceil(len(nigerian_stocks) / batch_size)
            
#             for batch_idx in range(num_batches):
#                 batch_start = batch_idx * batch_size
#                 batch_end = min((batch_idx + 1) * batch_size, len(nigerian_stocks))
#                 batch_stocks = nigerian_stocks[batch_start:batch_end]
                
#                 update_progress(processed_count, total_stocks, f'Nigerian Batch {batch_idx+1}', f'Processing Nigerian stocks batch {batch_idx+1}/{num_batches}')
#                 logger.info(f"Processing Nigerian batch {batch_idx+1}/{num_batches}: {[s['symbol'] for s in batch_stocks]}")
                
#                 for stock_info in batch_stocks:
#                     try:
#                         symbol = stock_info['symbol']
#                         update_progress(processed_count, total_stocks, symbol, f'Analyzing Nigerian stock: {symbol}')
#                         result = analyze_stock_unified(stock_info, "marketstack")
#                         if result:
#                             results.update(result)
#                             processed_count += 1
#                             logger.info(f"✓ {symbol} ({processed_count}/{total_stocks}) - Nigerian Stock (MarketStack)")
#                         else:
#                             logger.warning(f"✗ Failed to process {symbol} (Nigerian)")
#                     except Exception as e:
#                         logger.error(f"✗ Error processing {symbol} (Nigerian): {str(e)}")
                
#                 if batch_idx < num_batches - 1:
#                     update_progress(processed_count, total_stocks, 'Rate Limiting', f'Sleeping {MARKETSTACK_DELAY}s for rate limits...')
#                     logger.info(f"Sleeping {MARKETSTACK_DELAY}s...")
#                     time.sleep(MARKETSTACK_DELAY)
        
#         # Process Crypto assets with CryptoCompare
#         if crypto_stocks:
#             batch_size = CRYPTCOMPARE_BATCH_SIZE
#             num_batches = math.ceil(len(crypto_stocks) / batch_size)
            
#             for batch_idx in range(num_batches):
#                 batch_start = batch_idx * batch_size
#                 batch_end = min((batch_idx + 1) * batch_size, len(crypto_stocks))
#                 batch_stocks = crypto_stocks[batch_start:batch_end]
                
#                 update_progress(processed_count, total_stocks, f'Crypto Batch {batch_idx+1}', f'Processing crypto assets batch {batch_idx+1}/{num_batches}')
#                 logger.info(f"Processing Crypto batch {batch_idx+1}/{num_batches}: {[s['symbol'] for s in batch_stocks]}")
                
#                 for stock_info in batch_stocks:
#                     try:
#                         symbol = stock_info['symbol']
#                         update_progress(processed_count, total_stocks, symbol, f'Analyzing crypto: {symbol}')
#                         result = analyze_stock_unified(stock_info, "cryptcompare")
#                         if result:
#                             results.update(result)
#                             processed_count += 1
#                             logger.info(f"✓ {symbol} ({processed_count}/{total_stocks}) - Crypto (CryptoCompare)")
#                         else:
#                             logger.warning(f"✗ Failed to process {symbol} (Crypto)")
#                     except Exception as e:
#                         logger.error(f"✗ Error processing {symbol} (Crypto): {str(e)}")
                
#                 if batch_idx < num_batches - 1:
#                     update_progress(processed_count, total_stocks, 'Rate Limiting', f'Sleeping {CRYPTCOMPARE_DELAY}s for rate limits...')
#                     logger.info(f"Sleeping {CRYPTCOMPARE_DELAY}s...")
#                     time.sleep(CRYPTCOMPARE_DELAY)
        
#         # Calculate final statistics
#         us_stocks_count = len([k for k, v in results.items() if v.get('market') == 'US'])
#         nigerian_stocks_count = len([k for k, v in results.items() if v.get('market') == 'Nigerian'])
#         crypto_count = len([k for k, v in results.items() if v.get('market') == 'Crypto'])
        
#         update_progress(total_stocks, total_stocks, 'Complete', 'Unified analysis finished - results ready')
        
#         response = {
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#             'stocks_analyzed': len(results),
#             'total_requested': total_stocks,
#             'success_rate': round((len(results) / total_stocks) * 100, 1) if total_stocks > 0 else 0,
#             'status': 'success',
#             'data_source': 'unified_marketstack_cryptcompare',
#             'data_sources': {
#                 'marketstack_count': len([k for k, v in results.items() if v.get('data_source') == 'marketstack']),
#                 'cryptcompare_count': len([k for k, v in results.items() if v.get('data_source') == 'cryptcompare'])
#             },
#             'markets': {
#                 'us_stocks': us_stocks_count,
#                 'nigerian_stocks': nigerian_stocks_count,
#                 'crypto_assets': crypto_count
#             },
#             'processing_info': {
#                 'unified_analysis': True,
#                 'timeframes_analyzed': ['daily'],
#                 'ai_analysis_available': openrouter_client_available,
#                 'background_processing': True,
#                 'daily_auto_refresh': '5:00 PM',
#                 'marketstack_plan': 'Basic (10,000 requests/month)',
#                 'expanded_coverage': True
#             },
#             'note': f'Unified analysis complete using MarketStack + CryptoCompare. Successfully processed {len(results)} out of {total_stocks} assets.',
#             **results
#         }
        
#         logger.info(f"Unified analysis complete. Processed {len(results)}/{total_stocks} assets successfully.")
#         logger.info(f"US: {us_stocks_count}, Nigerian: {nigerian_stocks_count}, Crypto: {crypto_count}")
#         return response
        
#     except Exception as e:
#         logger.error(f"Error in unified analysis: {str(e)}")
#         return {
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#             'stocks_analyzed': 0,
#             'status': 'error',
#             'error': str(e)
#         }

# # ================= DATABASE FUNCTIONS =================
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
#             expiry_timestamp DATETIME NOT NULL,
#             UNIQUE(symbol) ON CONFLICT REPLACE
#         )
#     ''')
    
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS analysis_metadata (
#             id INTEGER PRIMARY KEY,
#             total_analyzed INTEGER,
#             success_rate REAL,
#             last_update DATETIME,
#             expiry_timestamp DATETIME,
#             status TEXT,
#             processing_time_minutes REAL
#         )
#     ''')
    
#     conn.commit()
#     conn.close()
#     logger.info("Database initialized successfully")

# def save_analysis_to_db(results):
#     """Save analysis results to database with expiry timestamp"""
#     conn = sqlite3.connect(DATABASE_PATH)
#     cursor = conn.cursor()

#     try:
#         expiry_time = (datetime.now() + timedelta(hours=CACHE_EXPIRY_HOURS)).isoformat()
        
#         saved_count = 0
#         for symbol, data in results.items():
#             if symbol not in ['timestamp', 'stocks_analyzed', 'total_requested', 'success_rate', 'status', 'data_sources', 'markets', 'processing_info', 'data_source', 'note', 'error', 'processing_time_minutes']:
#                 try:
#                     cursor.execute('''
#                         INSERT OR REPLACE INTO analysis_results 
#                         (symbol, market, data_source, analysis_data, expiry_timestamp)
#                         VALUES (?, ?, ?, ?, ?)
#                     ''', (
#                         symbol,
#                         data.get('market', 'Unknown'),
#                         data.get('data_source', 'Unknown'),
#                         json.dumps(data),
#                         expiry_time
#                     ))
#                     saved_count += 1
#                 except Exception as e:
#                     logger.error(f"Error saving {symbol}: {e}")
#                     continue
        
#         total_analyzed = results.get('stocks_analyzed', 0)
#         success_rate = results.get('success_rate', 0.0)
#         status = results.get('status', 'unknown')
#         processing_time_minutes = results.get('processing_time_minutes', 0.0)
        
#         cursor.execute('''
#             INSERT OR REPLACE INTO analysis_metadata 
#             (id, total_analyzed, success_rate, last_update, expiry_timestamp, status, processing_time_minutes)
#             VALUES (1, ?, ?, ?, ?, ?, ?)
#         ''', (
#             total_analyzed,
#             success_rate,
#             datetime.now().isoformat(),
#             expiry_time,
#             status,
#             processing_time_minutes
#         ))
        
#         conn.commit()
#         logger.info(f"Successfully saved {saved_count} analysis results and metadata to database")
        
#     except Exception as e:
#         logger.error(f"Error saving to database: {str(e)}")
#         conn.rollback()
#         raise e
#     finally:
#         conn.close()

# def load_analysis_from_db():
#     """Load latest analysis results from database if not expired"""
#     conn = sqlite3.connect(DATABASE_PATH)
#     cursor = conn.cursor()

#     try:
#         cursor.execute('SELECT * FROM analysis_metadata WHERE id = 1')
#         metadata = cursor.fetchone()
        
#         if not metadata:
#             logger.info("No metadata found in database")
#             return None
        
#         expiry_timestamp = datetime.fromisoformat(metadata[4])
#         if datetime.now() > expiry_timestamp:
#             logger.info("Cached data expired")
#             return None
        
#         cursor.execute('SELECT symbol, analysis_data FROM analysis_results WHERE expiry_timestamp > ?', (datetime.now().isoformat(),))
#         stock_results = cursor.fetchall()
        
#         if not stock_results:
#             logger.info("No valid stock results found")
#             return None
        
#         response = {
#             'timestamp': metadata[3],
#             'stocks_analyzed': metadata[1],
#             'success_rate': metadata[2],
#             'status': metadata[5] if metadata[5] else 'success',
#             'processing_time_minutes': metadata[6] if metadata[6] else 0.0,
#             'data_source': 'database_cache',
#             'markets': {'us_stocks': 0, 'nigerian_stocks': 0, 'crypto_assets': 0},
#             'data_sources': {'marketstack_count': 0, 'cryptcompare_count': 0}
#         }
        
#         for symbol, analysis_json in stock_results:
#             try:
#                 analysis_data = json.loads(analysis_json)
#                 response[symbol] = analysis_data
                
#                 market = analysis_data.get('market', 'Unknown')
#                 if market == 'US':
#                     response['markets']['us_stocks'] += 1
#                 elif market == 'Nigerian':
#                     response['markets']['nigerian_stocks'] += 1
#                 elif market == 'Crypto':
#                     response['markets']['crypto_assets'] += 1
                
#                 data_source = analysis_data.get('data_source', 'Unknown')
#                 if data_source == 'marketstack':
#                     response['data_sources']['marketstack_count'] += 1
#                 elif data_source == 'cryptcompare':
#                     response['data_sources']['cryptcompare_count'] += 1
                
#             except json.JSONDecodeError as e:
#                 logger.error(f"Error parsing analysis data for {symbol}: {e}")
#                 continue
        
#         logger.info(f"Loaded {len(stock_results)} fresh analysis results from database")
#         return response
        
#     except Exception as e:
#         logger.error(f"Error loading from database: {str(e)}")
#         return None
#     finally:
#         conn.close()

# # ================= AI ANALYSIS INTEGRATION =================
# def generate_ai_analysis(symbol, stock_data):
#     """Generate detailed AI analysis using OpenRouter API"""
#     if not openrouter_client_available:
#         return {
#             'error': 'OpenRouter API not configured',
#             'message': 'Please configure OPENROUTER_API_KEY to use AI analysis'
#         }
    
#     try:
#         context = f"""
#         Stock Symbol: {symbol}
#         Current Analysis Data:
#         - Current Price: ${stock_data.get('DAILY_TIMEFRAME', {}).get('PRICE', 'N/A')}
#         - Verdict: {stock_data.get('DAILY_TIMEFRAME', {}).get('VERDICT', 'N/A')}
#         - Confidence Score: {stock_data.get('DAILY_TIMEFRAME', {}).get('CONFIDENCE_SCORE', 'N/A')}
#         - Market: {stock_data.get('market', 'Unknown')}
#         - Data Source: {stock_data.get('data_source', 'Unknown')}
        
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
        
#         # OpenRouter API request
#         headers = {
#             "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#             "Content-Type": "application/json",
#             "HTTP-Referer": "https://your-app-domain.com",  # Optional: replace with your domain
#             "X-Title": "Stock Analysis AI"  # Optional: your app name
#         }
        
#         payload = {
#             "model": "anthropic/claude-3.5-sonnet",  # You can also use "openai/gpt-4o" or other models
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ],
#             "max_tokens": 2000,
#             "temperature": 0.3
#         }
        
#         response = requests.post(
#             f"{OPENROUTER_BASE_URL}/chat/completions",
#             headers=headers,
#             json=payload,
#             timeout=60
#         )
        
#         response.raise_for_status()
#         response_data = response.json()
        
#         if 'choices' in response_data and len(response_data['choices']) > 0:
#             analysis_text = response_data['choices'][0]['message']['content']
#             model_used = response_data.get('model', 'anthropic/claude-3.5-sonnet')
#         else:
#             analysis_text = "No analysis returned from OpenRouter API."
#             model_used = "unknown"

#         return {
#             'analysis': analysis_text,
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#             'model': model_used,
#             'symbol': symbol,
#             'provider': 'openrouter'
#         }
        
#     except requests.exceptions.RequestException as e:
#         logger.error(f"OpenRouter API request error for {symbol}: {str(e)}")
#         return {
#             'error': 'OpenRouter API request failed',
#             'message': str(e)
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
        
#         result = analyze_all_stocks_unified()
        
#         if result and result.get('status') == 'success':
#             save_analysis_to_db(result)
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

# # ================= FLASK ROUTES =================
# @app.route('/', methods=['GET'])
# def home():
#     """Enhanced home endpoint with unified data source info"""
#     try:
#         stock_config = get_expanded_stocks_config()
        
#         cached_data = load_analysis_from_db()
#         has_cached_data = cached_data is not None
        
#         return jsonify({
#             'message': 'Enhanced Multi-Asset Analysis API v5.0 - Unified MarketStack + CryptoCompare',
#             'version': '5.0 - Expanded Coverage (140 Assets) + Unified Data Sources',
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
#                 'unified_analysis': True,
#                 'timeframes': ['daily'],
#                 'ai_analysis': openrouter_client_available,
#                 'persistent_storage': True,
#                 'background_processing': True,
#                 'progress_tracking': True,
#                 'daily_auto_refresh': '5:00 PM',
#                 'data_sources': ['marketstack', 'cryptcompare'],
#                 'expanded_coverage': True,
#                 'marketstack_plan': 'Basic (10,000 requests/month)'
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
#     """List all available assets with expanded coverage"""
#     try:
#         stock_config = get_expanded_stocks_config()
#         return jsonify({
#             'us_stocks': [{'symbol': s['symbol'], 'name': s['name'], 'exchange': s['exchange']} for s in stock_config['us_stocks']],
#             'nigerian_stocks': [{'symbol': s['symbol'], 'name': s['name'], 'exchange': s['exchange']} for s in stock_config['nigerian_stocks']],
#             'crypto_assets': [{'symbol': s['symbol'], 'name': s['name']} for s in stock_config['crypto_stocks']],
#             'data_source_distribution': {
#                 'marketstack_us': len(stock_config['us_stocks']),
#                 'marketstack_nigerian': len(stock_config['nigerian_stocks']),
#                 'cryptcompare_crypto': len(stock_config['crypto_stocks'])
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
#             'version': '5.0 - Unified MarketStack + CryptoCompare',
#             'markets': ['US', 'Nigerian', 'Crypto'],
#             'features': {
#                 'unified_analysis': True,
#                 'ai_analysis': openrouter_client_available,
#                 'expanded_coverage': True,
#                 'persistent_storage': True,
#                 'background_processing': True,
#                 'progress_tracking': True,
#                 'marketstack_integration': True
#             },
#             'data_status': {
#                 'has_cached_data': cached_data is not None,
#                 'analysis_in_progress': analysis_in_progress,
#                 'last_update': cached_data.get('timestamp') if cached_data else None
#             },
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#             'service': 'Multi-Asset Analysis API v5.0 - Unified Data Sources (140 Assets)'
#         })
#     except Exception as e:
#         logger.error(f"Error in health endpoint: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/analyze', methods=['GET'])
# def analyze():
#     """Get latest analysis - from cache if available, otherwise return cached data"""
#     try:
#         cached_data = load_analysis_from_db()
        
#         if cached_data:
#             logger.info(f"Returning cached analysis data from {cached_data.get('timestamp')}")
#             cached_data['data_source'] = 'database_cache'
#             cached_data['note'] = 'This is cached data. Use /analyze/fresh for new analysis.'
#             return jsonify(cached_data)
#         else:
#             logger.info("No cached data found, running fresh analysis...")
#             json_response = analyze_all_stocks_unified()
            
#             if json_response and json_response.get('status') == 'success':
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
        
#         json_response = analyze_all_stocks_unified()
        
#         if json_response and json_response.get('status') == 'success':
#             processing_time = (time.time() - start_time) / 60
#             json_response['processing_time_minutes'] = round(processing_time, 2)
            
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
        
#         # Try to get from cache first
#         cached_data = load_analysis_from_db()
#         stock_analysis = None
        
#         if cached_data and symbol in cached_data:
#             stock_analysis = {symbol: cached_data[symbol]}
#             logger.info(f"Using cached data for AI analysis of {symbol}")
#         else:
#             # Determine data source and get fresh analysis
#             stock_config = get_expanded_stocks_config()
            
#             # Find the stock in our configuration
#             stock_info = None
#             data_source = None
            
#             for stock in stock_config['us_stocks']:
#                 if stock['symbol'] == symbol:
#                     stock_info = stock
#                     data_source = "marketstack"
#                     break
            
#             if not stock_info:
#                 for stock in stock_config['nigerian_stocks']:
#                     if stock['symbol'] == symbol:
#                         stock_info = stock
#                         data_source = "marketstack"
#                         break
            
#             if not stock_info:
#                 for stock in stock_config['crypto_stocks']:
#                     if stock['symbol'] == symbol:
#                         stock_info = stock
#                         data_source = "cryptcompare"
#                         break
            
#             if not stock_info:
#                 return jsonify({
#                     'error': 'Symbol not found',
#                     'message': f'Symbol {symbol} not found in our asset list'
#                 }), 404
            
#             # Get fresh analysis for this symbol
#             stock_analysis = analyze_stock_unified(stock_info, data_source)
        
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
#         scheduler.add_job(
#             func=analyze_all_stocks_background,
#             trigger=CronTrigger(hour=17, minute=0),
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
    
#     logger.info(f"Starting Enhanced Multi-Asset Analysis API v5.0 - Unified MarketStack + CryptoCompare on port {port}")
#     logger.info(f"Debug mode: {debug_mode}")
#     logger.info(f"Total assets configured: {get_expanded_stocks_config()['total_count']}")
#     logger.info(f"AI Analysis available: {openrouter_client_available}")
#     logger.info("Features: Unified Data Sources + Expanded Coverage (140 Assets) + MarketStack Basic Plan + OpenRouter AI")
    
#     try:
#         app.run(host='0.0.0.0', port=port, debug=debug_mode, threaded=True)
#     finally:
#         if scheduler.running:
#             scheduler.shutdown()
