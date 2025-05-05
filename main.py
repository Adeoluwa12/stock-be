from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple, Optional
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# # Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# Update this part in your main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # We'll update this with the specific frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/favicon.ico")
async def favicon():
    return FileResponse("favicon.ico", media_type="image/x-icon")

# ==================== MACD INDICATOR ====================
def get_macd_signal(symbol: str, interval: str = "1d") -> Dict[str, Union[str, float, None]]:
    try:
        # For weekly data, we need more history
        period = "1y" if interval == "1wk" else "6mo"
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        if data.empty or len(data) < 26:
            logger.warning(f"Not enough data for MACD calculation for {symbol}")
            return {"signal": "SELL", "price": None}  # Default to SELL if not enough data

        # Calculate MACD
        data["EMA12"] = data["Close"].ewm(span=12, adjust=False).mean()
        data["EMA26"] = data["Close"].ewm(span=26, adjust=False).mean()
        data["MACD"] = data["EMA12"] - data["EMA26"]
        data["Signal Line"] = data["MACD"].ewm(span=9, adjust=False).mean()
        data["MACD Histogram"] = data["MACD"] - data["Signal Line"]

        # Get latest values
        latest_macd = float(data["MACD"].iloc[-1])
        latest_signal = float(data["Signal Line"].iloc[-1])
        latest_hist = float(data["MACD Histogram"].iloc[-1])
        price = float(data["Close"].iloc[-1])

        # Determine signal
        if latest_macd > latest_signal:
            signal = "BUY"
        else:
            signal = "SELL"

        logger.info(f"MACD for {symbol}: {signal} (MACD: {latest_macd:.4f}, Signal: {latest_signal:.4f})")
        return {"signal": signal, "price": price}
    except Exception as e:
        logger.error(f"Error calculating MACD for {symbol}: {str(e)}")
        return {"signal": "SELL", "price": None}  # Default to SELL on error

# ==================== ELLIOTT WAVE ANALYSIS ====================
def detect_elliott_wave(symbol: str, interval: str = "1d") -> Dict[str, str]:
    try:
        # For weekly data, we need more history
        period = "5y" if interval == "1wk" else "2y"
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        if data.empty or len(data) < 50:
            logger.warning(f"Not enough data for Elliott Wave analysis for {symbol}")
            return {"elliott_wave": "INSUFFICIENT_DATA", "wave_verdict": "SELL"}
        
        # Calculate price momentum and trend
        data['SMA50'] = data['Close'].rolling(window=50).mean()
        data['SMA200'] = data['Close'].rolling(window=200).mean()
        
        # Calculate RSI for momentum
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Get latest values
        current_price = float(data['Close'].iloc[-1])
        sma50 = float(data['SMA50'].iloc[-1])
        sma200 = float(data['SMA200'].iloc[-1])
        rsi = float(data['RSI'].iloc[-1])
        
        # Calculate price change over different periods
        price_change_5d = (current_price / float(data['Close'].iloc[-6]) - 1) if len(data) > 5 else 0
        price_change_20d = (current_price / float(data['Close'].iloc[-21]) - 1) if len(data) > 20 else 0
        
        # Determine Elliott Wave pattern based on trend, momentum and price action
        wave_pattern = "NONE"
        verdict = "SELL"  # Default to SELL
        
        # Determine if we're in an uptrend or downtrend
        if current_price > sma50 and sma50 > sma200:
            # Uptrend
            if rsi > 70:
                # Overbought - potential end of wave 5 or 3
                wave_pattern = "IMPULSE_WAVE_5"
                verdict = "SELL"
            elif rsi < 30:
                # Oversold in uptrend - potential wave 2 or 4 retracement
                wave_pattern = "IMPULSE_WAVE_4"
                verdict = "BUY"
            elif price_change_5d > 0.03:
                # Strong recent momentum - potential wave 3
                wave_pattern = "IMPULSE_WAVE_3"
                verdict = "BUY"
            else:
                # Default in uptrend
                wave_pattern = "IMPULSE_WAVE_1"
                verdict = "BUY"
        else:
            # Downtrend
            if rsi < 30:
                # Oversold - potential end of wave 5 or 3
                wave_pattern = "CORRECTIVE_WAVE_C"
                verdict = "BUY"
            elif rsi > 70:
                # Overbought in downtrend - potential wave B
                wave_pattern = "CORRECTIVE_WAVE_B"
                verdict = "SELL"
            elif price_change_5d < -0.03:
                # Strong recent downward momentum - potential wave C
                wave_pattern = "CORRECTIVE_WAVE_C"
                verdict = "SELL"
            else:
                # Default in downtrend
                wave_pattern = "CORRECTIVE_WAVE_A"
                verdict = "SELL"
        
        logger.info(f"Elliott Wave for {symbol}: {wave_pattern} with verdict {verdict}")
        return {"elliott_wave": wave_pattern, "wave_verdict": verdict}
    except Exception as e:
        logger.error(f"Error detecting Elliott Wave for {symbol}: {str(e)}")
        return {"elliott_wave": "ERROR", "wave_verdict": "SELL"}

# ==================== AUTO CHART PATTERNS ====================
def detect_auto_chart_pattern(symbol: str, interval: str = "1d") -> Dict[str, str]:
    try:
        # For weekly data, we need more history
        period = "3y" if interval == "1wk" else "1y"
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        if data.empty or len(data) < 50:
            logger.warning(f"Not enough data for Auto Chart Pattern analysis for {symbol}")
            return {"auto_pattern": "INSUFFICIENT_DATA", "auto_verdict": "SELL"}
        
        # Calculate technical indicators
        # Moving averages
        data['SMA20'] = data['Close'].rolling(window=20).mean()
        data['SMA50'] = data['Close'].rolling(window=50).mean()
        data['SMA200'] = data['Close'].rolling(window=200).mean()
        
        # Bollinger Bands
        data['20dSTD'] = data['Close'].rolling(window=20).std()
        data['UpperBand'] = data['SMA20'] + (data['20dSTD'] * 2)
        data['LowerBand'] = data['SMA20'] - (data['20dSTD'] * 2)
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Get latest values
        current_price = float(data['Close'].iloc[-1])
        sma20 = float(data['SMA20'].iloc[-1])
        sma50 = float(data['SMA50'].iloc[-1])
        sma200 = float(data['SMA200'].iloc[-1])
        upper_band = float(data['UpperBand'].iloc[-1])
        lower_band = float(data['LowerBand'].iloc[-1])
        rsi = float(data['RSI'].iloc[-1])
        
        # Calculate price changes
        price_change_5d = (current_price / float(data['Close'].iloc[-6]) - 1) if len(data) > 5 else 0
        price_change_20d = (current_price / float(data['Close'].iloc[-21]) - 1) if len(data) > 20 else 0
        
        # Calculate volatility
        recent_volatility = float(data['20dSTD'].iloc[-1]) / sma20
        
        # Determine pattern and verdict
        pattern = "NONE"
        verdict = "SELL"  # Default to SELL
        
        # Check for bullish patterns
        if current_price > sma50 and sma50 > sma200:
            # Uptrend
            if current_price > upper_band and rsi > 70:
                pattern = "OVERBOUGHT_CHANNEL"
                verdict = "SELL"  # Overbought in uptrend
            elif current_price < sma20 and rsi < 40:
                pattern = "PULLBACK_OPPORTUNITY"
                verdict = "BUY"  # Pullback in uptrend
            elif price_change_5d > 0.03:
                pattern = "STRONG_UPTREND"
                verdict = "BUY"  # Strong momentum
            else:
                pattern = "BULLISH_TREND"
                verdict = "BUY"  # General uptrend
        
        # Check for bearish patterns
        elif current_price < sma50 and sma50 < sma200:
            # Downtrend
            if current_price < lower_band and rsi < 30:
                pattern = "OVERSOLD_CHANNEL"
                verdict = "BUY"  # Oversold in downtrend
            elif current_price > sma20 and rsi > 60:
                pattern = "RALLY_OPPORTUNITY"
                verdict = "SELL"  # Rally in downtrend
            elif price_change_5d < -0.03:
                pattern = "STRONG_DOWNTREND"
                verdict = "SELL"  # Strong downward momentum
            else:
                pattern = "BEARISH_TREND"
                verdict = "SELL"  # General downtrend
        
        # Check for transition patterns
        else:
            if current_price > sma50 and sma50 < sma200:
                pattern = "GOLDEN_CROSS_FORMING"
                verdict = "BUY"  # Potential trend reversal up
            elif current_price < sma50 and sma50 > sma200:
                pattern = "DEATH_CROSS_FORMING"
                verdict = "SELL"  # Potential trend reversal down
            elif rsi > 60:
                pattern = "MOMENTUM_SHIFT_UP"
                verdict = "BUY"
            else:
                pattern = "MOMENTUM_SHIFT_DOWN"
                verdict = "SELL"
        
        logger.info(f"Auto Chart Pattern for {symbol}: {pattern} with verdict {verdict}")
        return {"auto_pattern": pattern, "auto_verdict": verdict}
    except Exception as e:
        logger.error(f"Error detecting Auto Chart Pattern for {symbol}: {str(e)}")
        return {"auto_pattern": "ERROR", "auto_verdict": "SELL"}

# ==================== ALL CHART PATTERNS ====================
def detect_all_chart_patterns(symbol: str, interval: str = "1d") -> Dict[str, str]:
    try:
        # For weekly data, we need more history
        period = "3y" if interval == "1wk" else "1y"
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        if data.empty or len(data) < 50:
            logger.warning(f"Not enough data for All Chart Patterns analysis for {symbol}")
            return {"all_pattern": "INSUFFICIENT_DATA", "all_verdict": "SELL"}
        
        # Calculate technical indicators
        # Moving averages
        data['SMA10'] = data['Close'].rolling(window=10).mean()
        data['SMA20'] = data['Close'].rolling(window=20).mean()
        data['SMA50'] = data['Close'].rolling(window=50).mean()
        data['SMA200'] = data['Close'].rolling(window=200).mean()
        
        # MACD
        data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA12'] - data['EMA26']
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Hist'] = data['MACD'] - data['Signal']
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator
        low_14 = data['Low'].rolling(window=14).min()
        high_14 = data['High'].rolling(window=14).max()
        data['%K'] = 100 * ((data['Close'] - low_14) / (high_14 - low_14))
        data['%D'] = data['%K'].rolling(window=3).mean()
        
        # Get latest values
        current_price = float(data['Close'].iloc[-1])
        prev_price = float(data['Close'].iloc[-2]) if len(data) > 1 else current_price
        sma10 = float(data['SMA10'].iloc[-1])
        sma20 = float(data['SMA20'].iloc[-1])
        sma50 = float(data['SMA50'].iloc[-1])
        sma200 = float(data['SMA200'].iloc[-1])
        macd = float(data['MACD'].iloc[-1])
        signal = float(data['Signal'].iloc[-1])
        macd_hist = float(data['MACD_Hist'].iloc[-1])
        rsi = float(data['RSI'].iloc[-1])
        stoch_k = float(data['%K'].iloc[-1])
        stoch_d = float(data['%D'].iloc[-1])
        
        # Calculate price changes
        price_change_1d = (current_price / prev_price) - 1
        price_change_5d = (current_price / float(data['Close'].iloc[-6])) - 1 if len(data) > 5 else 0
        price_change_20d = (current_price / float(data['Close'].iloc[-21])) - 1 if len(data) > 20 else 0
        
        # Check for specific chart patterns
        pattern = "NONE"
        verdict = "SELL"  # Default to SELL
        
        # Double Top
        if len(data) > 20:
            recent_highs = data['High'].iloc[-20:].values.tolist()
            if max(recent_highs) == recent_highs[-1] and max(recent_highs[:-1]) > sma20:
                pattern = "DOUBLE_TOP"
                verdict = "SELL"
        
        # Double Bottom
        if len(data) > 20:
            recent_lows = data['Low'].iloc[-20:].values.tolist()
            if min(recent_lows) == recent_lows[-1] and min(recent_lows[:-1]) < sma20:
                pattern = "DOUBLE_BOTTOM"
                verdict = "BUY"
        
        # Head and Shoulders (bearish)
        if current_price < sma50 and macd < signal and rsi < 40:
            pattern = "HEAD_AND_SHOULDERS"
            verdict = "SELL"
        
        # Inverse Head and Shoulders (bullish)
        if current_price > sma50 and macd > signal and rsi > 60:
            pattern = "INV_HEAD_AND_SHOULDERS"
            verdict = "BUY"
        
        # Bullish Flag
        if price_change_20d > 0.1 and price_change_5d < 0 and price_change_5d > -0.05:
            pattern = "BULL_FLAG"
            verdict = "BUY"
        
        # Bearish Flag
        if price_change_20d < -0.1 and price_change_5d > 0 and price_change_5d < 0.05:
            pattern = "BEAR_FLAG"
            verdict = "SELL"
        
        # Cup and Handle
        if price_change_20d > 0.05 and current_price > sma50 and rsi > 50:
            pattern = "CUP_AND_HANDLE"
            verdict = "BUY"
        
        # If no specific pattern detected, use technical indicators
        if pattern == "NONE":
            # Use a combination of indicators for a more reliable signal
            bullish_signals = 0
            bearish_signals = 0
            
            # Moving Average signals
            if current_price > sma20:
                bullish_signals += 1
            else:
                bearish_signals += 1
                
            if sma20 > sma50:
                bullish_signals += 1
            else:
                bearish_signals += 1
                
            if sma50 > sma200:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            # MACD signal
            if macd > signal:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            # RSI signal
            if rsi > 50:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            # Stochastic signal
            if stoch_k > stoch_d:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            # Determine pattern and verdict based on signals
            if bullish_signals > bearish_signals:
                pattern = "TECHNICAL_BULLISH"
                verdict = "BUY"
            else:
                pattern = "TECHNICAL_BEARISH"
                verdict = "SELL"
        
        logger.info(f"All Chart Patterns for {symbol}: {pattern} with verdict {verdict}")
        return {"all_pattern": pattern, "all_verdict": verdict}
    except Exception as e:
        logger.error(f"Error detecting All Chart Patterns for {symbol}: {str(e)}")
        return {"all_pattern": "ERROR", "all_verdict": "SELL"}

# ==================== CYCLES ANALYSIS ====================
def detect_market_cycles(symbol: str, interval: str = "1d") -> Dict[str, str]:
    try:
        # For weekly data, we need more history
        period = "5y" if interval == "1wk" else "2y"
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        if data.empty or len(data) < 50:
            logger.warning(f"Not enough data for Market Cycles analysis for {symbol}")
            return {"cycle": "INSUFFICIENT_DATA", "cycle_verdict": "SELL"}
        
        # Calculate moving averages for cycle detection
        data['SMA20'] = data['Close'].rolling(window=20).mean()
        data['SMA50'] = data['Close'].rolling(window=50).mean()
        data['SMA200'] = data['Close'].rolling(window=200).mean()
        
        # Calculate momentum indicators
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Rate of Change
        data['ROC'] = data['Close'].pct_change(20) * 100
        
        # Get latest values
        current_price = float(data['Close'].iloc[-1])
        sma20 = float(data['SMA20'].iloc[-1])
        sma50 = float(data['SMA50'].iloc[-1])
        sma200 = float(data['SMA200'].iloc[-1])
        rsi = float(data['RSI'].iloc[-1])
        roc = float(data['ROC'].iloc[-1])
        
        # Determine market cycle
        cycle = "NONE"
        verdict = "SELL"  # Default to SELL
        
        # Bullish cycle (price above all MAs, strong momentum)
        if current_price > sma20 and sma20 > sma50 and sma50 > sma200 and rsi > 50 and roc > 0:
            cycle = "STRONG_BULLISH"
            verdict = "BUY"
        
        # Early bullish cycle (price above 50 MA, crossing above 200 MA)
        elif current_price > sma50 and sma50 > sma200 * 0.98 and rsi > 50:
            cycle = "EARLY_BULLISH"
            verdict = "BUY"
        
        # Bearish cycle (price below all MAs, weak momentum)
        elif current_price < sma20 and sma20 < sma50 and sma50 < sma200 and rsi < 50 and roc < 0:
            cycle = "STRONG_BEARISH"
            verdict = "SELL"
        
        # Early bearish cycle (price below 50 MA, crossing below 200 MA)
        elif current_price < sma50 and sma50 < sma200 * 1.02 and rsi < 50:
            cycle = "EARLY_BEARISH"
            verdict = "SELL"
        
        # Accumulation phase (price crossing above MAs after downtrend)
        elif current_price > sma20 and current_price > sma50 and sma20 > sma50 and sma50 < sma200:
            cycle = "ACCUMULATION"
            verdict = "BUY"
        
        # Distribution phase (price crossing below MAs after uptrend)
        elif current_price < sma20 and current_price < sma50 and sma20 < sma50 and sma50 > sma200:
            cycle = "DISTRIBUTION"
            verdict = "SELL"
        
        # Consolidation with bullish bias
        elif abs(sma20 - sma50) / sma50 < 0.02 and rsi > 50:
            cycle = "CONSOLIDATION_BULLISH"
            verdict = "BUY"
        
        # Consolidation with bearish bias
        elif abs(sma20 - sma50) / sma50 < 0.02 and rsi < 50:
            cycle = "CONSOLIDATION_BEARISH"
            verdict = "SELL"
        
        # Default based on price vs 200 MA
        elif current_price > sma200:
            cycle = "GENERAL_BULLISH"
            verdict = "BUY"
        else:
            cycle = "GENERAL_BEARISH"
            verdict = "SELL"
        
        logger.info(f"Market Cycles for {symbol}: {cycle} with verdict {verdict}")
        return {"cycle": cycle, "cycle_verdict": verdict}
    except Exception as e:
        logger.error(f"Error detecting Market Cycles for {symbol}: {str(e)}")
        return {"cycle": "ERROR", "cycle_verdict": "SELL"}

# ==================== MAIN API ENDPOINT ====================
@app.get("/analyze")
async def analyze(timeframe: str = "daily"):
    stock_list = [
        "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN",
        "NVDA", "META", "NFLX", "BABA", "AMD",
        "INTC", "UBER", "SHOP", "PYPL", "PEP",
        "KO", "DIS", "NKE", "WMT", "CRM"
    ]
    results = []
    
    interval = "1wk" if timeframe == "weekly" else "1d"
    logger.info(f"Analyzing stocks with timeframe: {timeframe}, interval: {interval}")

    for symbol in stock_list:
        logger.info(f"Analyzing {symbol}...")
        try:
            # Get all indicators
            macd = get_macd_signal(symbol, interval)
            elliott = detect_elliott_wave(symbol, interval)
            auto_pattern = detect_auto_chart_pattern(symbol, interval)
            all_pattern = detect_all_chart_patterns(symbol, interval)
            cycle_analysis = detect_market_cycles(symbol, interval)

            stock_data = {
                "symbol": symbol,
                "price": macd["price"],
                # MACD data
                "macd_signal": macd["signal"],
                # Elliott Wave data
                "elliott_wave": elliott["elliott_wave"],
                "wave_verdict": elliott["wave_verdict"],
                # Auto Chart Patterns data
                "auto_chart_pattern": auto_pattern["auto_pattern"],
                "auto_pattern_verdict": auto_pattern["auto_verdict"],
                # All Chart Patterns data
                "all_chart_pattern": all_pattern["all_pattern"],
                "all_pattern_verdict": all_pattern["all_verdict"],
                # Cycles Analysis data
                "market_cycle": cycle_analysis["cycle"],
                "cycle_verdict": cycle_analysis["cycle_verdict"]
            }

            results.append(stock_data)
            logger.info(f"Completed analysis for {symbol}")
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            # Add default data in case of error
            stock_data = {
                "symbol": symbol,
                "price": None,
                "macd_signal": "SELL",
                "elliott_wave": "ERROR",
                "wave_verdict": "SELL",
                "auto_chart_pattern": "ERROR",
                "auto_pattern_verdict": "SELL",
                "all_chart_pattern": "ERROR",
                "all_pattern_verdict": "SELL",
                "market_cycle": "ERROR",
                "cycle_verdict": "SELL"
            }
            results.append(stock_data)

    return JSONResponse(content=results)
