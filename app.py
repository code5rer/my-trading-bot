import streamlit as st
import pandas as pd
import numpy as np
import v20
from datetime import datetime
import time

# --- UI SETUP ---
st.set_page_config(page_title="Zarattini VWAP Bot", layout="wide")
st.title("🚀 Zarattini & Aziz VWAP Trading Bot")

# --- SIDEBAR: CONFIGURATION ---
st.sidebar.header("1. API Configuration")
api_key = st.sidebar.text_input("OANDA API Key", type="password")
account_id = st.sidebar.text_input("OANDA Account ID")
instrument = st.sidebar.selectbox("Instrument", ["NAS100_USD", "US30_USD", "EUR_USD"])

st.sidebar.header("2. Strategy Tuning")
risk_level = st.sidebar.slider("Risk per Trade (%)", 0.1, 2.0, 0.5)

# Initialize Session State for PnL tracking
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = pd.DataFrame(columns=['Time', 'Side', 'Entry', 'Exit', 'PnL', 'Type'])

# --- CORE LOGIC ---
def get_data(ctx, instrument):
    try:
        r = ctx.candle.get(instrument, granularity="M5", count=200)
        candles = r.get("candles", 200)
        data = []
        for c in candles:
            if c.complete:
                data.append({
                    'time': c.time,
                    'close': float(c.mid.c),
                    'high': float(c.mid.h),
                    'low': float(c.mid.l),
                    'volume': int(c.volume)
                })
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return None

def calculate_vwap(df):
    # Zarattini Strategy VWAP
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['pv'] = df['tp'] * df['volume']
    df['vwap'] = df['pv'].cumsum() / df['volume'].cumsum()
    return df

def optimize_logic(history):
    """Self-learning: Adjusts sensitivity based on recent win rate"""
    if len(history) < 5:
        return 1.0  # Base multiplier
    win_rate = len(history[history['PnL'] > 0]) / len(history)
    # If winning, take more risk; if losing, tighten the strategy
    return 1.2 if win_rate > 0.6 else 0.8

# --- MAIN DASHBOARD ---
if api_key and account_id:
    ctx = v20.Context("api-fxtrade.oanda.com", 443, token=api_key)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Market View & VWAP")
        df = get_data(ctx, instrument)
        if df is not None:
            df = calculate_vwap(df)
            st.line_chart(df[['close', 'vwap']])
            
            # SIGNAL LOGIC
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            
            if prev.close < prev.vwap and curr.close > curr.vwap:
                st.success(f"BULLISH SIGNAL DETECTED at {curr.close}")
                # Code to execute OANDA order would go here
            elif prev.close > prev.vwap and curr.close < curr.vwap:
                st.warning(f"BEARISH SIGNAL DETECTED at {curr.close}")

    with col2:
        st.subheader("PnL Decomposition")
        if not st.session_state.trade_history.empty:
            th = st.session_state.trade_history
            long_pnl = th[th['Side'] == 'Long']['PnL'].sum()
            short_pnl = th[th['Side'] == 'Short']['PnL'].sum()
            st.metric("Total PnL", f"${th['PnL'].sum():.2f}")
            st.write(f"🟢 Longs: ${long_pnl:.2f}")
            st.write(f"🔴 Shorts: ${short_pnl:.2f}")
        else:
            st.info("Waiting for first trade...")

    # Refresh Loop
    time.sleep(60)
    st.rerun()
else:
    st.info("Please enter your OANDA API details in the sidebar to start.")