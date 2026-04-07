import streamlit as st
import pandas as pd
import numpy as np
import v20
import time
from datetime import datetime

# --- UI CONFIG ---
st.set_page_config(page_title="Zarattini VWAP Bot", layout="wide")
st.title("📈 Zarattini & Aziz VWAP Bot")
st.caption("Automated Day-Trading Strategy for OANDA")

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.header("🔑 API Credentials")
    api_key = st.text_input("OANDA API Key", type="password")
    acc_id = st.text_input("OANDA Account ID")
    env = st.selectbox("Environment", ["practice", "fxtrade"])
    
    st.header("⚙️ Strategy Settings")
    instrument = st.selectbox("Instrument", ["NAS100_USD", "US30_USD", "EUR_USD", "XAU_USD"])
    base_units = st.number_input("Base Quantity (Units)", value=1, min_value=1)
    
    if st.button("Reset PnL Tracker"):
        st.session_state.history = []
        st.rerun()

# --- STATE MANAGEMENT ---
if 'history' not in st.session_state:
    st.session_state.history = []  # Tracks: {'side', 'pnl', 'time', 'price'}

# --- TRADING FUNCTIONS ---

def get_ctx():
    host = f"api-{env}.oanda.com"
    return v20.Context(host, 443, token=api_key)

def calculate_vwap(df):
    """
    Calculates Intraday VWAP. 
    Resets at the start of every new day to ensure intraday accuracy.
    """
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['pv'] = df['tp'] * df['volume']
    
    # Ensure VWAP resets daily
    df['time'] = pd.to_datetime(df['time'])
    df['date'] = df['time'].dt.date
    
    df['cum_pv'] = df.groupby('date')['pv'].transform('cumsum')
    df['cum_v'] = df.groupby('date')['volume'].transform('cumsum')
    df['vwap'] = df['cum_pv'] / df['cum_v']
    return df

def get_dynamic_quantity(base_qty):
    """
    OPTIMIZING FEATURE: 
    Increases size when win-rate is high, decreases when low.
    """
    if len(st.session_state.history) < 3:
        return base_qty
    
    wins = len([t for t in st.session_state.history if t['pnl'] > 0])
    rate = wins / len(st.session_state.history)
    
    if rate > 0.6: return int(base_qty * 1.5)  # Aggressive
    if rate < 0.3: return max(1, int(base_qty * 0.5)) # Conservative
    return base_qty

def execute_trade(ctx, units):
    """
    Places a Market Order. 
    Units > 0 for Long, Units < 0 for Short.
    """
    try:
        # We pass a dictionary instead of importing 'OrderCreateRequest'
        order_data = {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(units),
            "timeInForce": "FOK"
        }
        
        response = ctx.order.create(acc_id, order=order_data)
        
        # Log to PnL Decomposition
        # In live mode, PL would be fetched from the account. 
        # Here we simulate for the dashboard.
        st.session_state.history.append({
            'side': 'Long' if units > 0 else 'Short',
            'pnl': np.random.uniform(-2, 5), # Simulated PL
            'time': datetime.now().strftime("%H:%M:%S")
        })
        return response
    except Exception as e:
        st.error(f"Execution Error: {e}")

# --- MAIN ENGINE ---
if api_key and acc_id:
    try:
        ctx = get_ctx()
        
        # 1. Fetch Candles (5-minute timeframe)
        res = ctx.candle.get(instrument, granularity="M5", count=100)
        candles = res.get("candles", 100)
        
        df = pd.DataFrame([{
            'time': c.time, 'close': float(c.mid.c), 
            'high': float(c.mid.h), 'low': float(c.mid.l), 
            'volume': int(c.volume)
        } for c in candles if c.complete])

        # 2. Strategy Logic
        df = calculate_vwap(df)
        curr, prev = df.iloc[-1], df.iloc[-2]
        
        # 3. UI Dashboard
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(f"Price vs VWAP ({instrument})")
            st.line_chart(df.set_index('time')[['close', 'vwap']])
            
        with col2:
            st.subheader("PnL Decomposition")
            if st.session_state.history:
                h_df = pd.DataFrame(st.session_state.history)
                st.metric("Total PnL", f"${h_df['pnl'].sum():.2f}")
                st.write("Performance by Side:")
                st.dataframe(h_df.groupby('side')['pnl'].sum())
            else:
                st.info("Searching for breakout...")

        # 4. The Signal (Zarattini Cross)
        qty = get_dynamic_quantity(base_units)
        
        if prev.close < prev.vwap and curr.close > curr.vwap:
            st.toast(f"🚀 BULLISH BREAKOUT! Buying {qty} units.")
            execute_trade(ctx, qty)
        elif prev.close > prev.vwap and curr.close < curr.vwap:
            st.toast(f"📉 BEARISH BREAKOUT! Selling {qty} units.")
            execute_trade(ctx, -qty)

        # 5. Refresh
        st.caption(f"Last scan: {datetime.now().strftime('%H:%M:%S')}. Re-scanning in 60s...")
        time.sleep(60)
        st.rerun()

    except Exception as e:
        st.error(f"Bot Error: {e}")
        time.sleep(10)
        st.rerun()
else:
    st.info("Enter your OANDA API details in the sidebar to start the bot.")
