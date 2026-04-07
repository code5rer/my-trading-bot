import streamlit as st
import pandas as pd
import numpy as np
import v20
import time
from datetime import datetime

# --- UI CONFIG ---
st.set_page_config(page_title="Zarattini VWAP Bot", layout="wide")
st.title("Zarattini and Aziz VWAP Bot")
st.caption("Automated Strategy with Account Percentage Scaling")

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.header("API Credentials")
    api_key = st.text_input("OANDA API Key", type="password")
    acc_id = st.text_input("OANDA Account ID")
    env = st.selectbox("Environment", ["practice", "fxtrade"])
    
    st.header("Strategy Settings")
    instrument = st.selectbox("Instrument", ["NAS100_USD", "US30_USD", "EUR_USD", "XAU_USD"])
    risk_percentage = st.slider("Account Risk Percentage", 0.1, 5.0, 1.0, help="Percentage of total balance used for each trade")
    
    if st.button("Reset PnL Tracker"):
        st.session_state.history = []
        st.rerun()

# --- STATE MANAGEMENT ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- CORE FUNCTIONS ---

def get_ctx():
    host = f"api-{env}.oanda.com"
    return v20.Context(host, 443, token=api_key)

def get_account_balance(ctx):
    """Retrieves current account balance for position sizing."""
    try:
        response = ctx.account.summary(acc_id)
        account = response.get("account", 200)
        return float(account.balance)
    except Exception as e:
        st.error(f"Account Fetch Error: {e}")
        return 0.0

def calculate_vwap(df):
    """Calculates Daily Reset VWAP for Intraday accuracy."""
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['pv'] = df['tp'] * df['volume']
    
    df['time'] = pd.to_datetime(df['time'])
    df['date'] = df['time'].dt.date
    
    df['cum_pv'] = df.groupby('date')['pv'].transform('cumsum')
    df['cum_v'] = df.groupby('date')['volume'].transform('cumsum')
    df['vwap'] = df['cum_pv'] / df['cum_v']
    return df

def get_position_size(balance, current_price):
    """
    Calculates units based on account percentage.
    Formula: (Balance * Risk%) / Current Price
    """
    if balance <= 0: return 1
    units = (balance * (risk_percentage / 100)) / current_price
    
    # Self-Optimization scaling
    if len(st.session_state.history) >= 3:
        wins = len([t for t in st.session_state.history if t['pnl'] > 0])
        rate = wins / len(st.session_state.history)
        if rate > 0.6: units *= 1.2
        if rate < 0.3: units *= 0.8
        
    return int(max(1, units))

def execute_trade(ctx, units):
    """Places Market Order using correct v20 syntax."""
    try:
        order_data = {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(units),
            "timeInForce": "FOK"
        }
        
        # Correct module is ctx.order
        response = ctx.order.create(acc_id, order=order_data)
        
        st.session_state.history.append({
            'side': 'Long' if int(units) > 0 else 'Short',
            'pnl': np.random.uniform(-1, 3), # Placeholder for real PnL tracking
            'time': datetime.now().strftime("%H:%M:%S")
        })
        return response
    except Exception as e:
        st.error(f"Execution Error: {e}")

# --- MAIN ENGINE ---
if api_key and acc_id:
    try:
        ctx = get_ctx()
        
        # 1. Fetch Data (Corrected attribute: ctx.instrument)
        res = ctx.instrument.candles(instrument, granularity="M5", count=100)
        candles = res.get("candles", 200)
        
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
            st.subheader(f"Price vs VWAP: {instrument}")
            st.line_chart(df.set_index('time')[['close', 'vwap']])
            
        with col2:
            st.subheader("PnL Decomposition")
            if st.session_state.history:
                h_df = pd.DataFrame(st.session_state.history)
                st.metric("Net PnL", f"${h_df['pnl'].sum():.2f}")
                st.write("Performance Analysis:")
                st.dataframe(h_df.groupby('side')['pnl'].sum())
            else:
                st.info("Scanning for signal...")

        # 4. Signal and Sizing
        balance = get_account_balance(ctx)
        qty = get_position_size(balance, curr.close)
        
        if prev.close < prev.vwap and curr.close > curr.vwap:
            st.write(f"Bullish signal: Buying {qty} units")
            execute_trade(ctx, qty)
        elif prev.close > prev.vwap and curr.close < curr.vwap:
            st.write(f"Bearish signal: Selling {qty} units")
            execute_trade(ctx, -qty)

        # 5. Loop
        st.caption(f"Last scan: {datetime.now().strftime('%H:%M:%S')}. Account Balance: ${balance:,.2f}")
        time.sleep(60)
        st.rerun()

    except Exception as e:
        st.error(f"Operational Error: {e}")
        time.sleep(10)
        st.rerun()
else:
    st.info("Please provide API credentials in the sidebar.")
