import streamlit as st
import pandas as pd
import numpy as np
import v20
import time
from datetime import datetime

# --- UI CONFIG ---
st.set_page_config(page_title="Zarattini VWAP Multi-Bot", layout="wide")
st.title("Zarattini and Aziz Multi-Instrument Bot")

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.header("API Credentials")
    api_key = st.text_input("OANDA API Key", type="password")
    acc_id = st.text_input("OANDA Account ID")
    env = st.selectbox("Environment", ["practice", "fxtrade"])
    
    st.header("Strategy Settings")
    # Updated to include all previously discussed instruments
    instruments = st.multiselect(
        "Instruments to Trade", 
        ["NAS100_USD", "US30_USD", "EUR_USD", "XAU_USD"],
        default=["NAS100_USD", "US30_USD", "EUR_USD", "XAU_USD"]
    )
    risk_percentage = st.slider("Account Risk Percentage", 0.1, 5.0, 1.0)
    
    if st.button("Reset Session History"):
        st.session_state.history = []
        st.rerun()

if 'history' not in st.session_state:
    st.session_state.history = []

# --- CORE FUNCTIONS ---

def get_ctx():
    host = "api-fxpractice.oanda.com" if env == "practice" else "api-fxtrade.oanda.com"
    return v20.Context(host, 443, token=api_key)

def get_account_data(ctx):
    """Fetches balance and current open positions."""
    try:
        response = ctx.account.get(acc_id)
        account = response.get("account", 200)
        balance = float(account.balance)
        positions = account.positions
        
        active_trades = []
        for p in positions:
            # Check if there is an actual long or short position
            long_units = int(p.long.units)
            short_units = int(p.short.units)
            if long_units != 0 or short_units != 0:
                active_trades.append({
                    "Instrument": p.instrument,
                    "Long Units": long_units,
                    "Short Units": short_units,
                    "Unrealized PL": float(p.unrealizedPL)
                })
        return balance, pd.DataFrame(active_trades)
    except Exception:
        return 0.0, pd.DataFrame()

def calculate_vwap(df):
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['pv'] = df['tp'] * df['volume']
    df['time'] = pd.to_datetime(df['time'])
    df['date'] = df['time'].dt.date
    df['cum_pv'] = df.groupby('date')['pv'].transform('cumsum')
    df['cum_v'] = df.groupby('date')['volume'].transform('cumsum')
    df['vwap'] = df['cum_pv'] / df['cum_v']
    return df

def execute_trade(ctx, inst, units):
    try:
        order_data = {"type": "MARKET", "instrument": inst, "units": str(units), "timeInForce": "FOK"}
        ctx.order.create(acc_id, order=order_data)
        st.session_state.history.append({
            'instrument': inst,
            'side': 'Long' if int(units) > 0 else 'Short',
            'time': datetime.now().strftime("%H:%M:%S")
        })
    except Exception as e:
        st.error(f"Trade Error for {inst}: {e}")

# --- MAIN ENGINE ---
if api_key and acc_id:
    try:
        ctx = get_ctx()
        balance, positions_df = get_account_data(ctx)
        
        # Display Account Status
        col_a, col_b = st.columns(2)
        col_a.metric("Account Balance", f"${balance:,.2f}")
        
        st.subheader("Current Open Positions")
        if not positions_df.empty:
            st.table(positions_df)
        else:
            st.write("No active trades currently open.")

        st.divider()
        
        # Multi-Instrument Scanner
        for inst in instruments:
            res = ctx.instrument.candles(inst, granularity="M5", count=50)
            candles = res.get("candles", 200)
            
            if candles:
                df = pd.DataFrame([{
                    'time': c.time, 'close': float(c.mid.c), 
                    'high': float(c.mid.h), 'low': float(c.mid.l), 
                    'volume': int(c.volume)
                } for c in candles if c.complete])
                
                df = calculate_vwap(df)
                curr, prev = df.iloc[-1], df.iloc[-2]
                
                # Sizing logic
                qty = int((balance * (risk_percentage / 100)) / curr.close)
                qty = max(1, qty)

                # Signal Check
                if prev.close < prev.vwap and curr.close > curr.vwap:
                    st.write(f"Crossing UP on {inst}: Executing Long")
                    execute_trade(ctx, inst, qty)
                elif prev.close > prev.vwap and curr.close < curr.vwap:
                    st.write(f"Crossing DOWN on {inst}: Executing Short")
                    execute_trade(ctx, inst, -qty)
        
        st.caption(f"Last global scan: {datetime.now().strftime('%H:%M:%S')}")
        time.sleep(60)
        st.rerun()

    except Exception as e:
        st.error(f"Operational Error: {e}")
        time.sleep(10)
        st.rerun()
else:
    st.info("Enter API details in the sidebar to begin multi-instrument trading.")
