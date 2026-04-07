import streamlit as st
import pandas as pd
import numpy as np
import v20
import time
import threading
from datetime import datetime

# --- UI CONFIG ---
st.set_page_config(page_title="24/7 VWAP Cloud Bot", layout="wide")
st.title("Zarattini and Aziz 24/7 Cloud Bot")
st.caption("Background Worker Active: Trading will continue even if this tab is closed.")

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.header("API Credentials")
    api_key = st.text_input("OANDA API Key", type="password")
    acc_id = st.text_input("OANDA Account ID")
    env = st.selectbox("Environment", ["practice", "fxtrade"])
    
    st.header("Strategy Settings")
    instruments = st.multiselect(
        "Instruments to Trade", 
        ["NAS100_USD", "US30_USD", "EUR_USD", "XAU_USD"],
        default=["NAS100_USD", "US30_USD", "EUR_USD", "XAU_USD"]
    )
    risk_percentage = st.slider("Account Risk Percentage", 0.1, 5.0, 1.0)

# --- GLOBAL DATA STORE ---
# This keeps data alive across browser refreshes
if 'history' not in st.session_state:
    st.session_state.history = []

# --- CORE FUNCTIONS ---

def get_ctx(api_key, env):
    host = "api-fxpractice.oanda.com" if env == "practice" else "api-fxtrade.oanda.com"
    return v20.Context(host, 443, token=api_key)

def calculate_vwap(df):
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['pv'] = df['tp'] * df['volume']
    df['time'] = pd.to_datetime(df['time'])
    df['date'] = df['time'].dt.date
    df['cum_pv'] = df.groupby('date')['pv'].transform('cumsum')
    df['cum_v'] = df.groupby('date')['volume'].transform('cumsum')
    df['vwap'] = df['cum_pv'] / df['cum_v']
    return df

def trading_job(api_key, acc_id, env, instruments, risk_percentage):
    """The background task that runs 24/7 on the server."""
    ctx = get_ctx(api_key, env)
    
    while True:
        try:
            # 1. Get Account Summary
            response = ctx.account.get(acc_id)
            account = response.get("account", 200)
            balance = float(account.balance)
            
            pos_map = {}
            for p in account.positions:
                net = int(p.long.units) - int(p.short.units)
                if net != 0: pos_map[p.instrument] = net

            # 2. Scan Instruments
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
                    
                    qty = int((balance * (risk_percentage / 100)) / curr.close)
                    qty = max(1, qty)
                    current_units = pos_map.get(inst, 0)

                    # Signal Logic
                    if prev.close < prev.vwap and curr.close > curr.vwap and current_units <= 0:
                        if current_units != 0:
                            ctx.order.create(acc_id, order={"type":"MARKET","instrument":inst,"units":str(-current_units),"timeInForce":"FOK"})
                        ctx.order.create(acc_id, order={"type":"MARKET","instrument":inst,"units":str(qty),"timeInForce":"FOK"})
                        
                    elif prev.close > prev.vwap and curr.close < curr.vwap and current_units >= 0:
                        if current_units != 0:
                            ctx.order.create(acc_id, order={"type":"MARKET","instrument":inst,"units":str(-current_units),"timeInForce":"FOK"})
                        ctx.order.create(acc_id, order={"type":"MARKET","instrument":inst,"units":str(-qty),"timeInForce":"FOK"})

            time.sleep(60) # Wait 1 minute before next global scan
        except Exception as e:
            print(f"Background Error: {e}")
            time.sleep(30)

# --- BACKGROUND THREAD MANAGER ---
@st.cache_resource
def start_worker(api_key, acc_id, env, instruments, risk_percentage):
    """Starts the background thread only once."""
    thread = threading.Thread(
        target=trading_job, 
        args=(api_key, acc_id, env, instruments, risk_percentage), 
        daemon=True
    )
    thread.start()
    return "Worker Started"

# --- MAIN DASHBOARD ---
if api_key and acc_id:
    status = start_worker(api_key, acc_id, env, instruments, risk_percentage)
    st.success(f"Status: {status}. The bot is running on the server.")
    
    # Simple display for the user
    ctx = get_ctx(api_key, env)
    resp = ctx.account.get(acc_id)
    acc = resp.get("account", 200)
    
    st.metric("Cloud-Synced Balance", f"${float(acc.balance):,.2f}")
    
    st.write("Current Positions (Live from OANDA):")
    active = []
    for p in acc.positions:
        net = int(p.long.units) - int(p.short.units)
        if net != 0:
            active.append({"Instrument": p.instrument, "Net Units": net, "PL": p.unrealizedPL})
    
    if active:
        st.table(pd.DataFrame(active))
    else:
        st.info("No active trades. The background worker is monitoring the charts.")
    
    # Refresh the UI every 30s so you can watch, but the bot trades independently
    time.sleep(30)
    st.rerun()
else:
    st.warning("Enter API keys to launch the background worker.")
