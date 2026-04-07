import streamlit as st
import pandas as pd
import numpy as np
import v20
import time
from datetime import datetime

# --- UI CONFIG ---
st.set_page_config(page_title="High-Freq VWAP Bot", layout="wide")
st.title("Zarattini & Aziz High-Frequency Bot")

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
    # REDUCED SLEEP: Scan more often (Every 15 seconds)
    scan_interval = st.number_input("Scan Interval (Seconds)", value=15, min_value=5)

# --- CORE TRADING LOGIC ---

def get_ctx():
    host = "api-fxpractice.oanda.com" if env == "practice" else "api-fxtrade.oanda.com"
    return v20.Context(host, 443, token=api_key)

def run_scan(ctx):
    """Executes one full scan of all instruments."""
    try:
        # 1. Get Account State
        response = ctx.account.get(acc_id)
        account = response.get("account", 200)
        balance = float(account.balance)
        
        pos_map = {}
        for p in account.positions:
            net = int(p.long.units) - int(p.short.units)
            if net != 0: pos_map[p.instrument] = net

        # 2. Check Each Instrument
        for inst in instruments:
            res = ctx.instrument.candles(inst, granularity="M5", count=50)
            candles = res.get("candles", 200)
            
            if candles:
                df = pd.DataFrame([{
                    'close': float(c.mid.c), 'high': float(c.mid.h), 
                    'low': float(c.mid.l), 'volume': int(c.volume)
                } for c in candles if c.complete])
                
                # VWAP Calculation
                df['tp'] = (df['high'] + df['low'] + df['close']) / 3
                df['vwap'] = (df['tp'] * df['volume']).cumsum() / df['volume'].cumsum()
                
                curr, prev = df.iloc[-1], df.iloc[-2]
                qty = int((balance * (risk_percentage / 100)) / curr.close)
                current_units = pos_map.get(inst, 0)

                # Execute Reversal Logic
                if prev.close < prev.vwap and curr.close > curr.vwap and current_units <= 0:
                    if current_units != 0:
                        ctx.order.create(acc_id, order={"type":"MARKET","instrument":inst,"units":str(-current_units),"timeInForce":"FOK"})
                    ctx.order.create(acc_id, order={"type":"MARKET","instrument":inst,"units":str(max(1, qty)),"timeInForce":"FOK"})
                    st.toast(f"Long entry on {inst}")
                    
                elif prev.close > prev.vwap and curr.close < curr.vwap and current_units >= 0:
                    if current_units != 0:
                        ctx.order.create(acc_id, order={"type":"MARKET","instrument":inst,"units":str(-current_units),"timeInForce":"FOK"})
                    ctx.order.create(acc_id, order={"type":"MARKET","instrument":inst,"units":str(-max(1, qty)),"timeInForce":"FOK"})
                    st.toast(f"Short entry on {inst}")
                    
        return balance, pos_map
    except Exception as e:
        st.error(f"Scan Error: {e}")
        return 0.0, {}

# --- MAIN LOOP ---
if api_key and acc_id:
    ctx = get_ctx()
    
    # Run the scan
    balance, pos_map = run_scan(ctx)
    
    # UI Display
    st.metric("Live Balance", f"${balance:,.2f}")
    if pos_map:
        st.write("Active Positions:")
        st.dataframe(pd.DataFrame([{"Instrument": k, "Units": v} for k, v in pos_map.items()]))
    else:
        st.info("No active positions. Scanning charts...")

    # The "Force" Loop
    st.caption(f"Last heartbeat: {datetime.now().strftime('%H:%M:%S')}")
    time.sleep(scan_interval)
    st.rerun()
else:
    st.info("Enter API credentials to start the high-frequency scanner.")
