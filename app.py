import streamlit as st
import pandas as pd
import numpy as np
import v20
import time
from datetime import datetime

# --- UI CONFIGURATION ---
st.set_page_config(page_title="VWAP Notional Bot", layout="wide")
st.title("Zarattini & Aziz: Professional VWAP Suite")
st.caption("Consolidated High-Frequency Bot with Notional Risk Controls")

# --- SIDEBAR: CONFIGURATION & SAFETY ---
with st.sidebar:
    st.header("🔑 OANDA Authentication")
    api_key = st.text_input("OANDA API Key", type="password")
    acc_id = st.text_input("OANDA Account ID")
    env = st.selectbox("Environment", ["practice", "fxtrade"])
    
    st.header("🛡️ Risk Management")
    instruments = st.multiselect(
        "Instruments to Monitor", 
        ["NAS100_USD", "US30_USD", "EUR_USD", "XAU_USD", "GBP_USD"],
        default=["NAS100_USD"]
    )
    risk_percent = st.slider("Target Risk per Trade (%)", 0.1, 2.0, 1.0, 
                             help="Percentage of account equity allocated to the position value.")
    leverage_cap = st.slider("Hard Leverage Cap", 0.1, 5.0, 1.0, 
                             help="1.0 means position value cannot exceed account balance.")
    
    st.header("⏱️ Scanning Settings")
    scan_interval = st.number_input("Scan Interval (Seconds)", value=20, min_value=10)

# --- CORE TRADING ENGINE ---

def get_ctx():
    """Initializes the OANDA API Context based on environment."""
    host = "api-fxpractice.oanda.com" if env == "practice" else "api-fxtrade.oanda.com"
    return v20.Context(host, 443, token=api_key)

def calculate_safe_units(balance, price, inst):
    """
    Calculates units based on the dollar value (Notional).
    Formula: (Balance * Risk%) / Current Price
    Includes a Hard Leverage Cap.
    """
    # Step 1: Calculate target dollar value
    target_value = balance * (risk_percent / 100)
    
    # Step 2: Calculate maximum allowed dollar value (Leverage Cap)
    max_allowed_value = balance * leverage_cap
    
    # Step 3: Use the safer (smaller) of the two
    final_notional_value = min(target_value, max_allowed_value)
    
    # Step 4: Convert dollars to units
    units = final_notional_value / price
    
    # OANDA requires integer units; minimum 1
    return max(1, int(units))

def run_trading_cycle(ctx):
    """The main logic loop for scanning and executing trades."""
    try:
        # 1. Fetch current account state
        acc_res = ctx.account.get(acc_id)
        account = acc_res.get("account", 200)
        balance = float(account.balance)
        
        # 2. Map existing positions {Instrument: NetUnits}
        pos_map = {}
        for p in account.positions:
            net = int(p.long.units) - int(p.short.units)
            if net != 0:
                pos_map[p.instrument] = net

        # 3. Process each instrument
        for inst in instruments:
            # Fetch M5 candle data
            res = ctx.instrument.candles(inst, granularity="M5", count=100)
            candles = res.get("candles", 200)
            if not candles: continue
            
            # Prepare DataFrame
            df = pd.DataFrame([{
                'c': float(c.mid.c), 
                'h': float(c.mid.h), 
                'l': float(c.mid.l), 
                'v': int(c.volume)
            } for c in candles if c.complete])
            
            # Calculate Intraday VWAP
            df['tp'] = (df['h'] + df['l'] + df['c']) / 3
            df['vwap'] = (df['tp'] * df['v']).cumsum() / df['v'].cumsum()
            
            curr, prev = df.iloc[-1], df.iloc[-2]
            current_pos = pos_map.get(inst, 0)
            
            # Calculate Risk-Adjusted Units
            trade_units = calculate_safe_units(balance, curr['c'], inst)

            # --- SIGNAL EXECUTION ---
            
            # BULLISH SIGNAL: Price crosses UP over VWAP
            if prev['c'] < prev['vwap'] and curr['c'] > curr['vwap'] and current_pos <= 0:
                # Close existing Short if it exists
                if current_pos < 0:
                    ctx.order.create(acc_id, order={"type":"MARKET","instrument":inst,"units":str(-current_pos),"timeInForce":"FOK"})
                    st.toast(f"Closed Short on {inst}")
                
                # Open New Long
                ctx.order.create(acc_id, order={"type":"MARKET","instrument":inst,"units":str(trade_units),"timeInForce":"FOK"})
                st.success(f"LONG {trade_units} units of {inst} | Est. Value: ${trade_units * curr['c']:,.2f}")

            # BEARISH SIGNAL: Price crosses DOWN under VWAP
            elif prev['c'] > prev['vwap'] and curr['c'] < curr['vwap'] and current_pos >= 0:
                # Close existing Long if it exists
                if current_pos > 0:
                    ctx.order.create(acc_id, order={"type":"MARKET","instrument":inst,"units":str(-current_pos),"timeInForce":"FOK"})
                    st.toast(f"Closed Long on {inst}")
                
                # Open New Short
                ctx.order.create(acc_id, order={"type":"MARKET","instrument":inst,"units":str(-trade_units),"timeInForce":"FOK"})
                st.warning(f"SHORT {trade_units} units of {inst} | Est. Value: ${trade_units * curr['c']:,.2f}")

        return balance, pos_map

    except Exception as e:
        st.error(f"Cycle Error: {e}")
        return 0.0, {}

# --- MAIN UI LOOP ---
if api_key and acc_id:
    ctx = get_ctx()
    
    # Execute Scan
    bal, p_map = run_trading_cycle(ctx)
    
    # Dashboard Display
    col1, col2 = st.columns(2)
    col1.metric("Account Balance", f"${bal:,.2f}")
    
    with col2:
        st.write("**Active Positions**")
        if p_map:
            st.dataframe(pd.DataFrame([{"Instrument": k, "Units": v} for k, v in p_map.items()]))
        else:
            st.info("No active positions.")

    # Refresh Mechanism
    st.divider()
    st.caption(f"Last heartbeat: {datetime.now().strftime('%H:%M:%S')}. Next scan in {scan_interval}s...")
    time.sleep(scan_interval)
    st.rerun()
else:
    st.info("Please enter your OANDA API credentials in the sidebar to start the bot.")
