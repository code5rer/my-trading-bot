import streamlit as st
import pandas as pd
import numpy as np
import v20
import time
from datetime import datetime

# --- UI CONFIG ---
st.set_page_config(page_title="VWAP Auto-Reversal Bot", layout="wide")
st.title("Zarattini and Aziz Multi-Instrument Bot")
st.caption("No SL/TP - Trading by Signal Reversals")

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
    """Fetches balance and map of open positions."""
    try:
        response = ctx.account.get(acc_id)
        account = response.get("account", 200)
        balance = float(account.balance)
        positions = account.positions
        
        pos_map = {} # { 'INSTRUMENT': current_units }
        display_data = []
        
        for p in positions:
            l_units = int(p.long.units)
            s_units = int(p.short.units)
            net_units = l_units - s_units
            
            if net_units != 0:
                pos_map[p.instrument] = net_units
                display_data.append({
                    "Instrument": p.instrument,
                    "Net Units": net_units,
                    "Unrealized PL": float(p.unrealizedPL)
                })
        return balance, pd.DataFrame(display_data), pos_map
    except Exception:
        return 0.0, pd.DataFrame(), {}

def calculate_vwap(df):
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['pv'] = df['tp'] * df['volume']
    df['time'] = pd.to_datetime(df['time'])
    df['date'] = df['time'].dt.date
    df['cum_pv'] = df.groupby('date')['pv'].transform('cumsum')
    df['cum_v'] = df.groupby('date')['volume'].transform('cumsum')
    df['vwap'] = df['cum_pv'] / df['cum_v']
    return df

def execute_reversal(ctx, inst, units_to_open, current_net_units):
    """Closes existing position and opens the new signal direction."""
    try:
        # 1. Close current position by sending opposite units
        if current_net_units != 0:
            close_order = {
                "type": "MARKET",
                "instrument": inst,
                "units": str(-current_net_units),
                "timeInForce": "FOK"
            }
            ctx.order.create(acc_id, order=close_order)
            st.write(f"Closing existing position for {inst}")

        # 2. Open new position
        open_order = {
            "type": "MARKET",
            "instrument": inst,
            "units": str(units_to_open),
            "timeInForce": "FOK"
        }
        ctx.order.create(acc_id, order=open_order)
        
        st.session_state.history.append({
            'instrument': inst,
            'side': 'Long' if units_to_open > 0 else 'Short',
            'time': datetime.now().strftime("%H:%M:%S")
        })
    except Exception as e:
        st.error(f"Reversal Error for {inst}: {e}")

# --- MAIN ENGINE ---
if api_key and acc_id:
    try:
        ctx = get_ctx()
        balance, pos_df, pos_map = get_account_data(ctx)
        
        col_a, col_b = st.columns(2)
        col_a.metric("Account Balance", f"${balance:,.2f}")
        
        st.subheader("Live Portfolio Status")
        if not pos_df.empty:
            st.table(pos_df)
        else:
            st.write("Neutral - No open trades.")

        st.divider()
        
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
                
                # Position Sizing
                qty = int((balance * (risk_percentage / 100)) / curr.close)
                qty = max(1, qty)
                
                current_units = pos_map.get(inst, 0)

                # SIGNAL: CROSS UP (Buy)
                if prev.close < prev.vwap and curr.close > curr.vwap:
                    if current_units <= 0: # Only act if not already Long
                        st.write(f"Bullish Cross on {inst}")
                        execute_reversal(ctx, inst, qty, current_units)
                
                # SIGNAL: CROSS DOWN (Sell)
                elif prev.close > prev.vwap and curr.close < curr.vwap:
                    if current_units >= 0: # Only act if not already Short
                        st.write(f"Bearish Cross on {inst}")
                        execute_reversal(ctx, inst, -qty, current_units)
        
        st.caption(f"Last scan: {datetime.now().strftime('%H:%M:%S')}")
        time.sleep(60)
        st.rerun()

    except Exception as e:
        st.error(f"Operational Error: {e}")
        time.sleep(10)
        st.rerun()
else:
    st.info("Enter API details in the sidebar to begin.")
