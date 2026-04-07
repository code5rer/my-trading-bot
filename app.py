import streamlit as st
import pandas as pd
import numpy as np
import v20
from v20.operations import OrderCreateRequest
import time
from datetime import datetime

# --- SETTINGS & UI ---
st.set_page_config(page_title="Zarattini VWAP Bot", layout="wide")
st.title("📈 Zarattini-Aziz VWAP Automated Bot")
st.caption("A self-optimizing intraday strategy for OANDA")

# --- SIDEBAR CONFIG ---
with st.sidebar:
    st.header("🔑 API Credentials")
    api_key = st.text_input("OANDA API Key", type="password")
    acc_id = st.text_input("OANDA Account ID")
    env = st.selectbox("Environment", ["practice", "fxtrade"])
    
    st.header("⚙️ Strategy Parameters")
    instrument = st.selectbox("Instrument", ["NAS100_USD", "US30_USD", "EUR_USD", "XAU_USD"])
    trade_size = st.number_input("Base Units (e.g., 10)", value=10)
    
    if st.button("Reset Session PnL"):
        st.session_state.history = []
        st.rerun()

# --- INITIALIZE SESSION STATE ---
if 'history' not in st.session_state:
    st.session_state.history = [] # Tracks: {'side', 'pnl', 'time'}

# --- CORE FUNCTIONS ---

def get_ctx():
    host = f"api-{env}.oanda.com"
    return v20.Context(host, 443, token=api_key)

def calculate_vwap(df):
    """
    Implements the SSRN paper formula:
    $$VWAP = \frac{\sum (Typical Price \times Volume)}{\sum Volume}$$
    """
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    # Reset VWAP daily to align with Zarattini's intraday focus
    df['date'] = pd.to_datetime(df['time']).dt.date
    df['cum_pv'] = df.groupby('date')['tp'].transform(lambda x: (x * df.loc[x.index, 'volume']).cumsum())
    df['cum_v'] = df.groupby('date')['volume'].transform(lambda x: x.cumsum())
    df['vwap'] = df['cum_pv'] / df['cum_v']
    return df

def get_optimized_size(base_size):
    """SELF-OPTIMIZATION: Adjusts trade size based on recent win rate."""
    if len(st.session_state.history) < 3:
        return base_size
    
    wins = len([t for t in st.session_state.history if t['pnl'] > 0])
    win_rate = wins / len(st.session_state.history)
    
    if win_rate > 0.6:
        return int(base_size * 1.5) # Scale up on winning streaks
    elif win_rate < 0.4:
        return int(base_size * 0.5) # De-risk on losing streaks
    return base_size

def execute_trade(ctx, units):
    """Sends a market order to OANDA."""
    try:
        # units > 0 is Buy, units < 0 is Sell
        order_conf = dict(
            type="MARKET",
            instrument=instrument,
            units=str(units),
            timeInForce="FOK"
        )
        request = OrderCreateRequest(order=order_conf)
        response = ctx.order.create(acc_id, **request.body)
        
        # Log basic PnL (Mock PnL for visualization)
        # In a real bot, you'd fetch the transaction's actual PL from OANDA
        mock_pnl = np.random.uniform(-5, 10) 
        st.session_state.history.append({
            'side': 'Long' if units > 0 else 'Short',
            'pnl': mock_pnl,
            'time': datetime.now().strftime("%H:%M:%S")
        })
        return response
    except Exception as e:
        st.error(f"Trade Execution Error: {e}")

# --- MAIN ENGINE ---
if api_key and acc_id:
    try:
        ctx = get_ctx()
        
        # 1. Fetch & Process Data
        res = ctx.candle.get(instrument, granularity="M5", count=150)
        candles = res.get("candles", 150)
        
        raw_data = []
        for c in candles:
            if c.complete:
                raw_data.append({
                    'time': c.time,
                    'close': float(c.mid.c),
                    'high': float(c.mid.h),
                    'low': float(c.mid.l),
                    'volume': int(c.volume)
                })
        
        df = pd.DataFrame(raw_data)
        df = calculate_vwap(df)
        
        # 2. Visualizations
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"Live Chart: {instrument}")
            st.line_chart(df.set_index('time')[['close', 'vwap']])
            
        with col2:
            st.subheader("PnL Decomposition")
            if st.session_state.history:
                h_df = pd.DataFrame(st.session_state.history)
                total_pl = h_df['pnl'].sum()
                st.metric("Net PnL", f"${total_pl:.2f}", delta=f"{len(h_df)} Trades")
                
                # Breakdown by side
                breakdown = h_df.groupby('side')['pnl'].sum()
                st.dataframe(breakdown)
            else:
                st.info("Scanning for entries...")

        # 3. Strategy Logic (The Cross-Over)
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Optimization check
        dynamic_units = get_optimized_size(trade_size)
        
        # Entry Long: Price crosses above VWAP
        if prev.close < prev.vwap and curr.close > curr.vwap:
            st.toast(f"Buying {dynamic_units} units!", icon="🚀")
            execute_trade(ctx, dynamic_units)
            
        # Entry Short: Price crosses below VWAP
        elif prev.close > prev.vwap and curr.close < curr.vwap:
            st.toast(f"Selling {dynamic_units} units!", icon="📉")
            execute_trade(ctx, -dynamic_units)

        # 4. Auto-Refresh Logic
        st.caption(f"Last update: {datetime.now().strftime('%H:%M:%S')}. Next scan in 60s.")
        time.sleep(60)
        st.rerun()

    except Exception as e:
        st.error(f"Operational Error: {e}")
        time.sleep(10)
        st.rerun()
else:
    st.warning("Waiting for API Credentials... Enter them in the sidebar.")
