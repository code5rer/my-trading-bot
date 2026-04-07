import streamlit as st
import pandas as pd
import numpy as np
import v20
import time
from datetime import datetime

# --- UI CONFIGURATION ---
st.set_page_config(page_title="VWAP Portfolio Bot", layout="wide")
st.title("Zarattini & Aziz: Portfolio-Risk Controller")

# --- SIDEBAR: CONFIGURATION & SAFETY ---
with st.sidebar:
    st.header(" OANDA Authentication")
    api_key = st.text_input("OANDA API Key", type="password")
    acc_id = st.text_input("OANDA Account ID")
    env = st.selectbox("Environment", ["practice", "fxtrade"])
    
    st.header(" Strict Portfolio Risk")
    instruments = st.multiselect(
        "Instruments to Monitor", 
        ["NAS100_USD", "US30_USD", "EUR_USD", "XAU_USD", "GBP_USD"],
        default=["NAS100_USD", "US30_USD", "EUR_USD", "XAU_USD"]
    )
    # This 1% is now the TOTAL for the whole account, not per trade
    total_portfolio_risk = st.slider("Total Account Risk (%)", 0.1, 5.0, 1.0, 
                             help="The bot will divide this % by the number of instruments selected.")
    
    leverage_cap = st.slider("Hard Leverage Cap", 0.1, 2.0, 1.0)
    scan_interval = st.number_input("Scan Interval (Seconds)", value=20, min_value=10)

# --- CORE TRADING ENGINE ---

def get_ctx():
    host = "api-fxpractice.oanda.com" if env == "practice" else "api-fxtrade.oanda.com"
    return v20.Context(host, 443, token=api_key)

def calculate_portfolio_units(balance, price, num_inst):
    """
    Divides the total allowed risk by the number of active instruments.
    If 1% risk on $100k with 4 instruments: Each trade gets $250 of value.
    """
    # 1. Calculate the TOTAL dollar value allowed for the WHOLE portfolio
    total_allowed_value = balance * (total_portfolio_risk / 100)
    
    # 2. Divide that value by the number of instruments you want to trade
    # This ensures 4 trades = 1% total, NOT 1% each.
    value_per_trade = total_allowed_value / max(1, num_inst)
    
    # 3. Apply Hard Leverage Cap (Global safety)
    max_leverage_value = (balance * leverage_cap) / max(1, num_inst)
    
    final_value = min(value_per_trade, max_leverage_value)
    
    # 4. Convert to units
    units = final_value / price
    return max(1, int(units))

def run_trading_cycle(ctx):
    try:
        acc_res = ctx.account.get(acc_id)
        account = acc_res.get("account", 200)
        balance = float(account.balance)
        
        pos_map = {p.instrument: (int(p.long.units) - int(p.short.units)) 
                   for p in account.positions if (int(p.long.units) - int(p.short.units)) != 0}

        num_instruments = len(instruments)

        for inst in instruments:
            res = ctx.instrument.candles(inst, granularity="M5", count=100)
            candles = res.get("candles", 200)
            if not candles: continue
            
            df = pd.DataFrame([{'c': float(c.mid.c), 'h': float(c.mid.h), 'l': float(c.mid.l), 'v': int(c.volume)} 
                               for c in candles if c.complete])
            
            df['tp'] = (df['h'] + df['l'] + df['c']) / 3
            df['vwap'] = (df['tp'] * df['v']).cumsum() / df['v'].cumsum()
            
            curr, prev = df.iloc[-1], df.iloc[-2]
            current_pos = pos_map.get(inst, 0)
            
            # Use the NEW portfolio-aware unit calculation
            trade_units = calculate_portfolio_units(balance, curr['c'], num_instruments)

            # Signal Logic
            if prev['c'] < prev['vwap'] and curr['c'] > curr['vwap'] and current_pos <= 0:
                if current_pos < 0:
                    ctx.order.create(acc_id, order={"type":"MARKET","instrument":inst,"units":str(-current_pos),"timeInForce":"FOK"})
                ctx.order.create(acc_id, order={"type":"MARKET","instrument":inst,"units":str(trade_units),"timeInForce":"FOK"})
                st.success(f"LONG {inst} | Value: ${trade_units * curr['c']:,.2f} ({total_portfolio_risk/num_instruments:.2f}% of acct)")

            elif prev['c'] > prev['vwap'] and curr['c'] < curr['vwap'] and current_pos >= 0:
                if current_pos > 0:
                    ctx.order.create(acc_id, order={"type":"MARKET","instrument":inst,"units":str(-current_pos),"timeInForce":"FOK"})
                ctx.order.create(acc_id, order={"type":"MARKET","instrument":inst,"units":str(-trade_units),"timeInForce":"FOK"})
                st.warning(f"SHORT {inst} | Value: ${trade_units * curr['c']:,.2f} ({total_portfolio_risk/num_instruments:.2f}% of acct)")

        return balance, pos_map

    except Exception as e:
        st.error(f"Cycle Error: {e}")
        return 0.0, {}

# --- MAIN UI LOOP ---
if api_key and acc_id:
    ctx = get_ctx()
    bal, p_map = run_trading_cycle(ctx)
    
    st.metric("Total Equity", f"${bal:,.2f}")
    if p_map:
        st.write("**Open Trades**")
        st.dataframe(pd.DataFrame([{"Instrument": k, "Units": v, "Notional Value": "Calculating..."} for k, v in p_map.items()]))
    
    st.divider()
    st.caption(f"Last heartbeat: {datetime.now().strftime('%H:%M:%S')}. Risk split across {len(instruments)} assets.")
    time.sleep(scan_interval)
    st.rerun()
else:
    st.info("Enter API credentials to start.")
