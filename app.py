import streamlit as st
import pandas as pd
import numpy as np
import v20
import time
from datetime import datetime

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Safe Portfolio Bot", layout="wide")
st.title("Zarattini & Aziz: Emergency-Protected Bot")

# --- SIDEBAR: CONFIGURATION & EMERGENCY ---
with st.sidebar:
    st.header("OANDA Authentication")
    api_key = st.text_input("OANDA API Key", type="password")
    acc_id = st.text_input("OANDA Account ID")
    env = st.selectbox("Environment", ["practice", "fxtrade"])
    
    st.header("Emergency Settings")
    # THE CIRCUIT BREAKER: If total loss hits this %, kill everything.
    max_drawdown_pct = st.slider("Max Portfolio Loss % (Emergency)", 1.0, 10.0, 2.0, 
                                 help="If total unrealized loss hits this %, all trades close immediately.")
    
    st.header("Portfolio Risk")
    instruments = st.multiselect(
        "Instruments", 
        ["NAS100_USD", "US30_USD", "EUR_USD", "XAU_USD", "GBP_USD"],
        default=["NAS100_USD", "US30_USD", "EUR_USD", "XAU_USD"]
    )
    total_portfolio_risk = st.slider("Total Account Risk (%)", 0.1, 5.0, 1.0)
    leverage_cap = st.slider("Hard Leverage Cap", 0.1, 2.0, 1.0)
    scan_interval = st.number_input("Scan Interval (Seconds)", value=20, min_value=10)

    if st.button("Manual Emergency Liquidate"):
        st.session_state.kill_switch = True

# --- STATE MANAGEMENT ---
if 'kill_switch' not in st.session_state:
    st.session_state.kill_switch = False

# --- CORE TRADING ENGINE ---

def get_ctx():
    host = "api-fxpractice.oanda.com" if env == "practice" else "api-fxtrade.oanda.com"
    return v20.Context(host, 443, token=api_key)

def emergency_liquidate(ctx, positions):
    """Closes every single open position immediately."""
    st.error("EMERGENCY: Circuit Breaker Tripped. Liquidating all positions.")
    for p in positions:
        net = int(p.long.units) - int(p.short.units)
        if net != 0:
            ctx.order.create(acc_id, order={"type":"MARKET","instrument":p.instrument,"units":str(-net),"timeInForce":"FOK"})
    st.session_state.kill_switch = True

def calculate_portfolio_units(balance, price, num_inst):
    total_allowed_value = balance * (total_portfolio_risk / 100)
    value_per_trade = total_allowed_value / max(1, num_inst)
    max_leverage_value = (balance * leverage_cap) / max(1, num_inst)
    final_value = min(value_per_trade, max_leverage_value)
    return max(1, int(final_value / price))

def run_trading_cycle(ctx):
    try:
        # 1. Fetch current account state
        acc_res = ctx.account.get(acc_id)
        account = acc_res.get("account", 200)
        balance = float(account.balance)
        unrealized_pl = float(account.unrealizedPL)
        
        # 2. EMERGENCY CHECK: Drawdown Protection
        drawdown_pct = (abs(unrealized_pl) / balance) * 100 if unrealized_pl < 0 else 0
        
        if drawdown_pct >= max_drawdown_pct:
            emergency_liquidate(ctx, account.positions)
            return balance, {}, True

        # 3. Map existing positions
        pos_map = {}
        for p in account.positions:
            net = int(p.long.units) - int(p.short.units)
            if net != 0: pos_map[p.instrument] = net

        # 4. Process each instrument (Only if kill switch is OFF)
        if not st.session_state.kill_switch:
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
                trade_units = calculate_portfolio_units(balance, curr['c'], num_instruments)

                # Signal Logic
                if prev['c'] < prev['vwap'] and curr['c'] > curr['vwap'] and current_pos <= 0:
                    if current_pos < 0:
                        ctx.order.create(acc_id, order={"type":"MARKET","instrument":inst,"units":str(-current_pos),"timeInForce":"FOK"})
                    ctx.order.create(acc_id, order={"type":"MARKET","instrument":inst,"units":str(trade_units),"timeInForce":"FOK"})
                    st.success(f"LONG {inst} Executed")

                elif prev['c'] > prev['vwap'] and curr['c'] < curr['vwap'] and current_pos >= 0:
                    if current_pos > 0:
                        ctx.order.create(acc_id, order={"type":"MARKET","instrument":inst,"units":str(-current_pos),"timeInForce":"FOK"})
                    ctx.order.create(acc_id, order={"type":"MARKET","instrument":inst,"units":str(-trade_units),"timeInForce":"FOK"})
                    st.warning(f"SHORT {inst} Executed")

        return balance, pos_map, st.session_state.kill_switch

    except Exception as e:
        st.error(f"Cycle Error: {e}")
        return 0.0, {}, False

# --- MAIN UI LOOP ---
if api_key and acc_id:
    if st.session_state.kill_switch:
        st.error("BOT STOPPED: Emergency Liquidate was triggered. Refresh page to reset.")
        if st.button("Reset Bot"):
            st.session_state.kill_switch = False
            st.rerun()
    else:
        ctx = get_ctx()
        bal, p_map, killed = run_trading_cycle(ctx)
        
        st.metric("Portfolio Balance", f"${bal:,.2f}")
        if p_map:
            st.write("**Current Risk Distribution**")
            st.table(pd.DataFrame([{"Instrument": k, "Units": v} for k, v in p_map.items()]))
        
        st.caption(f"Heartbeat: {datetime.now().strftime('%H:%M:%S')}. Monitoring for >{max_drawdown_pct}% drawdown.")
        time.sleep(scan_interval)
        st.rerun()
else:
    st.info("Enter API credentials to start.")
