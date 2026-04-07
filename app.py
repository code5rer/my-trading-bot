import streamlit as st
import pandas as pd
import numpy as np
import v20
import time
from datetime import datetime

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Safe Portfolio Bot", layout="wide")
st.title("Zarattini & Aziz: High-Reliability Bot")

# --- INITIALIZE SESSION STATE ---
# This ensures variables exist before the bot tries to use them
if 'kill_switch' not in st.session_state:
    st.session_state.kill_switch = False
if 'last_action' not in st.session_state:
    st.session_state.last_action = "System Initialized"

# --- SIDEBAR: CONFIGURATION & EMERGENCY ---
with st.sidebar:
    st.header("OANDA Authentication")
    api_key = st.text_input("OANDA API Key", type="password")
    acc_id = st.text_input("OANDA Account ID")
    env = st.selectbox("Environment", ["practice", "fxtrade"])
    
    st.header("Emergency Settings")
    max_drawdown_pct = st.slider("Max Portfolio Loss % (Emergency)", 1.0, 10.0, 2.0)
    
    st.header("Portfolio Risk")
    instruments = st.multiselect(
        "Instruments", 
        ["NAS100_USD", "US30_USD", "EUR_USD", "XAU_USD", "GBP_USD"],
        default=["NAS100_USD"]
    )
    total_portfolio_risk = st.slider("Total Account Risk (%)", 0.1, 5.0, 1.0)
    leverage_cap = st.slider("Hard Leverage Cap", 0.1, 2.0, 1.0)
    scan_interval = st.number_input("Scan Interval (Seconds)", value=15, min_value=10)

    st.divider()
    if st.button("🔴 EMERGENCY STOP & LIQUIDATE", use_container_width=True):
        st.session_state.kill_switch = True
        st.rerun()
    
    if st.button("🟢 RESET & RESUME BOT", use_container_width=True):
        st.session_state.kill_switch = False
        st.session_state.last_action = f"Manual Reset at {datetime.now().strftime('%H:%M:%S')}"
        st.rerun()

# --- CORE TRADING ENGINE ---

def get_ctx():
    host = "api-fxpractice.oanda.com" if env == "practice" else "api-fxtrade.oanda.com"
    return v20.Context(host, 443, token=api_key)

def emergency_liquidate(ctx, positions):
    """Closes all positions and locks the bot."""
    for p in positions:
        net = int(p.long.units) - int(p.short.units)
        if net != 0:
            ctx.order.create(acc_id, order={"type":"MARKET","instrument":p.instrument,"units":str(-net),"timeInForce":"FOK"})
    st.session_state.kill_switch = True
    st.session_state.last_action = "EMERGENCY LIQUIDATION TRIGGERED"

def calculate_portfolio_units(balance, price, num_inst):
    total_allowed_value = balance * (total_portfolio_risk / 100)
    value_per_trade = total_allowed_value / max(1, num_inst)
    max_leverage_value = (balance * leverage_cap) / max(1, num_inst)
    final_value = min(value_per_trade, max_leverage_value)
    return max(1, int(final_value / price))

def run_trading_cycle(ctx):
    try:
        # 1. Fetch Account
        acc_res = ctx.account.get(acc_id)
        account = acc_res.get("account", 200)
        balance = float(account.balance)
        unrealized_pl = float(account.unrealizedPL)
        
        # 2. Check Drawdown
        drawdown = (abs(unrealized_pl) / balance) * 100 if unrealized_pl < 0 else 0
        if drawdown >= max_drawdown_pct:
            emergency_liquidate(ctx, account.positions)
            return balance, {}, True

        # 3. Process Trades (If not killed)
        pos_map = {p.instrument: (int(p.long.units) - int(p.short.units)) 
                   for p in account.positions if (int(p.long.units) - int(p.short.units)) != 0}

        if not st.session_state.kill_switch:
            num_inst = len(instruments)
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
                trade_units = calculate_portfolio_units(balance, curr['c'], num_inst)

                # Cross UP
                if prev['c'] < prev['vwap'] and curr['c'] > curr['vwap'] and current_pos <= 0:
                    if current_pos < 0:
                        ctx.order.create(acc_id, order={"type":"MARKET","instrument":inst,"units":str(-current_pos),"timeInForce":"FOK"})
                    ctx.order.create(acc_id, order={"type":"MARKET","instrument":inst,"units":str(trade_units),"timeInForce":"FOK"})
                    st.session_state.last_action = f"Bought {inst} at {datetime.now().strftime('%H:%M:%S')}"

                # Cross DOWN
                elif prev['c'] > prev['vwap'] and curr['c'] < curr['vwap'] and current_pos >= 0:
                    if current_pos > 0:
                        ctx.order.create(acc_id, order={"type":"MARKET","instrument":inst,"units":str(-current_pos),"timeInForce":"FOK"})
                    ctx.order.create(acc_id, order={"type":"MARKET","instrument":inst,"units":str(-trade_units),"timeInForce":"FOK"})
                    st.session_state.last_action = f"Sold {inst} at {datetime.now().strftime('%H:%M:%S')}"

        return balance, pos_map, st.session_state.kill_switch

    except Exception as e:
        st.error(f"Execution Error: {e}")
        return 0.0, {}, False

# --- MAIN UI ---
if api_key and acc_id:
    # 1. Status Header
    if st.session_state.kill_switch:
        st.error("🛑 BOT STATUS: LOCKED (Emergency Stop Active)")
    else:
        st.success("🟢 BOT STATUS: SCANNING (Active)")
    
    st.info(f"Last Action: {st.session_state.last_action}")

    # 2. Run Engine
    ctx = get_ctx()
    bal, p_map, killed = run_trading_cycle(ctx)
    
    # 3. Account Stats
    c1, c2 = st.columns(2)
    c1.metric("Balance", f"${bal:,.2f}")
    if p_map:
        c2.write("**Active Trades**")
        c2.dataframe(pd.DataFrame([{"Instrument": k, "Units": v} for k, v in p_map.items()]))

    # 4. Loop
    time.sleep(scan_interval)
    st.rerun()
else:
    st.warning("Awaiting API Credentials...")
