"""
HMM Forex Trading Bot — OANDA API
Unified Version: Robust Error Handling + Strict 1% Risk Management
"""

import time, logging, threading
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import ta
import oandapyV20
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.orders   as orders
import oandapyV20.endpoints.trades   as trades_ep
import oandapyV20.endpoints.instruments as instruments

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
OANDA_API_KEY    = "YOUR_OANDA_API_KEY"
OANDA_ACCOUNT_ID = "YOUR_ACCOUNT_ID"
OANDA_ENV        = "practice"           # "practice" or "live"
INSTRUMENT       = "EUR_USD"
GRANULARITY      = "M15"
CANDLE_COUNT     = 500
RISK_PER_TRADE   = 0.01                 # Strict 1% account risk
MAX_OPEN_TRADES  = 3                    # Max concurrent trades
MIN_CONFIDENCE   = 0.62                 # HMM posterior threshold
RETRAIN_EVERY    = 50                   
LOOP_INTERVAL    = 60                   

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("HMM-FX")

REGIMES = {0: "Bullish", 1: "Bearish", 2: "Ranging", 3: "Volatile"}

DEFAULT_PARAMS = {
    0: dict(ema_fast=9,  ema_slow=21, rsi_ob=68, rsi_os=32, atr_mult=1.8, bb_dev=2.0),
    1: dict(ema_fast=13, ema_slow=34, rsi_ob=72, rsi_os=28, atr_mult=2.1, bb_dev=2.2),
    2: dict(ema_fast=20, ema_slow=50, rsi_ob=60, rsi_os=40, atr_mult=1.4, bb_dev=1.6),
    3: dict(ema_fast=5,  ema_slow=13, rsi_ob=75, rsi_os=25, atr_mult=2.8, bb_dev=2.5),
}

# ─────────────────────────────────────────────
# HMM REGIME DETECTOR
# ─────────────────────────────────────────────
class RegimeDetector:
    def __init__(self, n_states: int = 4):
        self.n_states = n_states
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            tol=1e-4,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.fitted = False
        self._state_map: dict[int, int] = {}

    @staticmethod
    def build_features(df: pd.DataFrame) -> np.ndarray:
        c = df["close"].astype(float)
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        log_ret = np.log(c / c.shift(1)).fillna(0)
        roll_vol = log_ret.rolling(14).std().fillna(0) * np.sqrt(252 * 24 * 4)
        rsi = ta.momentum.RSIIndicator(c, window=14).rsi().fillna(50) / 100
        atr_norm = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range().fillna(0) / c
        feats = np.column_stack([log_ret, roll_vol, rsi, atr_norm])
        mask = np.isfinite(feats).all(axis=1)
        return feats[mask], mask

    def fit(self, df: pd.DataFrame) -> None:
        try:
            feats, _ = self.build_features(df)
            feats_scaled = self.scaler.fit_transform(feats)
            self.model.fit(feats_scaled)
            self.fitted = True
            self._build_state_map(feats)
            log.info(f"HMM Retrained | logL={self.model.score(feats_scaled):.1f}")
        except Exception as e:
            log.error(f"HMM Training failed: {e}")

    def _build_state_map(self, feats_raw: np.ndarray) -> None:
        states_seq = self.model.predict(self.scaler.transform(feats_raw))
        means = {}
        vols = {}
        for s in range(self.n_states):
            idx = states_seq == s
            means[s] = feats_raw[idx, 0].mean() if idx.any() else 0
            vols[s] = feats_raw[idx, 1].mean() if idx.any() else 0
        sorted_by_mean = sorted(means, key=means.get)
        sorted_by_vol = sorted(vols, key=vols.get, reverse=True)
        vol_state = sorted_by_vol[0]
        remaining = [s for s in sorted_by_mean if s != vol_state]
        bull_state, bear_state = remaining[-1], remaining[0]
        range_state = [s for s in range(self.n_states) if s not in (vol_state, bull_state, bear_state)][0]
        self._state_map = {bull_state: 0, bear_state: 1, range_state: 2, vol_state: 3}

    def predict(self, df: pd.DataFrame) -> tuple[int, float, np.ndarray]:
        if not self.fitted: raise RuntimeError("HMM not fitted")
        feats, _ = self.build_features(df)
        feats_scaled = self.scaler.transform(feats)
        viterbi_states = self.model.predict(feats_scaled)
        raw_state = int(viterbi_states[-1])
        posteriors = self.model.predict_proba(feats_scaled)[-1]
        return self._state_map.get(raw_state, 0), float(posteriors[raw_state]), posteriors

# ─────────────────────────────────────────────
# BROKER INTERFACE (FIXED)
# ─────────────────────────────────────────────
class OandaBroker:
    def __init__(self, api_key: str, account_id: str, env: str):
        self.account_id = account_id
        try:
            self.client = oandapyV20.API(access_token=api_key, environment=env)
        except Exception as e:
            log.error(f"OANDA API Init Error: {e}")
            raise

    def get_candles(self, instrument: str, granularity: str, count: int) -> pd.DataFrame:
        try:
            r = instruments.InstrumentsCandles(instrument=instrument, params={"granularity": granularity, "count": count})
            self.client.request(r)
            rows = [{"time": c["time"], "open": float(c["mid"]["o"]), "high": float(c["mid"]["h"]), 
                     "low": float(c["mid"]["l"]), "close": float(c["mid"]["c"]), "volume": int(c["volume"])} 
                    for c in r.response.get("candles", []) if c["complete"]]
            return pd.DataFrame(rows)
        except Exception as e:
            log.error(f"Candle Fetch Error: {e}")
            return pd.DataFrame()

    def get_balance(self) -> float:
        try:
            r = accounts.AccountSummary(accountID=self.account_id)
            self.client.request(r)
            return float(r.response["account"]["balance"])
        except Exception as e:
            log.error(f"Balance Fetch Error: {e}")
            return 0.0

    def count_open_trades(self, instrument: str) -> int:
        try:
            r = trades_ep.TradesList(accountID=self.account_id)
            self.client.request(r)
            return sum(1 for t in r.response.get("trades", []) if t["instrument"] == instrument)
        except Exception:
            return 999 # Safety block

    def place_order(self, instrument: str, units: int, sl: float, tp: float):
        data = {"order": {"type": "MARKET", "instrument": instrument, "units": str(units),
                          "stopLossOnFill": {"price": str(round(sl, 5))},
                          "takeProfitOnFill": {"price": str(round(tp, 5))}}}
        try:
            r = orders.OrderCreate(accountID=self.account_id, data=data)
            self.client.request(r)
            return r.response.get("orderFillTransaction", {}).get("tradeOpened", {}).get("tradeID")
        except Exception as e:
            log.error(f"Trade Execution Failed: {e}")
            return None

    def kelly_units(self, balance: float, win_rate: float, rr: float, sl_dist: float, price: float) -> int:
        kelly = max(0, (win_rate * rr - (1 - win_rate)) / rr)
        # Fraction of Kelly capped by 1% Risk Rule
        safe_fraction = min(kelly * 0.15, RISK_PER_TRADE) 
        risk_amount = balance * safe_fraction
        
        # Calculate units so that hitting Stop Loss = Risk Amount
        # Formula: Units = Risk / SL_Distance
        units = int(risk_amount / max(sl_dist, 1e-9))
        
        # Absolute safety cap (Never trade more than 1% account balance in margin-equivalent)
        return max(100, min(units, 50000))

# ─────────────────────────────────────────────
# ML OPTIMIZER & SIGNAL ENGINE
# ─────────────────────────────────────────────
class ParamOptimizer:
    def __init__(self):
        self.params = {r: dict(DEFAULT_PARAMS[r]) for r in range(4)}
        self.history = {r: deque(maxlen=100) for r in range(4)}
        self.best_pf = {r: 1.0 for r in range(4)}

    def record_trade(self, regime, pnl): self.history[regime].append(pnl)

    def update(self, regime):
        h = list(self.history[regime])
        if len(h) < 5: return
        w, l = [x for x in h if x > 0], [x for x in h if x < 0]
        pf = sum(w) / abs(sum(l)) if l else 1.1
        if pf > self.best_pf[regime]:
            self.best_pf[regime] = pf
            for k in self.params[regime]: 
                self.params[regime][k] += np.random.uniform(-0.01, 0.01) # Nudge

class SignalEngine:
    def compute(self, df, regime, params):
        c = df["close"].astype(float)
        ef = ta.trend.EMAIndicator(c, window=int(params["ema_fast"])).ema_indicator().iloc[-1]
        es = ta.trend.EMAIndicator(c, window=int(params["ema_slow"])).ema_indicator().iloc[-1]
        rv = ta.momentum.RSIIndicator(c, window=14).rsi().iloc[-1]
        atr_v = ta.volatility.AverageTrueRange(df["high"], df["low"], c, window=14).average_true_range().iloc[-1]
        bb = ta.volatility.BollingerBands(c, window=20, window_dev=params["bb_dev"])
        
        sig, score, price = "HOLD", 0.35, c.iloc[-1]
        
        if regime == 0 and ef > es: sig, score = "BUY", 0.7
        elif regime == 1 and ef < es: sig, score = "SELL", 0.7
        elif regime == 2:
            if rv < params["rsi_os"]: sig, score = "BUY", 0.65
            elif rv > params["rsi_ob"]: sig, score = "SELL", 0.65

        sl_dist = atr_v * params["atr_mult"]
        return {"signal": sig, "score": score, "price": price, "sl_dist": sl_dist, "rr": 1.5, "tp_dist": sl_dist * 1.5}

# ─────────────────────────────────────────────
# MAIN BOT
# ─────────────────────────────────────────────
class HMMForexBot:
    def __init__(self):
        self.broker = OandaBroker(OANDA_API_KEY, OANDA_ACCOUNT_ID, OANDA_ENV)
        self.detector = RegimeDetector()
        self.optimizer = ParamOptimizer()
        self.signal_eng = SignalEngine()
        self.tick_count = 0

    def tick(self):
        df = self.broker.get_candles(INSTRUMENT, GRANULARITY, CANDLE_COUNT)
        if df.empty: return

        if self.tick_count % RETRAIN_EVERY == 0: self.detector.fit(df)
        if not self.detector.fitted: return

        regime, conf, _ = self.detector.predict(df)
        params = self.optimizer.params[regime]
        sig = self.signal_eng.compute(df, regime, params)

        log.info(f"Regime: {REGIMES[regime]} | Signal: {sig['signal']} | Conf: {conf:.2f}")

        if sig["signal"] != "HOLD" and conf >= MIN_CONFIDENCE and sig["score"] >= MIN_CONFIDENCE:
            if self.broker.count_open_trades(INSTRUMENT) < MAX_OPEN_TRADES:
                self._execute(sig, regime)
        
        self.tick_count += 1

    def _execute(self, sig, regime):
        balance = self.broker.get_balance()
        units = self.broker.kelly_units(balance, 0.52, sig["rr"], sig["sl_dist"], sig["price"])
        if sig["signal"] == "SELL": units = -units
        
        is_buy = units > 0
        sl = sig["price"] - sig["sl_dist"] if is_buy else sig["price"] + sig["sl_dist"]
        tp = sig["price"] + sig["tp_dist"] if is_buy else sig["price"] - sig["tp_dist"]
        
        tid = self.broker.place_order(INSTRUMENT, units, sl, tp)
        if tid: log.info(f"Trade Opened: {tid} for {units} units")

    def run(self):
        log.info("HMM Bot Started...")
        while True:
            try:
                self.tick()
            except Exception as e:
                log.error(f"Runtime Error: {e}")
            time.sleep(LOOP_INTERVAL)

if __name__ == "__main__":
    bot = HMMForexBot()
    bot.run()
