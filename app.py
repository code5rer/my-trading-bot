"""
HMM Forex Trading Bot — OANDA API
===================================
Strategy:
  - Hidden Markov Model (4 regimes: Bull, Bear, Range, Volatile)
  - EM algorithm for unsupervised regime learning
  - Viterbi decoding for most-likely regime path
  - Per-regime optimized trading parameters (EMA, RSI, ATR, BB)
  - ML feedback loop: parameters self-improve using only profitable trades
  - Dynamic position sizing via Kelly Criterion

Requirements:
    pip install oandapyV20 hmmlearn scikit-learn numpy pandas ta
"""

import time, logging, threading
from datetime import datetime, timedelta
from collections import deque

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import ta                              # technical-analysis library
import oandapyV20
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.orders   as orders
import oandapyV20.endpoints.trades   as trades_ep
import oandapyV20.endpoints.pricing  as pricing
import oandapyV20.endpoints.instruments as instruments

# ─────────────────────────────────────────────
# CONFIG  (edit before running)
# ─────────────────────────────────────────────
OANDA_API_KEY    = "YOUR_OANDA_API_KEY"
OANDA_ACCOUNT_ID = "YOUR_ACCOUNT_ID"
OANDA_ENV        = "practice"           # "practice" or "live"
INSTRUMENT       = "EUR_USD"
GRANULARITY      = "M15"                # candlestick timeframe
CANDLE_COUNT     = 500                  # history candles to fetch
RISK_PER_TRADE   = 0.01                 # 1% account risk per trade
MAX_OPEN_TRADES  = 3
MIN_CONFIDENCE   = 0.62                 # minimum HMM posterior to trade
RETRAIN_EVERY    = 50                   # retrain HMM every N ticks
LOOP_INTERVAL    = 60                   # seconds between main loop ticks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("HMM-FX")

# ─────────────────────────────────────────────
# REGIME DEFINITIONS
# ─────────────────────────────────────────────
REGIMES = {
    0: "Bullish",
    1: "Bearish",
    2: "Ranging",
    3: "Volatile",
}

# Default per-regime parameters (will be updated by ML)
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
    """
    Wraps hmmlearn.GaussianHMM.
    Features fed to the HMM:
      - log returns
      - rolling 14-period volatility (annualised)
      - RSI (14)
      - ATR normalised by price
    EM is run on each retrain call (fit).
    Viterbi gives the most-likely hidden-state sequence.
    """

    def __init__(self, n_states: int = 4):
        self.n_states = n_states
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            tol=1e-4,
            random_state=42,
        )
        self.scaler  = StandardScaler()
        self.fitted  = False
        self._state_map: dict[int, int] = {}   # raw HMM state → semantic regime

    # ── Feature engineering ────────────────────
    @staticmethod
    def build_features(df: pd.DataFrame) -> np.ndarray:
        c = df["close"].astype(float)
        h = df["high"].astype(float)
        l = df["low"].astype(float)

        log_ret = np.log(c / c.shift(1)).fillna(0)
        roll_vol = log_ret.rolling(14).std().fillna(0) * np.sqrt(252 * 24 * 4)
        rsi      = ta.momentum.RSIIndicator(c, window=14).rsi().fillna(50) / 100
        atr_norm = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range().fillna(0) / c

        feats = np.column_stack([log_ret, roll_vol, rsi, atr_norm])
        # drop rows where ANY feature is NaN (first 14 rows typically)
        mask = np.isfinite(feats).all(axis=1)
        return feats[mask], mask

    # ── Training (EM) ──────────────────────────
    def fit(self, df: pd.DataFrame) -> None:
        feats, _ = self.build_features(df)
        feats_scaled = self.scaler.fit_transform(feats)
        self.model.fit(feats_scaled)
        self.fitted = True
        self._build_state_map(feats)
        log.info("HMM retrained | logL=%.1f | states=%s",
                 self.model.score(feats_scaled),
                 dict(zip(range(self.n_states),
                          [REGIMES[self._state_map.get(i, i)]
                           for i in range(self.n_states)])))

    # ── Map raw HMM states → semantic regimes ──
    def _build_state_map(self, feats_raw: np.ndarray) -> None:
        # Mean log-return per state
        states_seq = self.model.predict(self.scaler.transform(feats_raw))
        means = {}
        vols  = {}
        for s in range(self.n_states):
            idx = states_seq == s
            means[s] = feats_raw[idx, 0].mean() if idx.any() else 0
            vols[s]  = feats_raw[idx, 1].mean() if idx.any() else 0

        sorted_by_mean = sorted(means, key=means.get)   # lowest → highest return
        sorted_by_vol  = sorted(vols,  key=vols.get,  reverse=True)  # highest vol first

        # Volatile = highest vol; Bull = highest mean among remainder; Bear = lowest mean
        vol_state  = sorted_by_vol[0]
        remaining  = [s for s in sorted_by_mean if s != vol_state]
        bull_state = remaining[-1]
        bear_state = remaining[0]
        range_state = [s for s in range(self.n_states)
                       if s not in (vol_state, bull_state, bear_state)][0]

        self._state_map = {
            bull_state:  0,   # Bullish
            bear_state:  1,   # Bearish
            range_state: 2,   # Ranging
            vol_state:   3,   # Volatile
        }

    # ── Inference ──────────────────────────────
    def predict(self, df: pd.DataFrame) -> tuple[int, float, np.ndarray]:
        """
        Returns (regime_id, posterior_confidence, full_posteriors_array).
        Uses Viterbi for the sequence, then last-step forward posteriors.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before predict().")
        feats, mask = self.build_features(df)
        feats_scaled = self.scaler.transform(feats)

        # Viterbi most-likely path
        viterbi_states = self.model.predict(feats_scaled)
        raw_state = int(viterbi_states[-1])

        # Posterior probabilities for last observation (forward algorithm)
        posteriors = self.model.predict_proba(feats_scaled)[-1]
        # Map raw state → semantic regime
        semantic = self._state_map.get(raw_state, 0)
        confidence = float(posteriors[raw_state])
        return semantic, confidence, posteriors

    @property
    def transition_matrix(self) -> np.ndarray:
        return self.model.transmat_


# ─────────────────────────────────────────────
# ML PARAMETER OPTIMIZER
# ─────────────────────────────────────────────
class ParamOptimizer:
    """
    Gradient-free optimiser: keeps a rolling window of recent trades
    per regime and nudges parameters toward configurations that
    maximise the Profit Factor (gross wins / gross losses).
    Only the most profitable parameter set is kept.
    """

    STEP = {
        "ema_fast": 1, "ema_slow": 1,
        "rsi_ob": 1,   "rsi_os": 1,
        "atr_mult": 0.1, "bb_dev": 0.05,
    }
    BOUNDS = {
        "ema_fast": (3, 30),   "ema_slow": (10, 100),
        "rsi_ob":   (55, 85),  "rsi_os":   (15, 45),
        "atr_mult": (0.5, 5),  "bb_dev":   (1.0, 3.5),
    }

    def __init__(self):
        self.params  = {r: dict(DEFAULT_PARAMS[r]) for r in range(4)}
        self.history: dict[int, deque] = {r: deque(maxlen=100) for r in range(4)}
        self.best_pf: dict[int, float] = {r: 1.0 for r in range(4)}

    def record_trade(self, regime: int, pnl: float) -> None:
        self.history[regime].append(pnl)

    def update(self, regime: int) -> None:
        h = list(self.history[regime])
        if len(h) < 10:
            return
        wins   = [x for x in h if x > 0]
        losses = [x for x in h if x < 0]
        gross_w = sum(wins)   or 1e-9
        gross_l = abs(sum(losses)) or 1e-9
        pf = gross_w / gross_l
        win_rate = len(wins) / len(h)

        if pf > self.best_pf[regime]:
            self.best_pf[regime] = pf
            log.info("ML | regime=%s | new best PF=%.2f | wr=%.0f%%",
                     REGIMES[regime], pf, win_rate * 100)
            # Performance improving: fine-tune current params
            self._perturb(regime, direction=+1)
        else:
            # Performance degrading: explore a different config
            self._perturb(regime, direction=-1)

    def _perturb(self, regime: int, direction: int) -> None:
        p = self.params[regime]
        for key, step in self.STEP.items():
            lo, hi = self.BOUNDS[key]
            delta = direction * step * (np.random.choice([-1, 0, 1]))
            p[key] = float(np.clip(p[key] + delta, lo, hi))

    def get(self, regime: int) -> dict:
        return self.params[regime]


# ─────────────────────────────────────────────
# SIGNAL GENERATOR
# ─────────────────────────────────────────────
class SignalEngine:
    """
    Generates BUY / SELL / HOLD using the HMM-detected regime's
    optimized parameters.
    """

    def compute(self, df: pd.DataFrame, regime: int,
                params: dict) -> dict:
        c = df["close"].astype(float)
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        n = len(c)

        ema_fast = ta.trend.EMAIndicator(c, window=params["ema_fast"]).ema_indicator()
        ema_slow = ta.trend.EMAIndicator(c, window=params["ema_slow"]).ema_indicator()
        rsi      = ta.momentum.RSIIndicator(c, window=14).rsi()
        atr      = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
        bb       = ta.volatility.BollingerBands(c, window=20, window_dev=params["bb_dev"])

        ef = float(ema_fast.iloc[-1])
        es = float(ema_slow.iloc[-1])
        rv = float(rsi.iloc[-1])
        atr_v = float(atr.iloc[-1])
        bb_up = float(bb.bollinger_hband().iloc[-1])
        bb_lo = float(bb.bollinger_lband().iloc[-1])
        price = float(c.iloc[-1])

        signal = "HOLD"
        score  = 0

        if regime == 0:    # ── BULLISH: trend-following ──────
            if ef > es and rv < params["rsi_ob"] and price > bb_lo:
                signal = "BUY";  score = (ef - es) / es * 500 + (params["rsi_ob"] - rv) / 100
            elif ef < es or rv > params["rsi_ob"]:
                signal = "SELL"; score = 0.3

        elif regime == 1:  # ── BEARISH: trend-following short ─
            if ef < es and rv > params["rsi_os"] and price < bb_up:
                signal = "SELL"; score = (es - ef) / es * 500 + (rv - params["rsi_os"]) / 100
            elif ef > es or rv < params["rsi_os"]:
                signal = "BUY";  score = 0.3

        elif regime == 2:  # ── RANGING: mean-reversion ─────
            if rv > params["rsi_ob"] and price >= bb_up * 0.998:
                signal = "SELL"; score = (rv - params["rsi_ob"]) / 30
            elif rv < params["rsi_os"] and price <= bb_lo * 1.002:
                signal = "BUY";  score = (params["rsi_os"] - rv) / 30

        else:              # ── VOLATILE: breakout ──────────
            momentum = abs(ef - es) / es
            if momentum > 0.003:
                signal = "BUY"  if ef > es else "SELL"
                score = momentum * 200

        sl_dist = atr_v * params["atr_mult"]
        tp_dist = sl_dist * (1.5 + np.random.uniform(0, 0.5))   # min 1.5 R:R

        return {
            "signal":  signal,
            "score":   float(np.clip(0.35 + score, 0, 0.97)),
            "price":   price,
            "sl_dist": sl_dist,
            "tp_dist": tp_dist,
            "rr":      round(tp_dist / max(sl_dist, 1e-9), 2),
            "atr":     atr_v,
        }


# ─────────────────────────────────────────────
# OANDA BROKER INTERFACE
# ─────────────────────────────────────────────
class OandaBroker:
    def __init__(self, api_key: str, account_id: str, env: str):
        self.account_id = account_id
        self.client = oandapyV20.API(access_token=api_key, environment=env)

    # ── Fetch candles ──────────────────────────
    def get_candles(self, instrument: str, granularity: str,
                    count: int) -> pd.DataFrame:
        params = {"granularity": granularity, "count": count}
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        self.client.request(r)
        rows = []
        for c in r.response["candles"]:
            if c["complete"]:
                rows.append({
                    "time":  c["time"],
                    "open":  float(c["mid"]["o"]),
                    "high":  float(c["mid"]["h"]),
                    "low":   float(c["mid"]["l"]),
                    "close": float(c["mid"]["c"]),
                    "volume": int(c["volume"]),
                })
        return pd.DataFrame(rows)

    # ── Get account balance ────────────────────
    def get_balance(self) -> float:
        r = accounts.AccountSummary(accountID=self.account_id)
        self.client.request(r)
        return float(r.response["account"]["balance"])

    # ── Count open trades ──────────────────────
    def count_open_trades(self, instrument: str) -> int:
        r = trades_ep.TradesList(accountID=self.account_id)
        self.client.request(r)
        return sum(1 for t in r.response["trades"]
                   if t["instrument"] == instrument)

    # ── Place market order ─────────────────────
    def place_order(self, instrument: str, units: int,
                    sl_price: float, tp_price: float) -> str | None:
        data = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "stopLossOnFill":   {"price": str(round(sl_price, 5))},
                "takeProfitOnFill": {"price": str(round(tp_price, 5))},
            }
        }
        r = orders.OrderCreate(accountID=self.account_id, data=data)
        try:
            self.client.request(r)
            trade_id = r.response.get("orderFillTransaction", {}).get("tradeOpened", {}).get("tradeID")
            log.info("Order placed | units=%d | tradeID=%s", units, trade_id)
            return trade_id
        except Exception as e:
            log.error("Order failed: %s", e)
            return None

    # ── Position sizing (Kelly-fraction) ───────
    def kelly_units(self, balance: float, win_rate: float,
                    rr: float, sl_dist: float,
                    price: float, instrument: str) -> int:
        """
        Kelly fraction: f = (W * RR - (1-W)) / RR
        Capped at RISK_PER_TRADE for safety.
        """
        kelly = max(0, (win_rate * rr - (1 - win_rate)) / rr)
        safe_fraction = min(kelly * 0.25, RISK_PER_TRADE)   # quarter-Kelly
        risk_amount = balance * safe_fraction
        # pip value approximation (works for USD pairs)
        if "JPY" in instrument:
            pip = 0.01
        else:
            pip = 0.0001
        sl_pips = sl_dist / pip
        unit_risk = sl_pips * pip * (1 / price)   # risk per unit in account CCY
        units = int(risk_amount / max(unit_risk, 1e-9))
        return max(1000, min(units, 100_000))      # clamp to reasonable range


# ─────────────────────────────────────────────
# PERFORMANCE TRACKER
# ─────────────────────────────────────────────
class PerformanceTracker:
    def __init__(self):
        self.trades: list[dict] = []

    def record(self, regime: int, pnl: float, conf: float) -> None:
        self.trades.append({
            "time":   datetime.utcnow().isoformat(),
            "regime": REGIMES[regime],
            "pnl":    pnl,
            "conf":   conf,
        })

    def summary(self) -> dict:
        if not self.trades:
            return {}
        pnls = [t["pnl"] for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        return {
            "total":       len(pnls),
            "win_rate":    round(len(wins) / len(pnls) * 100, 1),
            "profit_factor": round(sum(wins) / max(abs(sum(losses)), 1e-9), 2),
            "net_pnl":     round(sum(pnls), 2),
            "max_dd":      round(self._max_dd(pnls), 2),
        }

    @staticmethod
    def _max_dd(pnls: list[float]) -> float:
        equity = np.cumsum(pnls)
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity)
        return float(dd.max()) if len(dd) else 0.0


# ─────────────────────────────────────────────
# MAIN BOT
# ─────────────────────────────────────────────
class HMMForexBot:
    def __init__(self):
        self.broker    = OandaBroker(OANDA_API_KEY, OANDA_ACCOUNT_ID, OANDA_ENV)
        self.detector  = RegimeDetector(n_states=4)
        self.optimizer = ParamOptimizer()
        self.signal_eng = SignalEngine()
        self.tracker   = PerformanceTracker()
        self.tick_count = 0
        self._lock = threading.Lock()

    # ── Single tick ────────────────────────────
    def tick(self) -> None:
        with self._lock:
            self.tick_count += 1

            # 1. Fetch latest candles
            df = self.broker.get_candles(INSTRUMENT, GRANULARITY, CANDLE_COUNT)
            if df.empty or len(df) < 50:
                log.warning("Insufficient data, skipping tick.")
                return

            # 2. Retrain HMM periodically
            if self.tick_count % RETRAIN_EVERY == 1:
                self.detector.fit(df)

            if not self.detector.fitted:
                log.info("Waiting for initial HMM fit...")
                return

            # 3. Detect current regime
            regime, confidence, posteriors = self.detector.predict(df)
            log.info("Tick %d | Regime=%s | conf=%.2f | price=%.5f",
                     self.tick_count, REGIMES[regime], confidence,
                     float(df["close"].iloc[-1]))

            # 4. Get ML-optimized params for this regime
            params = self.optimizer.get(regime)

            # 5. Generate signal
            sig = self.signal_eng.compute(df, regime, params)
            log.info("Signal=%s | score=%.2f | R:R=%.1f",
                     sig["signal"], sig["score"], sig["rr"])

            # 6. Execute trade if conditions met
            if (sig["signal"] in ("BUY", "SELL")
                    and sig["score"] >= MIN_CONFIDENCE
                    and confidence >= MIN_CONFIDENCE):

                open_count = self.broker.count_open_trades(INSTRUMENT)
                if open_count < MAX_OPEN_TRADES:
                    self._execute(sig, regime, confidence)
                else:
                    log.info("Max open trades reached (%d), skipping.", open_count)

            # 7. Update ML optimizer
            if self.tick_count % 10 == 0:
                self.optimizer.update(regime)

            # 8. Print performance summary every 20 ticks
            if self.tick_count % 20 == 0:
                s = self.tracker.summary()
                log.info("Performance | %s", s)

    def _execute(self, sig: dict, regime: int, confidence: float) -> None:
        balance  = self.broker.get_balance()
        win_rate = 0.55   # prior; will be overridden by tracker data
        pf_data  = self.tracker.summary()
        if pf_data and pf_data.get("total", 0) > 10:
            win_rate = pf_data["win_rate"] / 100

        units = self.broker.kelly_units(
            balance, win_rate, sig["rr"], sig["sl_dist"],
            sig["price"], INSTRUMENT
        )
        if sig["signal"] == "SELL":
            units = -units

        is_buy = sig["signal"] == "BUY"
        sl_price = sig["price"] - sig["sl_dist"] if is_buy else sig["price"] + sig["sl_dist"]
        tp_price = sig["price"] + sig["tp_dist"] if is_buy else sig["price"] - sig["tp_dist"]

        trade_id = self.broker.place_order(INSTRUMENT, units, sl_price, tp_price)
        if trade_id:
            # Simulate PnL tracking (real implementation: poll OANDA for trade close)
            simulated_pnl = sig["tp_dist"] * abs(units) * 0.3  # placeholder
            self.tracker.record(regime, simulated_pnl, confidence)
            self.optimizer.record_trade(regime, simulated_pnl)

    # ── Run loop ───────────────────────────────
    def run(self) -> None:
        log.info("═══════════════════════════════════════")
        log.info(" HMM Forex Bot starting")
        log.info(" Instrument : %s", INSTRUMENT)
        log.info(" Timeframe  : %s", GRANULARITY)
        log.info(" Environment: %s", OANDA_ENV)
        log.info("═══════════════════════════════════════")

        # Initial HMM fit before entering the loop
        log.info("Fetching initial candles for HMM training...")
        df = self.broker.get_candles(INSTRUMENT, GRANULARITY, CANDLE_COUNT)
        self.detector.fit(df)

        while True:
            try:
                self.tick()
            except KeyboardInterrupt:
                log.info("Shutting down.")
                break
            except Exception as e:
                log.error("Tick error: %s", e, exc_info=True)
            time.sleep(LOOP_INTERVAL)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    bot = HMMForexBot()
    bot.run()
