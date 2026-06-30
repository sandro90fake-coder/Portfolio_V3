"""
=======================================================================
 RSI + Moving-Average Trend-Following Strategie (Swing-Trading, Aktien)
=======================================================================

Ziel:
    Stabile, risikooptimierte Rendite mit klaren, regelbasierten
    Ein- und Ausstiegssignalen. Kein High-Frequency-Trading -
    es werden Tagesdaten (1d) verwendet.

Marktdaten:
    Yahoo Finance über das Paket `yfinance` (kostenlos, keine API-Keys nötig).

Strategie-Logik:
    LONG-Einstieg (Kauf):
        - RSI(14) < 30           -> Markt ist überverkauft
        - UND Close > SMA(200)   -> langfristiger Aufwärtstrend intakt
        => "Buy the dip im Bullenmarkt"

    Ausstieg (Verkauf):
        - RSI(14) > 70           -> Markt ist überkauft (Take-Profit-Signal)
        - ODER Close < SMA(50)   -> kurzfristiger Trendbruch (Stop/Exit)
        - ODER Stop-Loss wird getroffen (siehe Risk-Management)

    Zusatzfilter (optional, standardmäßig aktiv):
        - MACD-Linie > Signal-Linie für zusätzliche Bestätigung beim Einstieg

Risk-Management:
    - Risiko pro Trade: 1% des aktuellen Kapitals (konfigurierbar)
    - Stop-Loss: technisch (z. B. letztes Swing-Low / ATR-basiert)
    - Positionsgröße = (Kapital * Risiko%) / (Entry - StopLoss)

Abhängigkeiten (siehe requirements.txt):
    pandas, numpy, matplotlib, yfinance, pandas-ta
"""

from __future__ import annotations

import dataclasses
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False

import matplotlib
matplotlib.use("Agg")  # damit es auch headless (GitHub Actions) funktioniert
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("trading_bot")


# ----------------------------------------------------------------------
# 1) KONFIGURATION
# ----------------------------------------------------------------------
@dataclasses.dataclass
class Config:
    ticker: str = "AAPL"            # Aktien-Symbol (Yahoo Finance Format)
    start: str = "2015-01-01"
    end: str | None = None          # None = bis heute

    rsi_period: int = 14
    rsi_buy: float = 30.0
    rsi_sell: float = 70.0

    sma_short: int = 50
    sma_long: int = 200

    use_macd_filter: bool = True
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Risk Management
    initial_capital: float = 10_000.0
    risk_per_trade: float = 0.01    # 1% Risiko pro Trade
    atr_period: int = 14
    atr_stop_multiplier: float = 2.0  # Stop-Loss = Entry - 2*ATR
    commission_pct: float = 0.0005  # 0.05% pro Trade (Kauf & Verkauf)

    output_dir: str = "docs/output"


# ----------------------------------------------------------------------
# 2) DATEN LADEN
# ----------------------------------------------------------------------
def load_data(cfg: Config) -> pd.DataFrame:
    """Lädt historische Tagesdaten von Yahoo Finance."""
    log.info(f"Lade Daten für {cfg.ticker} ab {cfg.start} ...")
    df = yf.download(
        cfg.ticker, start=cfg.start, end=cfg.end,
        auto_adjust=True, progress=False,
    )
    if df.empty:
        raise ValueError(f"Keine Daten für Ticker {cfg.ticker} gefunden.")

    # yfinance liefert bei Einzeltickern manchmal MultiIndex-Spalten -> normalisieren
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.title)  # Open, High, Low, Close, Volume
    df = df.dropna()
    return df


# ----------------------------------------------------------------------
# 3) INDIKATOREN BERECHNEN
# ----------------------------------------------------------------------
def rsi(series: pd.Series, period: int) -> pd.Series:
    """Klassischer RSI nach Wilder (manuelle Implementierung, falls pandas-ta fehlt)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Average True Range - wird für den dynamischen Stop-Loss genutzt."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def macd(series: pd.Series, fast: int, slow: int, signal: int):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def add_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()

    if HAS_PANDAS_TA:
        df["RSI"] = ta.rsi(df["Close"], length=cfg.rsi_period)
        macd_df = ta.macd(
            df["Close"], fast=cfg.macd_fast, slow=cfg.macd_slow, signal=cfg.macd_signal
        )
        df["MACD"] = macd_df.iloc[:, 0]
        df["MACD_SIGNAL"] = macd_df.iloc[:, 2]
        df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=cfg.atr_period)
    else:
        df["RSI"] = rsi(df["Close"], cfg.rsi_period)
        df["MACD"], df["MACD_SIGNAL"] = macd(
            df["Close"], cfg.macd_fast, cfg.macd_slow, cfg.macd_signal
        )
        df["ATR"] = atr(df, cfg.atr_period)

    df["SMA_SHORT"] = df["Close"].rolling(cfg.sma_short).mean()
    df["SMA_LONG"] = df["Close"].rolling(cfg.sma_long).mean()

    df = df.dropna().copy()
    return df


# ----------------------------------------------------------------------
# 4) SIGNALE GENERIEREN
# ----------------------------------------------------------------------
def generate_signals(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()

    buy_condition = (
        (df["RSI"] < cfg.rsi_buy) &
        (df["Close"] > df["SMA_LONG"])
    )
    if cfg.use_macd_filter:
        buy_condition &= (df["MACD"] > df["MACD_SIGNAL"])

    sell_condition = (
        (df["RSI"] > cfg.rsi_sell) |
        (df["Close"] < df["SMA_SHORT"])
    )

    df["BUY_SIGNAL"] = buy_condition
    df["SELL_SIGNAL"] = sell_condition
    return df


# ----------------------------------------------------------------------
# 5) POSITIONSGRÖSSE (RISK MANAGEMENT)
# ----------------------------------------------------------------------
def position_size(capital: float, risk_pct: float, entry: float, stop_loss: float) -> float:
    """
    Berechnet die Anzahl der Aktien basierend auf dem Kapitalrisiko.
    Risiko pro Trade = capital * risk_pct
    Risiko pro Aktie = entry - stop_loss
    """
    risk_amount = capital * risk_pct
    risk_per_share = max(entry - stop_loss, 1e-6)  # Division durch 0 vermeiden
    shares = risk_amount / risk_per_share
    return max(shares, 0)


# ----------------------------------------------------------------------
# 6) BACKTEST-ENGINE
# ----------------------------------------------------------------------
@dataclasses.dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    stop_loss: float
    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str | None = None

    @property
    def pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) * self.shares


def backtest(df: pd.DataFrame, cfg: Config):
    capital = cfg.initial_capital
    equity_curve = []
    trades: list[Trade] = []
    position: Trade | None = None

    for date, row in df.iterrows():
        price = row["Close"]

        # --- Offene Position verwalten ---
        if position is not None:
            stop_hit = price <= position.stop_loss
            sell_signal = row["SELL_SIGNAL"]

            if stop_hit or sell_signal:
                exit_price = price * (1 - cfg.commission_pct)
                position.exit_date = date
                position.exit_price = exit_price
                position.exit_reason = "Stop-Loss" if stop_hit else "Signal"
                capital += position.shares * exit_price
                trades.append(position)
                position = None

        # --- Neue Position eröffnen ---
        elif row["BUY_SIGNAL"]:
            stop_loss = price - cfg.atr_stop_multiplier * row["ATR"]
            shares = position_size(capital, cfg.risk_per_trade, price, stop_loss)
            cost = shares * price * (1 + cfg.commission_pct)

            if shares > 0 and cost <= capital:
                capital -= cost
                position = Trade(
                    entry_date=date, entry_price=price,
                    shares=shares, stop_loss=stop_loss,
                )

        # --- Equity tracken (Cash + offene Position zu Marktpreis) ---
        open_value = position.shares * price if position else 0.0
        equity_curve.append({"Date": date, "Equity": capital + open_value})

    # Letzte offene Position zum Schlusskurs glattstellen (für saubere Auswertung)
    if position is not None:
        last_date = df.index[-1]
        last_price = df["Close"].iloc[-1]
        position.exit_date = last_date
        position.exit_price = last_price
        position.exit_reason = "Backtest-Ende"
        capital += position.shares * last_price
        trades.append(position)

    equity_df = pd.DataFrame(equity_curve).set_index("Date")
    return equity_df, trades


# ----------------------------------------------------------------------
# 7) PERFORMANCE-METRIKEN
# ----------------------------------------------------------------------
def performance_metrics(equity_df: pd.DataFrame, trades: list[Trade], cfg: Config) -> dict:
    equity = equity_df["Equity"]
    total_return = (equity.iloc[-1] / cfg.initial_capital - 1) * 100

    daily_returns = equity.pct_change().dropna()
    sharpe = (
        (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        if daily_returns.std() > 0 else 0.0
    )

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    wins = [t for t in trades if t.pnl > 0]
    win_rate = (len(wins) / len(trades) * 100) if trades else 0.0

    avg_win = np.mean([t.pnl for t in wins]) if wins else 0.0
    losses = [t for t in trades if t.pnl <= 0]
    avg_loss = np.mean([t.pnl for t in losses]) if losses else 0.0

    return {
        "Gesamtrendite (%)": round(total_return, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Max Drawdown (%)": round(max_drawdown, 2),
        "Anzahl Trades": len(trades),
        "Trefferquote (%)": round(win_rate, 2),
        "Ø Gewinn pro Trade": round(avg_win, 2),
        "Ø Verlust pro Trade": round(avg_loss, 2),
        "Endkapital": round(equity.iloc[-1], 2),
    }


# ----------------------------------------------------------------------
# 8) PLOTTING
# ----------------------------------------------------------------------
def plot_results(df: pd.DataFrame, equity_df: pd.DataFrame, trades: list[Trade], cfg: Config):
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)

    # --- Preis-Chart mit Trades ---
    ax1 = axes[0]
    ax1.plot(df.index, df["Close"], label="Close", color="black", linewidth=1)
    ax1.plot(df.index, df["SMA_SHORT"], label=f"SMA{cfg.sma_short}", color="orange", linewidth=1)
    ax1.plot(df.index, df["SMA_LONG"], label=f"SMA{cfg.sma_long}", color="blue", linewidth=1)

    for t in trades:
        ax1.scatter(t.entry_date, t.entry_price, marker="^", color="green", s=80, zorder=5)
        if t.exit_date is not None:
            color = "blue" if t.pnl > 0 else "red"
            ax1.scatter(t.exit_date, t.exit_price, marker="v", color=color, s=80, zorder=5)

    ax1.set_title(f"{cfg.ticker} - Kursverlauf & Trades")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # --- Equity-Kurve ---
    ax2 = axes[1]
    ax2.plot(equity_df.index, equity_df["Equity"], color="darkgreen")
    ax2.set_title("Equity Curve (Portfolio-Wert)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / f"{cfg.ticker}_backtest.png"
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)
    log.info(f"Chart gespeichert: {fig_path}")


# ----------------------------------------------------------------------
# 9) JSON-EXPORT FÜR DAS DASHBOARD (dashboard.html liest diese Datei)
# ----------------------------------------------------------------------
def export_dashboard_json(df: pd.DataFrame, equity_df: pd.DataFrame,
                           trades: list[Trade], metrics: dict, cfg: Config):
    """Schreibt alle Daten, die dashboard.html per fetch() benötigt, als JSON."""
    import json

    last = df.iloc[-1]
    if last["BUY_SIGNAL"]:
        signal, label = "buy", "KAUF"
    elif last["SELL_SIGNAL"]:
        signal, label = "sell", "VERKAUF"
    else:
        signal, label = "hold", "HALTEN"

    # Equity-Werte je df-Datum zuordnen (gleicher Index)
    equity_aligned = equity_df["Equity"].reindex(df.index).ffill().fillna(cfg.initial_capital)

    payload = {
        "generated_at": datetime.now().isoformat(),
        "ticker": cfg.ticker,
        "last_signal": {"code": signal, "label": label},
        "metrics": metrics,
        "series": [
            {
                "date": d.strftime("%Y-%m-%d"),
                "close": round(float(row["Close"]), 4),
                "sma50": round(float(row["SMA_SHORT"]), 4),
                "sma200": round(float(row["SMA_LONG"]), 4),
                "rsi": round(float(row["RSI"]), 2),
                "equity": round(float(equity_aligned.loc[d]), 2),
            }
            for d, row in df.iterrows()
        ],
        "trades": [
            {
                "entry_date": t.entry_date.strftime("%Y-%m-%d"),
                "entry_price": round(t.entry_price, 4),
                "exit_date": t.exit_date.strftime("%Y-%m-%d") if t.exit_date is not None else None,
                "exit_price": round(t.exit_price, 4) if t.exit_price is not None else None,
                "exit_reason": t.exit_reason,
                "pnl": round(t.pnl, 2),
            }
            for t in trades
        ],
    }

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "trading_data.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info(f"Dashboard-JSON gespeichert: {out_dir / 'trading_data.json'}")


# ----------------------------------------------------------------------
# 9b) TÄGLICHE SIGNAL-GENERIERUNG (für GitHub Actions Automatisierung)
# ----------------------------------------------------------------------
def generate_today_signal(df: pd.DataFrame, cfg: Config) -> str:
    """Gibt das aktuellste Signal (letzter Handelstag) als lesbaren String zurück."""
    last = df.iloc[-1]
    date_str = df.index[-1].strftime("%Y-%m-%d")

    if last["BUY_SIGNAL"]:
        signal = "KAUF"
    elif last["SELL_SIGNAL"]:
        signal = "VERKAUF"
    else:
        signal = "HALTEN / KEIN SIGNAL"

    msg = (
        f"Datum: {date_str}\n"
        f"Ticker: {cfg.ticker}\n"
        f"Schlusskurs: {last['Close']:.2f}\n"
        f"RSI({cfg.rsi_period}): {last['RSI']:.2f}\n"
        f"SMA{cfg.sma_short}: {last['SMA_SHORT']:.2f} | SMA{cfg.sma_long}: {last['SMA_LONG']:.2f}\n"
        f"Signal: {signal}\n"
    )
    return msg


def save_signal_log(msg: str, cfg: Config):
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(exist_ok=True)
    log_path = out_dir / "signals_log.txt"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n--- Lauf am {datetime.now().isoformat()} ---\n")
        f.write(msg)
    log.info(f"Signal in {log_path} gespeichert.")


# ----------------------------------------------------------------------
# 10) MAIN
# ----------------------------------------------------------------------
def run(cfg: Config):
    df = load_data(cfg)
    df = add_indicators(df, cfg)
    df = generate_signals(df, cfg)

    equity_df, trades = backtest(df, cfg)
    metrics = performance_metrics(equity_df, trades, cfg)

    log.info("=== Backtest-Ergebnisse ===")
    for k, v in metrics.items():
        log.info(f"{k}: {v}")

    plot_results(df, equity_df, trades, cfg)
    export_dashboard_json(df, equity_df, trades, metrics, cfg)

    signal_msg = generate_today_signal(df, cfg)
    log.info("=== Aktuelles Signal ===\n" + signal_msg)
    save_signal_log(signal_msg, cfg)

    # Ergebnisse als CSV/Text ablegen (z. B. für GitHub-Artifact / Commit)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(exist_ok=True)
    pd.Series(metrics).to_csv(out_dir / f"{cfg.ticker}_metrics.csv")
    equity_df.to_csv(out_dir / f"{cfg.ticker}_equity_curve.csv")

    return df, equity_df, trades, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RSI/SMA Trading-Strategie Backtest")
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--risk", type=float, default=0.01)
    args = parser.parse_args()

    cfg = Config(
        ticker=args.ticker,
        start=args.start,
        initial_capital=args.capital,
        risk_per_trade=args.risk,
    )
    run(cfg)
