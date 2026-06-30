"""
=======================================================================
 Portfolio-Tracker Backend
=======================================================================
Holt reale Marktdaten via yfinance fuer ein konfigurierbares Portfolio
(Aktien, ETFs, Krypto - alles was yfinance abdeckt), berechnet:

  - Aktuellen Depotwert & Performance (gesamt + pro Position)
  - Allokation nach Asset-Klasse
  - Dividendenhistorie & naechste erwartete Ausschuettungen
  - Zeitreihe des Portfolio-Werts fuer den Performance-Chart

Ergebnis wird als output/portfolio_data.json geschrieben und vom
Dashboard (dashboard.html) per fetch() eingelesen.

WICHTIG ZUM SELBST-AUSFUEHREN:
  1. pip install yfinance pandas numpy
  2. Eigene Positionen unten in PORTFOLIO eintragen
  3. python portfolio_tracker.py
  4. dashboard.html ueber einen lokalen Server oeffnen, z.B.:
       python -m http.server 8000
     dann im Browser: http://localhost:8000/dashboard.html
     (direktes Doppelklick-Oeffnen blockiert fetch() per CORS-Regel
     der Browser - ein lokaler Server umgeht das)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("portfolio_tracker")


# ----------------------------------------------------------------------
# 1) PORTFOLIO-DEFINITION - hier eigene Positionen eintragen
# ----------------------------------------------------------------------
@dataclass
class Position:
    ticker: str          # Yahoo-Finance-Symbol, z.B. "AAPL", "VWRL.SW", "BTC-USD"
    shares: float
    buy_price: float     # Einstandspreis pro Stueck in der Handelswaehrung
    buy_date: str         # "YYYY-MM-DD"
    asset_class: str      # "Aktie", "ETF", "Krypto", "Anleihe", ...
    currency: str = "USD"


PORTFOLIO: list[Position] = [
    Position("AAPL",      10, 165.00, "2023-03-15", "Aktie",  "USD"),
    Position("MSFT",       6, 290.00, "2023-01-10", "Aktie",  "USD"),
    Position("VWRL.SW",   25,  98.50, "2022-11-01", "ETF",    "CHF"),
    Position("VUSA.SW",   15,  78.20, "2023-06-20", "ETF",    "CHF"),
    Position("BTC-USD",    0.15, 28000.0, "2023-02-01", "Krypto", "USD"),
    Position("NESN.SW",   20, 102.00, "2022-09-05", "Aktie",  "CHF"),
]

BASE_CURRENCY = "CHF"
OUTPUT_PATH = Path("docs/output/portfolio_data.json")


# ----------------------------------------------------------------------
# 2) FX-RATEN (fuer Umrechnung in Basiswaehrung)
# ----------------------------------------------------------------------
def get_fx_rate(from_ccy: str, to_ccy: str) -> float:
    if from_ccy == to_ccy:
        return 1.0
    pair = f"{from_ccy}{to_ccy}=X"
    try:
        data = yf.Ticker(pair).history(period="5d")
        return float(data["Close"].iloc[-1])
    except Exception:
        log.warning(f"FX-Rate {pair} nicht abrufbar, nutze 1.0 als Fallback")
        return 1.0


# ----------------------------------------------------------------------
# 3) DATEN JE POSITION LADEN
# ----------------------------------------------------------------------
def fetch_position_data(pos: Position, fx_to_base: float) -> dict:
    log.info(f"Lade Daten fuer {pos.ticker} ...")
    tk = yf.Ticker(pos.ticker)

    hist = tk.history(start=pos.buy_date, auto_adjust=True)
    if hist.empty:
        raise ValueError(f"Keine Kursdaten fuer {pos.ticker}")

    current_price = float(hist["Close"].iloc[-1])
    cost_basis = pos.shares * pos.buy_price
    market_value = pos.shares * current_price
    pnl_abs = market_value - cost_basis
    pnl_pct = (pnl_abs / cost_basis) * 100 if cost_basis else 0.0

    # Dividenden seit Kaufdatum
    divs = tk.dividends
    if not divs.empty:
        divs = divs[divs.index >= pd.Timestamp(pos.buy_date, tz=divs.index.tz)]
    div_total = float((divs * pos.shares).sum()) if not divs.empty else 0.0
    last_div = None
    if not divs.empty:
        last_div = {
            "date": divs.index[-1].strftime("%Y-%m-%d"),
            "amount_per_share": float(divs.iloc[-1]),
            "amount_total": float(divs.iloc[-1] * pos.shares),
        }

    # Zeitreihe fuer Portfolio-Performance-Chart (taeglicher Marktwert in Basiswaehrung)
    series = (hist["Close"] * pos.shares * fx_to_base).rename(pos.ticker)

    return {
        "ticker": pos.ticker,
        "asset_class": pos.asset_class,
        "currency": pos.currency,
        "shares": pos.shares,
        "buy_price": pos.buy_price,
        "buy_date": pos.buy_date,
        "current_price": round(current_price, 2),
        "cost_basis_native": round(cost_basis, 2),
        "market_value_native": round(market_value, 2),
        "market_value_base": round(market_value * fx_to_base, 2),
        "pnl_abs_native": round(pnl_abs, 2),
        "pnl_pct": round(pnl_pct, 2),
        "dividends_received_native": round(div_total, 2),
        "last_dividend": last_div,
        "_series": series,
    }


# ----------------------------------------------------------------------
# 4) HAUPTLOGIK
# ----------------------------------------------------------------------
def build_portfolio_snapshot() -> dict:
    positions_out = []
    all_series = []

    fx_cache = {}
    for pos in PORTFOLIO:
        if pos.currency not in fx_cache:
            fx_cache[pos.currency] = get_fx_rate(pos.currency, BASE_CURRENCY)
        fx = fx_cache[pos.currency]

        data = fetch_position_data(pos, fx)
        all_series.append(data.pop("_series"))
        positions_out.append(data)

    # Portfolio-Zeitreihe: Summe aller Positionswerte je Tag (Index-Vereinigung, vorwaerts fuellen)
    combined = pd.concat(all_series, axis=1).sort_index()
    combined = combined.ffill().fillna(0)
    portfolio_series = combined.sum(axis=1)

    total_market_value = sum(p["market_value_base"] for p in positions_out)
    total_cost_basis = sum(p["cost_basis_native"] * fx_cache.get(p["currency"], 1.0) for p in positions_out)
    total_pnl = total_market_value - total_cost_basis
    total_pnl_pct = (total_pnl / total_cost_basis * 100) if total_cost_basis else 0.0
    total_dividends = sum(p["dividends_received_native"] * fx_cache.get(p["currency"], 1.0) for p in positions_out)

    # Allokation nach Asset-Klasse
    allocation: dict[str, float] = {}
    for p in positions_out:
        allocation[p["asset_class"]] = allocation.get(p["asset_class"], 0.0) + p["market_value_base"]
    allocation = {k: round(v, 2) for k, v in allocation.items()}

    # Dividenden-Liste (alle Positionen, sortiert nach letztem Ex-Datum)
    dividend_events = [
        {"ticker": p["ticker"], **p["last_dividend"]}
        for p in positions_out if p["last_dividend"]
    ]
    dividend_events.sort(key=lambda d: d["date"], reverse=True)

    snapshot = {
        "generated_at": datetime.now().isoformat(),
        "base_currency": BASE_CURRENCY,
        "summary": {
            "total_market_value": round(total_market_value, 2),
            "total_cost_basis": round(total_cost_basis, 2),
            "total_pnl_abs": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "total_dividends_received": round(total_dividends, 2),
            "position_count": len(positions_out),
        },
        "allocation": allocation,
        "positions": positions_out,
        "dividend_events": dividend_events,
        "performance_series": [
            {"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 2)}
            for d, v in portfolio_series.items()
        ],
    }
    return snapshot


def main():
    snapshot = build_portfolio_snapshot()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    log.info(f"Portfolio-Snapshot gespeichert: {OUTPUT_PATH}")
    log.info(f"Depotwert: {snapshot['summary']['total_market_value']} {BASE_CURRENCY} "
              f"({snapshot['summary']['total_pnl_pct']:+.2f}%)")


if __name__ == "__main__":
    main()
