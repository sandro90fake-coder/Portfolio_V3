# Senros Finance Tools

Zwei automatisierte Finanz-Dashboards, gehostet über GitHub Pages, täglich
aktualisiert mit echten Marktdaten via `yfinance`:

- **Signal-Board** (`docs/dashboard.html`) – RSI/SMA Trend-Strategie, Backtest
  und tägliches Kauf-/Verkaufssignal
- **Kontobuch** (`docs/dashboard_portfolio.html`) – Portfolio-Tracker über
  mehrere Asset-Klassen mit Performance, Allokation und Dividenden

## 1. Setup

```bash
git clone <dein-repo-url>
cd trading-bot
pip install -r requirements.txt
```

Eigene Portfolio-Positionen in `portfolio_tracker.py` unter `PORTFOLIO = [...]`
eintragen (Ticker, Stückzahl, Kaufpreis, Kaufdatum, Asset-Klasse, Währung).
Eigenen Trading-Ticker per `--ticker` an `strategy.py` übergeben.

```bash
python strategy.py --ticker AAPL --start 2015-01-01
python portfolio_tracker.py
```

Beide Skripte schreiben ihre Ergebnisse nach `docs/output/` – genau dort,
wo die Dashboards sie per `fetch()` erwarten.

## 2. Lokal ansehen

```bash
cd docs
python -m http.server 8000
```

Im Browser: `http://localhost:8000` (Landing-Page mit Links zu beiden
Dashboards). Wichtig: nicht die HTML-Dateien direkt per Doppelklick öffnen –
Browser blockieren dann das Nachladen der JSON-Daten (CORS). Ohne lokalen
Server zeigen die Dashboards automatisch Beispieldaten mit Hinweis-Banner.

## 3. Auf GitHub veröffentlichen (GitHub Pages)

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<dein-user>/trading-bot.git
git push -u origin main
```

Dann in den Repo-**Settings**:

1. **Settings → Actions → General → Workflow permissions** →
   *„Read and write permissions"* aktivieren (damit der Bot committen darf)
2. **Settings → Pages → Source** → *„Deploy from a branch"* → Branch `main`,
   Ordner `/docs` → Speichern

Nach ein paar Minuten ist die Seite live unter:
`https://<dein-user>.github.io/trading-bot/`

## 4. Automatisierung

`.github/workflows/daily-signal.yml` läuft werktags automatisch (Cron),
führt beide Skripte aus und committet die aktualisierten JSON-Dateien nach
`docs/output/`. Jeder Commit auf `/docs` löst automatisch ein Rebuild von
GitHub Pages aus – die Webseite aktualisiert sich also von selbst, ganz ohne
manuelles Eingreifen.

Manuell testen: Tab **„Actions"** → „Daily Update & Publish" → **„Run workflow"**.

## Wichtiger Hinweis

Bildungs-/Forschungswerkzeug, keine Anlageberatung. Backtest-Performance ist
keine Garantie für die Zukunft. Vor echtem Kapitaleinsatz: Paper-Trading und
eigene Risikoprüfung durchführen.
