from pathlib import Path
import importlib.metadata
import pandas as pd
from yfinance.scrapers.history import PriceHistory


def test_scipy_is_declared_and_installed():
    requirements = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text().splitlines()
    assert "scipy==" + importlib.metadata.version("scipy") in requirements


def test_offline_yahoo_price_repair():
    frame = pd.DataFrame({c: [100.0] * 5 for c in ["Open", "High", "Low", "Close", "Adj Close"]}, index=pd.date_range("2026-01-05", periods=5, tz="America/New_York"))
    frame["Volume"] = 1000
    history = PriceHistory(None, "SYNTHETIC", "America/New_York")
    result = history._fix_unit_random_mixups(frame, "1d", "America/New_York", False)
    assert len(result) == 5
    assert result["Close"].eq(100.0).all()


def test_synthetic_html_output(tmp_path, monkeypatch):
    import generate_daily_report as report
    monkeypatch.setattr(report, 'OUT', tmp_path)
    for name in ('biglist', 'watchlist'):
        pd.DataFrame(columns=['symbol', 'final_classification']).to_csv(tmp_path / (name + '_latest.csv'), index=False)
    report.main()
    html = (tmp_path / 'daily_report_latest.html').read_text()
    assert '<!doctype html>' in html
    assert html.count('No stocks in this list.') == 2
