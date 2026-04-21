# =============================================
# Binance USDⓈ-M Futures Public REST API - OHLCV 데이터 가져오기
# =============================================
import requests
import pandas as pd
from config import BINANCE_FUTURES_URL, CANDLE_LIMIT
from http_client import _session as _http  # 프록시 환경변수 무시 세션


def fetch_ohlcv(symbol: str, interval: str, limit: int = CANDLE_LIMIT) -> pd.DataFrame:
    """
    바이낸스 USDⓈ-M Futures 공개 API로 OHLCV 캔들 데이터를 가져옵니다.

    Parameters
    ----------
    symbol   : 'BTCUSDT' 형식
    interval : '15m' | '1h' | '4h' | '1d'
    limit    : 최대 1000, 기본 200

    Returns
    -------
    DataFrame(index=datetime, columns=[open, high, low, close, volume])
    """
    url = f"{BINANCE_FUTURES_URL}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    resp = _http.get(url, params=params, timeout=10)
    resp.raise_for_status()
    raw = resp.json()

    columns = [
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(raw, columns=columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df.set_index("timestamp", inplace=True)
    return df[["open", "high", "low", "close", "volume"]]


def fetch_current_price(symbol: str) -> float:
    """USDⓈ-M Futures 현재 체결가 반환."""
    url = f"{BINANCE_FUTURES_URL}/fapi/v1/ticker/price"
    resp = _http.get(url, params={"symbol": symbol}, timeout=5)
    resp.raise_for_status()
    return float(resp.json()["price"])
