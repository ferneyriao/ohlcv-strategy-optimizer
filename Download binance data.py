import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOL   = 'BTCUSDT'
INTERVAL = '1m'
LIMIT    = 1000          # velas por request (max Binance)
WORKERS  = 10            # requests paralelos; Binance soporta ~20 antes de 429
RETRY    = 3             # reintentos por chunk en caso de error

START_STR = '2026-01-01 00:00:00'
END_STR   = '2026-09-29 23:59:59'

INTERVAL_MS = {
    '1m': 60_000, '3m': 180_000, '5m': 300_000, '15m': 900_000,
    '30m': 1_800_000, '1h': 3_600_000, '4h': 14_400_000, '1d': 86_400_000,
}.get(INTERVAL, 60_000)

BASE_URL = 'https://fapi.binance.com/fapi/v1/klines'

# ── Helpers ───────────────────────────────────────────────────────────────────
def _to_ms(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)

def _fetch_chunk(start_ms: int, end_ms: int) -> list:
    """Descarga un bloque de hasta LIMIT velas; reintenta en errores 429/5xx."""
    params = {
        'symbol':    SYMBOL,
        'interval':  INTERVAL,
        'limit':     LIMIT,
        'startTime': start_ms,
        'endTime':   end_ms,
    }
    for attempt in range(RETRY):
        try:
            r = requests.get(BASE_URL, params=params, timeout=10)
            if r.status_code == 429:
                retry_after = int(r.headers.get('Retry-After', 2 ** (attempt + 1)))
                print(f"  [429] rate-limit, esperando {retry_after}s...")
                import time; time.sleep(retry_after)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == RETRY - 1:
                print(f"  [ERROR] chunk {start_ms}: {e}")
                return []
    return []

# ── Pre-calcular rangos (sin overlap) ─────────────────────────────────────────
def _build_chunks(start_ms: int, end_ms: int) -> list[tuple[int, int]]:
    chunk_span = LIMIT * INTERVAL_MS
    chunks = []
    s = start_ms
    while s <= end_ms:
        e = min(s + chunk_span - INTERVAL_MS, end_ms)
        chunks.append((s, e))
        s = e + INTERVAL_MS
    return chunks

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    start_ms = _to_ms(START_STR)
    end_ms   = _to_ms(END_STR)
    chunks   = _build_chunks(start_ms, end_ms)

    print(f"Descargando {SYMBOL} {INTERVAL}  {START_STR} → {END_STR}")
    print(f"Chunks: {len(chunks)}  |  Workers: {WORKERS}")

    results: dict[int, list] = {}

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(_fetch_chunk, s, e): s
            for s, e in chunks
        }
        done = 0
        for fut in as_completed(futures):
            start_key = futures[fut]
            data = fut.result()
            if data:
                results[start_key] = data
            done += 1
            print(f"  {done}/{len(chunks)} chunks completados", end='\r', flush=True)

    print()

    # Ordenar por timestamp de inicio de chunk, aplanar
    all_rows = []
    for key in sorted(results):
        all_rows.extend(results[key])

    if not all_rows:
        print("Sin datos descargados.")
        return

    cols = [
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ]
    df = pd.DataFrame(all_rows, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].astype({
        "open": float, "high": float, "low": float,
        "close": float, "volume": float,
    })
    df["price_change"] = df["close"].pct_change()

    # Filtro de rango + deduplicar (chunks adyacentes pueden solapar 1 vela)
    df = df[
        (df["timestamp"] >= pd.to_datetime(START_STR)) &
        (df["timestamp"] <= pd.to_datetime(END_STR))
    ].drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    filename = f"{SYMBOL.lower()}_{INTERVAL}.csv"
    df.to_csv(filename, index=False, float_format="%.10f")
    print(f"✅  {len(df)} velas  →  '{filename}'")

if __name__ == '__main__':
    main()
