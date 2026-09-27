import itertools, json, multiprocessing, os, time
from multiprocessing import shared_memory
import numpy as np
from numba import njit

# ── Load CSV ──────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
_CSV = os.path.join(_DIR, 'btcusdt_1m.csv')

with open(_CSV) as f:
    _header = f.readline().strip().split(',')
_ci = {name: idx for idx, name in enumerate(_header)}
_cols = (_ci['open'], _ci['high'], _ci['low'], _ci['close'], _ci['volume'])

_raw = np.genfromtxt(_CSV, delimiter=',', skip_header=1,
                      usecols=_cols, dtype=np.float64,
                      filling_values=np.nan)
O, H, L, C, V = (np.ascontiguousarray(_raw[:, i]) for i in range(5))
N = len(C)
print(f"CSV cargado: {N} velas  ({_CSV})", flush=True)

# ── Precálculo de indicadores (una vez, O(n)) ─────────────────────────────────

@njit(cache=True)
def _ema_arr(c, p):
    out = np.empty(len(c), dtype=np.float64)
    out[:] = np.nan
    if len(c) < p: return out
    k = 2.0 / (p + 1.0)
    v = 0.0
    for i in range(p): v += c[i]
    v /= p
    out[p - 1] = v
    for i in range(p, len(c)):
        v = c[i] * k + v * (1.0 - k)
        out[i] = v
    return out

@njit(cache=True)
def _atr_arr(h, l, c, p):
    n = len(c)
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan
    if n < p + 1: return out
    buf = np.empty(n - 1, dtype=np.float64)
    for i in range(1, n):
        buf[i-1] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    s = 0.0
    for i in range(p): s += buf[i]
    out[p] = s / p
    for i in range(p + 1, n):
        out[i] = (out[i-1] * (p - 1) + buf[i-1]) / p
    return out

@njit(cache=True)
def _rsi_arr(c, p):
    n = len(c)
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan
    if n <= p: return out
    ag = al = 0.0
    for i in range(1, p + 1):
        d = c[i] - c[i-1]
        if d > 0: ag += d
        else:     al -= d
    ag /= p; al /= p
    rs = ag / (al if al > 1e-10 else 1e-10)
    out[p] = 100.0 - 100.0 / (1.0 + rs)
    for i in range(p + 1, n):
        d = c[i] - c[i-1]
        ag = (ag * (p - 1) + max(d, 0.0)) / p
        al = (al * (p - 1) + max(-d, 0.0)) / p
        rs = ag / (al if al > 1e-10 else 1e-10)
        out[i] = 100.0 - 100.0 / (1.0 + rs)
    return out

@njit(cache=True)
def _rci_arr(c, p):
    # Ventana deslizante ordenada (valor, edad) mantenida por insertion sort
    # incremental: O(p) por paso (peor caso) en vez de O(p^2) recontando ranks.
    n = len(c)
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan
    if n < p:
        return out
    denom = p * (p * p - 1)

    sorted_val = np.empty(p, dtype=np.float64)  # ventana ordenada descendente
    sorted_age = np.empty(p, dtype=np.int64)     # edad (0=mas nuevo) por posicion

    for a in range(p):
        sorted_val[a] = c[p - 1 - a]
        sorted_age[a] = a
    # insertion sort inicial descendente por valor
    for a in range(1, p):
        v = sorted_val[a]; ag = sorted_age[a]
        b = a - 1
        while b >= 0 and sorted_val[b] < v:
            sorted_val[b+1] = sorted_val[b]
            sorted_age[b+1] = sorted_age[b]
            b -= 1
        sorted_val[b+1] = v
        sorted_age[b+1] = ag

    d2 = 0.0
    for pos in range(p):
        age = sorted_age[pos]
        date_rank = p - age
        val_rank = pos + 1
        diff = val_rank - date_rank
        d2 += diff * diff
    out[p - 1] = (1.0 - 6.0 * d2 / denom) * 100.0

    for i in range(p, n):
        # envejecer y descartar el mas viejo (age == p-1)
        drop_pos = -1
        for pos in range(p):
            sorted_age[pos] += 1
            if sorted_age[pos] == p:
                drop_pos = pos
        new_val = c[i]
        if drop_pos == -1:
            drop_pos = p - 1
        # compactar quitando drop_pos
        for pos in range(drop_pos, p - 1):
            sorted_val[pos] = sorted_val[pos + 1]
            sorted_age[pos] = sorted_age[pos + 1]
        # insertar new_val (age=0) manteniendo orden descendente
        ins = p - 1
        while ins > 0 and sorted_val[ins - 1] < new_val:
            sorted_val[ins] = sorted_val[ins - 1]
            sorted_age[ins] = sorted_age[ins - 1]
            ins -= 1
        sorted_val[ins] = new_val
        sorted_age[ins] = 0

        d2 = 0.0
        for pos in range(p):
            age = sorted_age[pos]
            date_rank = p - age
            val_rank = pos + 1
            diff = val_rank - date_rank
            d2 += diff * diff
        out[i] = (1.0 - 6.0 * d2 / denom) * 100.0

    return out

@njit(cache=True)
def _bb_arr(c, p, m):
    n = len(c)
    upper = np.empty(n, dtype=np.float64); upper[:] = np.nan
    mid   = np.empty(n, dtype=np.float64); mid[:]   = np.nan
    lower = np.empty(n, dtype=np.float64); lower[:] = np.nan
    for i in range(p - 1, n):
        sl = c[i - p + 1: i + 1]
        mu = 0.0
        for x in sl: mu += x
        mu /= p
        var = 0.0
        for x in sl: var += (x - mu) ** 2
        sd = (var / p) ** 0.5
        mid[i]   = mu
        upper[i] = mu + m * sd
        lower[i] = mu - m * sd
    return upper, mid, lower

@njit(cache=True)
def _vwap_arr(h, l, c, v, win):
    # Rolling VWAP de `win` velas (1440 = 24h en 1m) via suma deslizante O(n).
    n = len(c)
    out = np.empty(n, dtype=np.float64)
    tpv = np.empty(n, dtype=np.float64)
    for i in range(n):
        tpv[i] = (h[i] + l[i] + c[i]) / 3.0 * v[i]

    ctv = cv = 0.0
    for i in range(n):
        ctv += tpv[i]; cv += v[i]
        if i >= win:
            ctv -= tpv[i - win]; cv -= v[i - win]
        out[i] = ctv / cv if cv > 0 else np.nan
    return out

@njit(cache=True)
def _mom_arr(c, p):
    n = len(c)
    out = np.empty(n, dtype=np.float64); out[:] = np.nan
    for i in range(p, n): out[i] = c[i] - c[i - p]
    return out

@njit(cache=True)
def _avgvol_arr(v, p):
    n = len(v)
    out = np.empty(n, dtype=np.float64); out[:] = np.nan
    s = 0.0
    for i in range(p): s += v[i]
    out[p - 1] = s / p
    for i in range(p, n):
        s += v[i] - v[i - p]
        out[i] = s / p
    return out

# Warm-up numba (primera llamada compila)
print("Compilando JIT...", flush=True)
_ema_arr(C[:100], 5)
_atr_arr(H[:100], L[:100], C[:100], 14)
_rsi_arr(C[:100], 14)
_rci_arr(C[:100], 9)
_bb_arr(C[:100], 20, 2.0)
_vwap_arr(H[:100], L[:100], C[:100], V[:100], 50)
_mom_arr(C[:100], 5)
_avgvol_arr(V[:100], 10)
print("JIT listo.", flush=True)

# Precalcular todos los arrays fijos (params que NO varían en el grid)
EMA5  = _ema_arr(C, 5)
EMA20 = _ema_arr(C, 20)
ATR14 = _atr_arr(H, L, C, 14)
RSI14 = _rsi_arr(C, 14)
RCI9  = _rci_arr(C, 9)
BB_U, BB_M, BB_L = _bb_arr(C, 20, 2.0)
VWAP_WINDOW = 1440  # 24h en velas de 1m
VWAP  = _vwap_arr(H, L, C, V, VWAP_WINDOW)
MOM5  = _mom_arr(C, 5)
AVGV10 = _avgvol_arr(V, 10)
print("Indicadores precalculados.", flush=True)

# ── Shared memory: publicar arrays una sola vez, workers hacen attach ──────────
# En Windows multiprocessing.Pool no comparte memoria via fork (COW no existe);
# sin esto cada worker recibiria una copia pickled de los ~14 arrays completos.
_SHM_ARRAYS = {
    'O': O, 'H': H, 'L': L, 'C': C, 'V': V,
    'EMA5': EMA5, 'EMA20': EMA20, 'ATR14': ATR14, 'RSI14': RSI14, 'RCI9': RCI9,
    'BB_U': BB_U, 'BB_L': BB_L, 'VWAP': VWAP, 'MOM5': MOM5, 'AVGV10': AVGV10,
}
_shm_blocks = {}      # name -> SharedMemory (mantener vivos en el proceso main)
_shm_meta   = {}       # name -> (shm_name, shape, dtype_str)

def _publish_shared_arrays():
    for name, arr in _SHM_ARRAYS.items():
        arr = np.ascontiguousarray(arr, dtype=np.float64)
        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        dst = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        dst[:] = arr[:]
        _shm_blocks[name] = shm
        _shm_meta[name] = (shm.name, arr.shape, str(arr.dtype))

def _cleanup_shared_arrays():
    for shm in _shm_blocks.values():
        shm.close()
        shm.unlink()

# ── Replay JIT ────────────────────────────────────────────────────────────────
# Recibe arrays precalculados; solo itera el loop de trades.

@njit(cache=True)
def _replay(
    O, H, L, C, V,
    EMA5, EMA20, ATR14, RSI14, RCI9,
    BB_U, BB_L,
    VWAP, MOM5, AVGV10,
    # params del grid
    rsiOS, rsiOB, volThr, atrSLM, rr, maxH, thr,
    # fijos
    minBars, cooldown, beAtR, trailAtR, trailAtrM, spread, slip
):
    n = len(C)
    # weights: ema=1 vwap=1 bb=1 rsi=1 rci=2 mom=1 vol=1  → maxScore=8
    W_EMA=1; W_VWAP=1; W_BB=1; W_RSI=1; W_RCI=2; W_MOM=1; W_VOL=1

    wins=0; losses=0; gw=0.0; gl=0.0
    eq=0.0; pk=0.0; mxdd=0.0
    net=0.0; n_trades=0

    i = minBars; na = minBars
    while i < n - 1:
        if i < na: i += 1; continue

        # ── score ──
        ls = ss = 0
        # EMA
        if not np.isnan(EMA5[i]) and not np.isnan(EMA20[i]):
            if EMA5[i] > EMA20[i]: ls += W_EMA
            else:                   ss += W_EMA
        # VWAP
        if not np.isnan(VWAP[i]):
            if C[i] > VWAP[i]: ls += W_VWAP
            else:               ss += W_VWAP
        # BB
        if not np.isnan(BB_U[i]) and i > 0:
            if C[i-1] <= BB_L[i] and C[i] > BB_L[i]: ls += W_BB
            if C[i-1] >= BB_U[i] and C[i] < BB_U[i]: ss += W_BB
        # RSI
        if not np.isnan(RSI14[i]):
            if RSI14[i] < rsiOS: ls += W_RSI
            if RSI14[i] > rsiOB: ss += W_RSI
        # RCI cross
        if not np.isnan(RCI9[i]) and i > 0 and not np.isnan(RCI9[i-1]):
            if RCI9[i-1] <= -80.0 and RCI9[i] > -80.0: ls += W_RCI
            if RCI9[i-1] >=  80.0 and RCI9[i] <  80.0: ss += W_RCI
        # Momentum
        if not np.isnan(MOM5[i]):
            if MOM5[i] > 0: ls += W_MOM
            else:            ss += W_MOM
        # Volume
        if not np.isnan(AVGV10[i]) and AVGV10[i] > 0:
            if V[i] > AVGV10[i] * volThr:
                ls += W_VOL; ss += W_VOL

        gl_ = ls >= thr; gs_ = ss >= thr
        if not gl_ and not gs_: i += 1; continue

        if gl_ and gs_: d = 1 if ls >= ss else -1
        elif gl_:       d = 1
        else:           d = -1
        ib = d == 1

        # ── entry ──
        if i + 1 >= n: break
        adj   = spread / 2.0 + slip
        entry = O[i+1] + adj if ib else O[i+1] - adj

        if np.isnan(ATR14[i]): i += 1; continue
        sl_dist = ATR14[i] * atrSLM
        if sl_dist <= 0: i += 1; continue
        cur_sl  = entry - sl_dist if ib else entry + sl_dist
        risk    = sl_dist
        tp_p    = entry + risk * rr if ib else entry - risk * rr

        be_act = False; trail_act = False
        ep = np.nan; er = 0; ei = -1  # er: 1=TP 2=SL 3=BE 4=ES 5=MH

        for j in range(i + 1, min(n, i + 1 + maxH)):
            bars = j - (i + 1)
            if bars >= maxH:
                ep = O[j]; er = 5; ei = j; break

            ur = (C[j] - entry) / risk if ib else (entry - C[j]) / risk

            # BE
            if not be_act and ur >= beAtR:
                ns = entry
                if (ib and ns > cur_sl) or (not ib and ns < cur_sl):
                    cur_sl = ns; be_act = True

            # Trailing
            if not trail_act and ur >= trailAtR: trail_act = True
            if trail_act and not np.isnan(ATR14[j]):
                ts = C[j] - ATR14[j] * trailAtrM if ib else C[j] + ATR14[j] * trailAtrM
                if (ib and ts > cur_sl) or (not ib and ts < cur_sl):
                    cur_sl = ts

            # SL
            if ib and L[j] <= cur_sl:
                ep = cur_sl - slip; er = 3 if be_act else 2; ei = j; break
            if not ib and H[j] >= cur_sl:
                ep = cur_sl + slip; er = 3 if be_act else 2; ei = j; break

            # TP
            if ib and H[j] >= tp_p:
                ep = tp_p - slip; er = 1; ei = j; break
            if not ib and L[j] <= tp_p:
                ep = tp_p + slip; er = 1; ei = j; break

            # Exit signal (simplified): RCI + MOM + VWAP 2/3
            if j < n - 1:
                hits = 0
                if not np.isnan(RCI9[j]):
                    if ib and RCI9[j] < -60: hits += 1
                    if not ib and RCI9[j] > 60: hits += 1
                if not np.isnan(MOM5[j]):
                    if ib and MOM5[j] < 0: hits += 1
                    if not ib and MOM5[j] > 0: hits += 1
                if not np.isnan(VWAP[j]):
                    if ib and C[j] < VWAP[j]: hits += 1
                    if not ib and C[j] > VWAP[j]: hits += 1
                if hits >= 2:
                    ep = C[j]; er = 4; ei = j; break

        if np.isnan(ep):
            ep = C[n-1]; er = 5; ei = n - 1

        pnl_b = ep - entry if ib else entry - ep
        pnl_n = pnl_b - spread
        rm    = pnl_b / risk

        n_trades += 1
        net += pnl_n
        if pnl_n > 0: wins += 1; gw += pnl_n
        else:         losses += 1; gl -= pnl_n  # gl positive
        eq += rm
        if eq > pk: pk = eq
        dd = pk - eq
        if dd > mxdd: mxdd = dd

        i = ei + 1 + cooldown
        na = i

    if n_trades < 5:
        return 0.0, 0.0, 0.0, 0.0, 0, 0.0

    wr = wins / n_trades
    pf = gw / gl if gl > 0 else (1e9 if gw > 0 else 0.0)
    aw = gw / wins   if wins   > 0 else 0.0
    al = gl / losses if losses > 0 else 0.0
    exp = wr * aw - (1.0 - wr) * al
    return pf, wr, exp, mxdd, n_trades, net

# ── Grid ──────────────────────────────────────────────────────────────────────
grid = {
    'rsiOversold':   [30, 35, 40],
    'rsiOverbought': [60, 65, 70],
    'volThreshold':  [1.0, 1.2, 1.5, 2.0],
    'atrSLMult':     [1.0, 1.5, 2.0],
    'rr':            [1.5, 2.0, 2.5],
    'maxHoldBars':   [10, 15, 20],
    'scoreThreshold':[4, 5, 6],
}

FIXED = dict(minBars=20, cooldown=3, beAtR=1.0, trailAtR=1.5,
             trailAtrMult=1.0, spread=0.5, slip=0.1)

keys   = list(grid.keys())
vals   = list(grid.values())
combos = [c for c in itertools.product(*vals)
          if c[0] < c[1]]  # rsiOS < rsiOB

# ── Worker ────────────────────────────────────────────────────────────────────
# Estado por-proceso del worker: vistas np.ndarray sobre shared_memory + handles
# abiertos (deben permanecer vivos mientras el worker viva).
_W = {}
_W_SHM_HANDLES = []

def _worker_init(shm_meta):
    for name, (shm_name, shape, dtype_str) in shm_meta.items():
        shm = shared_memory.SharedMemory(name=shm_name)
        _W_SHM_HANDLES.append(shm)
        _W[name] = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)

def _score_combo(combo):
    rsiOS, rsiOB, volThr, atrSLM, rr, maxH, thr = combo
    pf, wr, exp, dd, n, net = _replay(
        _W['O'], _W['H'], _W['L'], _W['C'], _W['V'],
        _W['EMA5'], _W['EMA20'], _W['ATR14'], _W['RSI14'], _W['RCI9'],
        _W['BB_U'], _W['BB_L'], _W['VWAP'], _W['MOM5'], _W['AVGV10'],
        float(rsiOS), float(rsiOB), float(volThr),
        float(atrSLM), float(rr), int(maxH), int(thr),
        FIXED['minBars'], FIXED['cooldown'],
        FIXED['beAtR'], FIXED['trailAtR'], FIXED['trailAtrMult'],
        FIXED['spread'], FIXED['slip'],
    )
    if n < 5: return None
    rank = pf*0.4 + wr*0.3 + min(exp*10, 5)*0.3
    return {
        'cfg':   dict(zip(keys, combo)),
        'stats': {'pf': round(pf,3), 'wr': round(wr,3), 'exp': round(exp,4),
                  'dd': round(dd,2), 'n': n, 'net': round(net,4)},
        '_rank': rank,
    }

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    _publish_shared_arrays()
    try:
        total = len(combos)
        print(f"Grid: {total} combinaciones", flush=True)
        OUT_JSONL = os.path.join(_DIR, 'grid_results.jsonl')
        OUT_FINAL = os.path.join(_DIR, 'grid_results.json')

        t0 = time.time()
        WORKERS = max(1, (os.cpu_count() or 8) - 1)  # deja 1 thread libre para main/OS
        chunk = max(1, total // (WORKERS * 4))
        done = 0

        # Append-only: I/O constante por resultado, sin reescritura acumulativa.
        with open(OUT_JSONL, 'w') as fout, multiprocessing.Pool(
            processes=WORKERS,
            initializer=_worker_init,
            initargs=(_shm_meta,),
        ) as pool:
            for result in pool.imap_unordered(_score_combo, combos, chunksize=chunk):
                done += 1
                if result:
                    fout.write(json.dumps(result) + '\n')
                if done % 50 == 0 or done == total:
                    fout.flush()
                    elapsed = time.time() - t0
                    rate    = done / elapsed
                    eta     = (total - done) / rate if rate > 0 else 0.0
                    print(f"  {done}/{total} | {elapsed:.1f}s | ETA {eta:.0f}s | {rate:.1f} combo/s", flush=True)

        # Snapshot final: reordenar leyendo el JSONL (streaming), no desde memoria.
        raw = []
        with open(OUT_JSONL) as f:
            for line in f:
                raw.append(json.loads(line))

        results = sorted(raw, key=lambda x: x['_rank'], reverse=True)
        with open(OUT_FINAL, 'w') as f:
            json.dump([{k: v for k, v in r.items() if k != '_rank'} for r in results], f, indent=2)

        elapsed = time.time() - t0
        print(f"\nCompletado en {elapsed:.1f}s con {WORKERS} workers", flush=True)
    finally:
        _cleanup_shared_arrays()

    print(f"\nTop 10 configs:")
    for i, r in enumerate(results[:10]):
        s=r['stats']; c=r['cfg']
        print(f"\n#{i+1} PF={s['pf']} WR={s['wr']*100:.1f}% Exp={s['exp']:.4f} DD={s['dd']:.2f}R N={s['n']} Net={s['net']:.4f}")
        print(f"   rsiOS={c['rsiOversold']} rsiOB={c['rsiOverbought']} vol={c['volThreshold']} atrSL={c['atrSLMult']} rr={c['rr']} maxH={c['maxHoldBars']} thr={c['scoreThreshold']}")

    print(f"\nBEST:")
    print(json.dumps({k:v for k,v in results[0].items() if k!='_rank'}, indent=2))