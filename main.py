import yfinance as yf
import pandas as pd
import numpy as np
import itertools
import requests
import os
import warnings

warnings.filterwarnings("ignore")

# --- TELEGRAM AYARLARI (GitHub Secrets'tan okur) ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def send_telegram(msg):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("HATA: Telegram bilgileri eksik!")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        response = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}, timeout=10)
        if response.ok:
            print("✅ Telegram mesajı başarıyla gönderildi.")
        else:
            print(f"❌ Telegram gönderim hatası: {response.text}")
    except Exception as e:
        print(f"❌ Telegram bağlantı hatası: {e}")

# --- SADECE 15 DAKİKALIK SİNYAL İÇİN AYARLAR ---
symbols = ["AVAX-TRY", "ISCTR.IS"]
n1 = 150
n2 = 200
wave_prominence = 15.0

# --- HESAPLAMA FONKSİYONLARI ---
def pine_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def pine_sma(series, length):
    return series.rolling(window=length).mean()

def calc_wt(df, len1, len2):
    hlc3 = (df['High'] + df['Low'] + df['Close']) / 3.0
    esa = pine_ema(hlc3, len1)
    d = pine_ema((hlc3 - esa).abs(), len1)
    ci = (hlc3 - esa) / (0.015 * d.replace(0, np.nan))
    tci = pine_ema(ci, len2)
    wt2 = pine_sma(tci, 4)
    return tci, wt2

def detect_alternating_waves(scores, timestamps, prominence):
    n = len(scores)
    if n < 5:
        return []

    pivots = []
    direction = 0
    ext_val = scores[0]
    ext_idx = 0

    for i in range(1, n):
        val = scores[i]
        if direction == 0:
            if val - ext_val >= prominence:
                pivots.append({'type': 'AL', 'idx': ext_idx, 'ts': timestamps[ext_idx], 'val': ext_val})
                direction = 1
                ext_val = val
                ext_idx = i
            elif ext_val - val >= prominence:
                pivots.append({'type': 'SAT', 'idx': ext_idx, 'ts': timestamps[ext_idx], 'val': ext_val})
                direction = -1
                ext_val = val
                ext_idx = i
            else:
                if val < ext_val:
                    ext_val = val
                    ext_idx = i
        elif direction == 1:
            if val > ext_val:
                ext_val = val
                ext_idx = i
            elif ext_val - val >= prominence:
                pivots.append({'type': 'SAT', 'idx': ext_idx, 'ts': timestamps[ext_idx], 'val': ext_val})
                direction = -1
                ext_val = val
                ext_idx = i
        elif direction == -1:
            if val < ext_val:
                ext_val = val
                ext_idx = i
            elif val - ext_val >= prominence:
                pivots.append({'type': 'AL', 'idx': ext_idx, 'ts': timestamps[ext_idx], 'val': ext_val})
                direction = 1
                ext_val = val
                ext_idx = i

    if direction == 1:
        pivots.append({'type': 'SAT', 'idx': ext_idx, 'ts': timestamps[ext_idx], 'val': ext_val})
    elif direction == -1:
        pivots.append({'type': 'AL', 'idx': ext_idx, 'ts': timestamps[ext_idx], 'val': ext_val})
    else:
        min_i = int(np.argmin(scores))
        max_i = int(np.argmax(scores))
        if min_i < max_i:
            pivots.append({'type': 'AL', 'idx': min_i, 'ts': timestamps[min_i], 'val': scores[min_i]})
            pivots.append({'type': 'SAT', 'idx': max_i, 'ts': timestamps[max_i], 'val': scores[max_i]})
        else:
            pivots.append({'type': 'SAT', 'idx': max_i, 'ts': timestamps[max_i], 'val': scores[max_i]})
            pivots.append({'type': 'AL', 'idx': min_i, 'ts': timestamps[min_i], 'val': scores[min_i]})

    return pivots

def fetch_ohlc(ticker, interval, period):
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open', 'High', 'Low', 'Close']].dropna()
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('Europe/Istanbul')
        return df
    except Exception:
        return pd.DataFrame()

def get_symbol_data(sym, interval, period, usdtry_cache):
    if sym in ["AVAX-TRY", "SOL-TRY"]:
        base_sym = sym.replace("-TRY", "-USD")
        df_base = fetch_ohlc(base_sym, interval, period)
        df_fx = usdtry_cache.get(interval)
        if df_base.empty or df_fx is None or df_fx.empty:
            return pd.DataFrame()
        fx_aligned = df_fx['Close'].reindex(df_base.index, method='ffill').bfill()
        df_syn = pd.DataFrame(index=df_base.index)
        for col in ['Open', 'High', 'Low', 'Close']:
            df_syn[col] = df_base[col] * fx_aligned
        return df_syn.dropna()
    else:
        return fetch_ohlc(sym, interval, period)

def check_and_send_signal_15m(pivots, sym):
    """Sadece son 90 dakika içinde oluşan 15 dakikalık sinyalleri Telegram'a yollar."""
    if not pivots:
        return
    last = pivots[-1]
    
    # Son pivot son 90 dakika içinde mi oluştu?
    now = pd.Timestamp.now(tz='Europe/Istanbul')
    diff_min = (now - last['ts']).total_seconds() / 60
    
    if diff_min <= 90:
        emoji = "🟢" if last['type'] == "AL" else "🔴"
        msg = (f"{emoji} *YENİ 15 DK {last['type']} SİNYALİ!*\n\n"
               f"📈 *Sembol:* {sym}\n"
               f"⏰ *Zaman Dilimi:* 15 Dakika\n"
               f"📅 *Tarih:* {last['ts'].strftime('%Y-%m-%d %H:%M')}\n"
               f"📊 *Skor:* {last['val']:.2f}")
        print(f"[{sym}] Telegram'a 15 Dk {last['type']} sinyali gönderiliyor...")
        send_telegram(msg)
    else:
         print(f"[{sym}] 15 Dk için güncel (taze) bir sinyal yok (Son sinyal: {last['ts'].strftime('%H:%M')}).")

# --- ANA ÇALIŞMA ALANI ---
print("USDTRY=X kur verileri ön belleğe alınıyor...")
usdtry_cache = {
    "15m": fetch_ohlc("USDTRY=X", "15m", "40d"),
    "1d":  fetch_ohlc("USDTRY=X", "1d", "2y"),
    "60m": fetch_ohlc("USDTRY=X", "60m", "40d"),
    "5m":  fetch_ohlc("USDTRY=X", "5m", "40d"),
}

print(f"Toplam {len(symbols)} sembol taranıyor, 15 dakikalık sinyaller aranıyor...\n")

for sym in symbols:
    print(f"\n{'='*40} {sym} İÇİN SİNYAL ARANIYOR {'='*40}")
    
    df_15 = get_symbol_data(sym, "15m", "40d", usdtry_cache)
    df_d  = get_symbol_data(sym, "1d", "2y", usdtry_cache)
    df_60 = get_symbol_data(sym, "60m", "40d", usdtry_cache)
    df_5  = get_symbol_data(sym, "5m", "40d", usdtry_cache)

    if df_15.empty or df_d.empty or df_60.empty or df_5.empty:
        print(f"{sym} için yeterli veri alınamadı, atlanıyor.")
        continue

    wt1_d, _   = calc_wt(df_d, n1, n2)
    wt1_h, _   = calc_wt(df_60, n1, n2)
    wt1_15, _  = calc_wt(df_15, n1, n2)
    wt1_5, _   = calc_wt(df_5, n1, n2)

    wt1_d_aligned = wt1_d.reindex(df_15.index, method='ffill')
    wt1_h_aligned = wt1_h.reindex(df_15.index, method='ffill')
    wt1_5_aligned = wt1_5.reindex(df_15.index, method='ffill')

    temp_df = pd.DataFrame({
        'Close': df_15['Close'],
        'wt_d': wt1_d_aligned,
        'wt_h': wt1_h_aligned,
        'wt_15': wt1_15,
        'wt_5': wt1_5_aligned
    }, index=df_15.index).dropna()

    if len(temp_df) > 0:
        last_ts = temp_df.index[-1]
        temp_df = temp_df[temp_df.index >= (last_ts - pd.Timedelta(days=20))]

    weight_steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    best_correlation = -999
    optimal_weights = (0.4, 0.3, 0.2, 0.1)

    for wd, wh, w15, w5 in itertools.product(weight_steps, repeat=4):
        if abs((wd + wh + w15 + w5) - 1.0) < 1e-5:
            score = (temp_df['wt_d'] * wd) + (temp_df['wt_h'] * wh) + (temp_df['wt_15'] * w15) + (temp_df['wt_5'] * w5)
            corr = score.corr(temp_df['Close'])
            if not np.isnan(corr) and corr > best_correlation:
                best_correlation = corr
                optimal_weights = (wd, wh, w15, w5)

    wd, wh, w15, w5 = optimal_weights

    # --- SADECE 15 DAKİKALIK KISIM HESAPLANIYOR ---
    total_score_15 = (wt1_d_aligned * wd) + (wt1_h_aligned * wh) + (wt1_15 * w15) + (wt1_5_aligned * w5)
    
    res_df_15 = pd.DataFrame({
        'total_score': total_score_15
    }, index=df_15.index).dropna(subset=['total_score'])
    
    res_df_15 = res_df_15[(res_df_15['total_score'] >= -200) & (res_df_15['total_score'] <= 200)]
    
    if len(res_df_15) > 0:
        last_timestamp = res_df_15.index[-1]
        res_df_15 = res_df_15[res_df_15.index >= (last_timestamp - pd.Timedelta(days=20))]

    if len(res_df_15) >= 15:
        pivots_15 = detect_alternating_waves(res_df_15['total_score'].values, res_df_15.index, wave_prominence)
        check_and_send_signal_15m(pivots_15, sym)
    else:
        print(f"[{sym}] Hesaplama için yeterli 15 dakikalık veri yok.")

print("\nTüm taramalar tamamlandı. Sadece 15 dakikalık sinyaller kontrol edildi.")
