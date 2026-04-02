import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
from datetime import datetime, timedelta
import json
import numpy as np
import streamlit as st
# ... 其他 import

st.set_page_config(
    page_title="AI 股票趨勢分析系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    /* 強制顯示側邊欄按鈕，並將其放大、變色，方便 iPad 點擊 */
    [data-testid="collapsedControl"] {
        color: white;
        background-color: #ff4b4b; /* 變成紅色，讓你一定看得到 */
        border-radius: 50%;
        left: 10px;
        top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 主標題
st.title("📈 AI 股票趨勢分析系統")
st.divider()


# ─────────────────────────────────────────────
# 美股數據函數
# ─────────────────────────────────────────────

def get_us_stock_data(symbol, api_key, start_date, end_date):
    """
    從 Financial Modeling Prep API 獲取美股歷史數據

    Args:
        symbol: 股票代碼
        api_key: FMP API 金鑰
        start_date: 起始日期
        end_date: 結束日期

    Returns:
        DataFrame: 包含股票歷史數據的 DataFrame，或 None
    """
    try:
        url = "https://financialmodelingprep.com/stable/historical-price-eod/full"
        params = {
            'symbol': symbol,
            'apikey': api_key,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d')
        }

        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()

        # 新版 API 直接回傳陣列
        if not isinstance(data, list) or len(data) == 0:
            st.error(f"無法獲取股票 {symbol} 的數據，請檢查股票代碼是否正確。")
            return None

        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df

    except requests.exceptions.RequestException as e:
        st.error(f"FMP API 請求失敗：{str(e)}")
        return None
    except Exception as e:
        st.error(f"美股數據處理錯誤：{str(e)}")
        return None


# ─────────────────────────────────────────────
# 台股數據函數
# ─────────────────────────────────────────────

def get_tw_stock_price(symbol, api_key, start_date, end_date):
    """
    從 FinMind API 獲取台股歷史日 K 價格

    FinMind 欄位對應：
        max → high
        min → low
        Trading_Volume → volume

    Args:
        symbol: 台股代碼（純數字，如 2330）
        api_key: FinMind API 金鑰
        start_date: 起始日期
        end_date: 結束日期

    Returns:
        DataFrame: 統一欄位（date, open, high, low, close, volume），或 None
    """
    try:
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            'dataset': 'TaiwanStockPrice',
            'data_id': symbol,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'token': api_key
        }

        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        result = response.json()

        # 驗證 status
        if result.get('status') != 200:
            msg = result.get('msg', '未知錯誤')
            st.error(f"FinMind API 錯誤：{msg}")
            return None

        data = result.get('data', [])
        if not data:
            st.error(f"無法獲取台股 {symbol} 的數據，請確認代碼是否正確。")
            return None

        df = pd.DataFrame(data)

        # 驗證必要欄位
        required_cols = ['date', 'open', 'max', 'min', 'close', 'Trading_Volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"FinMind 回傳欄位缺失：{missing}")
            return None

        # 欄位對應
        df = df.rename(columns={
            'max': 'high',
            'min': 'low',
            'Trading_Volume': 'volume'
        })

        # 數值型別轉換
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # 保留必要欄位
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        return df

    except requests.exceptions.RequestException as e:
        st.error(f"FinMind API 請求失敗：{str(e)}")
        return None
    except Exception as e:
        st.error(f"台股數據處理錯誤：{str(e)}")
        return None


def get_tw_margin_trading(symbol, api_key, start_date, end_date):
    """
    從 FinMind API 獲取台股融資融券餘額
    靜默失敗，不影響主流程

    Returns:
        DataFrame or None
    """
    try:
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            'dataset': 'TaiwanStockMarginPurchaseShortSale',
            'data_id': symbol,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'token': api_key
        }

        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        result = response.json()

        if result.get('status') != 200 or not result.get('data'):
            return None

        df = pd.DataFrame(result['data'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # ── 自動偵測融資餘額欄位（FinMind 不同版本欄位名稱可能不同）──
        # 融資餘額候選欄位（優先順序）
        margin_candidates = [
            'MarginPurchaseRemaining',       # 規格書標準名
            'MarginPurchase',
            'margin_purchase_remaining',
            'FundingRemaining',
        ]
        # 融券餘額候選欄位
        short_candidates = [
            'ShortSaleRemaining',            # 規格書標準名
            'ShortSale',
            'short_sale_remaining',
            'ShortRemaining',
        ]

        def find_col(df_cols, candidates):
            """從候選清單中找出第一個存在的欄位名稱，若都沒有則嘗試模糊比對"""
            for c in candidates:
                if c in df_cols:
                    return c
            # 模糊比對：含關鍵字
            lower_cols = {c.lower(): c for c in df_cols}
            for keyword in ['marginpurchaseremaining', 'marginpurchase', 'shortsaleremaining', 'shortsale']:
                for lc, orig in lower_cols.items():
                    if keyword in lc:
                        return orig
            return None

        margin_col = find_col(df.columns.tolist(), margin_candidates)
        short_col  = find_col(df.columns.tolist(), short_candidates)

        # 若找不到任何融資欄位，靜默返回 None
        if margin_col is None and short_col is None:
            return None

        # 統一重命名為標準欄位名，方便後續使用
        rename_map = {}
        if margin_col and margin_col != 'MarginPurchaseRemaining':
            rename_map[margin_col] = 'MarginPurchaseRemaining'
        if short_col and short_col != 'ShortSaleRemaining':
            rename_map[short_col] = 'ShortSaleRemaining'
        if rename_map:
            df = df.rename(columns=rename_map)

        # 數值轉換
        for col in ['MarginPurchaseRemaining', 'ShortSaleRemaining']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    except Exception:
        return None


def get_tw_institutional(symbol, api_key, start_date, end_date):
    """
    從 FinMind API 獲取台股三大法人買賣超
    靜默失敗，不影響主流程

    Returns:
        DataFrame（含 date, name, buy, sell, net）or None
    """
    try:
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            'dataset': 'TaiwanStockInstitutionalInvestorsBuySell',
            'data_id': symbol,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'token': api_key
        }

        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        result = response.json()

        if result.get('status') != 200 or not result.get('data'):
            return None

        df = pd.DataFrame(result['data'])
        df['date'] = pd.to_datetime(df['date'])

        # 衍生欄位：買賣超張數
        for col in ['buy', 'sell']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        df['net'] = df['buy'] - df['sell']
        df = df.sort_values('date').reset_index(drop=True)
        return df

    except Exception:
        return None


def get_tw_financial_statements(symbol, api_key):
    """
    從 FinMind API 獲取台股財務報表（季報 + 年報）
    靜默失敗，各自獨立 try-except

    Returns:
        dict: {'quarterly': DataFrame or None, 'annual': DataFrame or None}
    """
    result = {'quarterly': None, 'annual': None}
    today = datetime.now()

    # ── 季報 ──
    try:
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            'dataset': 'TaiwanStockFinancialStatements',
            'data_id': symbol,
            'start_date': (today - timedelta(days=730)).strftime('%Y-%m-%d'),
            'token': api_key
        }
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        res = resp.json()

        if res.get('status') == 200 and res.get('data'):
            df = pd.DataFrame(res['data'])
            target_types = ['EPS', 'Revenue', 'OperatingIncome', 'NetIncome', 'GrossProfit']
            df = df[df['type'].isin(target_types)].tail(40)
            result['quarterly'] = df
    except Exception:
        pass

    # ── 年報（資產負債表）──
    try:
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            'dataset': 'TaiwanStockBalanceSheet',
            'data_id': symbol,
            'start_date': (today - timedelta(days=365 * 4)).strftime('%Y-%m-%d'),
            'token': api_key
        }
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        res = resp.json()

        if res.get('status') == 200 and res.get('data'):
            df = pd.DataFrame(res['data'])
            target_types = ['TotalAssets', 'TotalLiabilities', 'StockholdersEquity',
                            'CashAndEquivalents', 'DebtRatio']
            df = df[df['type'].isin(target_types)].tail(20)
            result['annual'] = df
    except Exception:
        pass

    return result


# ─────────────────────────────────────────────
# 通用數據處理函數
# ─────────────────────────────────────────────

def filter_by_date_range(df, start_date, end_date):
    """根據日期範圍過濾 DataFrame"""
    if df is None:
        return None
    mask = (df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))
    return df.loc[mask].copy().reset_index(drop=True)


def get_moving_averages(df):
    """計算 MA5, MA10, MA20, MA60"""
    if df is None or len(df) == 0:
        return None
    df = df.copy()
    for period in [5, 10, 20, 60]:
        df[f'MA{period}'] = df['close'].rolling(window=period, min_periods=1).mean()
    return df


def calculate_rsi(df, period=14):
    """
    計算 RSI（Wilder's Smoothing / EWM）

    RSI = 100 - (100 / (1 + RS))
    RS  = EWM(漲幅, com=period-1) / EWM(跌幅, com=period-1)
    初始 NaN 填補為 50（中性值）
    """
    try:
        if df is None or len(df) < period:
            df = df.copy() if df is not None else pd.DataFrame()
            df[f'RSI{period}'] = 50
            return df

        df = df.copy()
        delta = df['close'].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

        # 避免除以零
        avg_loss = avg_loss.replace(0, np.nan)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # 填補 NaN 為 50
        rsi = rsi.fillna(50)
        df[f'RSI{period}'] = rsi
        return df

    except Exception:
        if df is not None:
            df = df.copy()
            df[f'RSI{period}'] = 50
        return df


# ─────────────────────────────────────────────
# 進階技術指標計算函數
# ─────────────────────────────────────────────

def calculate_advanced_indicators(df):
    """
    計算進階技術指標：MACD(12,26,9)、布林通道(20,2SD)、OBV、DMI(14)

    Returns:
        DataFrame（新增欄位）or 原始 df（失敗時）
    """
    if df is None or len(df) < 30:
        return df

    df = df.copy()

    # ── MACD (12, 26, 9) ──
    try:
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD_DIF']  = ema12 - ema26
        df['MACD_LINE'] = df['MACD_DIF'].ewm(span=9, adjust=False).mean()
        df['MACD_HIST'] = (df['MACD_DIF'] - df['MACD_LINE']) * 2
        # 金叉偵測
        dif_prev  = df['MACD_DIF'].shift(1)
        line_prev = df['MACD_LINE'].shift(1)
        df['MACD_GOLDEN'] = (dif_prev < line_prev) & (df['MACD_DIF'] > df['MACD_LINE'])
    except Exception:
        pass

    # ── 布林通道 BB (20, 2SD) ──
    try:
        df['BB_MID']   = df['close'].rolling(20).mean()
        bb_std         = df['close'].rolling(20).std()
        df['BB_UPPER'] = df['BB_MID'] + 2 * bb_std
        df['BB_LOWER'] = df['BB_MID'] - 2 * bb_std
        df['BB_WIDTH'] = df['BB_UPPER'] - df['BB_LOWER']
    except Exception:
        pass

    # ── OBV（On-Balance Volume）──
    try:
        direction = np.sign(df['close'].diff())
        direction.iloc[0] = 0
        df['OBV'] = (direction * df['volume']).cumsum()
    except Exception:
        pass

    # ── DMI (14日) ──
    try:
        high  = df['high']
        low   = df['low']
        close = df['close']
        prev_high  = high.shift(1)
        prev_low   = low.shift(1)
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low  - prev_close).abs()
        TR  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        up_move   = high - prev_high
        down_move = prev_low - low
        plus_dm_arr  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm_arr = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        plus_dm_s  = pd.Series(plus_dm_arr,  index=df.index)
        minus_dm_s = pd.Series(minus_dm_arr, index=df.index)

        ATR14    = TR.ewm(com=13, adjust=False).mean()
        plus_di  = 100 * plus_dm_s.ewm(com=13, adjust=False).mean() / ATR14
        minus_di = 100 * minus_dm_s.ewm(com=13, adjust=False).mean() / ATR14
        DX       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        ADX      = DX.ewm(com=13, adjust=False).mean()

        df['DMI_PLUS']   = plus_di
        df['DMI_MINUS']  = minus_di
        df['DMI_ADX']    = ADX

        plus_prev  = df['DMI_PLUS'].shift(1)
        minus_prev = df['DMI_MINUS'].shift(1)
        df['DMI_GOLDEN'] = (plus_prev < minus_prev) & (df['DMI_PLUS'] > df['DMI_MINUS'])
    except Exception:
        pass

    return df


def calculate_bull_signals(df):
    """
    計算 8 項多頭訊號燈號評分（滿分 100）

    Returns:
        dict: { 'signals': list, 'total_score': float, 'conclusion': str, 'conclusion_level': str }
    """
    signals = []

    def add(name, status, desc):
        score = 12.5 if status == 'green' else (6.0 if status == 'yellow' else 0.0)
        signals.append({'name': name, 'status': status, 'desc': desc, 'score': score})

    # 1. MACD 轉正
    try:
        golden_recent = df['MACD_GOLDEN'].tail(5).any() if 'MACD_GOLDEN' in df.columns else False
        macd_pos = df['MACD_LINE'].iloc[-1] > 0 if 'MACD_LINE' in df.columns else False
        dif_pos  = df['MACD_DIF'].iloc[-1] > 0  if 'MACD_DIF'  in df.columns else False
        if golden_recent or (macd_pos and dif_pos):
            add('MACD 轉正', 'green', '最近5日金叉 或 MACD/DIF均>0')
        elif macd_pos:
            add('MACD 轉正', 'yellow', 'MACD線剛轉正，無金叉確認')
        else:
            add('MACD 轉正', 'red', 'MACD線<0，偏空')
    except Exception:
        add('MACD 轉正', 'red', '無法計算')

    # 2. BB 突破中軌
    try:
        if 'BB_MID' in df.columns:
            c   = df['close'].iloc[-1]
            mid = df['BB_MID'].iloc[-1]
            w1  = df['BB_WIDTH'].iloc[-1]
            w0  = df['BB_WIDTH'].iloc[-2] if len(df) > 1 else w1
            if c > mid and w1 > w0:
                add('BB 突破中軌', 'green', f'收盤>{mid:.2f}且通道擴張')
            elif c > mid:
                add('BB 突破中軌', 'yellow', f'收盤>{mid:.2f}，通道未擴張')
            else:
                add('BB 突破中軌', 'red', f'收盤<BB中軌({mid:.2f})')
        else:
            add('BB 突破中軌', 'red', '無法計算')
    except Exception:
        add('BB 突破中軌', 'red', '無法計算')

    # 3. BB 壓縮後突破
    try:
        if 'BB_WIDTH' in df.columns:
            w_mean = df['BB_WIDTH'].mean()
            compressed = df['BB_WIDTH'].tail(20).min() < w_mean * 0.5
            broken_up  = df['close'].iloc[-1] > df['BB_MID'].iloc[-1]
            if compressed and broken_up:
                add('BB 壓縮突破', 'green', '近期曾壓縮且已突破中軌')
            elif compressed:
                add('BB 壓縮突破', 'yellow', '通道壓縮中，尚未突破')
            else:
                add('BB 壓縮突破', 'red', '通道未壓縮')
        else:
            add('BB 壓縮突破', 'red', '無法計算')
    except Exception:
        add('BB 壓縮突破', 'red', '無法計算')

    # 4. OBV 資金流入
    try:
        if 'OBV' in df.columns:
            obv_last  = df['OBV'].iloc[-1]
            obv_prev_max = df['OBV'].iloc[:-1].max() if len(df) > 1 else obv_last
            new_high  = obv_last > obv_prev_max
            vol_surge = df['volume'].tail(5).mean() > df['volume'].mean() * 1.2
            if new_high and vol_surge:
                add('OBV 資金流入', 'green', 'OBV新高且近5日量放大>120%')
            elif new_high:
                add('OBV 資金流入', 'yellow', 'OBV新高，但量未明顯放大')
            else:
                add('OBV 資金流入', 'red', 'OBV未創新高')
        else:
            add('OBV 資金流入', 'red', '無法計算')
    except Exception:
        add('OBV 資金流入', 'red', '無法計算')

    # 5. RSI 動量
    try:
        rsi_col = next((c for c in df.columns if c.startswith('RSI')), None)
        if rsi_col:
            rv = df[rsi_col].iloc[-1]
            if 50 <= rv <= 70:
                add('RSI 動量', 'green', f'RSI={rv:.1f}，多頭動能區間')
            elif rv > 70:
                add('RSI 動量', 'yellow', f'RSI={rv:.1f}，超買警示')
            elif 40 <= rv < 50:
                add('RSI 動量', 'yellow', f'RSI={rv:.1f}，動能偏弱')
            else:
                add('RSI 動量', 'red', f'RSI={rv:.1f}，超賣區間')
        else:
            add('RSI 動量', 'red', '無法計算')
    except Exception:
        add('RSI 動量', 'red', '無法計算')

    # 6. DMI 多頭趨勢
    try:
        if 'DMI_PLUS' in df.columns:
            pv = df['DMI_PLUS'].iloc[-1]
            mv = df['DMI_MINUS'].iloc[-1]
            av = df['DMI_ADX'].iloc[-1]
            if pv > mv and av > 25:
                add('DMI 多頭趨勢', 'green', f'+DI({pv:.1f})>-DI({mv:.1f})，ADX={av:.1f}>25')
            elif pv > mv:
                add('DMI 多頭趨勢', 'yellow', f'+DI>{mv:.1f}，但ADX={av:.1f}≤25')
            else:
                add('DMI 多頭趨勢', 'red', f'-DI({mv:.1f})≥+DI({pv:.1f})，偏空')
        else:
            add('DMI 多頭趨勢', 'red', '無法計算')
    except Exception:
        add('DMI 多頭趨勢', 'red', '無法計算')

    # 7. DMI 黃金交叉
    try:
        if 'DMI_GOLDEN' in df.columns:
            if df['DMI_GOLDEN'].tail(5).any():
                add('DMI 黃金交叉', 'green', '近5日+DI上穿-DI，交叉確認')
            elif 'DMI_PLUS' in df.columns and df['DMI_PLUS'].iloc[-1] > df['DMI_MINUS'].iloc[-1]:
                add('DMI 黃金交叉', 'yellow', '+DI>-DI，但無近期交叉')
            else:
                add('DMI 黃金交叉', 'red', '無黃金交叉，-DI主導')
        else:
            add('DMI 黃金交叉', 'red', '無法計算')
    except Exception:
        add('DMI 黃金交叉', 'red', '無法計算')

    # 8. 均線多頭排列
    try:
        ma_cols = ['MA5', 'MA10', 'MA20', 'MA60']
        if all(c in df.columns for c in ma_cols):
            v5, v10, v20, v60 = [df[c].iloc[-1] for c in ma_cols]
            if v5 > v10 > v20 > v60:
                add('均線多頭排列', 'green', 'MA5>MA10>MA20>MA60 完全多頭')
            elif v5 > v20:
                add('均線多頭排列', 'yellow', 'MA5>MA20，部分多頭')
            else:
                add('均線多頭排列', 'red', '均線未呈多頭排列')
        else:
            add('均線多頭排列', 'red', '均線資料不足')
    except Exception:
        add('均線多頭排列', 'red', '無法計算')

    # 整體評分
    total_score = sum(s['score'] for s in signals)
    green_count = sum(1 for s in signals if s['status'] == 'green')

    if total_score >= 70:
        conclusion = f"🟢 多頭訊號確認（{green_count}/8項綠燈）—— 技術面偏多，條件符合"
        conclusion_level = 'success'
    elif total_score >= 40:
        conclusion = f"🟡 訊號混合（{green_count}/8項綠燈）—— 部分多頭條件成立，需審慎觀察"
        conclusion_level = 'warning'
    else:
        conclusion = f"🔴 條件不符（{green_count}/8項綠燈）—— 多頭訊號不足，技術面偏弱"
        conclusion_level = 'error'

    return {
        'signals': signals,
        'total_score': total_score,
        'conclusion': conclusion,
        'conclusion_level': conclusion_level
    }


def display_bull_dashboard(bull_signals, symbol):
    """渲染多頭訊號儀表板（8燈 + 評分）"""
    st.markdown("### 🚦 多頭訊號儀表板")
    emoji_map = {'green': '🟢', 'yellow': '🟡', 'red': '🔴'}
    signals = bull_signals['signals']

    for row_start in [0, 4]:
        cols = st.columns(4)
        for i, col in enumerate(cols):
            idx = row_start + i
            if idx < len(signals):
                s = signals[idx]
                with col:
                    st.metric(
                        label=f"{emoji_map[s['status']]} {s['name']}",
                        value=f"{s['score']:.0f} 分",
                        delta=s['desc'],
                        delta_color="normal"
                    )

    st.markdown("---")
    sc, dc = st.columns([1, 3])
    with sc:
        st.metric("整體評分", f"{bull_signals['total_score']:.0f} / 100")
    with dc:
        lvl = bull_signals['conclusion_level']
        if lvl == 'success':
            st.success(bull_signals['conclusion'])
        elif lvl == 'warning':
            st.warning(bull_signals['conclusion'])
        else:
            st.error(bull_signals['conclusion'])
        st.caption("⚠️ 此評分基於近期歷史數據的技術面統計，不構成任何投資建議。歷史表現不代表未來結果。")


# ─────────────────────────────────────────────
# 圖表函數
# ─────────────────────────────────────────────

def create_candlestick_chart(df, symbol, rsi_period, currency_symbol, institutional_df=None, market='us'):
    """
    創建多層 K 線圖
    - 美股：3 層（K線+MA / RSI / 成交量）
    - 台股（有法人數據）：4 層（K線+MA / RSI / 成交量 / 三大法人）

    Args:
        df: 包含 MA 與 RSI 的 DataFrame
        symbol: 股票代碼
        rsi_period: RSI 計算週期
        currency_symbol: 幣別（$ 或 NT$）
        institutional_df: 台股三大法人 DataFrame，可為 None
        market: 'us' 或 'tw'

    Returns:
        plotly Figure
    """
    # 判斷是否顯示法人子圖
    show_inst = market == 'tw' and institutional_df is not None and len(institutional_df) > 0

    if show_inst:
        rows = 4
        row_heights = [0.50, 0.18, 0.12, 0.20]
        subplot_titles = ('價格與移動平均線', f'RSI ({rsi_period}日)', '成交量', '三大法人買賣超（張）')
        chart_height = 1000
    else:
        rows = 3
        row_heights = [0.60, 0.22, 0.18]
        subplot_titles = ('價格與移動平均線', f'RSI ({rsi_period}日)', '成交量')
        chart_height = 900

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=subplot_titles,
        row_heights=row_heights
    )

    # ── Row 1：K線圖 + MA ──
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='K線圖',
            increasing_line_color='#ff4757',
            decreasing_line_color='#2ed573'
        ),
        row=1, col=1
    )

    ma_colors = {
        'MA5': '#ff6b6b',
        'MA10': '#4ecdc4',
        'MA20': '#45b7d1',
        'MA60': '#96ceb4'
    }
    for ma in ['MA5', 'MA10', 'MA20', 'MA60']:
        if ma in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df[ma],
                    mode='lines',
                    name=ma,
                    line=dict(color=ma_colors[ma], width=2)
                ),
                row=1, col=1
            )

    # ── Row 2：RSI ──
    rsi_col = f'RSI{rsi_period}'
    if rsi_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df[rsi_col],
                mode='lines',
                name=f'RSI({rsi_period})',
                line=dict(color='#1e90ff', width=2)
            ),
            row=2, col=1
        )

        # 超買線（70）
        fig.add_trace(
            go.Scatter(
                x=[df['date'].iloc[0], df['date'].iloc[-1]],
                y=[70, 70],
                mode='lines',
                name='超買(70)',
                line=dict(color='red', dash='dash', width=1),
                showlegend=True
            ),
            row=2, col=1
        )

        # 超賣線（30）
        fig.add_trace(
            go.Scatter(
                x=[df['date'].iloc[0], df['date'].iloc[-1]],
                y=[30, 30],
                mode='lines',
                name='超賣(30)',
                line=dict(color='green', dash='dash', width=1),
                showlegend=True
            ),
            row=2, col=1
        )

        # 超買背景色塊（70–100）
        fig.add_hrect(
            y0=70, y1=100,
            fillcolor='rgba(255,71,87,0.08)',
            line_width=0,
            row=2, col=1
        )

        # 超賣背景色塊（0–30）
        fig.add_hrect(
            y0=0, y1=30,
            fillcolor='rgba(46,213,115,0.08)',
            line_width=0,
            row=2, col=1
        )

        fig.update_yaxes(range=[0, 100], row=2, col=1)

    # ── Row 3：成交量 ──
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['volume'],
            name='成交量',
            marker_color='#a55eea',
            opacity=0.6
        ),
        row=3, col=1
    )

    # ── Row 4（台股）：三大法人買賣超 ──
    if show_inst:
        inst_colors = {
            'Foreign_Investor': '#e74c3c',
            'Investment_Trust': '#3498db',
            'Dealer': '#2ecc71',
            'Total': '#f39c12'
        }
        inst_labels = {
            'Foreign_Investor': '外資',
            'Investment_Trust': '投信',
            'Dealer': '自營商',
            'Total': '三大法人合計'
        }

        for name_key, label in inst_labels.items():
            sub = institutional_df[institutional_df['name'] == name_key].copy()
            if sub.empty:
                continue

            # 正負值分色
            colors = ['#e74c3c' if v >= 0 else '#2ed573' for v in sub['net']]

            fig.add_trace(
                go.Bar(
                    x=sub['date'],
                    y=sub['net'],
                    name=label,
                    marker_color=colors,
                    opacity=0.75,
                    visible='legendonly' if name_key == 'Total' else True
                ),
                row=4, col=1
            )

    # ── 佈局更新 ──
    price_label = f"價格 ({currency_symbol})"
    fig.update_layout(
        title=f'{symbol} 股價技術分析圖表',
        height=chart_height,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white'
    )

    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_yaxes(title_text=price_label, row=1, col=1)
    fig.update_yaxes(title_text=f"RSI({rsi_period})", row=2, col=1)
    fig.update_yaxes(title_text="成交量", row=3, col=1)
    if show_inst:
        fig.update_yaxes(title_text="買賣超（張）", row=4, col=1)

    return fig


def create_margin_chart(margin_df, symbol):
    """
    創建融資融券雙 Y 軸折線圖。
    欄位已由 get_tw_margin_trading 標準化為 MarginPurchaseRemaining / ShortSaleRemaining。
    若某欄位仍缺失，只繪製有資料的那條線；兩條都沒有則返回 None。
    """
    has_margin = 'MarginPurchaseRemaining' in margin_df.columns
    has_short  = 'ShortSaleRemaining' in margin_df.columns

    if not has_margin and not has_short:
        return None

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if has_margin:
        fig.add_trace(
            go.Scatter(
                x=margin_df['date'],
                y=margin_df['MarginPurchaseRemaining'],
                mode='lines',
                name='融資餘額（張）',
                line=dict(color='#e74c3c', width=2)
            ),
            secondary_y=False
        )

    if has_short:
        fig.add_trace(
            go.Scatter(
                x=margin_df['date'],
                y=margin_df['ShortSaleRemaining'],
                mode='lines',
                name='融券餘額（張）',
                line=dict(color='#3498db', width=2)
            ),
            secondary_y=True
        )

    fig.update_layout(
        title=f'{symbol} 融資融券餘額走勢',
        height=350,
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="融資餘額（張）", secondary_y=False)
    fig.update_yaxes(title_text="融券餘額（張）", secondary_y=True)

    return fig


def create_macd_chart(df, symbol):
    """
    創建 MACD 指標圖（2層子圖：DIF+MACD線 / 柱狀圖 HIST）

    Returns:
        plotly Figure or None
    """
    if not all(c in df.columns for c in ['MACD_DIF', 'MACD_LINE', 'MACD_HIST']):
        return None

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('MACD DIF / 訊號線', 'MACD 柱狀圖（HIST）'),
        row_heights=[0.55, 0.45]
    )

    # Row 1: DIF + MACD 訊號線
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['MACD_DIF'],
        mode='lines', name='DIF',
        line=dict(color='#ff6b35', width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['MACD_LINE'],
        mode='lines', name='MACD 訊號線',
        line=dict(color='#1e90ff', width=2)
    ), row=1, col=1)

    # 零軸
    fig.add_hline(y=0, line_dash='dash', line_color='gray', line_width=1, row=1, col=1)

    # 金叉標記
    if 'MACD_GOLDEN' in df.columns:
        golden_df = df[df['MACD_GOLDEN']]
        if not golden_df.empty:
            fig.add_trace(go.Scatter(
                x=golden_df['date'],
                y=golden_df['MACD_DIF'],
                mode='markers',
                name='金叉',
                marker=dict(symbol='triangle-up', size=10, color='#f39c12')
            ), row=1, col=1)

    # Row 2: HIST 柱狀圖（正紅負綠）
    hist_colors = ['#ff4757' if v >= 0 else '#2ed573' for v in df['MACD_HIST']]
    fig.add_trace(go.Bar(
        x=df['date'], y=df['MACD_HIST'],
        name='HIST', marker_color=hist_colors, opacity=0.8
    ), row=2, col=1)

    fig.add_hline(y=0, line_dash='dash', line_color='gray', line_width=1, row=2, col=1)

    fig.update_layout(
        title=f'{symbol} MACD 指標（12, 26, 9）',
        height=500,
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_bb_obv_chart(df, symbol, currency_symbol='$'):
    """
    創建布林通道 + OBV 雙層圖（上層BB+K線，下層OBV）

    Returns:
        plotly Figure or None
    """
    if 'BB_MID' not in df.columns:
        return None

    has_obv = 'OBV' in df.columns
    rows = 2 if has_obv else 1
    row_heights = [0.65, 0.35] if has_obv else [1.0]
    subplot_titles = ('布林通道 + 收盤價', 'OBV 量能指標') if has_obv else ('布林通道 + 收盤價',)

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=subplot_titles,
        row_heights=row_heights
    )

    # 上軌/下軌填色區域
    fig.add_trace(go.Scatter(
        x=pd.concat([df['date'], df['date'][::-1]]),
        y=pd.concat([df['BB_UPPER'], df['BB_LOWER'][::-1]]),
        fill='toself',
        fillcolor='rgba(100,149,237,0.08)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False, name='BB 通道'
    ), row=1, col=1)

    # BB 上軌 / 中軌 / 下軌
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['BB_UPPER'],
        mode='lines', name='BB 上軌',
        line=dict(color='#e74c3c', dash='dash', width=1)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['BB_MID'],
        mode='lines', name='BB 中軌',
        line=dict(color='#1e90ff', width=2)
    ), row=1, col=1)

    if 'BB_LOWER' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['BB_LOWER'],
            mode='lines', name='BB 下軌',
            line=dict(color='#2ecc71', dash='dash', width=1)
        ), row=1, col=1)

    # 收盤價線
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['close'],
        mode='lines', name='收盤價',
        line=dict(color='#2c3e50', width=1.5)
    ), row=1, col=1)

    # BB 壓縮期背景色塊（BB_WIDTH < 歷史均值50%）
    if 'BB_WIDTH' in df.columns:
        w_mean = df['BB_WIDTH'].mean()
        compressed_dates = df[df['BB_WIDTH'] < w_mean * 0.5]['date']
        for d in compressed_dates:
            fig.add_vrect(
                x0=d, x1=d,
                fillcolor='rgba(255,165,0,0.15)',
                line_width=0, row=1, col=1
            )

    # OBV 圖
    if has_obv:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['OBV'],
            mode='lines', name='OBV',
            line=dict(color='#9b59b6', width=2)
        ), row=2, col=1)

    fig.update_layout(
        title=f'{symbol} 布林通道（20, 2SD）+ OBV',
        height=600,
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text=f"價格 ({currency_symbol})", row=1, col=1)
    if has_obv:
        fig.update_yaxes(title_text="OBV", row=2, col=1)

    return fig


def create_dmi_chart(df, symbol):
    """
    創建 DMI 三線圖（+DI / -DI / ADX）

    Returns:
        plotly Figure or None
    """
    if not all(c in df.columns for c in ['DMI_PLUS', 'DMI_MINUS', 'DMI_ADX']):
        return None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['DMI_PLUS'],
        mode='lines', name='+DI（多方）',
        line=dict(color='#2ecc71', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['DMI_MINUS'],
        mode='lines', name='-DI（空方）',
        line=dict(color='#e74c3c', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['DMI_ADX'],
        mode='lines', name='ADX（趨勢強度）',
        line=dict(color='#f39c12', width=3)
    ))

    # ADX=25 水平虛線
    fig.add_hline(y=25, line_dash='dash', line_color='gray', line_width=1,
                  annotation_text='ADX=25', annotation_position='right')

    # DMI 黃金交叉標記
    if 'DMI_GOLDEN' in df.columns:
        golden_df = df[df['DMI_GOLDEN']]
        if not golden_df.empty:
            fig.add_trace(go.Scatter(
                x=golden_df['date'],
                y=golden_df['DMI_PLUS'],
                mode='markers', name='黃金交叉',
                marker=dict(symbol='triangle-up', size=10, color='#f39c12')
            ))

    fig.update_layout(
        title=f'{symbol} DMI 趨勢強度指標（14日）',
        height=400,
        template='plotly_white',
        yaxis_title='DMI 值',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def generate_ai_insights(symbol, stock_data, openai_api_key, start_date, end_date,
                          market='us', margin_df=None, institutional_df=None,
                          financial_data=None, rsi_period=14, bull_signals=None):
    """
    使用 OpenAI gpt-4o-mini 進行技術面（美股）或技術面+籌碼面+財務面（台股）分析

    Args:
        symbol: 股票代碼
        stock_data: 含 MA 與 RSI 的 DataFrame
        openai_api_key: OpenAI API 金鑰
        start_date, end_date: 分析期間
        market: 'us' 或 'tw'
        margin_df: 融資融券 DataFrame（台股）
        institutional_df: 三大法人 DataFrame（台股）
        financial_data: {'quarterly': df, 'annual': df}（台股）
        rsi_period: RSI 計算週期

    Returns:
        str: AI 分析報告
    """
    try:
        client = OpenAI(api_key=openai_api_key)

        # 基本數據準備
        first_date = stock_data['date'].iloc[0].strftime('%Y-%m-%d')
        last_date = stock_data['date'].iloc[-1].strftime('%Y-%m-%d')
        start_price = stock_data['close'].iloc[0]
        end_price = stock_data['close'].iloc[-1]
        price_change = ((end_price - start_price) / start_price) * 100

        rsi_col = f'RSI{rsi_period}'
        latest_rsi = stock_data[rsi_col].iloc[-1] if rsi_col in stock_data.columns else 50
        if latest_rsi > 70:
            rsi_status = f"超買區域，RSI={latest_rsi:.2f}（>70）"
        elif latest_rsi < 30:
            rsi_status = f"超賣區域，RSI={latest_rsi:.2f}（<30）"
        else:
            rsi_status = f"中性區域，RSI={latest_rsi:.2f}（30–70）"

        currency = "NT$" if market == 'tw' else "$"
        market_desc = "台灣股票市場" if market == 'tw' else "美國股票市場"

        # 最近 60 筆數據
        data_json = stock_data.tail(60).to_json(orient='records', date_format='iso')

        # ── System Message ──
        system_base = f"""你是一位專業的技術分析師，專精於{market_desc}股票技術分析和歷史數據解讀。你的職責包括：

1. 客觀描述股票價格的歷史走勢和技術指標狀態
2. 解讀歷史市場數據和交易量變化模式
3. 識別技術面的歷史支撐阻力位
4. 提供純教育性的技術分析知識，包含RSI動量指標解讀"""

        system_tw_extra = """
5. 解讀台股特有的籌碼面指標，包括三大法人買賣超動向、融資融券餘額變化
6. 結合財務報表數據進行基本面輔助解讀"""

        system_rules = """

重要原則：
- 僅提供歷史數據分析和技術指標解讀，絕不提供任何投資建議或預測
- 保持完全客觀中立的分析態度
- 使用專業術語但保持易懂
- 所有分析僅供研究目的
- 強調技術分析的局限性和不確定性
- 使用繁體中文回答

嚴格的表達方式要求：
- 使用「歷史數據顯示」、「技術指標反映」、「過去走勢呈現」等客觀描述
- 避免「可能性」、「預期」、「建議」、「關注」等暗示性用詞
- 禁用「如果...則...」的假設句型，改用「歷史上當...時，曾出現...現象」
- 不提供具體價位的操作參考點，僅描述技術位階的歷史表現
- 強調「歷史表現不代表未來結果」
- 避免任何可能被解讀為操作指引的表達

免責聲明：所提供的分析內容純粹基於歷史數據的技術解讀，僅供研究參考，不構成任何投資建議或未來走勢預測。歷史表現不代表未來結果。"""

        system_message = system_base + (system_tw_extra if market == 'tw' else '') + system_rules

        # ── User Prompt ──
        user_prompt = f"""請基於以下股票歷史數據進行深度技術分析：

### 基本資訊
- 股票代號：{symbol}
- 市場：{market_desc}
- 分析期間：{first_date} 至 {last_date}
- 期間價格變化：{price_change:.2f}% （從 {currency}{start_price:.2f} 變化到 {currency}{end_price:.2f}）
- 最新RSI狀態：{rsi_status}

### 完整交易數據（含 MA 與 RSI，最近60筆）
{data_json}
"""

        # 台股附加數據
        if market == 'tw':
            # 融資融券
            if margin_df is not None and len(margin_df) > 0:
                margin_json = margin_df.tail(10).to_json(orient='records', date_format='iso')
                user_prompt += f"\n### 融資融券餘額（最新10筆）\n{margin_json}\n"

            # 三大法人
            if institutional_df is not None and len(institutional_df) > 0:
                inst_json = institutional_df.tail(40).to_json(orient='records', date_format='iso')
                user_prompt += f"\n### 三大法人買賣超（外資/投信/自營商/合計，最新資料）\n{inst_json}\n"

            # 財務報表
            if financial_data:
                if financial_data.get('quarterly') is not None:
                    q_json = financial_data['quarterly'].to_json(orient='records', date_format='iso')
                    user_prompt += f"\n### 近期季度財務數據（EPS/Revenue/OperatingIncome 等）\n{q_json}\n"
                if financial_data.get('annual') is not None:
                    a_json = financial_data['annual'].to_json(orient='records', date_format='iso')
                    user_prompt += f"\n### 近期年度資產負債表摘要\n{a_json}\n"

        # 多頭訊號評分摘要（美股台股均加入）
        if bull_signals:
            signal_summary = "\n".join([
                f"- {s['name']}：{'🟢' if s['status']=='green' else ('🟡' if s['status']=='yellow' else '🔴')} {s['desc']}（{s['score']:.0f}分）"
                for s in bull_signals['signals']
            ])
            user_prompt += f"\n### 多頭訊號評分結果（整體{bull_signals['total_score']:.0f}/100分）\n{signal_summary}\n結論：{bull_signals['conclusion']}\n"

        # ── 分析架構 ──
        if market == 'tw':
            user_prompt += f"""
### 分析架構：技術面 + 籌碼面 + 財務面 + 多頭訊號完整分析

#### 1. 趨勢分析
- 整體趨勢方向（上升、下降、盤整）
- 關鍵支撐位和阻力位識別
- 趨勢強度評估

#### 2. 技術指標分析（MA 均線）
- 移動平均線分析（短期與長期MA的關係）
- 價格與移動平均線的相對位置

#### 3. RSI 分析（必要章節）
- 最新RSI狀態（{rsi_status}）
- RSI 歷史走勢描述
- 動量強度觀察

#### 4. MACD 分析（必要章節）
- DIF 與訊號線的相對位置（金叉/死叉）
- 柱狀圖（HIST）方向與強度
- 零軸上下位置判斷

#### 5. 布林通道分析（必要章節）
- 通道壓縮與突破型態描述
- 收盤價與中軌/上軌的相對位置
- 歷史上此型態曾出現的走勢現象

#### 6. OBV 分析（必要章節）
- OBV 與股價的歷史走勢關聯性
- 背離觀察（量價背離）
- 資金流向描述

#### 7. DMI 分析（必要章節）
- +DI 與 -DI 方向關係
- ADX 趨勢強度水位
- 黃金交叉或死亡交叉觀察

#### 8. 多頭訊號評分解讀（必要章節）
- 8項技術指標燈號總結
- 評分的歷史統計意涵
- 訊號密集度與時序重疊觀察

#### 9. 籌碼面分析（台股特有，必要章節）
- 三大法人（外資、投信、自營商）買賣超趨勢分析
- 外資動向與價格走勢的歷史關聯性
- 融資餘額變化與市場槓桿水位觀察
- 融券餘額與空方籌碼分布的歷史意涵
- 籌碼面指標的局限性說明

#### 10. 財務面輔助觀察（台股特有，必要章節）
- EPS 趨勢歷史觀察（季度變化）
- 營收成長性歷史數據描述
- 資產負債結構的歷史變化
- 財務面分析的局限性

#### 11. 價格行為分析
- 重要的價格突破點
- 波動性評估

#### 12. 風險評估
- 技術面風險因子
- 市場情緒指標

#### 13. 市場觀察
- 短期技術面觀察（1-2週）
- 中期技術面觀察（1-3個月）

分析目標：{symbol}（台股）"""
        else:
            user_prompt += f"""
### 分析架構：技術面完整分析

#### 1. 趨勢分析
- 整體趨勢方向（上升、下降、盤整）
- 關鍵支撐位和阻力位識別
- 趨勢強度評估

#### 2. 技術指標分析（MA 均線）
- 移動平均線分析（短期與長期MA的關係）
- 價格與移動平均線的相對位置
- 成交量與價格變動的關聯性

#### 3. RSI 分析（必要章節）
- 最新RSI狀態（{rsi_status}）
- RSI 歷史走勢描述
- 動量強度觀察

#### 4. MACD 分析（必要章節）
- DIF 與訊號線的相對位置（金叉/死叉）
- 柱狀圖（HIST）方向與強度
- 零軸上下位置判斷

#### 5. 布林通道分析（必要章節）
- 通道壓縮與突破型態描述
- 收盤價與中軌/上軌的相對位置
- 歷史上此型態曾出現的走勢現象

#### 6. OBV 分析（必要章節）
- OBV 與股價的歷史走勢關聯性
- 背離觀察（量價背離）
- 資金流向描述

#### 7. DMI 分析（必要章節）
- +DI 與 -DI 方向關係
- ADX 趨勢強度水位
- 黃金交叉或死亡交叉觀察

#### 8. 多頭訊號評分解讀（必要章節）
- 8項技術指標燈號總結
- 評分的歷史統計意涵
- 訊號密集度與時序重疊觀察

#### 9. 價格行為分析
- 重要的價格突破點
- 波動性評估
- 關鍵的轉折點識別

#### 10. 風險評估
- 當前價位的風險等級
- 潛在的支撐和阻力區間
- 市場情緒指標

#### 11. 市場觀察
- 短期技術面觀察（1-2週）
- 中期技術面觀察（1-3個月）
- 技術面風險因子

分析目標：{symbol}（美股）"""

        # 調用 OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=3000,
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        st.error(f"AI分析失敗：{str(e)}")
        return "AI分析暫時無法使用，請檢查API金鑰或稍後再試。"


# ─────────────────────────────────────────────
# 側邊欄 UI
# ─────────────────────────────────────────────

st.sidebar.markdown("## 🔧 分析設定")
st.sidebar.divider()

# 市場選擇下拉選單
market = st.sidebar.selectbox(
    "市場選擇",
    options=["美股 (US)", "台股 (TW)"],
    index=0,
    help="選擇要分析的市場"
)
is_tw = market == "台股 (TW)"

# 股票代碼輸入（依市場切換預設值與說明）
if is_tw:
    symbol = st.sidebar.text_input(
        "台股代碼",
        value="2330",
        help="輸入台股純數字代碼，例如：2330（台積電）、2317（鴻海）、0050（元大台灣50）"
    )
else:
    symbol = st.sidebar.text_input(
        "股票代碼",
        value="AAPL",
        help="輸入美股股票代碼，例如：AAPL, MSFT, GOOGL, TSLA, NVDA"
    )

# API 金鑰區塊（依市場動態顯示）
if is_tw:
    finmind_api_key = st.sidebar.text_input(
        "FinMind API Key",
        type="password",
        help="請輸入您的 FinMind API 金鑰（台股數據）"
    )
    fmp_api_key = ""
else:
    fmp_api_key = st.sidebar.text_input(
        "FMP API Key",
        type="password",
        help="請輸入您的 Financial Modeling Prep API 金鑰"
    )
    finmind_api_key = ""

# OpenAI API Key（兩種市場均顯示）
openai_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="請輸入您的 OpenAI API 金鑰"
)

# 日期選擇
default_start_date = datetime.now() - timedelta(days=365)
default_end_date = datetime.now()

start_date = st.sidebar.date_input(
    "起始日期",
    value=default_start_date,
    help="選擇分析的起始日期"
)

end_date = st.sidebar.date_input(
    "結束日期",
    value=default_end_date,
    help="選擇分析的結束日期"
)

# RSI 計算天數
rsi_period = st.sidebar.number_input(
    "RSI 計算天數",
    min_value=2,
    max_value=50,
    value=14,
    step=1,
    help="RSI 計算週期，預設 14 日，範圍 2–50"
)

# 分析按鈕
analyze_button = st.sidebar.button("🚀 開始分析", type="primary", use_container_width=True)

# 免責聲明
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📢 免責聲明
本系統僅供學術研究用途，AI 提供的數據與分析結果僅供參考，**不構成投資建議或財務建議**。

請使用者自行判斷投資決策，並承擔相關風險。本系統作者不對任何投資行為負責，亦不承擔任何損失責任。
""")


# ─────────────────────────────────────────────
# 主要分析邏輯
# ─────────────────────────────────────────────

if analyze_button:
    # ── 輸入驗證 ──
    if not symbol.strip():
        st.error("請輸入股票代碼")
    elif is_tw and not symbol.strip().isdigit():
        st.error("台股代碼請輸入純數字，例如：2330、0050")
    elif is_tw and not finmind_api_key.strip():
        st.error("請輸入 FinMind API Key")
    elif not is_tw and not fmp_api_key.strip():
        st.error("請輸入 FMP API Key")
    elif not openai_api_key.strip():
        st.error("請輸入 OpenAI API Key")
    elif start_date >= end_date:
        st.error("起始日期不能晚於或等於結束日期")
    else:
        market_key = 'tw' if is_tw else 'us'
        currency_symbol = "NT$" if is_tw else "$"

        # ── Step 1: 獲取日 K 價格數據 ──
        spinner_price = "正在獲取台股價格數據..." if is_tw else "正在獲取美股價格數據..."
        with st.spinner(spinner_price):
            if is_tw:
                stock_data = get_tw_stock_price(symbol.strip(), finmind_api_key, start_date, end_date)
            else:
                stock_data = get_us_stock_data(symbol.upper(), fmp_api_key, start_date, end_date)

        if stock_data is not None and len(stock_data) > 0:
            st.success(f"成功獲取 {len(stock_data)} 筆交易數據")

            # 過濾日期範圍
            filtered_data = filter_by_date_range(stock_data, start_date, end_date)

            if filtered_data is not None and len(filtered_data) > 0:

                # ── Step 2: 計算技術指標 ──
                with st.spinner("正在計算技術指標（MA & RSI）..."):
                    data_with_ma = get_moving_averages(filtered_data)
                    data_with_indicators = calculate_rsi(data_with_ma, period=rsi_period)

                # ── Step 2b: 計算進階指標（MACD / BB / OBV / DMI）──
                with st.spinner("正在計算進階技術指標（MACD / 布林通道 / OBV / DMI）..."):
                    data_with_indicators = calculate_advanced_indicators(data_with_indicators)

                # ── Step 2c: 計算多頭訊號評分 ──
                bull_signals = calculate_bull_signals(data_with_indicators)

                # ── Step 3 (台股): 籌碼數據 ──
                margin_df = None
                institutional_df = None
                financial_data = None

                if is_tw:
                    with st.spinner("正在獲取台股籌碼數據（融資融券、三大法人）..."):
                        margin_df = get_tw_margin_trading(symbol.strip(), finmind_api_key, start_date, end_date)
                        institutional_df = get_tw_institutional(symbol.strip(), finmind_api_key, start_date, end_date)

                    with st.spinner("正在獲取財務報表數據..."):
                        financial_data = get_tw_financial_statements(symbol.strip(), finmind_api_key)

                    # 附加數據狀態反饋
                    status_parts = []
                    if margin_df is not None:
                        status_parts.append("融資融券")
                    if institutional_df is not None:
                        status_parts.append("三大法人")
                    if financial_data and (financial_data.get('quarterly') is not None or financial_data.get('annual') is not None):
                        status_parts.append("財務報表")

                    if status_parts:
                        st.success(f"成功獲取台股附加數據：{'、'.join(status_parts)}")
                    else:
                        st.warning("台股附加數據（籌碼/財務）獲取失敗，將僅顯示技術面分析。")

                if data_with_indicators is not None:

                    # ── Step 4: K 線圖 ──
                    st.markdown("### 📊 股價K線圖與技術指標")
                    chart = create_candlestick_chart(
                        data_with_indicators,
                        symbol.upper() if not is_tw else symbol.strip(),
                        rsi_period,
                        currency_symbol,
                        institutional_df=institutional_df,
                        market=market_key
                    )
                    st.plotly_chart(chart, use_container_width=True)

                    # ── Step 5: RSI 即時警告 ──
                    rsi_col = f'RSI{rsi_period}'
                    latest_rsi = data_with_indicators[rsi_col].iloc[-1] if rsi_col in data_with_indicators.columns else 50

                    if latest_rsi > 70:
                        st.warning(f"⚠️ RSI 超買警告：目前 RSI = **{latest_rsi:.2f}**，已進入超買區域（>70）。歷史數據顯示此區域價格波動風險較高。")
                    elif latest_rsi < 30:
                        st.success(f"📉 RSI 超賣提示：目前 RSI = **{latest_rsi:.2f}**，已進入超賣區域（<30）。歷史數據顯示此區域曾出現反彈現象，但不代表未來走勢。")
                    else:
                        st.info(f"📊 RSI 中性：目前 RSI = **{latest_rsi:.2f}**，位於中性區域（30–70）。")

                    # ── Step 6 (台股): 融資融券走勢圖 ──
                    if is_tw and margin_df is not None and len(margin_df) > 0:
                        margin_chart = create_margin_chart(margin_df, symbol.strip())
                        if margin_chart is not None:
                            st.markdown("### 💳 融資融券餘額")
                            st.plotly_chart(margin_chart, use_container_width=True)
                        else:
                            st.info("ℹ️ 融資融券圖表無法顯示（欄位資料缺失）。")

                    # ── Step 7: 基本統計資訊（4 欄）──
                    st.markdown("### 📈 基本統計資訊")
                    col1, col2, col3, col4 = st.columns(4)

                    s_price = data_with_indicators['close'].iloc[0]
                    e_price = data_with_indicators['close'].iloc[-1]
                    price_change = e_price - s_price
                    price_change_pct = (price_change / s_price) * 100

                    with col1:
                        st.metric(
                            "起始價格",
                            f"{currency_symbol}{s_price:.2f}",
                            help="分析期間第一個交易日的收盤價"
                        )
                    with col2:
                        st.metric(
                            "結束價格",
                            f"{currency_symbol}{e_price:.2f}",
                            help="分析期間最後一個交易日的收盤價"
                        )
                    with col3:
                        st.metric(
                            "價格變化",
                            f"{currency_symbol}{price_change:.2f}",
                            f"{price_change_pct:.2f}%",
                            help="期間內的價格變化金額和百分比"
                        )
                    with col4:
                        rsi_delta = "超買🔴" if latest_rsi > 70 else ("超賣🟢" if latest_rsi < 30 else "中性🔵")
                        st.metric(
                            f"最新 RSI ({rsi_period}日)",
                            f"{latest_rsi:.2f}",
                            rsi_delta,
                            help="相對強弱指數：>70 超買，<30 超賣"
                        )

                    # ── Step 8 (台股): 三大法人買賣超表格 ──
                    if is_tw and institutional_df is not None and len(institutional_df) > 0:
                        st.markdown("### 🏦 三大法人買賣超（最近10個交易日）")

                        inst_display = institutional_df.copy()
                        label_map = {
                            'Foreign_Investor': '外資',
                            'Investment_Trust': '投信',
                            'Dealer': '自營商',
                            'Total': '三大法人合計'
                        }
                        inst_display['name'] = inst_display['name'].map(label_map).fillna(inst_display['name'])

                        try:
                            pivot_df = inst_display.pivot_table(
                                index='date',
                                columns='name',
                                values='net',
                                aggfunc='sum'
                            ).reset_index()
                            pivot_df['date'] = pivot_df['date'].dt.strftime('%Y-%m-%d')
                            pivot_df = pivot_df.sort_values('date', ascending=False).head(10)
                            st.dataframe(pivot_df, use_container_width=True, hide_index=True)
                        except Exception:
                            # 若 pivot 失敗，顯示原始表格
                            recent_inst = inst_display.tail(40).copy()
                            recent_inst['date'] = recent_inst['date'].dt.strftime('%Y-%m-%d')
                            st.dataframe(recent_inst[['date', 'name', 'buy', 'sell', 'net']],
                                         use_container_width=True, hide_index=True)

                    # ── Step 9 (台股): 財務報表 ──
                    if is_tw and financial_data:
                        if financial_data.get('quarterly') is not None:
                            st.markdown("### 📑 近期季度財務數據")
                            q_df = financial_data['quarterly'].copy().head(20)
                            st.dataframe(q_df, use_container_width=True, hide_index=True)

                        if financial_data.get('annual') is not None:
                            st.markdown("### 📊 近期年度資產負債表摘要")
                            a_df = financial_data['annual'].copy().head(10)
                            st.dataframe(a_df, use_container_width=True, hide_index=True)

                    # ── Step 9b: 多頭訊號儀表板 ──
                    display_bull_dashboard(bull_signals, symbol.strip() if is_tw else symbol.upper())

                    # ── Step 9c: MACD 圖 ──
                    macd_fig = create_macd_chart(
                        data_with_indicators,
                        symbol.strip() if is_tw else symbol.upper()
                    )
                    if macd_fig:
                        st.markdown("### 📉 MACD 指標（12, 26, 9）")
                        st.plotly_chart(macd_fig, use_container_width=True)

                    # ── Step 9d: 布林通道 + OBV 圖 ──
                    bb_obv_fig = create_bb_obv_chart(
                        data_with_indicators,
                        symbol.strip() if is_tw else symbol.upper(),
                        currency_symbol
                    )
                    if bb_obv_fig:
                        st.markdown("### 📊 布林通道（20, 2SD）+ OBV 量能指標")
                        st.plotly_chart(bb_obv_fig, use_container_width=True)

                    # ── Step 9e: DMI 圖 ──
                    dmi_fig = create_dmi_chart(
                        data_with_indicators,
                        symbol.strip() if is_tw else symbol.upper()
                    )
                    if dmi_fig:
                        st.markdown("### 📈 DMI 趨勢強度指標（14日）")
                        st.plotly_chart(dmi_fig, use_container_width=True)

                    # ── Step 10: AI 技術分析 ──
                    st.markdown("### 🤖 AI技術分析")
                    spinner_ai = "AI 正在分析中（含籌碼面、財務面、多頭訊號分析）..." if is_tw else "AI 正在分析中（含MACD/BB/OBV/DMI多頭訊號分析）..."
                    with st.spinner(spinner_ai):
                        ai_analysis = generate_ai_insights(
                            symbol.strip() if is_tw else symbol.upper(),
                            data_with_indicators,
                            openai_api_key,
                            start_date,
                            end_date,
                            market=market_key,
                            margin_df=margin_df,
                            institutional_df=institutional_df,
                            financial_data=financial_data,
                            rsi_period=rsi_period,
                            bull_signals=bull_signals
                        )

                    if ai_analysis:
                        st.markdown(ai_analysis)

                    # ── Step 11: 歷史數據表格 ──
                    st.markdown("### 📋 歷史數據表格（最近10筆）")
                    display_data = data_with_indicators.tail(10).copy()
                    display_data = display_data.sort_values('date', ascending=False)

                    rsi_col_label = f'RSI({rsi_period})'
                    display_cols = ['date', 'open', 'high', 'low', 'close', 'volume',
                                    'MA5', 'MA10', 'MA20', 'MA60', rsi_col]
                    display_cols = [c for c in display_cols if c in display_data.columns]
                    display_data_fmt = display_data[display_cols].copy()

                    col_rename = {
                        'date': '日期', 'open': '開盤', 'high': '最高',
                        'low': '最低', 'close': '收盤', 'volume': '成交量',
                        rsi_col: rsi_col_label
                    }
                    display_data_fmt = display_data_fmt.rename(columns=col_rename)

                    st.dataframe(display_data_fmt, use_container_width=True, hide_index=True)

                    st.success("✅ 分析完成！")

            else:
                st.warning("所選日期範圍內沒有交易數據，請調整日期範圍。")
        else:
            st.error("無法獲取股票數據，請檢查股票代碼和API金鑰。")


# ─────────────────────────────────────────────
# 初始歡迎頁面
# ─────────────────────────────────────────────

if not analyze_button:
    st.markdown("""
## 歡迎使用 AI 股票趨勢分析系統 👋

### 🚀 功能特色
- **雙市場支援**: 美股（FMP API）與台股（FinMind API），下拉選單一鍵切換
- **專業K線圖表**: 互動式多層子圖，含移動平均線、RSI、成交量；台股另含三大法人買賣超子圖
- **進階技術指標**: MACD（金叉偵測）、布林通道（壓縮突破）、OBV（量價背離）、DMI（趨勢強度）
- **🚦 多頭訊號儀表板**: 8項指標燈號（🟢🟡🔴）+ 整體評分（0–100分）+ 多頭確認結論
- **台股籌碼面分析**: 三大法人買賣超（外資/投信/自營商）、融資融券餘額走勢
- **台股財務面資訊**: 季度財務數據（EPS/營收等）、年度資產負債表摘要
- **AI智能分析**: 使用 gpt-4o-mini 進行技術面深度分析；台股版額外包含籌碼面、財務面與多頭訊號解讀
- **RSI 動量指標**: 可自訂計算週期，即時超買/超賣警告
- **教育導向**: 客觀的技術分析，僅供學習研究使用

### 📝 使用方法
1. 在左側選擇市場（美股 / 台股）
2. 輸入股票代碼（美股如：AAPL；台股純數字如：2330）
3. 輸入對應的 API 金鑰與 OpenAI API Key
4. 選擇分析的日期範圍（建議至少1年以利指標計算），設定 RSI 計算天數
5. 點擊「🚀 開始分析」按鈕

### 💡 技術指標說明
- **MA5 / MA10 / MA20 / MA60**: 移動平均線，分別反映短、短中、中、長期趨勢
- **RSI（相對強弱指數）**: >70 超買，<30 超賣，50 為中性分界線
- **MACD（12,26,9）**: DIF/訊號線金叉、柱狀圖方向、零軸上下位置
- **布林通道（20,2SD）**: 通道壓縮後突破為重要爆發型態
- **OBV（量能指標）**: 量價背離偵測，OBV新高配合量增為資金流入訊號
- **DMI（14日）**: +DI/-DI方向、ADX趨勢強度（>25為強趨勢）

### 🚦 多頭訊號儀表板
系統自動評估 8 項技術指標燈號：
MACD轉正 / BB突破中軌 / BB壓縮突破 / OBV資金流入 / RSI動量 / DMI多頭趨勢 / DMI黃金交叉 / 均線多頭排列
- **🟢 綠燈（12.5分）**: 強訊號
- **🟡 黃燈（6分）**: 弱訊號
- **🔴 紅燈（0分）**: 條件不符
- 整體評分 ≥70：多頭訊號確認 | 40–69：訊號混合 | <40：條件不符

### 🗂️ 台股特有指標說明
- **三大法人買賣超**: 外資、投信、自營商每日買賣超張數，反映法人籌碼動向
- **融資餘額**: 投資人以槓桿買進的總張數，反映市場槓桿水位
- **融券餘額**: 投資人放空的總張數，反映市場空方籌碼分布

### 🔑 API 金鑰獲取
- **FMP API（美股）**: [Financial Modeling Prep](https://financialmodelingprep.com/developer/docs)
- **FinMind API（台股）**: [FinMind Trade](https://finmindtrade.com/)（免費方案每日有請求次數限制）
- **OpenAI API**: [OpenAI Platform](https://platform.openai.com)

---
**開始您的技術分析之旅吧！** 📈
""")
