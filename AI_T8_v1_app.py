import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
from datetime import datetime, timedelta
import json
import numpy as np
import time
import random
try:
    from bs4 import BeautifulSoup as _BS4
    _BS4_OK = True
except ImportError:
    _BS4_OK = False

# 設置頁面配置
st.set_page_config(
    page_title="AI 股票趨勢分析系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 主標題
st.title("📈 AI 股票趨勢分析系統")
st.divider()


# ─────────────────────────────────────────────
# 美股數據函數
# ─────────────────────────────────────────────

def get_us_stock_data(symbol, api_key, start_date, end_date):
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
        if result.get('status') != 200:
            msg = result.get('msg', '未知錯誤')
            st.error(f"FinMind API 錯誤：{msg}")
            return None
        data = result.get('data', [])
        if not data:
            st.error(f"無法獲取台股 {symbol} 的數據，請確認代碼是否正確。")
            return None
        df = pd.DataFrame(data)
        required_cols = ['date', 'open', 'max', 'min', 'close', 'Trading_Volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"FinMind 回傳欄位缺失：{missing}")
            return None
        df = df.rename(columns={'max': 'high', 'min': 'low', 'Trading_Volume': 'volume'})
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"FinMind API 請求失敗：{str(e)}")
        return None
    except Exception as e:
        st.error(f"台股數據處理錯誤：{str(e)}")
        return None


def get_tw_margin_trading(symbol, api_key, start_date, end_date):
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

        margin_candidates = ['MarginPurchaseRemaining', 'MarginPurchase', 'margin_purchase_remaining', 'FundingRemaining']
        short_candidates  = ['ShortSaleRemaining', 'ShortSale', 'short_sale_remaining', 'ShortRemaining']

        def find_col(df_cols, candidates):
            for c in candidates:
                if c in df_cols:
                    return c
            lower_cols = {c.lower(): c for c in df_cols}
            for keyword in ['marginpurchaseremaining', 'marginpurchase', 'shortsaleremaining', 'shortsale']:
                for lc, orig in lower_cols.items():
                    if keyword in lc:
                        return orig
            return None

        margin_col = find_col(df.columns.tolist(), margin_candidates)
        short_col  = find_col(df.columns.tolist(), short_candidates)

        if margin_col is None and short_col is None:
            return None

        rename_map = {}
        if margin_col and margin_col != 'MarginPurchaseRemaining':
            rename_map[margin_col] = 'MarginPurchaseRemaining'
        if short_col and short_col != 'ShortSaleRemaining':
            rename_map[short_col] = 'ShortSaleRemaining'
        if rename_map:
            df = df.rename(columns=rename_map)

        for col in ['MarginPurchaseRemaining', 'ShortSaleRemaining']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception:
        return None


def get_tw_institutional(symbol, api_key, start_date, end_date):
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
        for col in ['buy', 'sell']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df['net'] = df['buy'] - df['sell']
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception:
        return None


def get_tw_broker_trading(symbol, api_key, date_str=None):
    """
    獲取台股券商分點進出明細。

    ⚠️ 重要說明：
    - FinMind TaiwanStockTradingDailyReport 是「sponsor 付費方案」專屬功能
    - 免費 / backer 方案完全無法使用
    - 本函數改用 TWSE 官方網站 (bsr.twse.com.tw) 爬取，完全免費

    來源：TWSE 買賣日報表
      上市: https://bsr.twse.com.tw/bshtm/bsContent.aspx
      上櫃: https://bsr.tpex.org.tw/bshtm/bsContent.aspx

    Args:
        symbol   : 台股代碼，如 '2330'
        api_key  : FinMind API Token（若有 sponsor 方案則嘗試，否則直接用 TWSE）
        date_str : 查詢日期 'YYYY-MM-DD'，None=今天

    Returns:
        dict 或 None
        {
          'date'       : '2026-04-10',
          'buy_df'     : DataFrame (broker_name, buy, sell, net, ratio),
          'sell_df'    : DataFrame (broker_name, buy, sell, net, ratio),
          'total_buy'  : int,
          'total_sell' : int,
          'source'     : str,   # 資料來源說明
        }
    """
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')

    twse_date = date_str.replace('-', '')  # YYYYMMDD

    # ── 嘗試 FinMind（僅 sponsor 方案有效）──
    if api_key and api_key.strip():
        try:
            url = "https://api.finmindtrade.com/api/v4/data"
            params = {
                'dataset': 'TaiwanStockTradingDailyReport',
                'data_id': symbol,
                'start_date': date_str,
                'end_date':   date_str,
                'token': api_key
            }
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                res = resp.json()
                # status=40x 或 data 為空時跳過
                if res.get('status') == 200 and res.get('data'):
                    raw = pd.DataFrame(res['data'])
                    result = _parse_broker_df(raw, date_str, 'FinMind')
                    if result:
                        return result
        except Exception:
            pass

    # ── TWSE 上市股票爬取（免費）──
    # 上市代碼通常是 4 碼純數字，≤ 6 碼；上櫃多為 5 碼或含字母
    headers_web = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/124.0.0.0 Safari/537.36'
        ),
        'Accept-Language': 'zh-TW,zh;q=0.9,en;q=0.8',
        'Referer': 'https://bsr.twse.com.tw/bshtm/',
    }

    # 先試上市 (TWSE)
    for base_url, market_label in [
        ("https://bsr.twse.com.tw/bshtm/bsContent.aspx", "上市"),
        ("https://bsr.tpex.org.tw/bshtm/bsContent.aspx", "上櫃"),
    ]:
        try:
            url = f"{base_url}?v=t&BHID=&StockNo={symbol}&StartDate={twse_date}&EndDate={twse_date}"
            resp = requests.get(url, headers=headers_web, timeout=20)
            resp.raise_for_status()

            # TWSE 回 big5 編碼
            try:
                resp.encoding = 'big5'
                html_text = resp.text
            except Exception:
                html_text = resp.content.decode('big5', errors='replace')

            # 檢查是否有實際資料（無資料時頁面會很短或包含特定文字）
            if len(html_text) < 500:
                continue
            if '查無資料' in html_text or 'no data' in html_text.lower():
                continue

            # 解析 HTML 表格
            try:
                tables = pd.read_html(html_text, header=0, flavor='lxml')
            except Exception:
                try:
                    tables = pd.read_html(html_text, header=0)
                except Exception:
                    continue

            if not tables:
                continue

            # 找到包含券商買賣資料的表格（通常欄位含「買進」「賣出」）
            target_df = None
            for t in tables:
                col_str = ' '.join(str(c) for c in t.columns)
                if '買進' in col_str and '賣出' in col_str:
                    target_df = t
                    break
                # 有些版本用英文或無標頭，嘗試看資料列數
                if len(t) > 5 and len(t.columns) >= 4:
                    target_df = t
                    break

            if target_df is None or target_df.empty:
                continue

            result = _parse_broker_df(target_df, date_str, f'TWSE {market_label}')
            if result:
                return result

        except Exception:
            continue

    return None


def _parse_broker_df(raw_df, date_str, source_label):
    """
    通用解析函數：將各來源的原始 DataFrame 轉為標準格式
    """
    try:
        df = raw_df.copy()

        # 自動偵測欄位名稱
        col_map = {}
        for c in df.columns:
            cs = str(c).strip().replace('\u3000', '').replace(' ', '')
            if any(k in cs for k in ['券商', 'broker', 'Broker', '分點', '名稱', '證券商']):
                col_map[c] = 'broker_name'
            elif '買進' in cs or cs in ('buy', 'Buy'):
                col_map[c] = 'buy'
            elif '賣出' in cs or cs in ('sell', 'Sell'):
                col_map[c] = 'sell'

        if col_map:
            df = df.rename(columns=col_map)

        # 若沒偵測到，嘗試位置推斷（第0欄=名稱, 第1欄=買進, 第2欄=賣出）
        if 'buy' not in df.columns and len(df.columns) >= 3:
            cols = list(df.columns)
            df = df.rename(columns={cols[0]: 'broker_name', cols[1]: 'buy', cols[2]: 'sell'})

        if 'buy' not in df.columns or 'sell' not in df.columns:
            return None

        if 'broker_name' not in df.columns:
            df['broker_name'] = '未知券商'

        # 清理數值（移除千分位逗號）
        def clean_num(v):
            try:
                return int(str(v).replace(',', '').replace('，', '').strip())
            except Exception:
                return 0

        df['buy']  = df['buy'].apply(clean_num)
        df['sell'] = df['sell'].apply(clean_num)
        df['net']  = df['buy'] - df['sell']

        # 過濾合計/小計列
        df = df[~df['broker_name'].astype(str).str.contains(
            r'合計|小計|平均|Total|total|^\s*$', na=False, regex=True
        )]
        df = df[df['buy'] + df['sell'] > 0]  # 過濾零值列

        if df.empty:
            return None

        total_buy  = int(df['buy'].sum())
        total_sell = int(df['sell'].sum())
        total_vol  = total_buy + total_sell

        # 估成交比重（百分比）
        df['ratio'] = df.apply(
            lambda r: round((r['buy'] + r['sell']) / total_vol * 100, 2) if total_vol > 0 else 0.0,
            axis=1
        )

        buy_df  = df[df['net'] > 0].sort_values('net', ascending=False).head(20).reset_index(drop=True)
        sell_df = df[df['net'] < 0].copy()
        sell_df['net'] = sell_df['net'].abs()
        sell_df = sell_df.sort_values('net', ascending=False).head(20).reset_index(drop=True)

        if buy_df.empty and sell_df.empty:
            return None

        return {
            'date':       date_str,
            'buy_df':     buy_df[['broker_name', 'buy', 'sell', 'net', 'ratio']],
            'sell_df':    sell_df[['broker_name', 'buy', 'sell', 'net', 'ratio']],
            'total_buy':  total_buy,
            'total_sell': total_sell,
            'source':     source_label,
        }

    except Exception:
        return None



def get_tw_financial_statements(symbol, api_key):
    """
    台股季報（5年20季度）
    年報（資產負債表）已省略
    """
    result = {'quarterly': None}
    today = datetime.now()

    try:
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            'dataset': 'TaiwanStockFinancialStatements',
            'data_id': symbol,
            'start_date': (today - timedelta(days=365 * 5)).strftime('%Y-%m-%d'),
            'token': api_key
        }
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        res = resp.json()
        if res.get('status') == 200 and res.get('data'):
            df = pd.DataFrame(res['data'])
            target_types = ['EPS', 'Revenue', 'OperatingIncome', 'NetIncome', 'GrossProfit']
            df = df[df['type'].isin(target_types)].tail(80)
            result['quarterly'] = df
    except Exception:
        pass

    return result


def get_tw_pe_ps_history(symbol, api_key):
    try:
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            'dataset': 'TaiwanStockPER',
            'data_id': symbol,
            'start_date': (datetime.now() - timedelta(days=365 * 5)).strftime('%Y-%m-%d'),
            'token': api_key
        }
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        result = resp.json()
        if result.get('status') != 200 or not result.get('data'):
            return None
        df = pd.DataFrame(result['data'])
        if 'date' not in df.columns:
            return None
        df['date'] = pd.to_datetime(df['date'])
        for col in ['PER', 'PBR', 'dividend_yield']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.sort_values('date').reset_index(drop=True)
        return {'ratios': df}
    except Exception:
        return None



# ─────────────────────────────────────────────────────────────────────────────
# 台股內部人資料來源 A：MOPS 公開資訊觀測站（按民國年度查詢）
# ─────────────────────────────────────────────────────────────────────────────

def get_mops_insider_changes(stock_code: str, year_roc: int = None) -> tuple:
    """
    公開資訊觀測站 — 董監事持股異動申報（ajax_t51sb06）
    year_roc：民國年，預設查當年度；同時也查上一年以補齊近6個月資料。
    回傳 (DataFrame, error_str | None)
    DataFrame 欄位統一為：申報日期、職稱、姓名、異動前持股數、異動後持股數、異動股數、買賣
    """
    if year_roc is None:
        year_roc = datetime.now().year - 1911

    _mops_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept":   "application/json, text/javascript, */*; q=0.01",
        "Referer":  "https://mops.twse.com.tw/mops/web/t51sb06",
        "X-Requested-With": "XMLHttpRequest",
    }

    def _fetch_year(yr):
        url = "https://mops.twse.com.tw/mops/web/ajax_t51sb06"
        form = {
            "encodeURIComponent": "1", "step": "1", "firstin": "1",
            "off": "1", "queryName": "co_id", "inpuType": "co_id",
            "TYPEK": "all", "isnew": "false",
            "co_id": str(stock_code), "year": str(yr),
        }
        try:
            resp = requests.post(url, data=form, headers=_mops_headers, timeout=20)
            if resp.status_code != 200:
                return pd.DataFrame(), f"MOPS HTTP {resp.status_code}"
            resp.encoding = "utf-8"
            if not _BS4_OK:
                return pd.DataFrame(), "缺少 beautifulsoup4（pip install beautifulsoup4）"
            soup = _BS4(resp.text, "html.parser")
            tables = soup.find_all("table")
            if not tables:
                return pd.DataFrame(), None   # 無資料（非錯誤）
            dfs = []
            for tbl in tables:
                try:
                    sub = pd.read_html(str(tbl))
                    for df in sub:
                        if df.shape[0] > 0 and df.shape[1] >= 4:
                            dfs.append(df)
                except Exception:
                    continue
            if not dfs:
                return pd.DataFrame(), None
            df = pd.concat(dfs, ignore_index=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [
                    " ".join(str(c) for c in col if "Unnamed" not in str(c)).strip()
                    for col in df.columns
                ]
            df.columns = [str(c).strip() for c in df.columns]
            df = df.dropna(how="all").reset_index(drop=True)
            return df, None
        except requests.exceptions.Timeout:
            return pd.DataFrame(), "MOPS 請求逾時"
        except requests.exceptions.ConnectionError:
            return pd.DataFrame(), "無法連線至 MOPS"
        except Exception as e:
            return pd.DataFrame(), str(e)

    # 查當年 + 上一年，合併後統一欄位
    all_dfs, last_err = [], None
    for yr in [year_roc, year_roc - 1]:
        df_yr, err = _fetch_year(yr)
        if not df_yr.empty:
            all_dfs.append(df_yr)
        if err:
            last_err = err

    if not all_dfs:
        return pd.DataFrame(), last_err or "MOPS 無申報資料"

    df = pd.concat(all_dfs, ignore_index=True).drop_duplicates()

    # 標準化欄位名稱
    col_map = {}
    for c in df.columns:
        cs = str(c).strip()
        if "申報" in cs and "日" in cs:   col_map[c] = "申報日期"
        elif "職稱" in cs or "職務" in cs: col_map[c] = "職稱"
        elif "姓名" in cs or "名稱" in cs: col_map[c] = "姓名"
        elif "異動前" in cs:               col_map[c] = "異動前持股數"
        elif "異動後" in cs:               col_map[c] = "異動後持股數"
        elif "異動" in cs and "股" in cs:  col_map[c] = "異動股數原始"
    if col_map:
        df = df.rename(columns=col_map)

    # 民國年 → 西元年
    def _roc_to_ad(s):
        try:
            parts = str(s).strip().split("/")
            if len(parts) == 3:
                return f"{int(parts[0]) + 1911}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
        except Exception:
            pass
        return s

    if "申報日期" in df.columns:
        df["申報日期"] = df["申報日期"].apply(_roc_to_ad)
        df["date"] = pd.to_datetime(df["申報日期"], errors="coerce")
    else:
        df["date"] = pd.NaT

    def _ci(v):
        try:
            return int(str(v).replace(",", "").replace("，", "").strip())
        except Exception:
            return 0

    for col in ["異動前持股數", "異動後持股數", "異動股數原始"]:
        if col in df.columns:
            df[col] = df[col].apply(_ci)

    if "異動前持股數" in df.columns and "異動後持股數" in df.columns:
        df["異動股數"] = df["異動後持股數"] - df["異動前持股數"]
    elif "異動股數原始" in df.columns:
        df["異動股數"] = df["異動股數原始"]
    else:
        df["異動股數"] = 0

    df["買賣"] = df["異動股數"].apply(
        lambda x: "🔴買入" if x > 0 else ("🟢賣出" if x < 0 else "－")
    )
    df = df.dropna(subset=["date"]).sort_values("date", ascending=False).reset_index(drop=True)
    return (df, None) if len(df) > 0 else (pd.DataFrame(), "MOPS 無有效申報記錄")


def get_tw_director_sharehold(symbol, api_key=None):
    """
    台股內部人持股異動 — 雙來源策略
    ────────────────────────────────
    主來源：MoneyDJ 董監質設異動清單（近3個月，HTML 爬取）
    備用來源：MOPS 公開資訊觀測站（當年度 + 上年度，POST API）

    防爬蟲措施（MoneyDJ）：
    - 隨機 User-Agent 輪換（6 組）
    - 隨機請求延遲 1.0~2.5 秒
    - Session 複用 + Cookie 暖身
    - Referer / DNT / Accept-Encoding 擬真標頭
    - WAF / Cloudflare 偵測後自動降級至 MOPS

    回傳 dict：
      {
        "moneydj": DataFrame | None,   # 近3個月申報（MoneyDJ）
        "mops":    DataFrame | None,   # 年度申報（MOPS，當年+上年）
        "source":  "moneydj" | "mops" | "both" | "none",
      }
    以 director_df["moneydj"] / director_df["mops"] 分別取用。
    """
    _USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    ]

    def _rh(referer="https://www.moneydj.com/"):
        return {
            "User-Agent":      random.choice(_USER_AGENTS),
            "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.7,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT":             "1",
            "Connection":      "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Referer":         referer,
            "Cache-Control":   "max-age=0",
        }

    def _jitter(lo=1.0, hi=2.5):
        time.sleep(random.uniform(lo, hi))

    def _decode(raw):
        for enc in ("big5", "cp950", "utf-8"):
            try:
                return raw.decode(enc, errors="replace")
            except Exception:
                pass
        return raw.decode("utf-8", errors="replace")

    def _fetch_moneydj():
        """爬取 MoneyDJ 個股董監頁，回傳 DataFrame | None"""
        if not _BS4_OK:
            return None
        try:
            session = requests.Session()
            session.max_redirects = 5
            try:
                session.get("https://www.moneydj.com/",
                            headers=_rh("https://www.google.com/"), timeout=10)
                _jitter(0.8, 1.8)
            except Exception:
                pass

            url = f"https://www.moneydj.com/z/zc/zck/zck_{symbol}.djhtm"
            resp = session.get(url, headers=_rh(), timeout=20)
            resp.raise_for_status()
            _jitter(0.5, 1.2)

            html = _decode(resp.content)
            if len(html) < 800:
                return None
            if any(kw in html for kw in ["Cloudflare", "Just a moment",
                                          "Enable JavaScript", "access denied"]):
                return None

            soup = _BS4(html, "html.parser")
            target_table, best_n = None, 0
            for tbl in soup.find_all("table"):
                txt = tbl.get_text()
                if any(kw in txt for kw in ["買進", "轉讓", "持股", "職稱", "申報"]):
                    n = len(tbl.find_all("tr"))
                    if n > best_n:
                        best_n, target_table = n, tbl

            if target_table is None or best_n < 3:
                return None

            rows = target_table.find_all("tr")
            ci = {k: None for k in ["date","role","name","before","after","buy","sell","pct"]}
            for i, cell in enumerate(rows[0].find_all(["th","td"])):
                t = cell.get_text(strip=True)
                if "申報" in t and "日" in t:           ci["date"]   = i
                elif "職" in t or "身份" in t:          ci["role"]   = i
                elif "姓名" in t or "代表" in t:         ci["name"]   = i
                elif "異動前" in t or "選任" in t:       ci["before"] = i
                elif "異動後" in t or "目前" in t:       ci["after"]  = i
                elif "買進" in t:                       ci["buy"]    = i
                elif "賣出" in t or "轉讓" in t:        ci["sell"]   = i
                elif "%" in t and "持股" in t:          ci["pct"]    = i

            def _ci_val(v):
                try:
                    return int(str(v).replace(",","").replace("，","")
                               .replace("-","0").strip() or "0")
                except Exception:
                    return 0

            now_yr = datetime.now().year
            cutoff = datetime.now() - timedelta(days=90)
            recs = []

            for row in rows[1:]:
                cells = row.find_all("td")
                if len(cells) < 3:
                    continue
                def _g(idx):
                    return cells[idx].get_text(strip=True) if idx is not None and idx < len(cells) else ""

                raw_d = _g(ci["date"]) if ci["date"] is not None else cells[0].get_text(strip=True)
                dt = None
                for fmt, s in [("%Y/%m/%d", raw_d), ("%Y/%m/%d", f"{now_yr}/{raw_d}")]:
                    try:
                        dt = datetime.strptime(s, fmt); break
                    except Exception:
                        pass
                if not dt or dt < cutoff:
                    continue

                buy_sh  = _ci_val(_g(ci["buy"]))    if ci["buy"]    is not None else 0
                sell_sh = _ci_val(_g(ci["sell"]))   if ci["sell"]   is not None else 0
                before  = _ci_val(_g(ci["before"])) if ci["before"] is not None else 0
                after   = _ci_val(_g(ci["after"]))  if ci["after"]  is not None else 0
                pct_str = _g(ci["pct"])             if ci["pct"]    is not None else ""

                if buy_sh > 0 or sell_sh > 0:
                    delta = (buy_sh - sell_sh) * 1000
                elif before > 0 or after > 0:
                    delta = (after - before) * 1000
                else:
                    continue
                if delta == 0:
                    continue

                recs.append({
                    "申報日期":    raw_d,
                    "職稱":       _g(ci["role"]),
                    "姓名":       _g(ci["name"]),
                    "異動前持股數": before * 1000,
                    "異動後持股數": after  * 1000,
                    "買進(千股)":  buy_sh,
                    "賣出(千股)":  sell_sh,
                    "異動股數":    delta,
                    "持股%":      pct_str,
                    "買賣":       "🔴買入" if delta > 0 else "🟢賣出",
                    "date":       dt,
                })

            if not recs:
                return None
            return (pd.DataFrame(recs)
                    .sort_values("date", ascending=False)
                    .reset_index(drop=True))

        except Exception:
            return None

    # ── 雙來源並行取得 ──────────────────────────────────────────
    moneydj_df = _fetch_moneydj()

    mops_df, _ = get_mops_insider_changes(symbol)
    if isinstance(mops_df, pd.DataFrame) and mops_df.empty:
        mops_df = None

    if moneydj_df is not None and mops_df is not None:
        source = "both"
    elif moneydj_df is not None:
        source = "moneydj"
    elif mops_df is not None:
        source = "mops"
    else:
        source = "none"

    return {"moneydj": moneydj_df, "mops": mops_df, "source": source}




def get_insider_trading(symbol, api_key):
    """美股內部人買賣（FMP API），近3個月，最新50筆"""
    try:
        url = "https://financialmodelingprep.com/stable/insider-trading"
        params = {'symbol': symbol, 'apikey': api_key, 'limit': 50}
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if not data or not isinstance(data, list):
            return None
        df = pd.DataFrame(data)
        keep = [c for c in ['transactionDate', 'reportingName', 'transactionType',
                             'securitiesTransacted', 'price', 'securitiesOwned',
                             'typeOfOwner'] if c in df.columns]
        if not keep:
            return None
        df = df[keep].copy()
        df['transactionDate'] = pd.to_datetime(df['transactionDate'], errors='coerce')
        df = df.dropna(subset=['transactionDate'])
        df = df.sort_values('transactionDate', ascending=False).reset_index(drop=True)
        cutoff = datetime.now() - timedelta(days=90)
        df = df[df['transactionDate'] >= cutoff]
        return df if len(df) > 0 else None
    except Exception:
        return None


def get_analyst_targets(symbol, api_key):
    result = {'targets': None, 'consensus': None}
    try:
        url = "https://financialmodelingprep.com/stable/price-target"
        params = {'symbol': symbol, 'apikey': api_key, 'limit': 20}
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if data and isinstance(data, list):
            df = pd.DataFrame(data)
            keep = [c for c in ['publishedDate', 'analystCompany', 'analystName',
                                 'priceTarget', 'priceWhenPosted', 'newsTitle'] if c in df.columns]
            if keep:
                df = df[keep].copy()
                df['publishedDate'] = pd.to_datetime(df['publishedDate'], errors='coerce')
                df = df.sort_values('publishedDate', ascending=False).head(20)
                result['targets'] = df
    except Exception:
        pass

    try:
        url2 = "https://financialmodelingprep.com/stable/price-target-consensus"
        params2 = {'symbol': symbol, 'apikey': api_key}
        resp2 = requests.get(url2, params=params2, timeout=20)
        resp2.raise_for_status()
        data2 = resp2.json()
        if data2 and isinstance(data2, list) and len(data2) > 0:
            result['consensus'] = data2[0]
        elif data2 and isinstance(data2, dict):
            result['consensus'] = data2
    except Exception:
        pass

    return result if (result['targets'] is not None or result['consensus'] is not None) else None


def get_analyst_targets_ai(symbol, openai_api_key, market='us', stock_name=None):
    """
    使用 OpenAI gpt-4o-mini + web_search 搜尋近一個月法人目標價新聞，
    並彙整為結構化表格（v4.4 規格）
    返回：dict { 'table': DataFrame or None, 'search_date': str }
    """
    try:
        from datetime import datetime
        import json

        today = datetime.now()
        search_date = today.strftime('%Y-%m-%d')
        yyyy_mm = today.strftime('%Y年%m月') if market == 'tw' else today.strftime('%Y-%m')

        if market == 'tw':
            name_part = f"{symbol} {stock_name}" if stock_name else symbol
            query = f"{name_part} 目標價 法人 {yyyy_mm}"
        else:
            query = f"{symbol} price target analyst {yyyy_mm}"

        client = __import__('openai').OpenAI(api_key=openai_api_key)

        system_msg = """你是一位專業的股票研究助理，負責從網路新聞中彙整法人目標價資訊。
請搜尋近一個月內各大券商、投行對指定股票的目標價報導，並以 JSON 格式回傳結構化資料。

回傳格式（僅回傳純 JSON，不含 markdown）：
{
  "targets": [
    {
      "institution": "券商/機構名稱",
      "target_price": "目標價（含幣別，如 NT$250 或 $185）",
      "rating": "評等（買入/中立/賣出/增持/買進 等，若無則填 N/A）",
      "date": "發布日期 YYYY-MM-DD（若不確定填 N/A）",
      "summary": "簡短說明依據（15字以內）"
    }
  ],
  "note": "若無資料則填 '近期無法人目標價資料'"
}

若找不到任何目標價資料，targets 請回傳空陣列 []。"""

        user_msg = f"請搜尋並彙整：{query}"

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            tools=[{"type": "web_search_preview"}],
            max_tokens=1500,
            temperature=0.2
        )

        # 解析回應
        content = ""
        for block in response.choices[0].message.content if isinstance(response.choices[0].message.content, list) else []:
            if hasattr(block, 'text'):
                content += block.text
        if not content:
            content = response.choices[0].message.content or ""

        # 清理 JSON fences
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[-2] if "```" in content else content
            content = content.replace("json", "", 1).strip()

        data = json.loads(content)
        targets = data.get("targets", [])

        if not targets:
            return {'table': None, 'search_date': search_date}

        df = pd.DataFrame(targets)
        df.columns = ['券商／機構名稱', '目標價', '評等', '更新日期', '來源摘要']
        return {'table': df, 'search_date': search_date}

    except Exception:
        return {'table': None, 'search_date': datetime.now().strftime('%Y-%m-%d')}


def get_pe_ps_history(symbol, api_key):
    """美股 P/E P/S P/B 歷年財務比率（FMP API）"""
    try:
        url = "https://financialmodelingprep.com/stable/ratios"
        params = {'symbol': symbol, 'apikey': api_key, 'period': 'annual', 'limit': 8}
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if not data or not isinstance(data, list):
            return None
        df = pd.DataFrame(data)
        keep = [c for c in ['date', 'priceEarningsRatio', 'priceToSalesRatio',
                             'priceToBookRatio', 'dividendYield'] if c in df.columns]
        if 'date' not in df.columns:
            return None
        df = df[keep].copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
        for col in keep:
            if col != 'date':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return {'ratios': df}
    except Exception:
        return None


# ─────────────────────────────────────────────
# 通用數據處理函數
# ─────────────────────────────────────────────

def filter_by_date_range(df, start_date, end_date):
    if df is None:
        return None
    mask = (df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))
    return df.loc[mask].copy().reset_index(drop=True)


def get_moving_averages(df):
    if df is None or len(df) == 0:
        return None
    df = df.copy()
    for period in [5, 10, 20, 60]:
        df[f'MA{period}'] = df['close'].rolling(window=period, min_periods=1).mean()
    return df


def calculate_rsi(df, period=14):
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
        avg_loss = avg_loss.replace(0, np.nan)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)
        df[f'RSI{period}'] = rsi
        return df
    except Exception:
        if df is not None:
            df = df.copy()
            df[f'RSI{period}'] = 50
        return df


def calculate_advanced_indicators(df):
    if df is None or len(df) < 30:
        return df
    df = df.copy()

    # MACD (12, 26, 9)
    try:
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD_DIF']  = ema12 - ema26
        df['MACD_LINE'] = df['MACD_DIF'].ewm(span=9, adjust=False).mean()
        df['MACD_HIST'] = (df['MACD_DIF'] - df['MACD_LINE']) * 2
        dif_prev  = df['MACD_DIF'].shift(1)
        line_prev = df['MACD_LINE'].shift(1)
        df['MACD_GOLDEN'] = (dif_prev < line_prev) & (df['MACD_DIF'] > df['MACD_LINE'])
    except Exception:
        pass

    # 布林通道 BB (20, 2SD)
    try:
        df['BB_MID']   = df['close'].rolling(20).mean()
        bb_std         = df['close'].rolling(20).std()
        df['BB_UPPER'] = df['BB_MID'] + 2 * bb_std
        df['BB_LOWER'] = df['BB_MID'] - 2 * bb_std
        df['BB_WIDTH'] = df['BB_UPPER'] - df['BB_LOWER']
    except Exception:
        pass

    # OBV
    try:
        direction = np.sign(df['close'].diff())
        direction.iloc[0] = 0
        df['OBV'] = (direction * df['volume']).cumsum()
    except Exception:
        pass

    # DMI (14日)
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


def generate_weekly_kline(df, rsi_period=14):
    """
    將日K線數據重新取樣生成周K線，並計算技術指標
    Returns:
        DataFrame or None
    """
    try:
        if df is None or len(df) < 10:
            return None
        df_temp = df.copy()
        df_temp = df_temp.set_index('date')
        df_weekly = df_temp.resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        df_weekly = df_weekly.reset_index()
        if len(df_weekly) < 5:
            return None

        # 計算MA
        for period in [5, 10, 20, 60]:
            df_weekly[f'MA{period}'] = df_weekly['close'].rolling(window=period, min_periods=1).mean()

        # 計算RSI
        df_weekly = calculate_rsi(df_weekly, period=rsi_period)

        # 計算MACD
        ema12 = df_weekly['close'].ewm(span=12, adjust=False).mean()
        ema26 = df_weekly['close'].ewm(span=26, adjust=False).mean()
        df_weekly['MACD_DIF']  = ema12 - ema26
        df_weekly['MACD_LINE'] = df_weekly['MACD_DIF'].ewm(span=9, adjust=False).mean()
        df_weekly['MACD_HIST'] = (df_weekly['MACD_DIF'] - df_weekly['MACD_LINE']) * 2
        dif_prev  = df_weekly['MACD_DIF'].shift(1)
        line_prev = df_weekly['MACD_LINE'].shift(1)
        df_weekly['MACD_GOLDEN'] = (dif_prev < line_prev) & (df_weekly['MACD_DIF'] > df_weekly['MACD_LINE'])

        # 布林通道
        df_weekly['BB_MID']   = df_weekly['close'].rolling(20, min_periods=5).mean()
        bb_std = df_weekly['close'].rolling(20, min_periods=5).std()
        df_weekly['BB_UPPER'] = df_weekly['BB_MID'] + 2 * bb_std
        df_weekly['BB_LOWER'] = df_weekly['BB_MID'] - 2 * bb_std

        return df_weekly
    except Exception:
        return None


# ─────────────────────────────────────────────
# 多頭訊號計算
# ─────────────────────────────────────────────

def calculate_bull_signals(df):
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
            obv_last     = df['OBV'].iloc[-1]
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

def create_candlestick_chart(df, symbol, rsi_period, currency_symbol,
                              institutional_df=None, market='us', selected_mas=None):
    """
    v3 主K線多層圖：K線+BB+可切換MA / RSI / OBV / 成交量 / 三大法人（台股）
    美股5層，台股6層
    """
    if selected_mas is None:
        selected_mas = ['MA5', 'MA10', 'MA20', 'MA60']

    show_inst = market == 'tw' and institutional_df is not None and len(institutional_df) > 0

    if show_inst:
        rows = 6
        row_heights = [0.38, 0.12, 0.12, 0.10, 0.15, 0.13]
        subplot_titles = (
            f'{symbol} K線 + 布林通道 + MA',
            f'RSI ({rsi_period}日)',
            'OBV 量能指標',
            '成交量',
            '三大法人買賣超（張）',
            '（預留）'
        )
        chart_height = 1200
    else:
        rows = 5
        row_heights = [0.45, 0.15, 0.13, 0.12, 0.15]
        subplot_titles = (
            f'{symbol} K線 + 布林通道 + MA',
            f'RSI ({rsi_period}日)',
            'OBV 量能指標',
            '成交量',
            '（預留）'
        )
        chart_height = 1100

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=subplot_titles,
        row_heights=row_heights
    )

    # ── Row 1：布林通道填色 + BB三軌 + K線 + 可切換MA ──

    # 1a. BB 上下軌填色
    if 'BB_UPPER' in df.columns and 'BB_LOWER' in df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([df['date'], df['date'][::-1]]),
            y=pd.concat([df['BB_UPPER'], df['BB_LOWER'][::-1]]),
            fill='toself',
            fillcolor='rgba(100,149,237,0.08)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False, name='BB 通道填色',
            hoverinfo='skip'
        ), row=1, col=1)

    # 1b. BB 三軌
    if 'BB_UPPER' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['BB_UPPER'],
            mode='lines', name='BB上軌',
            line=dict(color='#e74c3c', dash='dash', width=1.2)
        ), row=1, col=1)
    if 'BB_MID' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['BB_MID'],
            mode='lines', name='BB中軌',
            line=dict(color='#1e90ff', width=2)
        ), row=1, col=1)
    if 'BB_LOWER' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['BB_LOWER'],
            mode='lines', name='BB下軌',
            line=dict(color='#2ecc71', dash='dash', width=1.2)
        ), row=1, col=1)

    # 1c. BB 壓縮期橙色矩形
    if 'BB_WIDTH' in df.columns:
        w_mean = df['BB_WIDTH'].mean()
        compressed = df[df['BB_WIDTH'] < w_mean * 0.5].copy()
        if len(compressed) > 0:
            compressed['gap'] = (compressed.index.to_series().diff() > 1)
            block_id = compressed['gap'].cumsum()
            for _, block in compressed.groupby(block_id):
                x0 = block['date'].iloc[0]
                x1 = block['date'].iloc[-1]
                fig.add_vrect(x0=x0, x1=x1,
                              fillcolor='rgba(255,165,0,0.12)',
                              line_width=0, row=1, col=1)

    # 1d. K線
    fig.add_trace(go.Candlestick(
        x=df['date'],
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name='K線圖',
        increasing_line_color='#ff4757',
        decreasing_line_color='#2ed573',
        increasing_fillcolor='#ff4757',
        decreasing_fillcolor='#2ed573'
    ), row=1, col=1)

    # 1e. 可切換MA
    ma_colors = {'MA5': '#ff6b6b', 'MA10': '#4ecdc4', 'MA20': '#45b7d1', 'MA60': '#96ceb4'}
    for ma in ['MA5', 'MA10', 'MA20', 'MA60']:
        if ma in selected_mas and ma in df.columns:
            fig.add_trace(go.Scatter(
                x=df['date'], y=df[ma],
                mode='lines', name=ma,
                line=dict(color=ma_colors[ma], width=2)
            ), row=1, col=1)

    # ── Row 2：RSI ──
    rsi_col = f'RSI{rsi_period}'
    if rsi_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df[rsi_col],
            mode='lines', name=f'RSI({rsi_period})',
            line=dict(color='#1e90ff', width=2)
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=[df['date'].iloc[0], df['date'].iloc[-1]], y=[70, 70],
            mode='lines', name='超買(70)',
            line=dict(color='red', dash='dash', width=1), showlegend=False
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=[df['date'].iloc[0], df['date'].iloc[-1]], y=[30, 30],
            mode='lines', name='超賣(30)',
            line=dict(color='green', dash='dash', width=1), showlegend=False
        ), row=2, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor='rgba(255,71,87,0.08)', line_width=0, row=2, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor='rgba(46,213,115,0.08)', line_width=0, row=2, col=1)
        fig.update_yaxes(range=[0, 100], row=2, col=1)

    # ── Row 3：OBV ──
    if 'OBV' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['OBV'],
            mode='lines', name='OBV',
            line=dict(color='#9b59b6', width=2)
        ), row=3, col=1)

    # ── Row 4：成交量 ──
    fig.add_trace(go.Bar(
        x=df['date'], y=df['volume'],
        name='成交量',
        marker_color='#a55eea', opacity=0.6
    ), row=4, col=1)

    # ── Row 5（台股）：三大法人買賣超 ──
    if show_inst:
        inst_colors = {
            'Foreign_Investor': '#e74c3c',
            'Investment_Trust': '#3498db',
            'Dealer_self': '#2ecc71',
            'Dealer_Hedging': '#1abc9c',
            'Dealer': '#27ae60',
            'Total': '#f39c12'
        }
        inst_labels = {
            'Foreign_Investor': '外資',
            'Investment_Trust': '投信',
            'Dealer_self': '自營商(自行買賣)',
            'Dealer_Hedging': '自營商(避險)',
            'Dealer': '自營商合計',
            'Total': '三大法人合計'
        }
        for name_key, label in inst_labels.items():
            sub = institutional_df[institutional_df['name'] == name_key].copy()
            if sub.empty:
                continue
            base_color = inst_colors.get(name_key, '#e74c3c')
            colors = [base_color if v >= 0 else '#95a5a6' for v in sub['net']]
            fig.add_trace(go.Bar(
                x=sub['date'], y=sub['net'],
                name=label,
                marker_color=colors,
                opacity=0.75,
                visible='legendonly' if name_key == 'Total' else True
            ), row=5, col=1)

    # ── 佈局更新 ──
    fig.update_layout(
        title=f'{symbol} 主K線圖（含布林通道、MA、RSI、OBV）',
        height=chart_height,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white'
    )
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text=f"價格 ({currency_symbol})", row=1, col=1)
    fig.update_yaxes(title_text=f"RSI({rsi_period})", row=2, col=1)
    fig.update_yaxes(title_text="OBV", row=3, col=1)
    fig.update_yaxes(title_text="成交量", row=4, col=1)
    if show_inst:
        fig.update_yaxes(title_text="買賣超（張）", row=5, col=1)

    return fig


def create_weekly_chart(df_weekly, symbol, selected_mas=None):
    """
    周K線圖（圖2）：上層周K線+MA+BB，下層MACD
    """
    if df_weekly is None or len(df_weekly) < 5:
        return None
    if selected_mas is None:
        selected_mas = ['MA5', 'MA10', 'MA20', 'MA60']

    try:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{symbol} 周K線 + MA + 布林通道', 'MACD（12, 26, 9）'),
            row_heights=[0.60, 0.40]
        )

        # ── 上層：布林通道填色 ──
        if 'BB_UPPER' in df_weekly.columns and 'BB_LOWER' in df_weekly.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([df_weekly['date'], df_weekly['date'][::-1]]),
                y=pd.concat([df_weekly['BB_UPPER'], df_weekly['BB_LOWER'][::-1]]),
                fill='toself', fillcolor='rgba(100,149,237,0.08)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False, hoverinfo='skip'
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df_weekly['date'], y=df_weekly['BB_UPPER'],
                mode='lines', name='週BB上軌',
                line=dict(color='#e74c3c', dash='dash', width=1)
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df_weekly['date'], y=df_weekly['BB_MID'],
                mode='lines', name='週BB中軌',
                line=dict(color='#1e90ff', width=1.5)
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df_weekly['date'], y=df_weekly['BB_LOWER'],
                mode='lines', name='週BB下軌',
                line=dict(color='#2ecc71', dash='dash', width=1)
            ), row=1, col=1)

        # 周K線
        fig.add_trace(go.Candlestick(
            x=df_weekly['date'],
            open=df_weekly['open'], high=df_weekly['high'],
            low=df_weekly['low'], close=df_weekly['close'],
            name='周K線',
            increasing_line_color='#ff4757',
            decreasing_line_color='#2ed573'
        ), row=1, col=1)

        # 可切換MA
        ma_colors = {'MA5': '#ff6b6b', 'MA10': '#4ecdc4', 'MA20': '#45b7d1', 'MA60': '#96ceb4'}
        for ma in ['MA5', 'MA10', 'MA20', 'MA60']:
            if ma in selected_mas and ma in df_weekly.columns:
                fig.add_trace(go.Scatter(
                    x=df_weekly['date'], y=df_weekly[ma],
                    mode='lines', name=f'週{ma}',
                    line=dict(color=ma_colors[ma], width=1.5)
                ), row=1, col=1)

        # ── 下層：MACD ──
        if all(c in df_weekly.columns for c in ['MACD_DIF', 'MACD_LINE', 'MACD_HIST']):
            fig.add_trace(go.Scatter(
                x=df_weekly['date'], y=df_weekly['MACD_DIF'],
                mode='lines', name='週DIF',
                line=dict(color='#ff6b35', width=2)
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=df_weekly['date'], y=df_weekly['MACD_LINE'],
                mode='lines', name='週MACD訊號線',
                line=dict(color='#1e90ff', width=2)
            ), row=2, col=1)

            hist_colors = ['#ff4757' if v >= 0 else '#2ed573' for v in df_weekly['MACD_HIST']]
            fig.add_trace(go.Bar(
                x=df_weekly['date'], y=df_weekly['MACD_HIST'],
                name='週MACD HIST', marker_color=hist_colors, opacity=0.8
            ), row=2, col=1)

            fig.add_hline(y=0, line_dash='dash', line_color='gray', line_width=1, row=2, col=1)

            if 'MACD_GOLDEN' in df_weekly.columns:
                golden_df = df_weekly[df_weekly['MACD_GOLDEN']]
                if not golden_df.empty:
                    fig.add_trace(go.Scatter(
                        x=golden_df['date'], y=golden_df['MACD_DIF'],
                        mode='markers', name='週金叉',
                        marker=dict(symbol='triangle-up', size=10, color='#f39c12')
                    ), row=2, col=1)

        fig.update_layout(
            title=f'{symbol} 周K線圖與MACD指標',
            height=700,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(rangeslider_visible=False)
        return fig
    except Exception:
        return None


def create_margin_chart(margin_df, symbol):
    """融資融券雙Y軸折線圖"""
    try:
        has_margin = 'MarginPurchaseRemaining' in margin_df.columns and margin_df['MarginPurchaseRemaining'].notna().sum() > 0
        has_short  = 'ShortSaleRemaining' in margin_df.columns and margin_df['ShortSaleRemaining'].notna().sum() > 0
        if not has_margin and not has_short:
            return None

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        if has_margin:
            fig.add_trace(go.Scatter(
                x=margin_df['date'], y=margin_df['MarginPurchaseRemaining'],
                mode='lines', name='融資餘額',
                line=dict(color='#e74c3c', width=2)
            ), secondary_y=False)
        if has_short:
            fig.add_trace(go.Scatter(
                x=margin_df['date'], y=margin_df['ShortSaleRemaining'],
                mode='lines', name='融券餘額',
                line=dict(color='#3498db', width=2)
            ), secondary_y=True)

        fig.update_layout(
            title=f'{symbol} 融資融券餘額走勢',
            height=350,
            template='plotly_white'
        )
        fig.update_yaxes(title_text="融資餘額（張）", secondary_y=False)
        fig.update_yaxes(title_text="融券餘額（張）", secondary_y=True)
        return fig
    except Exception:
        return None


def display_institutional_table(institutional_df):
    """
    三大法人買賣超表格（v4.4 規格 Step 1-6）
    - 只保留 5 個 name（排除 Dealer 合計避免重複）
    - pivot 轉寬表，中文欄位，日期降序，最近 10 交易日
    - 最下方附 10日合計列
    - Total 直接使用 FinMind 官方合計，不自行加總
    """
    try:
        df = institutional_df.copy()

        # Step 2: 確保 net 欄位
        if 'net' not in df.columns:
            df['net'] = df['buy'] - df['sell']

        # Step 3: 篩選法人與日期
        target_names = ['Foreign_Investor', 'Investment_Trust',
                        'Dealer_self', 'Dealer_Hedging', 'Total']
        df = df[df['name'].isin(target_names)]
        top10_dates = sorted(df['date'].unique())[-10:]
        df = df[df['date'].isin(top10_dates)]

        # Step 4: pivot 轉寬表
        pivot = df.pivot_table(
            index='date',
            columns='name',
            values='net',
            aggfunc='sum'
        ).reset_index()

        pivot.rename(columns={
            'date':             '日期',
            'Foreign_Investor': '外資(張)',
            'Investment_Trust': '投信(張)',
            'Dealer_self':      '自營商-自行(張)',
            'Dealer_Hedging':   '自營商-避險(張)',
            'Total':            '三大法人合計(張)'
        }, inplace=True)

        pivot = pivot.sort_values('日期', ascending=False).reset_index(drop=True)
        pivot['日期'] = pd.to_datetime(pivot['日期']).dt.strftime('%Y-%m-%d')

        # Step 5: 10日合計列
        num_cols = [c for c in pivot.columns if c != '日期']
        sum_row = pivot[num_cols].sum(numeric_only=True).to_dict()
        sum_row['日期'] = '10日合計'
        pivot = pd.concat([pivot, pd.DataFrame([sum_row])], ignore_index=True)

        # 整數格式
        for col in num_cols:
            if col in pivot.columns:
                pivot[col] = pivot[col].apply(
                    lambda x: int(x) if pd.notna(x) and x != '' else 0
                )

        # Step 6: 顯示
        st.dataframe(pivot, use_container_width=True, hide_index=True)

    except Exception:
        try:
            simple = institutional_df.tail(40).copy()
            simple['date'] = simple['date'].dt.strftime('%Y-%m-%d')
            st.dataframe(simple[['date', 'name', 'buy', 'sell', 'net']],
                         use_container_width=True, hide_index=True)
        except Exception:
            pass


def display_broker_table(broker_data, symbol, query_date=None):
    """
    券商分點進出明細 — 仿照富邦/元大分點查詢格式
    接收 get_tw_broker_trading() 回傳的 dict

    Args:
        broker_data: dict { date, buy_df, sell_df, total_buy, total_sell }
                     或舊格式 DataFrame（兼容）
        symbol: 股票代碼
        query_date: 日期字串（可由 broker_data['date'] 自動帶入）
    """
    # 兼容舊格式 DataFrame
    if isinstance(broker_data, pd.DataFrame):
        if broker_data.empty:
            st.info("ℹ️ 目前無券商分點資料（可能為非交易日或資料尚未更新）。")
            return
        total_vol = broker_data['buy'].sum() + broker_data['sell'].sum()
        buy_df  = broker_data[broker_data['net'] > 0].sort_values('net', ascending=False).head(20).reset_index(drop=True)
        sell_df = broker_data[broker_data['net'] < 0].copy()
        sell_df['net'] = sell_df['net'].abs()
        sell_df = sell_df.sort_values('net', ascending=False).head(20).reset_index(drop=True)
        def _ratio(r):
            return f"{(r['buy']+r['sell'])/total_vol*100:.2f}%" if total_vol > 0 else '—'
        buy_df['ratio']  = buy_df.apply(_ratio, axis=1)
        sell_df['ratio'] = sell_df.apply(_ratio, axis=1)
        total_buy  = int(broker_data['buy'].sum())
        total_sell = int(broker_data['sell'].sum())
        date_str   = query_date or datetime.now().strftime('%Y-%m-%d')
    elif isinstance(broker_data, dict):
        if not broker_data:
            st.info("ℹ️ 目前無券商分點資料（可能為非交易日或資料尚未更新）。")
            return
        buy_df     = broker_data.get('buy_df', pd.DataFrame())
        sell_df    = broker_data.get('sell_df', pd.DataFrame())
        total_buy  = broker_data.get('total_buy', 0)
        total_sell = broker_data.get('total_sell', 0)
        date_str   = broker_data.get('date', query_date or datetime.now().strftime('%Y-%m-%d'))
        # 確保有 ratio 欄位
        total_vol = total_buy + total_sell
        if 'ratio' not in buy_df.columns:
            def _ratio(r):
                return f"{(r['buy']+r['sell'])/total_vol*100:.2f}%" if total_vol > 0 else '—'
            if not buy_df.empty:  buy_df['ratio']  = buy_df.apply(_ratio, axis=1)
            if not sell_df.empty: sell_df['ratio'] = sell_df.apply(_ratio, axis=1)
        else:
            buy_df['ratio']  = buy_df['ratio'].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else str(x))
            sell_df['ratio'] = sell_df['ratio'].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else str(x))
    else:
        st.info("ℹ️ 目前無券商分點資料（可能為非交易日或資料尚未更新）。")
        return

    if buy_df.empty and sell_df.empty:
        st.info("ℹ️ 目前無券商分點資料（可能為非交易日或資料尚未更新）。")
        return

    try:
        total_vol  = total_buy + total_sell
        date_disp  = date_str.replace('-', '/')

        def fmt_num(v):
            try:    return f"{int(v):,}"
            except: return str(v)

        max_rows = max(len(buy_df), len(sell_df))

        html = f"""
<style>
.broker-wrap {{ overflow-x: auto; }}
.broker-table {{
    width: 100%; border-collapse: collapse;
    font-size: 13px;
    font-family: 'Microsoft JhengHei', '微軟正黑體', Arial, sans-serif;
}}
.broker-table th {{
    background-color: #d0e8f5; padding: 5px 10px;
    border: 1px solid #aac8d8; text-align: center;
    font-weight: bold; white-space: nowrap;
}}
.broker-table td {{
    padding: 4px 10px; border: 1px solid #dde;
    text-align: right; white-space: nowrap;
}}
.broker-table td.name-cell {{
    text-align: left; color: #1a5fa8; font-weight: 500; min-width: 110px;
}}
.broker-table tr:nth-child(even) {{ background-color: #f5faff; }}
.broker-table tr:hover {{ background-color: #e8f4ff; }}
.buy-net  {{ color: #cc0000; font-weight: bold; }}
.sell-net {{ color: #007700; font-weight: bold; }}
.total-row td {{
    font-weight: bold; background-color: #ddeedd !important;
    border-top: 2px solid #88aa88;
}}
.avg-row td {{
    font-weight: bold; background-color: #eef3ff !important;
}}
.divider-col {{
    width: 8px; min-width: 8px; background: #e8eef4; border: none !important; padding: 0 !important;
}}
.broker-header {{
    background: #e8f4f8; padding: 6px 14px; border-radius: 4px;
    font-size: 13px; margin-bottom: 8px; border-left: 4px solid #3498db;
}}
</style>

<div class="broker-header">
  <b>📊 {symbol} 券商分點 - 進出明細</b>&ensp;
  ｜&ensp;單位：張&ensp;
  ｜&ensp;最後更新日：{date_disp}&ensp;
  ｜&ensp;全體合計買進：{fmt_num(total_buy)}　賣出：{fmt_num(total_sell)}
</div>

<div class="broker-wrap">
<table class="broker-table">
  <thead>
    <tr>
      <th colspan="5" style="background:#fde8e8; color:#800000; font-size:14px;">🔴 買超券商（前20）</th>
      <th class="divider-col"></th>
      <th colspan="5" style="background:#e8fde8; color:#006600; font-size:14px;">🟢 賣超券商（前20）</th>
    </tr>
    <tr>
      <th>買超券商</th><th>買進</th><th>賣出</th><th style="color:#cc0000;">買超</th><th>估成交<br>比重</th>
      <th class="divider-col"></th>
      <th>賣超券商</th><th>買進</th><th>賣出</th><th style="color:#007700;">賣超</th><th>估成交<br>比重</th>
    </tr>
  </thead>
  <tbody>
"""
        for i in range(max_rows):
            if i < len(buy_df):
                b = buy_df.iloc[i]
                b_name  = str(b['broker_name'])[:14]
                b_ratio = str(b.get('ratio', '—'))
                buy_cells = (
                    f'<td class="name-cell">{b_name}</td>'
                    f'<td>{fmt_num(b["buy"])}</td>'
                    f'<td>{fmt_num(b["sell"])}</td>'
                    f'<td class="buy-net">{fmt_num(b["net"])}</td>'
                    f'<td>{b_ratio}</td>'
                )
            else:
                buy_cells = '<td></td><td></td><td></td><td></td><td></td>'

            if i < len(sell_df):
                s = sell_df.iloc[i]
                s_name  = str(s['broker_name'])[:14]
                s_ratio = str(s.get('ratio', '—'))
                sell_cells = (
                    f'<td class="name-cell">{s_name}</td>'
                    f'<td>{fmt_num(s["buy"])}</td>'
                    f'<td>{fmt_num(s["sell"])}</td>'
                    f'<td class="sell-net">{fmt_num(s["net"])}</td>'
                    f'<td>{s_ratio}</td>'
                )
            else:
                sell_cells = '<td></td><td></td><td></td><td></td><td></td>'

            html += f'    <tr>{buy_cells}<td class="divider-col"></td>{sell_cells}</tr>\n'

        # 合計列
        buy_net_total  = buy_df['net'].sum()  if not buy_df.empty  else 0
        sell_net_total = sell_df['net'].sum() if not sell_df.empty else 0
        buy_total_ratio_str  = f"{buy_net_total  / total_vol * 100:.2f}%" if total_vol > 0 else '—'
        sell_total_ratio_str = f"{sell_net_total / total_vol * 100:.2f}%" if total_vol > 0 else '—'

        # 計算平均買進成本（加權平均）
        def avg_cost(df_side, col):
            total = df_side[col].sum()
            if total > 0:
                return '—'  # 無價格資料時顯示 —
            return '—'

        html += f"""
    <tr class="total-row">
      <td class="name-cell">合計買超張數</td>
      <td colspan="2" style="text-align:center;">（上述{len(buy_df)}家）</td>
      <td class="buy-net">{fmt_num(buy_net_total)}</td>
      <td>{buy_total_ratio_str}</td>
      <td class="divider-col"></td>
      <td class="name-cell">合計賣超張數</td>
      <td colspan="2" style="text-align:center;">（上述{len(sell_df)}家）</td>
      <td class="sell-net">{fmt_num(sell_net_total)}</td>
      <td>{sell_total_ratio_str}</td>
    </tr>
  </tbody>
</table>
</div>
<div style="font-size:11px; color:#777; margin-top:6px; padding:0 4px;">
  【註1】合計買超或賣超，為上述家數之合計。&emsp;
  【註2】估成交比重 = 該券商(買進+賣出) ÷ 全體券商(買進+賣出)合計。
</div>
"""
        st.markdown(html, unsafe_allow_html=True)

    except Exception:
        try:
            # 降級顯示
            if not buy_df.empty:
                st.markdown("**🔴 買超券商**")
                st.dataframe(buy_df.head(20), use_container_width=True, hide_index=True)
            if not sell_df.empty:
                st.markdown("**🟢 賣超券商**")
                st.dataframe(sell_df.head(20), use_container_width=True, hide_index=True)
        except Exception:
            pass




def create_financial_bar_chart(financial_data, price_df, symbol):
    """
    近期季度財務數據長條圖（5年20季度，含股價趨勢線）
    """
    if financial_data is None or financial_data.get('quarterly') is None:
        return None
    try:
        q_df = financial_data['quarterly'].copy()
        if q_df is None or len(q_df) == 0:
            return None

        q_df['date'] = pd.to_datetime(q_df['date'], errors='coerce')
        q_df = q_df.dropna(subset=['date'])
        q_df['value'] = pd.to_numeric(q_df['value'], errors='coerce')

        # 季度標籤
        q_df['quarter'] = q_df['date'].dt.year.astype(str) + '-Q' + q_df['date'].dt.quarter.astype(str)

        type_colors = {
            'OperatingIncome': '#3498db',
            'Revenue': '#2ecc71',
            'GrossProfit': '#9b59b6',
            'EPS': '#e74c3c',
            'NetIncome': '#f39c12'
        }
        type_names = {
            'OperatingIncome': '營業利益',
            'Revenue': '營業收入',
            'GrossProfit': '營業毛利',
            'EPS': '基本每股盈餘(EPS)',
            'NetIncome': '淨利'
        }

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        has_data = False
        for t in ['Revenue', 'GrossProfit', 'OperatingIncome', 'NetIncome', 'EPS']:
            sub = q_df[q_df['type'] == t].sort_values('date').tail(20)
            if sub.empty:
                continue
            has_data = True
            fig.add_trace(go.Bar(
                x=sub['quarter'], y=sub['value'],
                name=type_names.get(t, t),
                marker_color=type_colors.get(t, '#7f8c8d'),
                opacity=0.8
            ), secondary_y=False)

        if not has_data:
            return None

        # 股價趨勢線（每季末收盤）
        if price_df is not None and len(price_df) > 0:
            try:
                p_copy = price_df.copy()
                p_copy['quarter'] = p_copy['date'].dt.year.astype(str) + '-Q' + p_copy['date'].dt.quarter.astype(str)
                quarterly_price = p_copy.groupby('quarter')['close'].last().reset_index()
                quarterly_price = quarterly_price.sort_values('quarter').tail(20)
                fig.add_trace(go.Scatter(
                    x=quarterly_price['quarter'], y=quarterly_price['close'],
                    mode='lines+markers', name='季末股價',
                    line=dict(color='#e74c3c', width=2),
                    marker=dict(size=5)
                ), secondary_y=True)
            except Exception:
                pass

        fig.update_layout(
            title=f'{symbol} 近期季度財務數據比較（5年）',
            height=600,
            template='plotly_white',
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_yaxes(title_text="財務數值（千元/元）", secondary_y=False)
        fig.update_yaxes(title_text="股價（NT$）", secondary_y=True)
        return fig
    except Exception:
        return None


def create_insider_chart(insider_df, symbol):
    """內部人買賣橫條圖"""
    if insider_df is None or len(insider_df) == 0:
        return None
    try:
        df = insider_df.head(20).copy()
        buy_types = ['P-Purchase', 'Buy', 'A-Award', 'M-Exempt', 'P - Purchase']
        shares_col = next((c for c in ['securitiesTransacted', 'shares'] if c in df.columns), None)
        type_col   = next((c for c in ['transactionType', 'type'] if c in df.columns), None)
        date_col   = next((c for c in ['transactionDate', 'date'] if c in df.columns), None)
        name_col   = next((c for c in ['reportingName', 'name'] if c in df.columns), None)

        if shares_col is None:
            return None

        df['shares'] = pd.to_numeric(df[shares_col], errors='coerce').fillna(0)
        df['is_buy'] = df[type_col].str.contains('P-Purchase|Purchase|Buy|Award', case=False, na=False) if type_col else True
        df['signed_shares'] = df.apply(lambda r: r['shares'] if r['is_buy'] else -r['shares'], axis=1)

        def make_label(r):
            n = str(r[name_col])[:18] if name_col else '未知'
            if date_col and pd.notna(r[date_col]):
                try:
                    return n + '  ' + pd.to_datetime(r[date_col]).strftime('%m/%d')
                except Exception:
                    pass
            return n
        df['label'] = df.apply(make_label, axis=1)
        df['color'] = df['is_buy'].map({True: '#2ecc71', False: '#e74c3c'})
        df = df.sort_values(date_col if date_col else df.columns[0], ascending=True)

        fig = go.Figure(go.Bar(
            x=df['signed_shares'],
            y=df['label'],
            orientation='h',
            marker_color=df['color'],
            text=df.apply(lambda r: f"{'買進' if r['is_buy'] else '賣出'} {abs(r['shares']):,.0f}股", axis=1),
            textposition='outside',
            hovertemplate='%{y}<br>%{x:,.0f} 股<extra></extra>'
        ))
        fig.update_layout(
            title=f'{symbol} 內部人買賣（近3個月，正值=買進，負值=賣出）',
            height=max(350, len(df) * 28 + 120),
            template='plotly_white',
            xaxis_title='交易股數',
            xaxis=dict(tickformat=','),
            margin=dict(l=200)
        )
        fig.add_vline(x=0, line_dash='dash', line_color='gray', line_width=1)
        return fig
    except Exception:
        return None


def create_analyst_chart(analyst_data, symbol, currency_symbol, current_price):
    """法人目標價散點圖"""
    if analyst_data is None:
        return None
    targets_df = analyst_data.get('targets')
    consensus  = analyst_data.get('consensus')
    if targets_df is None or len(targets_df) == 0:
        return None
    try:
        df = targets_df.copy()
        df['priceTarget'] = pd.to_numeric(df['priceTarget'], errors='coerce')
        df = df.dropna(subset=['priceTarget'])
        if len(df) == 0:
            return None

        df['label'] = df.apply(lambda r: (
            r.get('analystCompany', '未知')[:20] + ' ' +
            (r['publishedDate'].strftime('%Y/%m/%d') if pd.notna(r['publishedDate']) else '')
        ), axis=1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['publishedDate'],
            y=df['priceTarget'],
            mode='markers+text',
            marker=dict(size=10, color='#3498db', symbol='diamond', line=dict(color='white', width=1)),
            text=df['label'],
            textposition='top center',
            name='目標價',
            hovertemplate='%{text}<br>目標價：' + currency_symbol + '%{y:.2f}<extra></extra>'
        ))

        fig.add_hline(y=current_price, line_dash='dash', line_color='#e74c3c',
                      annotation_text=f'現價 {currency_symbol}{current_price:.2f}',
                      annotation_position='right')

        if consensus:
            target_mean = consensus.get('targetConsensus') or consensus.get('targetMean') or consensus.get('priceTarget')
            if target_mean:
                target_mean = float(target_mean)
                fig.add_hline(y=target_mean, line_dash='dot', line_color='#f39c12',
                              annotation_text=f'共識目標 {currency_symbol}{target_mean:.2f}',
                              annotation_position='left')

        avg_target = df['priceTarget'].mean()
        upside = ((avg_target - current_price) / current_price * 100) if current_price > 0 else 0
        title_extra = f'  |  平均目標 {currency_symbol}{avg_target:.2f}（{upside:+.1f}%）'

        fig.update_layout(
            title=f'{symbol} 法人目標價分布' + title_extra,
            height=450,
            template='plotly_white',
            yaxis_title=f'目標價（{currency_symbol}）',
            showlegend=False
        )
        return fig
    except Exception:
        return None


def create_pe_ps_chart(pe_data, symbol, price_df):
    """
    P/E 等估值歷年趨勢圖 — 6張分開子圖，共享X軸季度
    每張子圖：估值指標折線 + 股價折線（右軸），X軸季度格式
    每季取最高值，每點顯示數值標籤
    """
    if pe_data is None or pe_data.get('ratios') is None:
        return None
    try:
        ratios = pe_data['ratios'].copy()

        pe_col  = next((c for c in ['PER', 'priceEarningsRatio'] if c in ratios.columns), None)
        ps_col  = next((c for c in ['priceToSalesRatio'] if c in ratios.columns), None)
        pb_col  = next((c for c in ['PBR', 'priceToBookRatio'] if c in ratios.columns), None)
        dy_col  = next((c for c in ['dividend_yield', 'dividendYield'] if c in ratios.columns), None)
        peg_col = next((c for c in ['priceEarningsToGrowthRatio', 'pegRatio', 'PEG'] if c in ratios.columns), None)

        if pe_col is None and ps_col is None and pb_col is None:
            return None

        # 按季度重新取樣，每季取最高值
        ratios['quarter'] = pd.to_datetime(ratios['date']).dt.to_period('Q').astype(str)
        ratios['quarter'] = ratios['quarter'].str.replace('Q', '-Q')
        agg_cols = {c: 'max' for c in ratios.columns if c not in ['date', 'quarter']}
        ratios = ratios.groupby('quarter', as_index=False).agg(agg_cols)
        ratios = ratios.sort_values('quarter').reset_index(drop=True)
        x_vals = ratios['quarter']

        # 季末股價（季度版）
        quarterly_close = None
        if price_df is not None and len(price_df) > 0:
            p_copy = price_df.copy()
            p_copy['quarter'] = p_copy['date'].dt.to_period('Q').astype(str)
            p_copy['quarter'] = p_copy['quarter'].str.replace('Q', '-Q')
            qp = p_copy.groupby('quarter')['close'].last().reset_index()
            qp = qp.sort_values('quarter').reset_index(drop=True)
            # 對齊 ratios 的季度
            qp = qp[qp['quarter'].isin(x_vals)].reset_index(drop=True)
            quarterly_close = qp

        # 6張子圖，每張雙Y軸（左=估值，右=股價）
        specs = [[{"secondary_y": True}]] * 6
        fig = make_subplots(
            rows=6, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=(
                'PE（本益比）',
                'PEG（本益成長比）',
                'PS（股價銷售比）',
                'PBR（股價淨值比）',
                'PER / 殖利率（%）',
                '季末股價'
            ),
            specs=specs,
            row_heights=[1/6]*6
        )

        def add_val_line(row, y, name, color, fmt='.1f'):
            """估值指標折線（左軸）+ 每季數值標籤"""
            y_series = pd.Series(y).reset_index(drop=True)
            text_vals = [f"{v:{fmt}}" if pd.notna(v) else '' for v in y_series]
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_series,
                mode='lines+markers+text',
                name=name,
                line=dict(color=color, width=2),
                marker=dict(size=5),
                text=text_vals,
                textposition='top center',
                textfont=dict(size=8, color=color),
                showlegend=(row == 1)  # legend 只在第一張顯示一次
            ), row=row, col=1, secondary_y=False)

        def add_price_line(row, qc):
            """股價折線（右軸）每季數值標籤"""
            if qc is None or len(qc) == 0:
                return
            y_p = qc['close']
            text_p = [f"{v:.0f}" if pd.notna(v) else '' for v in y_p]
            fig.add_trace(go.Scatter(
                x=qc['quarter'], y=y_p,
                mode='lines+markers+text',
                name='季末股價',
                line=dict(color='#7f8c8d', width=1.5, dash='dot'),
                marker=dict(size=4),
                text=text_p,
                textposition='bottom center',
                textfont=dict(size=8, color='#7f8c8d'),
                showlegend=(row == 1)
            ), row=row, col=1, secondary_y=True)
            fig.update_yaxes(title_text='股價', secondary_y=True, row=row, col=1)

        # Row 1: PE
        if pe_col and ratios[pe_col].notna().sum() > 1:
            add_val_line(1, ratios[pe_col], 'PE 本益比', '#e74c3c', fmt='.1f')
            add_price_line(1, quarterly_close)
            fig.update_yaxes(title_text='PE 倍數', secondary_y=False, row=1, col=1)

        # Row 2: PEG
        if peg_col and ratios[peg_col].notna().sum() > 1:
            add_val_line(2, ratios[peg_col], 'PEG 本益成長比', '#9b59b6', fmt='.2f')
            add_price_line(2, quarterly_close)
            fig.update_yaxes(title_text='PEG 倍數', secondary_y=False, row=2, col=1)
        else:
            fig.add_trace(go.Scatter(x=[], y=[], name='PEG（資料不足）',
                                     line=dict(color='#9b59b6'), showlegend=False),
                          row=2, col=1, secondary_y=False)

        # Row 3: PS
        if ps_col and ratios[ps_col].notna().sum() > 1:
            add_val_line(3, ratios[ps_col], 'PS 股價銷售比', '#3498db', fmt='.2f')
            add_price_line(3, quarterly_close)
            fig.update_yaxes(title_text='PS 倍數', secondary_y=False, row=3, col=1)
        else:
            fig.add_trace(go.Scatter(x=[], y=[], name='PS（資料不足）',
                                     line=dict(color='#3498db'), showlegend=False),
                          row=3, col=1, secondary_y=False)

        # Row 4: PBR
        if pb_col and ratios[pb_col].notna().sum() > 1:
            add_val_line(4, ratios[pb_col], 'PBR 股價淨值比', '#27ae60', fmt='.2f')
            add_price_line(4, quarterly_close)
            fig.update_yaxes(title_text='PBR 倍數', secondary_y=False, row=4, col=1)
        else:
            fig.add_trace(go.Scatter(x=[], y=[], name='PBR（資料不足）',
                                     line=dict(color='#27ae60'), showlegend=False),
                          row=4, col=1, secondary_y=False)

        # Row 5: PER（台股）或殖利率（美股）
        per_col_tw = 'PER' if 'PER' in ratios.columns else None
        if per_col_tw and ratios[per_col_tw].notna().sum() > 1 and dy_col is None:
            add_val_line(5, ratios[per_col_tw], 'PER 本益比', '#f39c12', fmt='.1f')
            add_price_line(5, quarterly_close)
            fig.update_yaxes(title_text='PER 倍數', secondary_y=False, row=5, col=1)
        elif dy_col and ratios[dy_col].notna().sum() > 1:
            add_val_line(5, ratios[dy_col] * 100, '殖利率（%）', '#f39c12', fmt='.2f')
            add_price_line(5, quarterly_close)
            fig.update_yaxes(title_text='殖利率%', secondary_y=False, row=5, col=1)
        else:
            fig.add_trace(go.Scatter(x=[], y=[], name='PER/殖利率（資料不足）',
                                     line=dict(color='#f39c12'), showlegend=False),
                          row=5, col=1, secondary_y=False)

        # Row 6: 季末股價（單獨一張）
        if quarterly_close is not None and len(quarterly_close) > 0:
            y_p6 = quarterly_close['close']
            text_p6 = [f"{v:.0f}" if pd.notna(v) else '' for v in y_p6]
            fig.add_trace(go.Scatter(
                x=quarterly_close['quarter'], y=y_p6,
                mode='lines+markers+text',
                name='季末股價（獨立）',
                line=dict(color='#7f8c8d', width=2),
                marker=dict(size=5),
                text=text_p6,
                textposition='top center',
                textfont=dict(size=8, color='#7f8c8d'),
                showlegend=False
            ), row=6, col=1, secondary_y=False)
            fig.update_yaxes(title_text='股價', secondary_y=False, row=6, col=1)

        # X 軸：只有最下方（row=6）顯示季度標籤
        for r in range(1, 6):
            fig.update_xaxes(showticklabels=False, row=r, col=1)
        fig.update_xaxes(tickangle=45, showticklabels=True, row=6, col=1)

        fig.update_layout(
            title=f'{symbol} P/E 等估值歷年趨勢（6張，X軸每季，含相對股價）',
            height=2400,
            template='plotly_white',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1)
        )

        return fig
    except Exception:
        return None


def create_dmi_chart(df, symbol):
    """
    DMI 三線圖（含顏色索引 Legend）
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

    fig.add_hline(y=25, line_dash='dash', line_color='gray', line_width=1,
                  annotation_text='ADX=25（強趨勢門檻）', annotation_position='right')

    if 'DMI_GOLDEN' in df.columns:
        golden_df = df[df['DMI_GOLDEN']]
        if not golden_df.empty:
            fig.add_trace(go.Scatter(
                x=golden_df['date'], y=golden_df['DMI_PLUS'],
                mode='markers', name='DMI黃金交叉',
                marker=dict(symbol='triangle-up', size=10, color='#f39c12')
            ))

    fig.update_layout(
        title=f'{symbol} DMI 趨勢強度指標（14日）',
        height=400,
        template='plotly_white',
        yaxis_title='DMI 值',
        legend=dict(
            orientation="v",
            yanchor="top", y=1,
            xanchor="right", x=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray', borderwidth=1
        )
    )
    return fig


# ─────────────────────────────────────────────
# AI 分析函數
# ─────────────────────────────────────────────

def generate_ai_insights(symbol, stock_data, openai_api_key, start_date, end_date,
                          market='us', margin_df=None, institutional_df=None,
                          financial_data=None, rsi_period=14, bull_signals=None,
                          insider_df=None, analyst_data=None, pe_data=None,
                          weekly_df=None, director_df=None):
    try:
        client = OpenAI(api_key=openai_api_key)

        first_date = stock_data['date'].iloc[0].strftime('%Y-%m-%d')
        last_date  = stock_data['date'].iloc[-1].strftime('%Y-%m-%d')
        start_price = stock_data['close'].iloc[0]
        end_price   = stock_data['close'].iloc[-1]
        price_change = ((end_price - start_price) / start_price) * 100

        rsi_col = f'RSI{rsi_period}'
        latest_rsi = stock_data[rsi_col].iloc[-1] if rsi_col in stock_data.columns else 50
        if latest_rsi > 70:
            rsi_status = f"超買區域，RSI={latest_rsi:.2f}（>70）"
        elif latest_rsi < 30:
            rsi_status = f"超賣區域，RSI={latest_rsi:.2f}（<30）"
        else:
            rsi_status = f"中性區域，RSI={latest_rsi:.2f}（30–70）"

        currency    = "NT$" if market == 'tw' else "$"
        market_desc = "台灣股票市場" if market == 'tw' else "美國股票市場"

        data_json = stock_data.tail(60).to_json(orient='records', date_format='iso')

        # System message
        system_base = f"""你是一位專業的股票分析師，專精於{market_desc}技術面、籌碼面與基本面分析。

職責：
1. 客觀描述歷史走勢和技術指標狀態
2. 解讀歷史市場數據和交易量變化模式
3. 識別技術面歷史支撐阻力位
4. 教育性技術分析知識（RSI/MACD/BB/OBV/DMI）
5. 解讀內部人買賣動向的歷史意涵
6. 描述分析師目標價分布與共識統計
7. 解讀 P/E、P/S 歷年估值變化特徵，並對應股價走勢同步分析"""

        system_tw_extra = """
8. 解讀台股籌碼面（三大法人/融資融券）
9. 財務報表輔助解讀"""

        system_rules = """

重要原則：
- 使用「歷史數據顯示」、「技術指標反映」、「過去走勢呈現」等客觀描述語
- 禁用：「可能性」、「預期」、「建議」、「關注」
- 禁用「如果…則…」，改用「歷史上當…時，曾出現…現象」
- 不提供具體操作指引；強調「歷史表現不代表未來結果」
- 使用繁體中文回答
- 保持客觀中立，禁止提供投資建議"""

        system_message = system_base + (system_tw_extra if market == 'tw' else '') + system_rules

        # User prompt
        user_prompt = f"""請基於以下股票歷史數據進行深度技術分析：

### 基本資訊
- 股票代號：{symbol}
- 市場：{market_desc}
- 分析期間：{first_date} 至 {last_date}
- 期間價格變化：{price_change:.2f}% （從 {currency}{start_price:.2f} 到 {currency}{end_price:.2f}）
- 最新RSI狀態：{rsi_status}

### 完整交易數據（含 MA 與 RSI，最近60筆）
{data_json}
"""

        # 周K線數據
        if weekly_df is not None and len(weekly_df) > 0:
            try:
                w_json = weekly_df.tail(20).to_json(orient='records', date_format='iso')
                user_prompt += f"\n### 周K線最近20筆數據（含MACD）\n{w_json}\n"
            except Exception:
                pass

        # 內部人買賣
        if insider_df is not None and len(insider_df) > 0:
            try:
                ins_json = insider_df.head(20).to_json(orient='records', date_format='iso')
                user_prompt += f"\n### 內部人買賣紀錄（近3個月，最新20筆）\n{ins_json}\n"
            except Exception:
                pass

        # 法人目標價
        if analyst_data is not None:
            try:
                tgt = analyst_data.get('targets')
                con = analyst_data.get('consensus')
                if tgt is not None and len(tgt) > 0:
                    tgt_json = tgt.head(10).to_json(orient='records', date_format='iso')
                    user_prompt += f"\n### 分析師目標價（最新10筆）\n{tgt_json}\n"
                if con:
                    user_prompt += f"\n### 分析師共識評級\n{json.dumps(con, ensure_ascii=False)}\n"
            except Exception:
                pass

        # P/E P/S
        if pe_data is not None and pe_data.get('ratios') is not None:
            try:
                pe_json = pe_data['ratios'].to_json(orient='records', date_format='iso')
                user_prompt += f"\n### 歷年 P/E、P/S、P/B 比率\n{pe_json}\n"
            except Exception:
                pass

        # 台股附加數據
        if market == 'tw':
            if margin_df is not None and len(margin_df) > 0:
                margin_json = margin_df.tail(10).to_json(orient='records', date_format='iso')
                user_prompt += f"\n### 融資融券餘額（最新10筆）\n{margin_json}\n"
            if institutional_df is not None and len(institutional_df) > 0:
                inst_json = institutional_df.tail(40).to_json(orient='records', date_format='iso')
                user_prompt += f"\n### 三大法人買賣超（最新資料）\n{inst_json}\n"
            if director_df is not None and isinstance(director_df, dict):
                try:
                    # 優先用 MoneyDJ（近3月），其次用 MOPS
                    _dir_src = director_df.get("moneydj") or director_df.get("mops")
                    if _dir_src is not None and len(_dir_src) > 0:
                        dir_json = _dir_src.head(20).to_json(orient='records', date_format='iso')
                        user_prompt += f"\n### 近3個月董監持股異動明細（最新20筆）\n{dir_json}\n"
                except Exception:
                    pass
            if financial_data and financial_data.get('quarterly') is not None:
                q_json = financial_data['quarterly'].to_json(orient='records', date_format='iso')
                user_prompt += f"\n### 近期季度財務數據（5年20季度，EPS/Revenue/OperatingIncome等）\n{q_json}\n"

        # 多頭訊號
        if bull_signals:
            signal_summary = "\n".join([
                f"- {s['name']}：{'🟢' if s['status']=='green' else ('🟡' if s['status']=='yellow' else '🔴')} {s['desc']}（{s['score']:.0f}分）"
                for s in bull_signals['signals']
            ])
            user_prompt += f"\n### 多頭訊號評分結果（整體{bull_signals['total_score']:.0f}/100分）\n{signal_summary}\n結論：{bull_signals['conclusion']}\n"

        # 分析架構
        conclusion_extra = """

#### 結論（必要）
請在結論中包含以下三點評估：
1. **目前股價位階**：根據歷史價格區間、布林通道位置、均線排列，評估目前股價處於歷史高位/中位/低位
2. **多頭訊號的強弱**：根據8項多頭訊號評分結果，綜合評估目前多頭動能的強弱程度
3. **中長線勝率**：依照基本面、技術面、籌碼面，給出歷史統計的中長線勝率參考（格式如「歷史上類似條件組合的中長線表現約 XX%」），並強調此為歷史統計非未來保證"""

        if market == 'tw':
            user_prompt += f"""
### 分析架構：技術面 + 籌碼面 + 財務面 + 多頭訊號完整分析（共18章節）

#### 1. 趨勢分析
- 整體趨勢方向（上升、下降、盤整）、關鍵支撐位和阻力位識別、趨勢強度評估

#### 2. 技術指標分析（MA 均線）
- 移動平均線分析（短期與長期MA的關係）、價格與均線相對位置

#### 3. RSI 分析（必要）
- 最新RSI狀態（{rsi_status}）、RSI歷史走勢、動量強度觀察

#### 4. MACD 分析（必要，含周K線MACD解讀）
- DIF與訊號線相對位置、柱狀圖方向與強度、零軸位置、周K線MACD同步解讀

#### 5. OBV 分析（必要）
- OBV與股價歷史走勢關聯、背離觀察、資金流向描述

#### 6. DMI 分析（必要）
- +DI/-DI方向關係、ADX趨勢強度、黃金/死亡交叉

#### 7. 多頭訊號評分解讀（必要）
- 8項指標燈號總結、評分歷史統計意涵

#### 8. 量能分析（必要）
- 解讀成交量與價格的關係（量增價漲 / 量縮價跌 / 量價背離）
- 近5日均量 vs 全期均量，判斷量能是否放大或萎縮
- OBV趨勢與成交量配合度
- 歷史上量能異常放大時（>全期均量×1.5）的價格後續表現

#### 9. 布林通道深度分析（必要）
- 目前股價位於布林通道上軌／中軌／下軌的相對位置
- 布林通道寬窄（BB_WIDTH）變化：是否處於壓縮期（<均值50%）
- 壓縮期後突破方向的歷史統計（向上突破 vs 向下突破比例）
- 股價穿越中軌（BB_MID）的方向與持續性
- 上下軌觸碰次數與回歸中軌的歷史規律

#### 10. 內部人買賣分析（必要；台股顯示GoodInfo查詢連結說明）
- 近3個月董監事買賣動向、淨買超/賣超歷史意涵
- 台股查詢建議：https://goodinfo.tw/tw/StockList.asp（全體董監持股比例）

#### 11. 法人目標價分析（必要；顯示鉅亨網外部連結）
- 目標價分布描述、共識評級
- 外部連結：https://cmnews.com.tw/report 或請搜尋近一個月相關新聞

#### 12. P/E、P/B 歷年估值分析（必要，FinMind PER；需對應股價同步分析）
- 當前估值與歷史均值比較、估值高低歷史脈絡

#### 13. 籌碼面分析（台股必要）
- 三大法人買賣超趨勢、外資動向與股價關聯、融資融券餘額觀察

#### 14. 財務面輔助觀察（台股必要；使用長條圖數據）
- EPS趨勢、營收成長性、財務指標歷史觀察

#### 15. 價格行為分析
- 重要突破點、波動性評估

#### 16. 風險評估
- 技術面風險因子、市場情緒指標

#### 17. 市場觀察
- 短期技術面觀察（1-2週）、中期技術面觀察（1-3個月）

#### 18. 結論（股價位階 / 多頭訊號強弱 / 中長線勝率）
{conclusion_extra}

分析目標：{symbol}（台股）"""
        else:
            user_prompt += f"""
### 分析架構：技術面完整分析（共16章節）

#### 1. 趨勢分析
- 整體趨勢方向（上升、下降、盤整）、關鍵支撐位和阻力位、趨勢強度評估

#### 2. 技術指標分析（MA 均線）
- 移動平均線分析（短期與長期MA）、價格與均線相對位置、成交量與價格關聯

#### 3. RSI 分析（必要）
- 最新RSI狀態（{rsi_status}）、歷史走勢、動量強度觀察

#### 4. MACD 分析（必要，含周K線MACD解讀）
- DIF與訊號線相對位置、柱狀圖方向、零軸位置、周K線MACD同步解讀

#### 5. OBV 分析（必要）
- OBV與股價歷史走勢關聯、背離觀察、資金流向

#### 6. DMI 分析（必要）
- +DI/-DI方向、ADX趨勢強度、黃金/死亡交叉

#### 7. 多頭訊號評分解讀（必要）
- 8項指標燈號總結、評分歷史統計意涵

#### 8. 量能分析（必要）
- 解讀成交量與價格的關係（量增價漲 / 量縮價跌 / 量價背離）
- 近5日均量 vs 全期均量，判斷量能是否放大或萎縮
- OBV趨勢與成交量配合度
- 歷史上量能異常放大時（>全期均量×1.5）的價格後續表現

#### 9. 布林通道深度分析（必要）
- 目前股價位於布林通道上軌／中軌／下軌的相對位置
- 布林通道寬窄（BB_WIDTH）變化：是否處於壓縮期（<均值50%）
- 壓縮期後突破方向的歷史統計（向上突破 vs 向下突破比例）
- 股價穿越中軌（BB_MID）的方向與持續性
- 上下軌觸碰次數與回歸中軌的歷史規律

#### 10. 內部人買賣分析（必要，若有資料；增加搜尋網站連結參考）
- 近3個月董監事、大股東買賣動向、淨買超歷史意涵
- 相關搜尋可參考：SEC EDGAR 或各大財經網站

#### 11. 法人目標價分析（必要，若有資料；或請AI搜尋近一個月新聞）
- 目標價分布、共識評級
- 外部連結：https://cmnews.com.tw/report 或請搜尋近一個月相關新聞

#### 12. P/E、P/S 歷年估值分析（必要，若有資料；需對應股價同步分析）
- 當前估值與歷史均值比較、估值高低歷史脈絡

#### 13. 價格行為分析
- 重要突破點、波動性評估、關鍵轉折點

#### 14. 風險評估
- 技術面風險因子、支撐阻力區間、市場情緒指標

#### 15. 市場觀察
- 短期技術面觀察（1-2週）、中期技術面觀察（1-3個月）

#### 16. 結論（股價位階 / 多頭訊號強弱 / 中長線勝率）
{conclusion_extra}

分析目標：{symbol}（美股）"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=3500,
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

market = st.sidebar.selectbox(
    "市場選擇",
    options=["台股 (TW)", "美股 (US)"],
    index=0,
    help="選擇要分析的市場"
)
is_tw = market == "台股 (TW)"

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

if is_tw:
    finmind_api_key = st.sidebar.text_input("FinMind API Key", type="password",
                                             help="請輸入您的 FinMind API 金鑰（台股數據）")
    fmp_api_key = ""
else:
    fmp_api_key = st.sidebar.text_input("FMP API Key", type="password",
                                         help="請輸入您的 Financial Modeling Prep API 金鑰")
    finmind_api_key = ""

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password",
                                         help="請輸入您的 OpenAI API 金鑰")

default_start_date = datetime.now() - timedelta(days=365)
default_end_date   = datetime.now()

start_date = st.sidebar.date_input("起始日期", value=default_start_date)
end_date   = st.sidebar.date_input("結束日期", value=default_end_date)

rsi_period = st.sidebar.number_input(
    "RSI 計算天數", min_value=2, max_value=50, value=14, step=1,
    help="RSI 計算週期，預設 14 日，範圍 2–50"
)

# 移動平均線可切換顯示
selected_mas = st.sidebar.multiselect(
    "顯示移動平均線",
    options=["MA5", "MA10", "MA20", "MA60"],
    default=["MA5", "MA10", "MA20", "MA60"],
    help="選擇要在K線圖上顯示的移動平均線"
)

analyze_button = st.sidebar.button("🚀 開始分析", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📢 免責聲明

本系統僅供學術研究與教育用途，AI 提供的數據與分析結果僅供參考，**不構成投資建議或財務建議**。

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
        market_key      = 'tw' if is_tw else 'us'
        currency_symbol = "NT$" if is_tw else "$"

        # ── Step 1: 獲取日K價格數據 ──
        spinner_price = "正在獲取台股價格數據..." if is_tw else "正在獲取美股價格數據..."
        with st.spinner(spinner_price):
            if is_tw:
                stock_data = get_tw_stock_price(symbol.strip(), finmind_api_key, start_date, end_date)
            else:
                stock_data = get_us_stock_data(symbol.upper(), fmp_api_key, start_date, end_date)

        if stock_data is not None and len(stock_data) > 0:
            st.success(f"成功獲取 {len(stock_data)} 筆交易數據")
            filtered_data = filter_by_date_range(stock_data, start_date, end_date)

            if filtered_data is not None and len(filtered_data) > 0:

                # ── Step 2: 計算技術指標 ──
                with st.spinner("正在計算技術指標（MA & RSI）..."):
                    data_with_ma = get_moving_averages(filtered_data)
                    data_with_indicators = calculate_rsi(data_with_ma, period=rsi_period)

                # ── Step 3: 計算進階指標 ──
                with st.spinner("正在計算進階技術指標（MACD / 布林通道 / OBV / DMI）..."):
                    data_with_indicators = calculate_advanced_indicators(data_with_indicators)

                # ── Step 4: 生成周K線數據 ──
                weekly_df = None
                with st.spinner("正在生成周K線數據..."):
                    try:
                        weekly_df = generate_weekly_kline(data_with_indicators, rsi_period)
                    except Exception:
                        pass

                # ── Step 5: 計算多頭訊號 ──
                bull_signals = calculate_bull_signals(data_with_indicators)

                # ── Step 6: 籌碼/附加數據 ──
                margin_df        = None
                institutional_df = None
                financial_data   = None
                insider_df       = None
                analyst_data     = None
                pe_data          = None
                broker_df        = None
                broker_date      = None
                director_df      = None

                if is_tw:
                    with st.spinner("正在獲取台股籌碼數據（融資融券、三大法人）..."):
                        margin_df        = get_tw_margin_trading(symbol.strip(), finmind_api_key, start_date, end_date)
                        institutional_df = get_tw_institutional(symbol.strip(), finmind_api_key, start_date, end_date)

                    with st.spinner("正在從 TWSE 獲取券商分點進出明細..."):
                        # 往回找最近 5 個交易日（週末/假日無資料）
                        for days_back in range(0, 8):
                            _d = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
                            # 跳過周末
                            _weekday = (datetime.now() - timedelta(days=days_back)).weekday()
                            if _weekday >= 5:  # 5=週六, 6=週日
                                continue
                            _result = get_tw_broker_trading(symbol.strip(), finmind_api_key, _d)
                            if _result is not None:
                                broker_df   = _result
                                broker_date = _result.get('date', _d)
                                break

                    with st.spinner("正在獲取財務報表數據（5年）..."):
                        financial_data = get_tw_financial_statements(symbol.strip(), finmind_api_key)

                    with st.spinner("正在從 MoneyDJ + MOPS 獲取台股董監持股異動（含防爬蟲延遲）..."):
                        _director_result = get_tw_director_sharehold(symbol.strip(), finmind_api_key)
                        # _director_result 為 dict：{"moneydj": df|None, "mops": df|None, "source": str}
                        director_df       = _director_result   # 傳遞 dict 給後續顯示邏輯
                        _dj_df  = _director_result.get("moneydj")
                        _mp_df  = _director_result.get("mops")
                        _src    = _director_result.get("source", "none")

                    with st.spinner("正在獲取 P/E 歷史估值數據..."):
                        pe_data = get_tw_pe_ps_history(symbol.strip(), finmind_api_key)

                    status_parts = []
                    if margin_df is not None:         status_parts.append("融資融券")
                    if institutional_df is not None:   status_parts.append("三大法人")
                    if broker_df is not None:          status_parts.append(f"券商分點({broker_date})")
                    if director_df is not None:
                        _dj_n  = len(_dj_df) if _dj_df is not None else 0
                        _mp_n  = len(_mp_df) if _mp_df is not None else 0
                        if _src == "both":
                            status_parts.append(f"董監持股異動 MoneyDJ({_dj_n}筆)+MOPS({_mp_n}筆)")
                        elif _src == "moneydj":
                            status_parts.append(f"董監持股異動-MoneyDJ({_dj_n}筆)")
                        elif _src == "mops":
                            status_parts.append(f"董監持股異動-MOPS({_mp_n}筆)")
                    if financial_data and financial_data.get('quarterly') is not None:
                        status_parts.append("財務報表（5年）")
                    if pe_data is not None:            status_parts.append("P/E歷年估值")

                    if status_parts:
                        st.success(f"成功獲取台股附加數據：{'、'.join(status_parts)}")
                    else:
                        st.warning("台股附加數據獲取失敗，將僅顯示技術面分析。")
                else:
                    with st.spinner("正在獲取內部人買賣紀錄..."):
                        insider_df = get_insider_trading(symbol.upper(), fmp_api_key)
                    with st.spinner("正在獲取法人目標價與評級..."):
                        analyst_data = get_analyst_targets(symbol.upper(), fmp_api_key)
                    with st.spinner("正在獲取 P/E、P/S 歷年估值數據..."):
                        pe_data = get_pe_ps_history(symbol.upper(), fmp_api_key)

                    status_parts = []
                    if insider_df is not None:    status_parts.append(f"內部人買賣（{len(insider_df)}筆）")
                    if analyst_data is not None:  status_parts.append("法人目標價")
                    if pe_data is not None:        status_parts.append("P/E歷年估值")
                    if status_parts:
                        st.success(f"成功獲取附加數據：{'、'.join(status_parts)}")

                if data_with_indicators is not None:

                    # ── 顯示 1：主K線圖（圖1）──
                    st.markdown("### 📊 主K線圖（含布林通道、OBV、成交量）")
                    chart = create_candlestick_chart(
                        data_with_indicators,
                        symbol.upper() if not is_tw else symbol.strip(),
                        rsi_period,
                        currency_symbol,
                        institutional_df=institutional_df,
                        market=market_key,
                        selected_mas=selected_mas
                    )
                    st.plotly_chart(chart, use_container_width=True)

                    # ── 顯示 2：RSI 即時警告 ──
                    rsi_col = f'RSI{rsi_period}'
                    latest_rsi = data_with_indicators[rsi_col].iloc[-1] if rsi_col in data_with_indicators.columns else 50
                    if latest_rsi > 70:
                        st.warning(f"⚠️ RSI 超買警告：目前 RSI = **{latest_rsi:.2f}**，已進入超買區域（>70）。歷史數據顯示此區域價格波動風險較高。")
                    elif latest_rsi < 30:
                        st.success(f"📉 RSI 超賣提示：目前 RSI = **{latest_rsi:.2f}**，已進入超賣區域（<30）。歷史數據顯示此區域曾出現反彈現象，但不代表未來走勢。")
                    else:
                        st.info(f"📊 RSI 中性：目前 RSI = **{latest_rsi:.2f}**，位於中性區域（30–70）。")

                    # ── 顯示 3：周K線圖（圖2）──
                    if weekly_df is not None:
                        weekly_chart = create_weekly_chart(
                            weekly_df,
                            symbol.upper() if not is_tw else symbol.strip(),
                            selected_mas=selected_mas
                        )
                        if weekly_chart:
                            st.markdown("### 📅 周K線圖（含MACD）")
                            st.plotly_chart(weekly_chart, use_container_width=True)

                    # ── 顯示 4（台股）：融資融券 ──
                    if is_tw and margin_df is not None and len(margin_df) > 0:
                        margin_chart = create_margin_chart(margin_df, symbol.strip())
                        if margin_chart is not None:
                            st.markdown("### 💳 融資融券餘額")
                            st.plotly_chart(margin_chart, use_container_width=True)
                        else:
                            st.info("ℹ️ 融資融券圖表無法顯示（欄位資料缺失）。")

                    # ── 顯示 5：基本統計4欄 ──
                    st.markdown("### 📈 基本統計資訊")
                    col1, col2, col3, col4 = st.columns(4)
                    s_price = data_with_indicators['close'].iloc[0]
                    e_price = data_with_indicators['close'].iloc[-1]
                    price_change_val = e_price - s_price
                    price_change_pct = (price_change_val / s_price) * 100
                    with col1:
                        st.metric("起始價格", f"{currency_symbol}{s_price:.2f}")
                    with col2:
                        st.metric("結束價格", f"{currency_symbol}{e_price:.2f}")
                    with col3:
                        st.metric("價格變化", f"{currency_symbol}{price_change_val:.2f}", f"{price_change_pct:.2f}%")
                    with col4:
                        rsi_delta = "超買🔴" if latest_rsi > 70 else ("超賣🟢" if latest_rsi < 30 else "中性🔵")
                        st.metric(f"最新 RSI ({rsi_period}日)", f"{latest_rsi:.2f}", rsi_delta)

                    # ── 顯示 6（台股）：三大法人 + 券商分點進出明細 ──
                    if is_tw:
                        # 6a. 三大法人買賣超表格
                        if institutional_df is not None and len(institutional_df) > 0:
                            st.markdown("### 🏦 三大法人買賣超（最近10個交易日）")
                            display_institutional_table(institutional_df)

                        # 6b. 券商分點進出明細（主力進出）
                        st.markdown("### 🏢 券商分點-進出明細（主力買賣超）")
                        if broker_df is not None:
                            # 顯示資料來源
                            _src = broker_df.get('source', 'TWSE') if isinstance(broker_df, dict) else 'TWSE'
                            st.caption(f"📡 資料來源：{_src}（TWSE 官方買賣日報表）")
                            display_broker_table(
                                broker_df,
                                symbol.strip(),
                                query_date=broker_date
                            )
                        else:
                            st.warning(
                                "⚠️ 無法獲取券商分點資料。\n\n"
                                "**可能原因：**\n"
                                "- 今日為非交易日（週末／假日）\n"
                                "- TWSE 資料尚未更新（通常收盤後約1小時更新）\n"
                                "- 網路連線問題\n\n"
                                "**備用查詢連結（手動開啟）：**"
                            )
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.markdown(
                                    f"🔗 [富邦券商分點查詢](https://fubon-ebrokerdj.fbs.com.tw/z/zc/zco/zco_{symbol.strip()}_8.djhtm)"
                                )
                            with col_b:
                                st.markdown(
                                    f"🔗 [GoodInfo 主力進出](https://goodinfo.tw/tw/StockBuySaleByBroker.asp?STOCK_ID={symbol.strip()})"
                                )
                            st.markdown(
                                f"🔗 [TWSE 買賣日報表](https://bsr.twse.com.tw/bshtm/bsMain.aspx) "
                                f"（輸入股號 {symbol.strip()} 查詢）"
                            )


                    # ── 顯示 7（台股）：季度財務長條圖 ──
                    if is_tw and financial_data and financial_data.get('quarterly') is not None:
                        st.markdown("### 📑 近期季度財務數據比較（5年20季度）")
                        fin_chart = create_financial_bar_chart(
                            financial_data,
                            data_with_indicators,
                            symbol.strip()
                        )
                        if fin_chart:
                            st.plotly_chart(fin_chart, use_container_width=True)

                    # ── 顯示 8：內部人買賣儀表板（MoneyDJ 近3月 + MOPS 年度申報）──
                    if is_tw:
                        st.markdown("### 🔍 內部人買賣分析（MoneyDJ + MOPS 雙來源）")

                        _dj   = director_df.get("moneydj") if isinstance(director_df, dict) else None
                        _mp   = director_df.get("mops")    if isinstance(director_df, dict) else None
                        _srck = director_df.get("source", "none") if isinstance(director_df, dict) else "none"

                        # ── 來源狀態徽章 ──────────────────────────────
                        _src_badge = {
                            "both":     "✅ MoneyDJ（近3月）＋ MOPS（年度申報）雙來源",
                            "moneydj":  "✅ MoneyDJ（近3月）｜ ⚠️ MOPS 暫無資料",
                            "mops":     "⚠️ MoneyDJ 無法取得（防爬機制）｜ ✅ MOPS（年度申報）",
                            "none":     "❌ MoneyDJ 與 MOPS 均無法取得資料",
                        }.get(_srck, "—")
                        st.caption(f"📡 資料來源狀態：{_src_badge}")

                        # ── 頂部統計卡（整合雙來源）──────────────────
                        _any_df = _dj if _dj is not None else _mp
                        if _any_df is not None and len(_any_df) > 0:
                            _buy_col = "異動股數" if "異動股數" in _any_df.columns else None
                            if _buy_col:
                                _buy_t  = _any_df[_any_df[_buy_col] > 0][_buy_col].sum()
                                _sel_t  = abs(_any_df[_any_df[_buy_col] < 0][_buy_col].sum())
                                _net_t  = _any_df[_buy_col].sum()
                                _buy_n  = int((_any_df[_buy_col] > 0).sum())
                                _sel_n  = int((_any_df[_buy_col] < 0).sum())
                                dc1, dc2, dc3, dc4 = st.columns(4)
                                with dc1:
                                    st.metric("買入合計（股）", f"{_buy_t:,.0f}",
                                              delta=f"🔴 {_buy_n} 筆")
                                with dc2:
                                    st.metric("賣出合計（股）", f"{_sel_t:,.0f}",
                                              delta=f"🟢 {_sel_n} 筆")
                                with dc3:
                                    st.metric("淨異動（股）", f"{_net_t:,.0f}",
                                              delta="淨買超 ▲" if _net_t > 0 else "淨賣超 ▼")
                                with dc4:
                                    _src_label = "MoneyDJ 近3月" if _dj is not None else "MOPS 年度"
                                    st.metric("申報筆數", f"{len(_any_df)} 筆", delta=_src_label)

                        # ── 頁籤：MoneyDJ ／ MOPS ／ 買賣走勢 ────────
                        _tab_labels = []
                        if _dj is not None:   _tab_labels.append("📅 MoneyDJ 近3月申報")
                        if _mp is not None:   _tab_labels.append("🏛️ MOPS 年度申報")
                        _tab_labels.append("📊 買賣走勢圖")

                        if _tab_labels:
                            _tabs = st.tabs(_tab_labels)
                            _tab_idx = 0

                            # ── 頁籤 A：MoneyDJ ──────────────────────
                            if _dj is not None:
                                with _tabs[_tab_idx]:
                                    st.caption("資料來源：MoneyDJ 董監質設異動清單（近90天申報）")
                                    _dj_disp_cols = [c for c in [
                                        "申報日期","職稱","姓名",
                                        "買進(千股)","賣出(千股)",
                                        "異動前持股數","異動後持股數","異動股數","持股%","買賣"
                                    ] if c in _dj.columns]
                                    if not _dj_disp_cols:
                                        _dj_disp_cols = [c for c in _dj.columns if c != "date"]
                                    _dj_disp = _dj[_dj_disp_cols].copy()
                                    for nc in ["異動前持股數","異動後持股數","異動股數"]:
                                        if nc in _dj_disp.columns:
                                            _dj_disp[nc] = _dj_disp[nc].apply(
                                                lambda v: f"{int(v):,}" if pd.notna(v) else "—")
                                    st.dataframe(_dj_disp, use_container_width=True, hide_index=True)
                                _tab_idx += 1

                            # ── 頁籤 B：MOPS ─────────────────────────
                            if _mp is not None:
                                with _tabs[_tab_idx]:
                                    st.caption(f"資料來源：公開資訊觀測站（MOPS）— 董監事持股異動申報，民國 {datetime.now().year - 1911} 年及上年度")
                                    _mp_disp_cols = [c for c in [
                                        "申報日期","職稱","姓名",
                                        "異動前持股數","異動後持股數","異動股數","買賣"
                                    ] if c in _mp.columns]
                                    if not _mp_disp_cols:
                                        _mp_disp_cols = [c for c in _mp.columns if c != "date"]
                                    _mp_disp = _mp[_mp_disp_cols].copy()
                                    for nc in ["異動前持股數","異動後持股數","異動股數"]:
                                        if nc in _mp_disp.columns:
                                            _mp_disp[nc] = _mp_disp[nc].apply(
                                                lambda v: f"{int(v):,}" if pd.notna(v) else "—")
                                    st.dataframe(_mp_disp, use_container_width=True, hide_index=True)

                                    # MOPS CSV 下載
                                    _mp_csv = _mp_disp.to_csv(index=False).encode("utf-8-sig")
                                    st.download_button(
                                        "📥 下載 MOPS 申報 CSV", _mp_csv,
                                        file_name=f"mops_{symbol.strip()}_{datetime.now().strftime('%Y%m%d')}.csv",
                                        mime="text/csv", key="mops_csv_dl"
                                    )
                                    st.markdown(f"🔗 [至公開資訊觀測站查看原始資料](https://mops.twse.com.tw/mops/web/t51sb06)")
                                _tab_idx += 1

                            # ── 頁籤 C：買賣走勢圖 ───────────────────
                            with _tabs[_tab_idx]:
                                _plot_df = _dj if _dj is not None else _mp
                                if _plot_df is not None and len(_plot_df) > 0 and "異動股數" in _plot_df.columns:
                                    _plot_df2 = _plot_df.copy()
                                    _plot_df2["date2"] = pd.to_datetime(
                                        _plot_df2.get("date", _plot_df2.get("申報日期")), errors="coerce")
                                    _plot_df2 = _plot_df2.dropna(subset=["date2"]).sort_values("date2")

                                    _buy_pts  = _plot_df2[_plot_df2["異動股數"] > 0]
                                    _sell_pts = _plot_df2[_plot_df2["異動股數"] < 0]

                                    _fig_ins = go.Figure()
                                    if len(_buy_pts) > 0:
                                        _fig_ins.add_trace(go.Bar(
                                            x=_buy_pts["date2"],
                                            y=_buy_pts["異動股數"] / 1000,
                                            name="買入（千股）",
                                            marker_color="rgba(34,139,34,0.75)",
                                            hovertemplate="%{x|%Y-%m-%d}<br>買入：%{y:,.0f} 千股<extra></extra>",
                                        ))
                                    if len(_sell_pts) > 0:
                                        _fig_ins.add_trace(go.Bar(
                                            x=_sell_pts["date2"],
                                            y=_sell_pts["異動股數"] / 1000,
                                            name="賣出（千股）",
                                            marker_color="rgba(220,53,69,0.75)",
                                            hovertemplate="%{x|%Y-%m-%d}<br>賣出：%{y:,.0f} 千股<extra></extra>",
                                        ))
                                    _fig_ins.update_layout(
                                        title=f"{symbol.strip()} 董監持股異動走勢（千股）",
                                        barmode="relative",
                                        xaxis_title="申報日期",
                                        yaxis_title="異動千股（正=買入，負=賣出）",
                                        height=360,
                                        showlegend=True,
                                        plot_bgcolor="rgba(0,0,0,0)",
                                        paper_bgcolor="rgba(0,0,0,0)",
                                        margin=dict(l=50, r=20, t=50, b=40),
                                        hovermode="x unified",
                                    )
                                    _fig_ins.add_hline(y=0, line_dash="dash",
                                                       line_color="gray", line_width=1)
                                    st.plotly_chart(_fig_ins, use_container_width=True)
                                else:
                                    st.info("ℹ️ 無足夠資料繪製走勢圖")

                        else:
                            st.info("ℹ️ MoneyDJ 與 MOPS 均無申報異動資料（近90天 / 今年）")
                            st.caption("可能原因：該股近期無申報紀錄、MoneyDJ 防爬機制、或 MOPS 查無資料。")

                        st.markdown(f"""
🔗 [MoneyDJ 董監持股明細](https://www.moneydj.com/z/zc/zck/zck_{symbol.strip()}.djhtm) ｜
🔗 [GoodInfo 董監持股查詢](https://goodinfo.tw/tw/StockDirectorSharehold.asp?STOCK_ID={symbol.strip()}) ｜
🔗 [MOPS 公開資訊觀測站](https://mops.twse.com.tw/mops/web/t51sb06)
""")
                    elif insider_df is not None and len(insider_df) > 0:
                        insider_fig = create_insider_chart(insider_df, symbol.upper())
                        if insider_fig:
                            st.markdown("### 👤 內部人買賣紀錄（近3個月）")
                            st.plotly_chart(insider_fig, use_container_width=True)

                    # ── 顯示 9：法人目標價（AI 搜尋新聞彙整）──
                    st.markdown("### 🎯 法人目標價分析（近一個月新聞彙整）")
                    with st.spinner("AI 正在搜尋近一個月法人目標價新聞..."):
                        ai_targets = get_analyst_targets_ai(
                            symbol.upper() if not is_tw else symbol.strip(),
                            openai_api_key,
                            market=market_key
                        )

                    st.caption(f"📅 資料搜尋時間：{ai_targets.get('search_date', '')}，涵蓋近一個月新聞")

                    tgt_table = ai_targets.get('table')
                    if tgt_table is not None and len(tgt_table) > 0:
                        # 統計摘要 4 欄
                        prices_raw = []
                        for v in tgt_table['目標價']:
                            try:
                                num = float(str(v).replace('NT$', '').replace('$', '')
                                            .replace(',', '').strip())
                                prices_raw.append(num)
                            except Exception:
                                pass

                        m1, m2, m3, m4 = st.columns(4)
                        with m1:
                            st.metric("目標價最高",
                                      f"{currency_symbol}{max(prices_raw):.2f}" if prices_raw else "—")
                        with m2:
                            st.metric("目標價最低",
                                      f"{currency_symbol}{min(prices_raw):.2f}" if prices_raw else "—")
                        with m3:
                            avg_p = sum(prices_raw) / len(prices_raw) if prices_raw else 0
                            st.metric("目標價均值",
                                      f"{currency_symbol}{avg_p:.2f}" if prices_raw else "—")
                        with m4:
                            st.metric("機構家數", f"{len(tgt_table)} 家")

                        st.dataframe(tgt_table, use_container_width=True, hide_index=True)
                    else:
                        st.warning("⚠️ 目前無法取得近期法人目標價，建議手動查詢")

                    # 美股同時保留 FMP 散點圖（若有資料）
                    if not is_tw and analyst_data is not None:
                        current_price = data_with_indicators['close'].iloc[-1]
                        analyst_fig = create_analyst_chart(
                            analyst_data, symbol.upper(), currency_symbol, current_price)
                        if analyst_fig:
                            with st.expander("📊 FMP 法人目標價散點圖（展開查看）"):
                                st.plotly_chart(analyst_fig, use_container_width=True)
                                con = analyst_data.get('consensus')
                                if con:
                                    ec1, ec2, ec3, ec4 = st.columns(4)
                                    tgt_hi  = con.get('targetHigh') or con.get('targetHighPrice')
                                    tgt_lo  = con.get('targetLow')  or con.get('targetLowPrice')
                                    tgt_avg = con.get('targetConsensus') or con.get('targetMean') or con.get('priceTarget')
                                    rating  = con.get('consensus') or con.get('rating') or '—'
                                    with ec1: st.metric("目標價最高", f"{currency_symbol}{float(tgt_hi):.2f}" if tgt_hi else "—")
                                    with ec2: st.metric("目標價最低", f"{currency_symbol}{float(tgt_lo):.2f}" if tgt_lo else "—")
                                    with ec3: st.metric("目標價均值", f"{currency_symbol}{float(tgt_avg):.2f}" if tgt_avg else "—")
                                    with ec4: st.metric("共識評級", str(rating))

                    # ── 顯示 10：P/E等6張估值趨勢圖 ──
                    if pe_data is not None:
                        pe_fig = create_pe_ps_chart(
                            pe_data,
                            symbol.upper() if not is_tw else symbol.strip(),
                            data_with_indicators
                        )
                        if pe_fig:
                            st.markdown("### 📐 P/E 等估值歷年趨勢圖（6張，共享時間軸）")
                            st.plotly_chart(pe_fig, use_container_width=True)

                    # ── 顯示 11：多頭訊號儀表板 ──
                    display_bull_dashboard(bull_signals, symbol.strip() if is_tw else symbol.upper())

                    # ── 顯示 12：DMI 圖（含顏色索引）──
                    dmi_fig = create_dmi_chart(
                        data_with_indicators,
                        symbol.strip() if is_tw else symbol.upper()
                    )
                    if dmi_fig:
                        st.markdown("### 📈 DMI 趨勢強度指標（14日）")
                        st.plotly_chart(dmi_fig, use_container_width=True)

                    # ── 顯示 13：AI 技術分析 ──
                    st.markdown("### 🤖 AI 技術分析")
                    spinner_ai = "AI 正在分析中（含內部人/目標價/估值/多頭訊號/位階勝率分析）..."
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
                            bull_signals=bull_signals,
                            insider_df=insider_df,
                            analyst_data=analyst_data,
                            pe_data=pe_data,
                            weekly_df=weekly_df,
                            director_df=director_df
                        )
                    if ai_analysis:
                        st.markdown(ai_analysis)

                    # ── 顯示 14：歷史數據表格 ──
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
- **主K線圖（圖1）**: 布林通道整合 + 可切換MA線 + RSI + OBV + 成交量 + 台股三大法人
- **周K線圖（圖2）**: 周K線 + 布林通道 + 可切換MA + MACD（整合圖2下方）
- **進階技術指標**: MACD、布林通道（壓縮突破）、OBV（量價背離）、DMI（含顏色索引）
- **🚦 多頭訊號儀表板**: 8項指標燈號（🟢🟡🔴）+ 整體評分（0–100分）
- **台股特有**: 三大法人中文表格+10日加總、季度財務長條圖（5年20季度+股價）
- **估值分析**: P/E等6張獨立趨勢圖（PE/PEG/PS/PBR/殖利率/年均股價），共享時間軸
- **內部人分析**: 美股橫條圖 + 台股GoodInfo連結；法人目標價 + 鉅亨網連結
- **AI智能分析**: gpt-4o-mini 深度分析，結論含股價位階、多頭訊號強弱、中長線勝率

### 📝 使用方法
1. 在左側選擇市場（美股 / 台股）
2. 輸入股票代碼（美股如：AAPL；台股純數字如：2330）
3. 輸入對應的 API 金鑰與 OpenAI API Key
4. 選擇分析的日期範圍（預設1年）、設定 RSI 計算天數
5. 選擇要顯示的移動平均線（MA5/MA10/MA20/MA60）
6. 點擊「🚀 開始分析」按鈕

### 💡 技術指標說明
- **MA5 / MA10 / MA20 / MA60**: 移動平均線（可切換顯示/隱藏）
- **RSI（相對強弱指數）**: >70 超買，<30 超賣，50 為中性分界線
- **MACD（12,26,9）**: 整合至周K線圖，DIF/訊號線金叉、柱狀圖方向
- **布林通道（20,2SD）**: 整合至主K線圖，通道壓縮後突破為爆發型態
- **OBV（量能指標）**: 整合至主K線圖，量價背離偵測
- **DMI（14日）**: +DI/-DI方向、ADX趨勢強度（>25強趨勢），含顏色索引

### 🚦 多頭訊號儀表板（8項）
MACD轉正 / BB突破中軌 / BB壓縮突破 / OBV資金流入 / RSI動量 / DMI多頭趨勢 / DMI黃金交叉 / 均線多頭排列
- **🟢 綠燈（12.5分）** | **🟡 黃燈（6分）** | **🔴 紅燈（0分）**
- 評分 ≥70：多頭確認 | 40–69：訊號混合 | <40：條件不符

### 🗂️ 台股特有指標
- **三大法人**: 外資/投信/自營商每日買賣超，中文欄位+10日加總
- **融資/融券餘額**: 市場槓桿水位與空方籌碼
- **季度財務長條圖**: EPS/營收/毛利/營業利益，5年20季度含股價趨勢線

### 🔑 API 金鑰獲取
- **FMP API（美股）**: [Financial Modeling Prep](https://financialmodelingprep.com/developer/docs)
- **FinMind API（台股）**: [FinMind Trade](https://finmindtrade.com/)（免費方案每日有請求次數限制）
- **OpenAI API**: [OpenAI Platform](https://platform.openai.com)

---
**開始您的技術分析之旅吧！** 📈
""")
