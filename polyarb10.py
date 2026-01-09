import os
import re
import time
import json
import base64
import datetime
import requests
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

from dotenv import load_dotenv

import sys
import io
from contextlib import redirect_stdout, redirect_stderr

from difflib import SequenceMatcher

# --- Optional UI / notifications ---
try:
    import streamlit as st
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    UI_MODE = True
except ImportError:
    UI_MODE = False

try:
    from plyer import notification
    NOTIFICATIONS_ENABLED = True
except ImportError:
    NOTIFICATIONS_ENABLED = False

# --- Kalshi signing (kept) ---
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend


# =============================================================================
# CONFIG
# =============================================================================

load_dotenv()

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
KALSHI_API = "https://trading-api.kalshi.com/trade-api/v2"
PREDICTIT_API = "https://www.predictit.org/api/marketdata/all/"

KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID")
KALSHI_PRIVATE_KEY_PATH = "kalshi_private.key"

# Scan behavior
SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "30"))
POLY_LIMIT = int(os.getenv("POLY_LIMIT", "100"))  # gamma /markets limit
KALSHI_LIMIT = int(os.getenv("KALSHI_LIMIT", "100"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "15"))
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.82"))

# Performance guardrails
MAX_OUTCOMES_TO_FETCH = int(os.getenv("MAX_OUTCOMES_TO_FETCH", "12"))  # per market, to avoid explosion
MAX_MARKETS_PER_PLATFORM = int(os.getenv("MAX_MARKETS_PER_PLATFORM", "100"))

# Prefilter / prioritization (Stage A)
PREFILTER_TOPK_POLY = int(os.getenv("PREFILTER_TOPK_POLY", "40"))
PREFILTER_TOPK_PREDICTIT = int(os.getenv("PREFILTER_TOPK_PREDICTIT", "80"))
PREFILTER_TOPK_KALSHI = int(os.getenv("PREFILTER_TOPK_KALSHI", "80"))

WEIGHT_ACTIVITY = float(os.getenv("WEIGHT_ACTIVITY", "0.70"))
WEIGHT_RECENCY = float(os.getenv("WEIGHT_RECENCY", "0.25"))
WEIGHT_OUTCOMES_PENALTY = float(os.getenv("WEIGHT_OUTCOMES_PENALTY", "0.05"))

# NEW: depth metrics configuration (Polymarket only, based on full orderbook)
DEPTH_TOP_LEVELS = int(os.getenv("DEPTH_TOP_LEVELS", "10"))         # sum sizes of top N levels
DEPTH_PRICE_WINDOW = float(os.getenv("DEPTH_PRICE_WINDOW", "0.02")) # sum sizes within +/- window of best price

# Inefficiency thresholds (gross baseline)
FULL_SET_EDGE_MIN = float(os.getenv("FULL_SET_EDGE_MIN", "0.01"))  # e.g., 0.01 = 1%
FULL_SET_EDGE_MIN_NONEXHAUSTIVE = float(os.getenv("FULL_SET_EDGE_MIN_NONEXHAUSTIVE", str(FULL_SET_EDGE_MIN * 2)))
BINARY_EDGE_MIN = float(os.getenv("BINARY_EDGE_MIN", "0.005"))     # e.g., 0.5%

# PredictIt fee model (heuristic; used to avoid false-positive "edges")
PREDICTIT_PROFIT_FEE = float(os.getenv("PREDICTIT_PROFIT_FEE", "0.10"))
PREDICTIT_WITHDRAWAL_FEE = float(os.getenv("PREDICTIT_WITHDRAWAL_FEE", "0.05"))
PREDICTIT_FEE_MODE = os.getenv("PREDICTIT_FEE_MODE", "conservative").lower()  # conservative | profit_only | none

# Trading mode (safe defaults)
TRADING_ENABLED = os.getenv("TRADING_ENABLED", "false").lower() == "true"
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true"
MAX_TRADE_USD = float(os.getenv("MAX_TRADE_USD", "25"))
AUTO_TRADE_MIN_EDGE = float(os.getenv("AUTO_TRADE_MIN_EDGE", "0.02"))


# =============================================================================
# Polymarket: request hardening + correct CLOB orderbook endpoint
# =============================================================================

DEFAULT_HTTP_HEADERS = {
    "User-Agent": os.getenv(
        "HTTP_USER_AGENT",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

CLOB_ORDERBOOK_PATH = "/book"

HTTP_MAX_RETRIES = int(os.getenv("HTTP_MAX_RETRIES", "3"))
HTTP_BACKOFF_SECONDS = float(os.getenv("HTTP_BACKOFF_SECONDS", "1.2"))


# =============================================================================
# DATA MODELS
# =============================================================================

BET_BINARY_YN = "BINARY_YESNO"
BET_BINARY_2WAY = "BINARY_2WAY"
BET_MULTI = "MULTI_OUTCOME"
BET_BUCKET = "SCALAR_BUCKETED"
BET_THRESHOLD = "SCALAR_THRESHOLD"
BET_UNKNOWN = "UNKNOWN"

OPP_FULL_SET = "FULL_SET_CHEAP"
OPP_BINARY = "BINARY_2WAY_CHEAP"
OPP_CROSS = "CROSS_PLATFORM"


@dataclass
class OutcomeQuote:
    name: str
    token_id: Optional[str] = None
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    raw: Optional[dict] = None


@dataclass
class MarketSnapshot:
    platform: str
    market_id: Optional[str]
    question: str
    bet_type: str
    outcomes: List[OutcomeQuote]

    # Volume/attention proxies (platform dependent)
    volume: Optional[float] = None  # kept (Polymarket "volume" when present)
    activity_raw: Optional[float] = None  # attention proxy: Poly volume24h/volume, PI shares traded, Kalshi vol/OI proxy

    url: Optional[str] = None
    mutually_exclusive: bool = True
    exhaustive_hint: str = "UNKNOWN"
    has_other_outcome: bool = False

    # Recency / timestamps (best-effort)
    created_ts: Optional[str] = None
    updated_ts: Optional[str] = None
    close_ts: Optional[str] = None
    end_ts: Optional[str] = None
    recency_ts: Optional[str] = None              # chosen dt used for scoring (updated > created > start, etc.)
    recency_age_seconds: Optional[float] = None   # now - recency_ts

    # Priority metadata (Stage A)
    priority_score: Optional[float] = None
    outcomes_count_raw: Optional[int] = None

    # Spread metrics (derived from best bid/ask)
    spread_avg: Optional[float] = None
    spread_min: Optional[float] = None
    spread_max: Optional[float] = None

    # Depth metrics (Polymarket only; derived from orderbook sizes)
    depth_bid_top: Optional[float] = None
    depth_ask_top: Optional[float] = None
    depth_total_top: Optional[float] = None

    depth_bid_window: Optional[float] = None
    depth_ask_window: Optional[float] = None
    depth_total_window: Optional[float] = None

    fetched_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())


@dataclass
class Opportunity:
    opp_type: str
    platform: str
    question: str
    bet_type: str
    edge_gross: float
    edge_net: float
    cost_sum_asks: Optional[float]
    details: str
    suggested_size_usd: float
    confidence: str
    warnings: List[str]
    actions: List[dict]

    # NEW: carry market linkage so the dashboard can always link the question
    market_id: Optional[str] = None
    url: Optional[str] = None


# =============================================================================
# UTILS
# =============================================================================

def send_alert(message: str):
    global NOTIFICATIONS_ENABLED
    if NOTIFICATIONS_ENABLED and os.getenv("DISABLE_NOTIFICATIONS", "false").lower() != "true":
        try:
            notification.notify(title="Arb / Inefficiency Alert", message=message, timeout=10)
        except Exception as e:
            NOTIFICATIONS_ENABLED = False
            print(f"[WARN] Desktop notifications disabled (plyer backend error: {e})")
    print(message)


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def is_yes_no_pair(outcomes: List[str]) -> bool:
    s = set([o.strip().lower() for o in outcomes])
    return s == {"yes", "no"} or ("yes" in s and "no" in s and len(s) == 2)


RANGE_RE = re.compile(r"(^\s*<\s*\d+(\.\d+)?\s*$)|(^\s*>\s*\d+(\.\d+)?\s*$)|(\d+(\.\d+)?\s*[-–]\s*\d+(\.\d+)?)")
THRESH_RE = re.compile(r"(over|under|>=|<=|>|<)\s*\d+(\.\d+)?", re.IGNORECASE)

def looks_like_bucket_or_range(name: str) -> bool:
    return bool(RANGE_RE.search(name))

def looks_like_threshold(name: str) -> bool:
    return bool(THRESH_RE.search(name))


OTHER_ALIASES = {
    "other",
    "any other",
    "someone else",
    "field",
    "none of the above",
    "other (specify)",
    "other candidate",
    "other team",
    "other outcome",
}

def has_other_outcome(outcome_names: List[str]) -> bool:
    for n in outcome_names:
        if n is None:
            continue
        s = str(n).strip().lower()
        if s in OTHER_ALIASES:
            return True
        if "other" in s and "mother" not in s:
            return True
    return False


def compute_exhaustiveness_hint(outcome_names: List[str]) -> Tuple[str, bool]:
    n = len(outcome_names)
    other = has_other_outcome(outcome_names)
    if n == 2:
        return "TWO_OUTCOME_ASSUME_EXHAUSTIVE", other
    if other:
        return "HAS_OTHER_LIKELY_EXHAUSTIVE", other
    return "UNKNOWN_EXHAUSTIVENESS", other


def classify_bet_type(outcome_names: List[str]) -> str:
    outcome_names = [str(x) for x in (outcome_names or []) if x is not None]
    n = len(outcome_names)

    if n == 2:
        if is_yes_no_pair(outcome_names):
            return BET_BINARY_YN
        threshish = sum(1 for name in outcome_names if looks_like_threshold(name) or name.strip().lower() in {"over", "under"})
        if threshish >= 1:
            return BET_THRESHOLD
        return BET_BINARY_2WAY

    bucketish = sum(1 for nm in outcome_names if looks_like_bucket_or_range(nm))
    threshish = sum(1 for nm in outcome_names if looks_like_threshold(nm))

    if n >= 3 and bucketish >= max(2, int(0.5 * n)):
        return BET_BUCKET

    if n >= 2 and threshish >= max(1, int(0.4 * n)):
        return BET_THRESHOLD

    if n >= 3:
        return BET_MULTI

    return BET_UNKNOWN


def atomic_write_json(path: str, obj: dict):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def export_dashboard_snapshot(
    out_dir: str,
    market_rows: List[dict],
    opportunities_rows: List[dict],
    counts: dict,
    arbs_found_msgs: List[str],
):
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    payload = {
        "generated_at": now,
        "counts": counts,
        "opportunities": opportunities_rows,
        "markets": market_rows,
        "alerts": arbs_found_msgs[-50:],  # keep last 50 messages
    }

    atomic_write_json(os.path.join(out_dir, "latest.json"), payload)

    # Optional: append history (one JSON per line)
    hist_path = os.path.join(out_dir, "history.jsonl")
    try:
        with open(hist_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[WARN] Could not append history.jsonl: {e}")


def slugify(text: str) -> str:
    """
    Simple slugifier for URLs. Good enough for PredictIt detail URLs.
    """
    if text is None:
        return ""
    s = str(text).strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


def predictit_market_url(market_id: Optional[str], name: str) -> Optional[str]:
    """
    PredictIt usually supports:
      https://www.predictit.org/markets/detail/<id>/<slug>
    If this pattern ever changes, you'll still have market_id exported to rebuild later.
    """
    if not market_id:
        return None
    sl = slugify(name)
    if sl:
        return f"https://www.predictit.org/markets/detail/{market_id}/{sl}"
    return f"https://www.predictit.org/markets/detail/{market_id}"


# =============================================================================
# Robust HTTP GET
# =============================================================================

def robust_get(
    session: requests.Session,
    url: str,
    *,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: float = REQUEST_TIMEOUT,
    max_retries: int = HTTP_MAX_RETRIES,
    backoff_seconds: float = HTTP_BACKOFF_SECONDS,
) -> requests.Response:
    h = dict(DEFAULT_HTTP_HEADERS)
    if headers:
        h.update(headers)

    last_resp = None
    for attempt in range(max_retries):
        try:
            resp = session.get(url, params=params, headers=h, timeout=timeout)
            last_resp = resp

            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff_seconds * (1.5 ** attempt))
                continue

            return resp
        except requests.RequestException:
            time.sleep(backoff_seconds * (1.5 ** attempt))
            continue

    if last_resp is None:
        raise requests.RequestException(f"robust_get failed with no response for {url}")
    return last_resp


# =============================================================================
# DATETIME PARSING (best-effort)
# =============================================================================

def _parse_iso_dt(s: str) -> Optional[datetime.datetime]:
    """Best-effort ISO8601 parser (handles trailing 'Z'). Returns timezone-aware UTC if possible."""
    if not s:
        return None
    try:
        ss = str(s).strip()
        if ss.endswith("Z"):
            ss = ss[:-1] + "+00:00"
        dt = datetime.datetime.fromisoformat(ss)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.astimezone(datetime.timezone.utc)
    except Exception:
        return None

def _parse_dt_any(v: Any) -> Optional[datetime.datetime]:
    """
    Parse a datetime from:
      - ISO strings
      - epoch seconds or ms (int/float or numeric strings)
    Returns tz-aware UTC datetime if possible.
    """
    if v is None:
        return None

    # numeric epoch?
    if isinstance(v, (int, float)):
        try:
            x = float(v)
            # ms threshold (rough)
            if x > 1e12:
                x = x / 1000.0
            dt = datetime.datetime.fromtimestamp(x, tz=datetime.timezone.utc)
            return dt
        except Exception:
            return None

    # numeric string?
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        # try iso first
        dt = _parse_iso_dt(s)
        if dt:
            return dt
        # try numeric
        try:
            x = float(s)
            if x > 1e12:
                x = x / 1000.0
            return datetime.datetime.fromtimestamp(x, tz=datetime.timezone.utc)
        except Exception:
            return None

    return None


# =============================================================================
# ORDERBOOK HELPERS (prices + sizes)
# =============================================================================

def _iter_level_prices(levels: Any):
    """
    Extract prices from CLOB levels robustly.
    Levels are typically a list of dicts: {"price": "...", "size": "..."}
    but sometimes can be [price, size] style arrays.
    """
    if not levels:
        return
    for lvl in levels:
        try:
            if isinstance(lvl, dict):
                p = safe_float(lvl.get("price"))
                if p is not None:
                    yield p
                continue

            if isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                a = safe_float(lvl[0])
                b = safe_float(lvl[1])

                # Prefer the value that looks like a probability price in [0,1]
                cand = None
                if a is not None and 0.0 <= a <= 1.0:
                    cand = a
                elif b is not None and 0.0 <= b <= 1.0:
                    cand = b
                elif a is not None:
                    cand = a

                if cand is not None:
                    yield cand
        except Exception:
            continue

def _iter_level_price_size(levels: Any):
    """
    Yield (price, size) from orderbook levels.
    Supports dict levels {"price": "...", "size": "..."} and list/tuple [price, size].
    """
    if not levels:
        return
    for lvl in levels:
        try:
            if isinstance(lvl, dict):
                p = safe_float(lvl.get("price"))
                s = safe_float(lvl.get("size"))  # Polymarket book levels typically include "size"
                if p is None:
                    continue
                if s is None:
                    # sometimes different keys, best-effort
                    s = safe_float(lvl.get("qty")) or safe_float(lvl.get("quantity")) or safe_float(lvl.get("amount"))
                yield (p, s if s is not None else 0.0)
                continue

            if isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                a = safe_float(lvl[0])
                b = safe_float(lvl[1])
                if a is None and b is None:
                    continue

                # choose which is price (in [0,1]) and which is size
                if a is not None and 0.0 <= a <= 1.0:
                    price = a
                    size = b if b is not None else 0.0
                elif b is not None and 0.0 <= b <= 1.0:
                    price = b
                    size = a if a is not None else 0.0
                else:
                    # fallback: assume first is price
                    price = a if a is not None else b
                    size = b if a is not None else 0.0
                yield (price, size if size is not None else 0.0)

        except Exception:
            continue


def best_bid_ask_from_clob_book(book: dict) -> Tuple[Optional[float], Optional[float]]:
    if not book or not isinstance(book, dict):
        return None, None

    bids = book.get("bids") or []
    asks = book.get("asks") or []

    bid_prices = list(_iter_level_prices(bids))
    ask_prices = list(_iter_level_prices(asks))

    best_bid = max(bid_prices) if bid_prices else None   # highest bid
    best_ask = min(ask_prices) if ask_prices else None   # lowest ask

    return best_bid, best_ask


def compute_depth_from_book(book: Optional[dict], *, top_levels: int, window: float) -> Dict[str, Optional[float]]:
    """
    Compute depth metrics from a Polymarket CLOB orderbook.
    - top depth: sum sizes of first N levels per side
    - window depth: sum sizes within `window` of the best price per side
    """
    if not book or not isinstance(book, dict):
        return {
            "bid_top": None, "ask_top": None, "total_top": None,
            "bid_win": None, "ask_win": None, "total_win": None,
        }

    bids = book.get("bids") or []
    asks = book.get("asks") or []

    bid_levels = list(_iter_level_price_size(bids))
    ask_levels = list(_iter_level_price_size(asks))

    # If book is empty, return None
    if not bid_levels and not ask_levels:
        return {
            "bid_top": None, "ask_top": None, "total_top": None,
            "bid_win": None, "ask_win": None, "total_win": None,
        }

    # Best prices
    best_bid = max([p for (p, _) in bid_levels], default=None)
    best_ask = min([p for (p, _) in ask_levels], default=None)

    # Sort levels as they appear logically
    bid_levels_sorted = sorted(bid_levels, key=lambda x: x[0], reverse=True)
    ask_levels_sorted = sorted(ask_levels, key=lambda x: x[0], reverse=False)

    bid_top = sum([s for (_, s) in bid_levels_sorted[:max(0, top_levels)]]) if bid_levels_sorted else 0.0
    ask_top = sum([s for (_, s) in ask_levels_sorted[:max(0, top_levels)]]) if ask_levels_sorted else 0.0

    bid_win = None
    ask_win = None
    if best_bid is not None:
        bid_win = sum([s for (p, s) in bid_levels_sorted if p >= (best_bid - window)])
    if best_ask is not None:
        ask_win = sum([s for (p, s) in ask_levels_sorted if p <= (best_ask + window)])

    total_top = (bid_top + ask_top) if (bid_levels_sorted or ask_levels_sorted) else None
    total_win = None
    if bid_win is not None or ask_win is not None:
        total_win = (bid_win or 0.0) + (ask_win or 0.0)

    return {
        "bid_top": float(bid_top) if bid_levels_sorted else None,
        "ask_top": float(ask_top) if ask_levels_sorted else None,
        "total_top": float(total_top) if total_top is not None else None,
        "bid_win": float(bid_win) if bid_win is not None else None,
        "ask_win": float(ask_win) if ask_win is not None else None,
        "total_win": float(total_win) if total_win is not None else None,
    }


def best_bid_ask_from_predictit_contract(contract: dict) -> Tuple[Optional[float], Optional[float]]:
    bid = safe_float(contract.get("bestSellYesCost"))
    ask = safe_float(contract.get("bestBuyYesCost"))
    return bid, ask


def estimate_predictit_net_edge(cost_sum_asks: float) -> float:
    gross_edge = 1.0 - cost_sum_asks
    if gross_edge <= 0:
        return gross_edge

    mode = PREDICTIT_FEE_MODE
    if mode == "none":
        return gross_edge

    net = gross_edge * (1.0 - max(0.0, min(PREDICTIT_PROFIT_FEE, 1.0)))
    if mode == "conservative":
        net = net * (1.0 - max(0.0, min(PREDICTIT_WITHDRAWAL_FEE, 1.0)))
    return net


# =============================================================================
# KALSHI AUTH
# =============================================================================

def load_private_key(key_path):
    with open(key_path, "rb") as f:
        return serialization.load_pem_private_key(
            f.read(),
            password=None,
            backend=default_backend()
        )

def create_kalshi_signature(private_key, timestamp, method, path):
    path_without_query = path.split('?')[0]
    message = f"{timestamp}{method.upper()}{path_without_query}".encode('utf-8')
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH
        ),
        hashes.SHA256()
    )
    return base64.b64encode(signature).decode('utf-8')


# =============================================================================
# FETCHERS
# =============================================================================

def is_polymarket_market_live(market: dict) -> bool:
    """
    Gamma filters are usually enough, but some resolved/expired markets can still leak in.
    This is a best-effort client-side filter.
    """
    for k in ("closed", "is_closed", "archived", "is_archived", "resolved", "is_resolved"):
        v = market.get(k)
        if isinstance(v, bool) and v:
            return False

    now = datetime.datetime.now(datetime.timezone.utc)

    for k in ("endDate", "end_date", "end_time", "closeTime", "closingTime", "resolveDate", "resolve_date"):
        dt = _parse_dt_any(market.get(k)) if market.get(k) else None
        if dt and dt < now:
            return False

    return True


def fetch_active_markets(session: requests.Session, limit=100) -> List[dict]:
    params = {"closed": "false", "limit": limit, "order": "volume", "ascending": "false"}
    r = robust_get(session, f"{GAMMA_API}/markets", params=params, timeout=REQUEST_TIMEOUT)
    if r.status_code == 200:
        data = r.json()

        if isinstance(data, dict) and "markets" in data:
            data = data.get("markets", [])

        if isinstance(data, list):
            live = [m for m in data if is_polymarket_market_live(m)]
            return live[:MAX_MARKETS_PER_PLATFORM]

    print(f"Error fetching Polymarket markets: {r.status_code} {r.text[:200]}")
    return []


def fetch_order_book(session: requests.Session, token_id: str) -> Optional[dict]:
    r = robust_get(
        session,
        f"{CLOB_API}{CLOB_ORDERBOOK_PATH}",
        params={"token_id": token_id},
        timeout=REQUEST_TIMEOUT,
    )
    if r.status_code == 200:
        return r.json()

    if r.status_code == 404 and "No orderbook exists" in (r.text or ""):
        return None

    print(f"Error fetching Polymarket order book for {token_id}: {r.status_code} {r.text[:120]}")
    return None


def fetch_predictit_markets(session: requests.Session) -> List[dict]:
    r = robust_get(session, PREDICTIT_API, timeout=REQUEST_TIMEOUT)
    if r.status_code == 200:
        mkts = [m for m in r.json().get("markets", []) if m.get("status") == "Open"]
        return mkts[:MAX_MARKETS_PER_PLATFORM]
    print(f"Error fetching PredictIt markets: {r.status_code}")
    return []


def fetch_kalshi_markets(session: requests.Session, limit=100) -> List[dict]:
    if not KALSHI_API_KEY_ID or not os.path.exists(KALSHI_PRIVATE_KEY_PATH):
        print("Kalshi auth not configured. Set KALSHI_API_KEY_ID in .env and place kalshi_private.key in directory.")
        return []
    private_key = load_private_key(KALSHI_PRIVATE_KEY_PATH)
    path = f"/markets?status=open&limit={limit}"
    timestamp = str(int(datetime.datetime.now().timestamp() * 1000))
    signature = create_kalshi_signature(private_key, timestamp, "GET", path)
    headers = {
        'KALSHI-ACCESS-KEY': KALSHI_API_KEY_ID,
        'KALSHI-ACCESS-SIGNATURE': signature,
        'KALSHI-ACCESS-TIMESTAMP': timestamp
    }
    r = robust_get(session, f"{KALSHI_API}{path}", headers=headers, timeout=REQUEST_TIMEOUT)
    if r.status_code == 200:
        return r.json().get("markets", [])[:MAX_MARKETS_PER_PLATFORM]
    print(f"Error fetching Kalshi markets: {r.status_code} - {r.text[:200]}")
    return []


# =============================================================================
# PRIORITIZATION (Stage A)
# =============================================================================

def _to_epoch(dt: Optional[datetime.datetime]) -> Optional[float]:
    if not dt:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.timestamp()

def _minmax_norm(values: List[float], v: Optional[float]) -> float:
    if v is None:
        return 0.5
    if not values:
        return 0.5
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return 0.5
    return (v - lo) / (hi - lo)

def _log1p(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        return math.log1p(max(0.0, float(x)))
    except Exception:
        return None

def polymarket_outcomes_count(m: dict) -> int:
    outs = m.get("outcomes")
    if isinstance(outs, list):
        return len(outs)
    tokens = m.get("tokens")
    if isinstance(tokens, list):
        return len(tokens)
    return 0

def predictit_outcomes_count(m: dict) -> int:
    contracts = m.get("contracts")
    return len(contracts) if isinstance(contracts, list) else 0

def kalshi_outcomes_count(_: dict) -> int:
    return 2

def extract_activity_raw(platform: str, m: dict) -> Optional[float]:
    """
    Attention proxy per your strategy:
    - Polymarket: volume24hr (or similar) fallback to volume
    - PredictIt: totalSharesTraded (market) fallback to sum(contracts.totalSharesTraded)
    - Kalshi: best-effort from list fields (volume/open_interest/etc.)
    """
    if platform == "Polymarket":
        for k in ("volume24hr", "volume_24hr", "volume24h", "volume_24h", "volume"):
            v = safe_float(m.get(k))
            if v is not None:
                return v
        return None

    if platform == "PredictIt":
        v = safe_float(m.get("totalSharesTraded"))
        if v is not None:
            return v
        total = 0.0
        ok = False
        for c in (m.get("contracts") or []):
            tv = safe_float(c.get("totalSharesTraded"))
            if tv is not None:
                total += tv
                ok = True
        return total if ok else None

    if platform == "Kalshi":
        for k in ("volume", "volume_24h", "volume24h", "open_interest", "openInterest", "openinterest"):
            v = safe_float(m.get(k))
            if v is not None:
                return v
        return None

    return None

def extract_times(platform: str, m: dict) -> Dict[str, Optional[datetime.datetime]]:
    """
    Extract created/updated/close/end times best-effort from each platform payload.
    """
    created = None
    updated = None
    close = None
    end = None

    if platform == "Polymarket":
        for k in ("createdAt", "created_at"):
            created = created or _parse_dt_any(m.get(k))
        for k in ("updatedAt", "updated_at"):
            updated = updated or _parse_dt_any(m.get(k))
        for k in ("closeTime", "closingTime", "close_time"):
            close = close or _parse_dt_any(m.get(k))
        for k in ("endDate", "end_date", "end_time", "resolveDate", "resolve_date"):
            end = end or _parse_dt_any(m.get(k))

    elif platform == "PredictIt":
        for k in ("createdAt", "created_at", "createdDate", "created_date"):
            created = created or _parse_dt_any(m.get(k))
        for k in ("updatedAt", "updated_at", "timeStamp", "timestamp"):
            updated = updated or _parse_dt_any(m.get(k))
        for k in ("closeTime", "close_time", "dateEnd", "endDate", "end_date"):
            end = end or _parse_dt_any(m.get(k))

    elif platform == "Kalshi":
        for k in ("created_time", "createdAt", "created_at"):
            created = created or _parse_dt_any(m.get(k))
        for k in ("updated_time", "updatedAt", "updated_at"):
            updated = updated or _parse_dt_any(m.get(k))
        for k in ("close_time", "closeTime"):
            close = close or _parse_dt_any(m.get(k))
        for k in ("expiration_time", "expirationTime", "settlement_time", "settlementTime"):
            end = end or _parse_dt_any(m.get(k))

    return {"created": created, "updated": updated, "close": close, "end": end}

def choose_recency_dt(times: Dict[str, Optional[datetime.datetime]]) -> Optional[datetime.datetime]:
    """
    Recency for scoring: prefer updated, then created, then close, then end.
    """
    for k in ("updated", "created", "close", "end"):
        dt = times.get(k)
        if isinstance(dt, datetime.datetime):
            return dt
    return None

def score_markets(platform: str, markets: List[dict]) -> Dict[str, dict]:
    """
    Returns {market_id_key: {...meta...}}
    Uses per-platform min-max normalization so each venue can be ranked internally.
    """
    rows = []
    for idx, m in enumerate(markets):
        if platform == "Polymarket":
            mid = str(m.get("id", idx))
            n_out = polymarket_outcomes_count(m)
        elif platform == "PredictIt":
            mid = str(m.get("id", idx))
            n_out = predictit_outcomes_count(m)
        else:
            mid = str(m.get("ticker") or m.get("id") or idx)
            n_out = kalshi_outcomes_count(m)

        activity = extract_activity_raw(platform, m)
        times = extract_times(platform, m)
        recency_dt = choose_recency_dt(times)

        rows.append((mid, activity, recency_dt, n_out, times))

    act_logs = [x for x in (_log1p(r[1]) for r in rows) if x is not None]
    rec_epochs = [x for x in (_to_epoch(r[2]) for r in rows) if x is not None]

    out = {}
    for mid, activity, recency_dt, n_out, times in rows:
        act_n = _minmax_norm(act_logs, _log1p(activity))
        rec_n = _minmax_norm(rec_epochs, _to_epoch(recency_dt))

        out_n = 0.0
        if n_out and n_out > 0:
            out_n = min(n_out, MAX_OUTCOMES_TO_FETCH) / float(MAX_OUTCOMES_TO_FETCH)

        priority = (
            WEIGHT_ACTIVITY * act_n
            + WEIGHT_RECENCY * rec_n
            - WEIGHT_OUTCOMES_PENALTY * out_n
        )

        out[mid] = {
            "activity_raw": activity,
            "times": times,
            "recency_dt": recency_dt,
            "priority": float(priority),
            "outcomes_n": int(n_out),
        }
    return out


# =============================================================================
# NORMALIZATION
# =============================================================================

def _ensure_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str) and x.strip().startswith("[") and x.strip().endswith("]"):
        try:
            return json.loads(x)
        except Exception:
            return []
    return []

def extract_polymarket_outcomes(market: dict) -> List[Dict[str, Optional[str]]]:
    """
    Prefer the explicit outcomes + clobTokenIds mapping if present.
    Only fall back to tokens[] if necessary.
    """
    out: List[Dict[str, Optional[str]]] = []

    outcomes_arr = _ensure_list(market.get("outcomes"))
    clob_ids = _ensure_list(market.get("clobTokenIds")) or _ensure_list(market.get("clob_token_ids"))
    token_ids = _ensure_list(market.get("tokenIds")) or _ensure_list(market.get("token_ids"))

    if outcomes_arr and clob_ids and len(outcomes_arr) == len(clob_ids):
        for name, tid in zip(outcomes_arr, clob_ids):
            out.append({"name": str(name), "token_id": str(tid) if tid is not None else None})
        return out

    if outcomes_arr and token_ids and len(outcomes_arr) == len(token_ids):
        for name, tid in zip(outcomes_arr, token_ids):
            out.append({"name": str(name), "token_id": str(tid) if tid is not None else None})
        return out

    tokens = _ensure_list(market.get("tokens"))
    if tokens:
        for t in tokens:
            if not isinstance(t, dict):
                continue

            name = t.get("outcome") or t.get("name") or t.get("title") or "UNKNOWN"
            token_id = (
                t.get("clobTokenId")
                or t.get("clob_token_id")
                or t.get("clobTokenID")
            )
            if token_id is None:
                token_id = t.get("token_id") or t.get("tokenId") or t.get("tokenID")

            if isinstance(token_id, list):
                token_id = token_id[0] if token_id else None

            out.append({"name": str(name), "token_id": str(token_id) if token_id is not None else None})

        return out

    return out


def compute_spread_metrics(outcomes: List[OutcomeQuote]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    spreads = []
    for o in outcomes:
        if o.best_bid is None or o.best_ask is None:
            continue
        if o.best_ask <= 0:
            continue
        s = o.best_ask - o.best_bid
        if s is None:
            continue
        spreads.append(float(s))
    if not spreads:
        return None, None, None
    return float(sum(spreads) / len(spreads)), float(min(spreads)), float(max(spreads))


def _iso_or_none(dt: Optional[datetime.datetime]) -> Optional[str]:
    if isinstance(dt, datetime.datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.astimezone(datetime.timezone.utc).isoformat()
    return None


def normalize_polymarket(session: requests.Session, market: dict, meta: Optional[dict] = None) -> Optional[MarketSnapshot]:
    q = market.get("question") or market.get("title") or "Untitled"
    mid = str(market.get("id")) if market.get("id") is not None else None

    pairs = extract_polymarket_outcomes(market)
    if not pairs:
        return None

    pairs = pairs[:MAX_OUTCOMES_TO_FETCH]
    outcomes: List[OutcomeQuote] = [OutcomeQuote(name=p["name"], token_id=p["token_id"]) for p in pairs]

    bet_type = classify_bet_type([o.name for o in outcomes])
    hint, other = compute_exhaustiveness_hint([o.name for o in outcomes])

    slug = market.get("slug") or market.get("marketSlug") or None
    url = f"https://polymarket.com/market/{slug}" if slug else None

    # Orderbooks + depth (Polymarket only)
    depth_bid_top_sum = 0.0
    depth_ask_top_sum = 0.0
    depth_bid_win_sum = 0.0
    depth_ask_win_sum = 0.0
    depth_any_top = False
    depth_any_win = False

    for oq in outcomes:
        if not oq.token_id:
            continue
        book = fetch_order_book(session, oq.token_id)
        bid, ask = best_bid_ask_from_clob_book(book)
        oq.best_bid = bid
        oq.best_ask = ask

        depth = compute_depth_from_book(book, top_levels=DEPTH_TOP_LEVELS, window=DEPTH_PRICE_WINDOW)
        oq.raw = {"book": book, "depth": depth} if book else {"depth": depth}

        if depth.get("bid_top") is not None:
            depth_bid_top_sum += float(depth["bid_top"])
            depth_any_top = True
        if depth.get("ask_top") is not None:
            depth_ask_top_sum += float(depth["ask_top"])
            depth_any_top = True

        if depth.get("bid_win") is not None:
            depth_bid_win_sum += float(depth["bid_win"])
            depth_any_win = True
        if depth.get("ask_win") is not None:
            depth_ask_win_sum += float(depth["ask_win"])
            depth_any_win = True

    spread_avg, spread_min, spread_max = compute_spread_metrics(outcomes)

    vol = safe_float(market.get("volume"))

    times = (meta or {}).get("times") or {}
    created_dt = times.get("created")
    updated_dt = times.get("updated")
    close_dt = times.get("close")
    end_dt = times.get("end")

    recency_dt = (meta or {}).get("recency_dt")
    recency_ts = _iso_or_none(recency_dt)
    now = datetime.datetime.now(datetime.timezone.utc)
    recency_age_seconds = None
    if isinstance(recency_dt, datetime.datetime):
        recency_age_seconds = float((now - recency_dt).total_seconds())

    return MarketSnapshot(
        platform="Polymarket",
        market_id=mid,
        question=q,
        bet_type=bet_type,
        outcomes=outcomes,
        volume=vol,
        activity_raw=(meta or {}).get("activity_raw"),
        url=url,
        mutually_exclusive=True,
        exhaustive_hint=hint,
        has_other_outcome=other,

        created_ts=_iso_or_none(created_dt),
        updated_ts=_iso_or_none(updated_dt),
        close_ts=_iso_or_none(close_dt),
        end_ts=_iso_or_none(end_dt),
        recency_ts=recency_ts,
        recency_age_seconds=recency_age_seconds,

        priority_score=(meta or {}).get("priority"),
        outcomes_count_raw=(meta or {}).get("outcomes_n"),

        spread_avg=spread_avg,
        spread_min=spread_min,
        spread_max=spread_max,

        depth_bid_top=(float(depth_bid_top_sum) if depth_any_top else None),
        depth_ask_top=(float(depth_ask_top_sum) if depth_any_top else None),
        depth_total_top=(float(depth_bid_top_sum + depth_ask_top_sum) if depth_any_top else None),

        depth_bid_window=(float(depth_bid_win_sum) if depth_any_win else None),
        depth_ask_window=(float(depth_ask_win_sum) if depth_any_win else None),
        depth_total_window=(float(depth_bid_win_sum + depth_ask_win_sum) if depth_any_win else None),

        fetched_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


def normalize_predictit(market: dict, meta: Optional[dict] = None) -> Optional[MarketSnapshot]:
    q = market.get("name") or "Untitled"
    contracts = market.get("contracts", []) or []
    if len(contracts) == 0:
        return None

    contracts = contracts[:MAX_OUTCOMES_TO_FETCH]

    outcomes: List[OutcomeQuote] = []
    for c in contracts:
        name = str(c.get("name", "UNKNOWN"))
        bid, ask = best_bid_ask_from_predictit_contract(c)
        outcomes.append(OutcomeQuote(name=name, best_bid=bid, best_ask=ask, raw=c))

    bet_type = classify_bet_type([o.name for o in outcomes])
    hint, other = compute_exhaustiveness_hint([o.name for o in outcomes])
    mid = str(market.get("id")) if market.get("id") is not None else None

    spread_avg, spread_min, spread_max = compute_spread_metrics(outcomes)

    times = (meta or {}).get("times") or {}
    created_dt = times.get("created")
    updated_dt = times.get("updated")
    close_dt = times.get("close")
    end_dt = times.get("end")

    recency_dt = (meta or {}).get("recency_dt")
    recency_ts = _iso_or_none(recency_dt)
    now = datetime.datetime.now(datetime.timezone.utc)
    recency_age_seconds = None
    if isinstance(recency_dt, datetime.datetime):
        recency_age_seconds = float((now - recency_dt).total_seconds())

    # NEW: PredictIt URL is usually NOT present in the API payload — construct it
    url = market.get("url") or predictit_market_url(mid, q)

    return MarketSnapshot(
        platform="PredictIt",
        market_id=mid,
        question=q,
        bet_type=bet_type,
        outcomes=outcomes,
        volume=None,
        activity_raw=(meta or {}).get("activity_raw"),
        url=url,
        mutually_exclusive=True,
        exhaustive_hint=hint,
        has_other_outcome=other,

        created_ts=_iso_or_none(created_dt),
        updated_ts=_iso_or_none(updated_dt),
        close_ts=_iso_or_none(close_dt),
        end_ts=_iso_or_none(end_dt),
        recency_ts=recency_ts,
        recency_age_seconds=recency_age_seconds,

        priority_score=(meta or {}).get("priority"),
        outcomes_count_raw=(meta or {}).get("outcomes_n"),

        spread_avg=spread_avg,
        spread_min=spread_min,
        spread_max=spread_max,

        # PredictIt: no true orderbook depth from this endpoint
        depth_bid_top=None,
        depth_ask_top=None,
        depth_total_top=None,
        depth_bid_window=None,
        depth_ask_window=None,
        depth_total_window=None,

        fetched_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


def normalize_kalshi(market: dict, meta: Optional[dict] = None) -> Optional[MarketSnapshot]:
    q = market.get("title") or "Untitled"
    mid = str(market.get("ticker")) if market.get("ticker") else str(market.get("id")) if market.get("id") else None

    outcomes = [
        OutcomeQuote(
            name="YES",
            best_bid=(safe_float(market.get("yes_bid")) / 100.0) if market.get("yes_bid") is not None else None,
            best_ask=(safe_float(market.get("yes_ask")) / 100.0) if market.get("yes_ask") is not None else None,
            raw=market
        ),
        OutcomeQuote(name="NO", best_bid=None, best_ask=None, raw=None)
    ]

    bet_type = BET_BINARY_YN
    hint, other = compute_exhaustiveness_hint([o.name for o in outcomes])

    spread_avg, spread_min, spread_max = compute_spread_metrics(outcomes)

    times = (meta or {}).get("times") or {}
    created_dt = times.get("created")
    updated_dt = times.get("updated")
    close_dt = times.get("close")
    end_dt = times.get("end")

    recency_dt = (meta or {}).get("recency_dt")
    recency_ts = _iso_or_none(recency_dt)
    now = datetime.datetime.now(datetime.timezone.utc)
    recency_age_seconds = None
    if isinstance(recency_dt, datetime.datetime):
        recency_age_seconds = float((now - recency_dt).total_seconds())

    return MarketSnapshot(
        platform="Kalshi",
        market_id=mid,
        question=q,
        bet_type=bet_type,
        outcomes=outcomes,
        volume=None,
        activity_raw=(meta or {}).get("activity_raw"),
        url=None,
        mutually_exclusive=True,
        exhaustive_hint=hint,
        has_other_outcome=other,

        created_ts=_iso_or_none(created_dt),
        updated_ts=_iso_or_none(updated_dt),
        close_ts=_iso_or_none(close_dt),
        end_ts=_iso_or_none(end_dt),
        recency_ts=recency_ts,
        recency_age_seconds=recency_age_seconds,

        priority_score=(meta or {}).get("priority"),
        outcomes_count_raw=(meta or {}).get("outcomes_n"),

        spread_avg=spread_avg,
        spread_min=spread_min,
        spread_max=spread_max,

        # Kalshi: depth not fetched here
        depth_bid_top=None,
        depth_ask_top=None,
        depth_total_top=None,
        depth_bid_window=None,
        depth_ask_window=None,
        depth_total_window=None,

        fetched_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


# =============================================================================
# INEFFICIENCY DETECTION
# =============================================================================

def sum_best_asks(outcomes: List[OutcomeQuote]) -> Optional[float]:
    if not outcomes:
        return None
    asks = []
    for o in outcomes:
        if o.best_ask is None or o.best_ask <= 0:
            return None
        asks.append(o.best_ask)
    return float(sum(asks))


def min_full_set_edge_required(snapshot: MarketSnapshot) -> float:
    base = FULL_SET_EDGE_MIN
    if snapshot.bet_type in (BET_MULTI, BET_BUCKET) and snapshot.platform == "PredictIt":
        if not snapshot.has_other_outcome and snapshot.exhaustive_hint.startswith("UNKNOWN"):
            return FULL_SET_EDGE_MIN_NONEXHAUSTIVE
    return base


def detect_structural_opps(snapshot: MarketSnapshot) -> List[Opportunity]:
    opps: List[Opportunity] = []
    n = len(snapshot.outcomes)

    cost = sum_best_asks(snapshot.outcomes)

    if snapshot.bet_type in (BET_BINARY_YN, BET_BINARY_2WAY, BET_THRESHOLD) and n == 2 and cost is not None:
        gross_edge = 1.0 - cost
        if gross_edge > 0:
            net_edge = gross_edge
            warnings = []
            conf = "HIGH"

            if snapshot.platform == "PredictIt":
                net_edge = estimate_predictit_net_edge(cost)
                if net_edge < gross_edge:
                    warnings.append(f"PredictIt fee model applied: gross={gross_edge*100:.2f}% → net≈{net_edge*100:.2f}%")
                    conf = "MED"

            if net_edge >= BINARY_EDGE_MIN:
                opps.append(Opportunity(
                    opp_type=OPP_BINARY,
                    platform=snapshot.platform,
                    question=snapshot.question,
                    bet_type=snapshot.bet_type,
                    edge_gross=gross_edge,
                    edge_net=net_edge,
                    cost_sum_asks=cost,
                    details=f"2-outcome full-set sum_asks={cost:.4f} gross_edge={gross_edge*100:.2f}% net_edge≈{net_edge*100:.2f}% | exhaust={snapshot.exhaustive_hint}",
                    suggested_size_usd=min(
                        MAX_TRADE_USD,
                        MAX_TRADE_USD * (net_edge / max(AUTO_TRADE_MIN_EDGE, 1e-6))
                    ),
                    confidence=conf,
                    warnings=warnings,
                    actions=[
                        {"action": "BUY_OUTCOME", "outcome": snapshot.outcomes[0].name, "price": snapshot.outcomes[0].best_ask},
                        {"action": "BUY_OUTCOME", "outcome": snapshot.outcomes[1].name, "price": snapshot.outcomes[1].best_ask},
                    ],
                    market_id=snapshot.market_id,
                    url=snapshot.url,
                ))

    if snapshot.bet_type in (BET_MULTI, BET_BUCKET) and n >= 3 and cost is not None:
        gross_edge = 1.0 - cost
        if gross_edge > 0:
            net_edge = gross_edge
            warnings = []
            conf = "MED"

            req = min_full_set_edge_required(snapshot)
            if req > FULL_SET_EDGE_MIN:
                warnings.append(f"Exhaustiveness uncertain (no 'Other'); required edge raised to {req*100:.2f}%")
                conf = "LOW"

            if snapshot.platform == "PredictIt":
                net_edge = estimate_predictit_net_edge(cost)
                if net_edge < gross_edge:
                    warnings.append(f"PredictIt fee model applied: gross={gross_edge*100:.2f}% → net≈{net_edge*100:.2f}%")
                    conf = "LOW" if conf == "LOW" else "MED"

            if net_edge >= req:
                actions = [{"action": "BUY_OUTCOME", "outcome": o.name, "price": o.best_ask} for o in snapshot.outcomes]
                opps.append(Opportunity(
                    opp_type=OPP_FULL_SET,
                    platform=snapshot.platform,
                    question=snapshot.question,
                    bet_type=snapshot.bet_type,
                    edge_gross=gross_edge,
                    edge_net=net_edge,
                    cost_sum_asks=cost,
                    details=f"Full-set sum_asks={cost:.4f} gross_edge={gross_edge*100:.2f}% net_edge≈{net_edge*100:.2f}% | exhaust={snapshot.exhaustive_hint}",
                    suggested_size_usd=min(
                        MAX_TRADE_USD,
                        MAX_TRADE_USD * (net_edge / max(AUTO_TRADE_MIN_EDGE, 1e-6))
                    ),
                    confidence=conf,
                    warnings=warnings,
                    actions=actions,
                    market_id=snapshot.market_id,
                    url=snapshot.url,
                ))

    return opps


# =============================================================================
# MATCHING (CROSS PLATFORM) - placeholder
# =============================================================================

def find_matching_market(question: str, other_snaps: List[MarketSnapshot], threshold=MATCH_THRESHOLD) -> Tuple[Optional[MarketSnapshot], float]:
    best = None
    best_ratio = 0.0
    for s in other_snaps:
        ratio = SequenceMatcher(None, question.lower(), s.question.lower()).ratio()
        if ratio > best_ratio and ratio >= threshold:
            best_ratio = ratio
            best = s
    return best, best_ratio


# =============================================================================
# TRADE EXECUTION (SAFE: PAPER BY DEFAULT)
# =============================================================================

class TradeExecutor:
    def __init__(self, trading_enabled: bool, paper: bool):
        self.trading_enabled = trading_enabled
        self.paper = paper

    def execute_opportunity(self, opp: Opportunity):
        if not self.trading_enabled:
            return

        if opp.edge_net < AUTO_TRADE_MIN_EDGE:
            return

        if self.paper:
            print(f"[PAPER TRADE] Would execute: {opp.opp_type} on {opp.platform} | net_edge≈{opp.edge_net*100:.2f}% | size≈${opp.suggested_size_usd:.2f}")
            for a in opp.actions[:20]:
                print(f"  - {a}")
            if opp.warnings:
                for w in opp.warnings[:10]:
                    print(f"  [WARN] {w}")
            return

        raise NotImplementedError("Real execution not implemented. Wire platform brokers here.")


# =============================================================================
# SCAN ENGINE
# =============================================================================

def snapshot_to_row(snapshot: MarketSnapshot) -> dict:
    preview = []
    for o in snapshot.outcomes[:6]:
        a = "NA" if o.best_ask is None else f"{o.best_ask:.3f}"
        b = "NA" if o.best_bid is None else f"{o.best_bid:.3f}"
        preview.append(f"{o.name} (bid {b} / ask {a})")

    cost = sum_best_asks(snapshot.outcomes)
    gross_edge = (1.0 - cost) if cost is not None else None
    net_edge = None
    if cost is not None:
        net_edge = estimate_predictit_net_edge(cost) if snapshot.platform == "PredictIt" else gross_edge

    return {
        "bet_type": snapshot.bet_type,
        "platform": snapshot.platform,
        "question": snapshot.question,
        "market_id": snapshot.market_id,
        "outcomes_n": len(snapshot.outcomes),

        "sum_best_asks": cost,
        "edge_gross": gross_edge,
        "edge_net": net_edge,

        "exhaustive_hint": snapshot.exhaustive_hint,
        "has_other_outcome": snapshot.has_other_outcome,

        # Volume/attention proxies
        "volume": snapshot.volume,
        "activity_raw": snapshot.activity_raw,

        # Recency fields (newest/oldest sorting can use recency_age_seconds)
        "created_ts": snapshot.created_ts,
        "updated_ts": snapshot.updated_ts,
        "close_ts": snapshot.close_ts,
        "end_ts": snapshot.end_ts,
        "recency_ts": snapshot.recency_ts,
        "recency_age_seconds": snapshot.recency_age_seconds,

        # Priority
        "priority_score": snapshot.priority_score,
        "outcomes_count_raw": snapshot.outcomes_count_raw,

        # Spread
        "spread_avg": snapshot.spread_avg,
        "spread_min": snapshot.spread_min,
        "spread_max": snapshot.spread_max,

        # Depth (Polymarket only)
        "depth_bid_top": snapshot.depth_bid_top,
        "depth_ask_top": snapshot.depth_ask_top,
        "depth_total_top": snapshot.depth_total_top,
        "depth_bid_window": snapshot.depth_bid_window,
        "depth_ask_window": snapshot.depth_ask_window,
        "depth_total_window": snapshot.depth_total_window,

        "outcomes_preview": " | ".join(preview),
        "url": snapshot.url,
        "fetched_at": snapshot.fetched_at,
    }


def scan_markets(enable_polymarket=True, enable_kalshi=False, enable_predictit=True):
    print("Starting scan...")

    session = requests.Session()
    session.headers.update(DEFAULT_HTTP_HEADERS)

    poly_markets = fetch_active_markets(session, limit=POLY_LIMIT) if enable_polymarket else []
    kalshi_markets = fetch_kalshi_markets(session, limit=KALSHI_LIMIT) if enable_kalshi else []
    predictit_markets = fetch_predictit_markets(session) if enable_predictit else []

    # compute per-platform priority metadata (Stage A)
    poly_meta = score_markets("Polymarket", poly_markets) if enable_polymarket else {}
    pi_meta = score_markets("PredictIt", predictit_markets) if enable_predictit else {}
    kalshi_meta = score_markets("Kalshi", kalshi_markets) if enable_kalshi else {}

    # prefilter top-K per platform by priority (Stage B reduces orderbook calls)
    if enable_polymarket and poly_markets:
        poly_markets = sorted(
            poly_markets,
            key=lambda m: poly_meta.get(str(m.get("id")), {}).get("priority", 0.0),
            reverse=True
        )[:min(PREFILTER_TOPK_POLY, len(poly_markets))]

    if enable_predictit and predictit_markets:
        predictit_markets = sorted(
            predictit_markets,
            key=lambda m: pi_meta.get(str(m.get("id")), {}).get("priority", 0.0),
            reverse=True
        )[:min(PREFILTER_TOPK_PREDICTIT, len(predictit_markets))]

    if enable_kalshi and kalshi_markets:
        kalshi_markets = sorted(
            kalshi_markets,
            key=lambda m: kalshi_meta.get(str(m.get("ticker") or m.get("id")), {}).get("priority", 0.0),
            reverse=True
        )[:min(PREFILTER_TOPK_KALSHI, len(kalshi_markets))]

    snapshots: List[MarketSnapshot] = []

    if enable_polymarket:
        for m in poly_markets:
            mid = str(m.get("id")) if m.get("id") is not None else None
            snap = normalize_polymarket(session, m, meta=poly_meta.get(mid, {}))
            if snap:
                snapshots.append(snap)

    if enable_predictit:
        for m in predictit_markets:
            mid = str(m.get("id")) if m.get("id") is not None else None
            snap = normalize_predictit(m, meta=pi_meta.get(mid, {}))
            if snap:
                snapshots.append(snap)

    if enable_kalshi:
        for m in kalshi_markets:
            mid = str(m.get("ticker") or m.get("id")) if (m.get("ticker") or m.get("id")) else None
            snap = normalize_kalshi(m, meta=kalshi_meta.get(mid, {}))
            if snap:
                snapshots.append(snap)

    opportunities: List[Opportunity] = []
    opportunities_rows: List[dict] = []
    arbs_found_msgs: List[str] = []
    executor = TradeExecutor(trading_enabled=TRADING_ENABLED, paper=PAPER_TRADING)

    # NEW: build opportunities export rows by merging the SAME market row schema + opp fields
    for s in snapshots:
        base_row = snapshot_to_row(s)

        opps = detect_structural_opps(s)
        for opp in opps:
            opportunities.append(opp)

            msg = (
                f"[{opp.opp_type}] {opp.platform} | {opp.question} | "
                f"net_edge≈{opp.edge_net*100:.2f}% (gross {opp.edge_gross*100:.2f}%) | "
                f"sum_asks={opp.cost_sum_asks:.4f} | conf={opp.confidence}"
            )
            send_alert(msg)
            arbs_found_msgs.append(msg)
            executor.execute_opportunity(opp)

            opp_row = dict(base_row)
            opp_row.update({
                # opportunity-specific fields
                "opp_type": opp.opp_type,
                "edge_net_pct": opp.edge_net * 100.0,
                "edge_gross_pct": opp.edge_gross * 100.0,
                "confidence": opp.confidence,
                "warnings": " | ".join(opp.warnings[:3]) if opp.warnings else "",
                "suggested_size_usd": opp.suggested_size_usd,
                "details": opp.details,
                # explicit linkage (redundant but convenient)
                "url": opp.url or base_row.get("url"),
                "market_id": opp.market_id or base_row.get("market_id"),
            })
            opportunities_rows.append(opp_row)

    market_rows = [snapshot_to_row(s) for s in snapshots]

    counts = {
        "polymarket_fetched": len(poly_markets),
        "predictit_fetched": len(predictit_markets),
        "kalshi_fetched": len(kalshi_markets),
        "snapshots_total": len(snapshots),
        "opps_total": len(opportunities),
        "by_bet_type": {},
        "by_platform": {},
    }
    for s in snapshots:
        counts["by_bet_type"][s.bet_type] = counts["by_bet_type"].get(s.bet_type, 0) + 1
        counts["by_platform"][s.platform] = counts["by_platform"].get(s.platform, 0) + 1

    print_grouped_terminal(snapshots, opportunities)

    print("Scan complete.")
    out_dir = os.getenv("DASHBOARD_OUTDIR", "./dashboard")
    export_dashboard_snapshot(out_dir, market_rows, opportunities_rows, counts, arbs_found_msgs)
    return market_rows, opportunities_rows, arbs_found_msgs, counts, snapshots


def print_grouped_terminal(snapshots: List[MarketSnapshot], opportunities: List[Opportunity]):
    def sort_key(s: MarketSnapshot):
        ps = s.priority_score if s.priority_score is not None else -999
        return (s.bet_type, s.platform, -ps, (s.volume or 0.0) * -1)

    snaps_sorted = sorted(snapshots, key=sort_key)

    print("\n==============================")
    print("SCAN OUTPUT (bet type → platform → market)")
    print("==============================")

    current_bt = None
    current_pf = None

    for s in snaps_sorted:
        if s.bet_type != current_bt:
            current_bt = s.bet_type
            current_pf = None
            print(f"\n--- BET TYPE: {current_bt} ---")

        if s.platform != current_pf:
            current_pf = s.platform
            print(f"\n  > PLATFORM: {current_pf}")

        cost = sum_best_asks(s.outcomes)
        gross_edge = (1.0 - cost) if cost is not None else None
        net_edge = None
        if cost is not None:
            net_edge = estimate_predictit_net_edge(cost) if s.platform == "PredictIt" else gross_edge

        gross_str = "NA" if gross_edge is None else f"{gross_edge*100:.2f}%"
        net_str = "NA" if net_edge is None else f"{net_edge*100:.2f}%"
        cost_str = "NA" if cost is None else f"{cost:.4f}"
        pr_str = "NA" if s.priority_score is None else f"{s.priority_score:.3f}"
        spr_str = "NA" if s.spread_avg is None else f"{s.spread_avg:.4f}"

        print(f"    - {s.question}")
        print(f"      priority={pr_str} activity={s.activity_raw if s.activity_raw is not None else 'NA'} recency={s.recency_ts or 'NA'}")
        print(f"      outcomes={len(s.outcomes)} sum_best_asks={cost_str} gross_edge={gross_str} net_edge≈{net_str} spread_avg={spr_str} | exhaust={s.exhaustive_hint}")

        if s.platform == "Polymarket":
            dt_top = s.depth_total_top if s.depth_total_top is not None else "NA"
            dt_win = s.depth_total_window if s.depth_total_window is not None else "NA"
            print(f"      depth_top(N={DEPTH_TOP_LEVELS})={dt_top} | depth_window(±{DEPTH_PRICE_WINDOW})={dt_win}")

        for o in s.outcomes[:6]:
            a = "NA" if o.best_ask is None else f"{o.best_ask:.3f}"
            b = "NA" if o.best_bid is None else f"{o.best_bid:.3f}"
            print(f"        • {o.name}: bid={b} ask={a}")

    if opportunities:
        print("\n==============================")
        print("OPPORTUNITIES FOUND (sorted by net edge)")
        print("==============================")
        for o in sorted(opportunities, key=lambda x: x.edge_net, reverse=True):
            warn = f" | WARN: {o.warnings[0]}" if o.warnings else ""
            print(f"- [{o.opp_type}] {o.platform} | net≈{o.edge_net*100:.2f}% (gross {o.edge_gross*100:.2f}%) | {o.question} | conf={o.confidence}{warn}")


# =============================================================================
# TERMINAL MODE
# =============================================================================

def main_terminal():
    enable_polymarket = os.getenv("ENABLE_POLYMARKET", "true").lower() == "true"
    enable_kalshi = os.getenv("ENABLE_KALSHI", "false").lower() == "true"
    enable_predictit = os.getenv("ENABLE_PREDICTIT", "true").lower() == "true"

    print("Starting Inefficiency Scanner (Terminal Mode)...")
    print(f"TRADING_ENABLED={TRADING_ENABLED} PAPER_TRADING={PAPER_TRADING} AUTO_TRADE_MIN_EDGE={AUTO_TRADE_MIN_EDGE}")
    print(f"PredictIt fees mode={PREDICTIT_FEE_MODE} profit_fee={PREDICTIT_PROFIT_FEE} withdrawal_fee={PREDICTIT_WITHDRAWAL_FEE}")
    print(f"Prefilter: Poly={PREFILTER_TOPK_POLY} | PredictIt={PREFILTER_TOPK_PREDICTIT} | Kalshi={PREFILTER_TOPK_KALSHI}")
    print(f"Depth: topN={DEPTH_TOP_LEVELS} | window=±{DEPTH_PRICE_WINDOW}")

    while True:
        _market_rows, _opportunities_rows, _arbs_found, counts, _snaps = scan_markets(
            enable_polymarket=enable_polymarket,
            enable_kalshi=enable_kalshi,
            enable_predictit=enable_predictit
        )
        print(f"\nSummary: snapshots={counts['snapshots_total']} opps={counts['opps_total']}")
        print(f"Fetched: Poly={counts['polymarket_fetched']} | Kalshi={counts['kalshi_fetched']} | PredictIt={counts['predictit_fetched']}")
        print(f"By bet type: {counts['by_bet_type']}")
        print(f"By platform: {counts['by_platform']}")
        print(f"Sleeping {SCAN_INTERVAL_SECONDS}s...\n")
        time.sleep(SCAN_INTERVAL_SECONDS)


# =============================================================================
# STREAMLIT UI MODE
# =============================================================================

def main_ui():
    st.title("Prediction Markets Inefficiency Dashboard")
    st.caption("bet type → platform → market | structural inefficiencies + (paper) execution hooks")
    st.caption("Note: PredictIt 'net edge' applies a conservative fee heuristic to reduce false positives.")

    try:
        from streamlit_autorefresh import st_autorefresh
        tick = st_autorefresh(interval=SCAN_INTERVAL_SECONDS * 1000, key="auto_refresh_tick")
    except ImportError:
        tick = None
        st.warning("Auto-refresh not enabled. Run: pip install streamlit-autorefresh")

    for k, v in {
        "market_rows": [],
        "opportunities_rows": [],
        "arbs_found": [],
        "counts": {},
        "logs": "",
        "last_tick": -1,
        "snapshots": [],
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

    colA, colB, colC = st.columns(3)
    with colA:
        enable_polymarket = st.checkbox("Enable Polymarket", value=True)
    with colB:
        enable_kalshi = st.checkbox("Enable Kalshi", value=False)
    with colC:
        enable_predictit = st.checkbox("Enable PredictIt", value=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Scan interval (s)", SCAN_INTERVAL_SECONDS)
    with col2:
        st.metric("Trading enabled", "YES" if TRADING_ENABLED else "NO")
    with col3:
        st.metric("Paper trading", "YES" if PAPER_TRADING else "NO")
    with col4:
        st.metric("Auto-trade min net edge", f"{AUTO_TRADE_MIN_EDGE*100:.2f}%")

    c5, c6, c7 = st.columns(3)
    with c5:
        st.metric("Prefilter Poly (top K)", PREFILTER_TOPK_POLY)
    with c6:
        st.metric("Prefilter PredictIt (top K)", PREFILTER_TOPK_PREDICTIT)
    with c7:
        st.metric("Prefilter Kalshi (top K)", PREFILTER_TOPK_KALSHI)

    c8, c9 = st.columns(2)
    with c8:
        st.metric("Depth top N", DEPTH_TOP_LEVELS)
    with c9:
        st.metric("Depth window (±price)", DEPTH_PRICE_WINDOW)

    scan_button = st.button("Scan Now")
    should_scan = scan_button or (tick is not None and tick != st.session_state.last_tick)

    if should_scan:
        st.session_state.last_tick = tick if tick is not None else st.session_state.last_tick

        class Tee(io.TextIOBase):
            def __init__(self, *streams):
                self.streams = streams
            def write(self, s):
                for stream in self.streams:
                    stream.write(s)
                    stream.flush()
                return len(s)
            def flush(self):
                for stream in self.streams:
                    stream.flush()

        buf = io.StringIO()
        tee = Tee(sys.stdout, buf)

        with st.spinner("Scanning markets..."):
            with redirect_stdout(tee), redirect_stderr(tee):
                market_rows, opportunities_rows, arbs_found, counts, snaps = scan_markets(
                    enable_polymarket=enable_polymarket,
                    enable_kalshi=enable_kalshi,
                    enable_predictit=enable_predictit
                )

        scan_logs = buf.getvalue()
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.logs = f"[{ts}] Scan run\n{scan_logs}\n" + st.session_state.logs

        st.session_state.market_rows = market_rows
        st.session_state.opportunities_rows = opportunities_rows
        st.session_state.arbs_found = arbs_found
        st.session_state.counts = counts
        st.session_state.snapshots = snaps

        st.success(f"Scan complete. Snapshots={counts.get('snapshots_total', 0)} | Opps={counts.get('opps_total', 0)}")

    counts = st.session_state.counts or {}
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Snapshots", counts.get("snapshots_total", 0))
    m2.metric("Opps", counts.get("opps_total", 0))
    m3.metric("Polymarket fetched (post-prefilter)", counts.get("polymarket_fetched", 0))
    m4.metric("PredictIt fetched (post-prefilter)", counts.get("predictit_fetched", 0))

    st.subheader("Opportunities (sorted by net edge)")
    if st.session_state.opportunities_rows:
        opps_sorted = sorted(st.session_state.opportunities_rows, key=lambda r: -(r.get("edge_net_pct") or -999))
        st.dataframe(opps_sorted, use_container_width=True)
    else:
        st.info("No structural opportunities detected this scan (based on current thresholds).")

    st.subheader("Markets (includes activity/recency/spread/depth where available)")
    rows = st.session_state.market_rows or []
    if rows:
        rows_sorted = sorted(
            rows,
            key=lambda x: (
                -(x.get("priority_score") or -999),
                (x.get("recency_age_seconds") or 1e18),
            )
        )
        st.dataframe(rows_sorted, use_container_width=True)
    else:
        st.info("No markets yet.")

    with st.expander("Log output (same as terminal)", expanded=False):
        st.code(st.session_state.logs[:30000], language="text")


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            main_ui()
        else:
            main_terminal()
    except ImportError:
        main_terminal()
