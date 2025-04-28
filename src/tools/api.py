import os
import pandas as pd
import requests
import sys
from pathlib import Path
import time
import threading

# Load environment variables first
from dotenv import load_dotenv

# Get the project root directory
project_root = Path(__file__).resolve().parent.parent.parent
dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path=dotenv_path)

from data.cache import get_cache
from data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
)

# Import both Alpaca packages
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, TimeFrame

# Also import the newer alpaca-py package
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest, StockLatestQuoteRequest, CryptoLatestQuoteRequest
from alpaca.trading.requests import GetOrdersRequest
from alpaca.data.timeframe import TimeFrame as AlpacaTimeFrame
from datetime import datetime, timedelta
from pydantic import ValidationError

# Import Kraken library
import krakenex

# Alpaca API configuration
APCA_API_KEY_ID = os.environ.get("APCA_API_KEY_ID")
APCA_API_SECRET_KEY = os.environ.get("APCA_API_SECRET_KEY")
ALPACA_PAPER_TRADING = os.environ.get("ALPACA_PAPER_TRADING", "True").lower() == "true"

# Function to set paper trading mode
def set_paper_trading(paper_mode: bool):
    """Set the paper trading mode for Alpaca API.
    
    Args:
        paper_mode: True for paper trading, False for live trading
    """
    global ALPACA_PAPER_TRADING, api, trading_client
    
    # Update the global variable
    ALPACA_PAPER_TRADING = paper_mode
    
    # Re-initialize the legacy API client with the new mode
    if APCA_API_KEY_ID and APCA_API_SECRET_KEY:
        try:
            api = tradeapi.REST(
                APCA_API_KEY_ID,
                APCA_API_SECRET_KEY,
                base_url="https://paper-api.alpaca.markets" if ALPACA_PAPER_TRADING else "https://api.alpaca.markets"
            )
            print(f"Alpaca API (legacy) re-initialized with paper trading mode: {ALPACA_PAPER_TRADING}")
        except Exception as e:
            print(f"Error re-initializing Alpaca API (legacy): {e}")
    
    # Re-initialize the new API client with the new mode
    if APCA_API_KEY_ID and APCA_API_SECRET_KEY:
        try:
            trading_client = TradingClient(
                APCA_API_KEY_ID,
                APCA_API_SECRET_KEY,
                paper=ALPACA_PAPER_TRADING
            )
            print(f"Alpaca API (new) re-initialized with paper trading mode: {ALPACA_PAPER_TRADING}")
        except Exception as e:
            print(f"Error re-initializing Alpaca API (new): {e}")

# Kraken API configuration
KRAKEN_API_KEY = os.environ.get("KRAKEN_API_KEY")
KRAKEN_SECRET_KEY = os.environ.get("KRAKEN_SECRET_KEY")

# Initialize Alpaca API clients
api = None
trading_client = None
stock_client = None
crypto_client = None

# Initialize Kraken API client
kraken_client = None

# Initialize Alpaca API (legacy client)
try:
    if APCA_API_KEY_ID and APCA_API_SECRET_KEY:
        api = tradeapi.REST(
            APCA_API_KEY_ID,
            APCA_API_SECRET_KEY,
            base_url="https://paper-api.alpaca.markets" if ALPACA_PAPER_TRADING else "https://api.alpaca.markets"
        )
        print("Alpaca API (legacy) initialized successfully.")
    else:
        print("Alpaca API keys not found. Some functionality may be limited.")
        api = None
except Exception as e:
    print(f"Error initializing Alpaca API (legacy): {e}")
    api = None

# Initialize Alpaca API (new client)
try:
    if APCA_API_KEY_ID and APCA_API_SECRET_KEY:
        trading_client = TradingClient(
            APCA_API_KEY_ID,
            APCA_API_SECRET_KEY,
            paper=ALPACA_PAPER_TRADING
        )
        stock_client = StockHistoricalDataClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY)
        crypto_client = CryptoHistoricalDataClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY)
        print("Alpaca API (new) initialized successfully.")
        print(f"Paper trading mode: {ALPACA_PAPER_TRADING}")
    else:
        print("Alpaca API keys not found. Some functionality may be limited.")
        trading_client = None
        stock_client = None
        crypto_client = None
except Exception as e:
    print(f"Error initializing Alpaca API (new): {e}")
    trading_client = None
    stock_client = None
    crypto_client = None

# Initialize Kraken API
try:
    if KRAKEN_API_KEY and KRAKEN_SECRET_KEY:
        kraken_client = krakenex.API(KRAKEN_API_KEY, KRAKEN_SECRET_KEY)
        print("Kraken API initialized successfully.")
    else:
        print("Kraken API keys not found. Kraken functionality will be disabled.")
        kraken_client = None
except Exception as e:
    print(f"Error initializing Kraken API: {e}")
    kraken_client = None

# Global cache instance
_cache = get_cache()

# Helper function to identify asset type (crypto or stock)
def is_crypto_pair(ticker: str) -> bool:
    """Determine if a ticker is a crypto pair (e.g., BTC/USD) or a stock symbol."""
    return '/' in ticker

# Function to get crypto assets from Alpaca
def get_tradable_crypto_assets():
    """Get a list of tradable crypto assets from Alpaca."""
    if not trading_client:
        print("Alpaca trading client not initialized. Cannot fetch crypto assets.")
        return []
    
    try:
        # Get all assets
        assets = trading_client.get_all_assets()
        
        # Filter for tradable crypto assets
        crypto_assets = [asset for asset in assets if 
                        asset.asset_class == 'crypto' and 
                        asset.tradable and 
                        asset.status == 'active']
        
        return crypto_assets
    except Exception as e:
        print(f"Error fetching tradable crypto assets: {e}")
        return []

@alpaca_rate_limiter
def place_alpaca_crypto_order(pair: str, quantity: float, side: str, type: str):
    """Place a crypto order with Alpaca.
    
    Args:
        pair: Crypto pair in format 'BTC/USD'
        quantity: Quantity to trade (can be fractional)
        side: 'buy' or 'sell'
        type: 'market' or 'limit'
    """
    if not trading_client:
        print("Alpaca trading client not initialized. Cannot place crypto order.")
        return None

    try:
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
        
        # Convert parameters to the format expected by Alpaca
        order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
        order_tif = TimeInForce.GTC  # Good 'til canceled
        
        # Cancel existing open orders for this pair
        try:
            cancel_request = GetOrdersRequest(symbol=pair, status='open')
            open_orders = trading_client.get_orders(filter=cancel_request)
            cancelled_ids = []
            for order in open_orders:
                try:
                    trading_client.cancel_order_by_id(order.id)
                    cancelled_ids.append(order.id)
                except Exception as cancel_err:
                    print(f"Warning: Failed to cancel order {order.id} for {pair}: {cancel_err}")
            if cancelled_ids:
                print(f"Cancelled open orders for {pair}: {cancelled_ids}")
        except Exception as e:
            print(f"Error checking/cancelling existing orders for {pair}: {e}")
        
        # Create the appropriate order request based on order type
        if type.lower() == 'market':
            order_request = MarketOrderRequest(
                symbol=pair,
                qty=quantity,
                side=order_side,
                time_in_force=order_tif
            )
        elif type.lower() == 'limit':
            # For limit orders, we would need a price
            # This is a placeholder - in a real implementation, you would use a provided limit price
            raise ValueError("Limit orders for crypto not implemented yet")
        else:
            raise ValueError(f"Unsupported order type: {type}")
        
        # Submit the order
        order = trading_client.submit_order(order_request)
        print(f"Placed {side} order for {quantity} {pair} via Alpaca")
        return order
    except Exception as e:
        print(f"Error placing Alpaca crypto order for {pair}: {e}")
        return None

@alpaca_rate_limiter
def get_alpaca_crypto_price_data(pair: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical price data for a crypto pair from Alpaca."""
    if not crypto_client:
        print("Alpaca crypto client not initialized. Cannot fetch price data.")
        return pd.DataFrame()

    try:
        # Convert dates to datetime objects
        start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
        end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Create request parameters
        request_params = CryptoBarsRequest(
            symbol_or_symbols=pair,
            timeframe=AlpacaTimeFrame.Day,
            start=start_datetime,
            end=end_datetime
        )
        
        # Get the bars
        bars = crypto_client.get_crypto_bars(request_params)
        
        # Convert to dataframe
        if bars and bars.data and pair in bars.data:
            df = pd.DataFrame([bar.dict() for bar in bars.data[pair]])
            df["Date"] = pd.to_datetime(df["timestamp"])
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)
            print(f"Retrieved {len(df)} days of crypto price data for {pair} from Alpaca.")
            return df
        else:
            print(f"No data returned from Alpaca for {pair}")
            return pd.DataFrame()

    except Exception as e:
        print(f"Exception fetching Alpaca crypto price data for {pair}: {e}")
        return pd.DataFrame()

def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch price data from cache or API."""
    # Check cache first
    if cached_data := _cache.get_prices(ticker):
        # Filter cached data by date range and convert to Price objects
        filtered_data = [Price(**price) for price in cached_data if start_date <= price["time"] <= end_date]
        if filtered_data:
            return filtered_data

    # If not in cache or no data in range, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = f"https://api.financialdatasets.ai/prices/?ticker={ticker}&interval=day&interval_multiplier=1&start_date={start_date}&end_date={end_date}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

    # Parse response with Pydantic model
    price_response = PriceResponse(**response.json())
    prices = price_response.prices

    if not prices:
        return []

    # Cache the results as dicts
    _cache.set_prices(ticker, [p.model_dump() for p in prices])
    return prices


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[FinancialMetrics]:
    """Fetch financial metrics from cache or API."""
    # Check cache first
    if cached_data := _cache.get_financial_metrics(ticker):
        # Filter cached data by date and limit
        filtered_data = [FinancialMetrics(**metric) for metric in cached_data if metric["report_period"] <= end_date]
        filtered_data.sort(key=lambda x: x.report_period, reverse=True)
        if filtered_data:
            return filtered_data[:limit]

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = f"https://api.financialdatasets.ai/financial-metrics/?ticker={ticker}&report_period_lte={end_date}&limit={limit}&period={period}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

    # Parse response with Pydantic model
    metrics_response = FinancialMetricsResponse(**response.json())
    # Return the FinancialMetrics objects directly instead of converting to dict
    financial_metrics = metrics_response.financial_metrics

    if not financial_metrics:
        return []

    # Cache the results as dicts
    _cache.set_financial_metrics(ticker, [m.model_dump() for m in financial_metrics])
    return financial_metrics


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[LineItem]:
    """Fetch line items from API."""
    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = "https://api.financialdatasets.ai/financials/search/line-items"

    body = {
        "tickers": [ticker],
        "line_items": line_items,
        "end_date": end_date,
        "period": period,
        "limit": limit,
    }
    response = requests.post(url, headers=headers, json=body)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
    data = response.json()
    response_model = LineItemResponse(**data)
    search_results = response_model.search_results
    if not search_results:
        return []

    # Cache the results
    return search_results[:limit]


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[InsiderTrade]:
    """Fetch insider trades from cache or API."""
    # Check cache first
    if cached_data := _cache.get_insider_trades(ticker):
        # Filter cached data by date range
        filtered_data = [InsiderTrade(**trade) for trade in cached_data 
                        if (start_date is None or (trade.get("transaction_date") or trade["filing_date"]) >= start_date)
                        and (trade.get("transaction_date") or trade["filing_date"]) <= end_date]
        filtered_data.sort(key=lambda x: x.transaction_date or x.filing_date, reverse=True)
        if filtered_data:
            return filtered_data

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    all_trades = []
    current_end_date = end_date
    
    while True:
        url = f"https://api.financialdatasets.ai/insider-trades/?ticker={ticker}&filing_date_lte={current_end_date}"
        if start_date:
            url += f"&filing_date_gte={start_date}"
        url += f"&limit={limit}"
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
        
        data = response.json()
        response_model = InsiderTradeResponse(**data)
        insider_trades = response_model.insider_trades
        
        if not insider_trades:
            break
            
        all_trades.extend(insider_trades)
        
        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(insider_trades) < limit:
            break
            
        # Update end_date to the oldest filing date from current batch for next iteration
        current_end_date = min(trade.filing_date for trade in insider_trades).split('T')[0]
        
        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_trades:
        return []

    # Cache the results
    _cache.set_insider_trades(ticker, [trade.model_dump() for trade in all_trades])
    return all_trades


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[CompanyNews]:
    """Fetch company news from cache or API."""
    # Check cache first
    if cached_data := _cache.get_company_news(ticker):
        # Filter cached data by date range
        filtered_data = [CompanyNews(**news) for news in cached_data 
                        if (start_date is None or news["date"] >= start_date)
                        and news["date"] <= end_date]
        filtered_data.sort(key=lambda x: x.date, reverse=True)
        if filtered_data:
            return filtered_data

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    all_news = []
    current_end_date = end_date
    
    while True:
        url = f"https://api.financialdatasets.ai/news/?ticker={ticker}&end_date={current_end_date}"
        if start_date:
            url += f"&start_date={start_date}"
        url += f"&limit={limit}"
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
        
        data = response.json()
        response_model = CompanyNewsResponse(**data)
        company_news = response_model.news
        
        if not company_news:
            break
            
        all_news.extend(company_news)
        
        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(company_news) < limit:
            break
            
        # Update end_date to the oldest date from current batch for next iteration
        current_end_date = min(news.date for news in company_news).split('T')[0]
        
        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_news:
        return []

    # Cache the results
    _cache.set_company_news(ticker, [news.model_dump() for news in all_news])
    return all_news



def get_market_cap(
    ticker: str,
    end_date: str,
) -> float | None:
    """Fetch market cap from the API."""
    financial_metrics = get_financial_metrics(ticker, end_date)
    market_cap = financial_metrics[0].market_cap
    if not market_cap:
        return None

    return market_cap


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


# --- Kraken API Functions ---

# Helper function to identify asset type (crypto or stock)
def is_crypto_pair(ticker: str) -> bool:
    """Determine if a ticker is a crypto pair (e.g., BTC/USD) or a stock symbol."""
    return '/' in ticker

def get_kraken_price_data(pair: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical price data for a crypto pair from Kraken."""
    if not kraken_client:
        print("Kraken client not initialized. Cannot fetch price data.")
        return pd.DataFrame()

    try:
        # Convert standard pair format to Kraken format
        kraken_pair = _convert_pair_to_kraken_format(pair)
        if not kraken_pair:
            print(f"Could not convert pair '{pair}' to Kraken format. Cannot fetch price data.")
            return pd.DataFrame()

        # Convert dates to UNIX timestamps (Kraken API requirement)
        start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

        # Prepare parameters for OHLC (Open, High, Low, Close) data request
        params = {
            'pair': kraken_pair,
            'interval': 1440,  # Daily data (in minutes)
            'since': start_timestamp
        }

        # Query the Kraken API
        response = kraken_client.query_public('OHLC', params)
        print(f"Kraken OHLC response received. Success: {not response.get('error')}")

        if response.get('error'):
            error_msg = ", ".join(response['error'])
            print(f"Error fetching Kraken price data for {pair}: {error_msg}")
            return pd.DataFrame()

        # Extract and format the price data
        if response.get('result') and kraken_pair in response['result']:
            ohlc_data = response['result'][kraken_pair]
            
            # Create DataFrame with proper column names
            df = pd.DataFrame(ohlc_data, columns=[
                'time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
            ])
            
            # Convert timestamp to datetime
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Filter by date range
            df = df[(df['time'] >= start_date) & (df['time'] <= end_date)]
            
            # Convert string values to numeric
            for col in ['open', 'high', 'low', 'close', 'vwap', 'volume']:
                df[col] = pd.to_numeric(df[col])
                
            # Rename columns to match the format used elsewhere
            df.rename(columns={'time': 'Date'}, inplace=True)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            print(f"Retrieved {len(df)} days of price data for {pair} from Kraken.")
            return df
        else:
            print(f"No price data found for {pair} or unexpected response format.")
            return pd.DataFrame()

    except Exception as e:
        print(f"Exception fetching Kraken price data for {pair}: {e}")
        return pd.DataFrame()

def _convert_pair_to_kraken_format(standard_pair: str) -> str | None:
    """Converts a standard pair like 'BTC/USD' to Kraken's format like 'XXBTZUSD'."""
    # Basic mapping for common pairs (can be expanded or fetched dynamically)
    # Kraken often prefixes assets with X or Z (X for crypto, Z for fiat)
    mapping = {
        "BTC": "XBT", # Kraken uses XBT for Bitcoin
        "ETH": "ETH", 
        "USD": "USD",
        "EUR": "EUR",
        "GBP": "GBP",
        # Add more mappings as needed
    }
    
    # Kraken uses 'X' prefix for crypto, 'Z' for fiat, but the API often needs the prefixed versions
    # for the pair string. We'll construct common pairs directly for now.
    common_kraken_pairs = {
        "BTC/USD": "XXBTZUSD",
        "ETH/USD": "XETHZUSD",
        "BTC/EUR": "XXBTZEUR",
        "ETH/EUR": "XETHZEUR",
        # Add more common pairs directly
    }

    standard_pair_upper = standard_pair.upper()
    if standard_pair_upper in common_kraken_pairs:
        return common_kraken_pairs[standard_pair_upper]

    # Fallback logic (may not always work perfectly)
    try:
        base, quote = standard_pair_upper.split('/')
        kraken_base = f"X{mapping.get(base, base)}" # Assume crypto prefix if unknown
        kraken_quote = f"Z{mapping.get(quote, quote)}" # Assume fiat prefix if unknown
        # Handle cases like ETH/BTC -> XETHXXBT
        if base in mapping and mapping[base] != base: # It's a known crypto
             kraken_base = f"X{mapping[base]}"
        if quote in mapping and mapping[quote] != quote: # It's a known crypto
             kraken_quote = f"X{mapping[quote]}"
             
        # Special case for BTC
        if base == "BTC": kraken_base = "XXBT"
        if quote == "BTC": kraken_quote = "XXBT"
            
        # Adjust for fiat prefixes
        if base not in mapping or mapping[base] == base: # Assume fiat if not in crypto mapping
             kraken_base = f"Z{base}"
        if quote not in mapping or mapping[quote] == quote:
             kraken_quote = f"Z{quote}"
             
        # Final check for XXBT
        if kraken_base == "ZXBT": kraken_base = "XXBT"
        if kraken_quote == "ZXBT": kraken_quote = "XXBT"
        
        # Reconstruct pair, ensuring Z goes last if present
        if kraken_base.startswith('Z') and kraken_quote.startswith('X'):
            return f"{kraken_quote}{kraken_base}"
        else:
            return f"{kraken_base}{kraken_quote}"

    except ValueError:
        print(f"Warning: Could not parse standard pair format: {standard_pair}")
        return None
    except Exception as e:
        print(f"Error converting pair {standard_pair} to Kraken format: {e}")
        return None

def get_kraken_balance():
    """Fetch account balance from Kraken."""
    if not kraken_client:
        print("Kraken client not initialized. Cannot fetch balance.")
        return None

    try:
        response = kraken_client.query_private('Balance')
        print(f"Kraken Balance response received. Success: {not response.get('error')}")

        if response.get('error'):
            error_msg = ", ".join(response['error'])
            print(f"Error fetching Kraken balance: {error_msg}")
            return None

        if response.get('result'):
            # Format the balance data for easier consumption
            balances = response['result']
            formatted_balances = {}

            # Kraken uses asset codes like XXBT, ZUSD - we'll try to convert them back
            # to standard format where possible
            for asset, amount in balances.items():
                # Handle common prefixes
                standard_asset = asset
                if asset.startswith('X') and len(asset) > 1:
                    if asset == 'XXBT':
                        standard_asset = 'BTC'
                    else:
                        standard_asset = asset[1:]  # Remove X prefix
                elif asset.startswith('Z') and len(asset) > 1:
                    standard_asset = asset[1:]  # Remove Z prefix

                # Convert to float for easier use
                formatted_balances[standard_asset] = float(amount)

            print(f"Kraken balance retrieved successfully. {len(formatted_balances)} assets found.")
            return formatted_balances
        else:
            print("Kraken balance response has unexpected format.")
            return None

    except Exception as e:
        print(f"Exception fetching Kraken balance: {e}")
        return None

@kraken_rate_limiter
def get_kraken_open_orders(pair: str | None = None):
    """Fetch open orders from Kraken, optionally filtering by pair."""
    if not kraken_client:
        print("Kraken client not initialized. Cannot fetch open orders.")
        return []

    try:
        # Prepare parameters
        params = {}
        if pair:
            kraken_pair = _convert_pair_to_kraken_format(pair)
            if kraken_pair:
                params['pair'] = kraken_pair

        response = kraken_client.query_private('OpenOrders', params)
        print(f"Kraken OpenOrders response received. Success: {not response.get('error')}")

        if response.get('error'):
            error_msg = ", ".join(response['error'])
            print(f"Error fetching Kraken open orders: {error_msg}")
            return []

        if response.get('result') and response['result'].get('open'):
            # Format the orders data for easier consumption
            open_orders = response['result']['open']
            formatted_orders = []

            for order_id, order_data in open_orders.items():
                # Extract and format order information
                formatted_order = {
                    'order_id': order_id,
                    'pair': order_data.get('descr', {}).get('pair', ''),
                    'type': order_data.get('descr', {}).get('type', ''),  # buy/sell
                    'ordertype': order_data.get('descr', {}).get('ordertype', ''),  # market/limit
                    'price': order_data.get('descr', {}).get('price', ''),
                    'volume': order_data.get('vol', ''),
                    'status': order_data.get('status', ''),
                    'open_time': order_data.get('opentm', 0)
                }

                # Only include orders for the specified pair if provided
                if not pair or (pair and formatted_order['pair'] == kraken_pair):
                    formatted_orders.append(formatted_order)

            print(f"Retrieved {len(formatted_orders)} open orders from Kraken.")
            return formatted_orders
        else:
            print("No open orders found or unexpected response format.")
            return []

    except Exception as e:
        print(f"Exception fetching Kraken open orders: {e}")
        return []

@kraken_rate_limiter
def cancel_kraken_order(order_id: str):
    """Cancel a specific open order on Kraken."""
    if not kraken_client:
        print("Kraken client not initialized. Cannot cancel order.")
        return False

    try:
        params = {'txid': order_id}
        response = kraken_client.query_private('CancelOrder', params)
        print(f"Kraken CancelOrder response received. Success: {not response.get('error')}")

        if response.get('error'):
            error_msg = ", ".join(response['error'])
            print(f"Error cancelling Kraken order {order_id}: {error_msg}")
            return False

        if response.get('result'):
            result = response['result']
            # Check if the count of cancelled orders is as expected
            if result.get('count') and int(result['count']) > 0:
                print(f"Successfully cancelled Kraken order: {order_id}")
                return True
            else:
                print(f"Kraken order {order_id} cancellation may have failed. Response: {result}")
                return False
        else:
            print(f"Unexpected response format when cancelling Kraken order {order_id}")
            return False

    except Exception as e:
        print(f"Exception cancelling Kraken order {order_id}: {e}")
        return False

@kraken_rate_limiter
def place_kraken_order(pair: str, quantity: float, side: str, type: str):
    """Place an order with Kraken."""
    if not kraken_client:
        print("Kraken client not initialized. Cannot place order.")
        return None

    kraken_pair = _convert_pair_to_kraken_format(pair)
    if not kraken_pair:
        print(f"Could not convert pair '{pair}' to Kraken format. Aborting order.")
        return None

    # Map parameters to Kraken API requirements
    kraken_side = side.lower() # 'buy' or 'sell'
    kraken_type = type.lower() # 'market', 'limit', etc.

    params = {
        'pair': kraken_pair,
        'type': kraken_side,
        'ordertype': kraken_type,
        'volume': str(quantity), # Volume must be a string
        # Add other parameters like 'price' for limit orders if needed
        # 'validate': True # Use validate=True to test order without executing
    }

    print(f"Placing Kraken order with params: {params}")

    try:
        response = kraken_client.query_private('AddOrder', params)
        print(f"Kraken AddOrder response: {response}")

        if response.get('error'):
            error_msg = ", ".join(response['error'])
            print(f"Error placing Kraken order for {pair}: {error_msg}")
            # TODO: Add more specific error handling (e.g., insufficient funds)
            return None

        if response.get('result') and response['result'].get('txid'):
            tx_ids = response['result']['txid']
            print(f"Kraken order placed successfully for {pair}. Transaction ID(s): {tx_ids}")
            # Return the transaction ID(s) or a success indicator
            return {"success": True, "txid": tx_ids}
        else:
            print(f"Kraken order placement for {pair} might have failed or response format unexpected.")
            return None

    except Exception as e:
        print(f"Exception placing Kraken order for {pair}: {e}")
        return None

# Update the get_price_data function to use the new functions
def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Get price data using either the Financial Datasets API, Alpaca API, or Kraken API"""
    # Check if this is a crypto pair
    if is_crypto_pair(ticker):
        # Use Alpaca for crypto pairs if paper trading is enabled
        if ALPACA_PAPER_TRADING:
            try:
                return get_alpaca_crypto_price_data(ticker, start_date, end_date)
            except Exception as e:
                print(f"Error fetching crypto price data from Alpaca for {ticker}: {e}")
                # Fall back to Kraken if Alpaca fails
                try:
                    return get_kraken_price_data(ticker, start_date, end_date)
                except Exception as kraken_error:
                    print(f"Error fetching crypto price data from Kraken for {ticker}: {kraken_error}")
                    return pd.DataFrame()
        else:
            # Use Kraken API for crypto pairs in live trading
            try:
                return get_kraken_price_data(ticker, start_date, end_date)
            except Exception as e:
                print(f"Error fetching crypto price data for {ticker}: {e}")
                return pd.DataFrame()
    else:
        # Try Financial Datasets API first for stocks
        try:
            prices = get_prices(ticker, start_date, end_date)
            return prices_to_df(prices)
        except Exception as e:
            # If that fails, try Alpaca API if available
            if stock_client is not None:
                try:
                    # Convert string dates to datetime objects
                    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
                    end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
                    
                    # Create request parameters
                    request_params = StockBarsRequest(
                        symbol_or_symbols=ticker,
                        timeframe=AlpacaTimeFrame.Day,
                        start=start_datetime,
                        end=end_datetime
                    )
                    
                    # Get the bars
                    bars = stock_client.get_stock_bars(request_params)
                    
                    # Convert to dataframe
                    if bars and bars.data and ticker in bars.data:
                        df = pd.DataFrame([bar.dict() for bar in bars.data[ticker]])
                        df["Date"] = pd.to_datetime(df["timestamp"])
                        df.set_index("Date", inplace=True)
                        df.sort_index(inplace=True)
                        return df
                    else:
                        print(f"No data returned from Alpaca for {ticker}")
                        return pd.DataFrame()
                except Exception as alpaca_error:
                    print(f"Error fetching data from Alpaca for {ticker}: {alpaca_error}")
                    return pd.DataFrame()
            else:
                print(f"Error fetching data for {ticker} and Alpaca client not available: {e}")
                return pd.DataFrame()

# Rate limiting implementation
class RateLimiter:
    """Simple rate limiter to prevent excessive API calls"""
    def __init__(self, max_calls, time_frame):
        self.max_calls = max_calls  # Maximum number of calls allowed in the time frame
        self.time_frame = time_frame  # Time frame in seconds
        self.calls = []  # List to store timestamps of calls
        self.lock = threading.Lock()  # Lock for thread safety
    
    def __call__(self, func):
        """Decorator to rate limit a function"""
        def wrapper(*args, **kwargs):
            with self.lock:
                # Remove timestamps older than time_frame
                current_time = time.time()
                self.calls = [t for t in self.calls if current_time - t < self.time_frame]
                
                # Check if we've exceeded the rate limit
                if len(self.calls) >= self.max_calls:
                    # Calculate time to wait
                    oldest_call = min(self.calls)
                    sleep_time = self.time_frame - (current_time - oldest_call)
                    if sleep_time > 0:
                        print(f"Rate limit reached. Waiting {sleep_time:.2f} seconds before next API call.")
                        time.sleep(sleep_time)
                        # After waiting, update current time and clean calls again
                        current_time = time.time()
                        self.calls = [t for t in self.calls if current_time - t < self.time_frame]
                
                # Add current timestamp to calls
                self.calls.append(current_time)
                
                # Call the function
                return func(*args, **kwargs)
        return wrapper

# Create rate limiters for different APIs
alpaca_rate_limiter = RateLimiter(max_calls=200, time_frame=60)  # 200 calls per minute
kraken_rate_limiter = RateLimiter(max_calls=15, time_frame=60)  # 15 calls per minute
alpha_vantage_rate_limiter = RateLimiter(max_calls=5, time_frame=60)  # 5 calls per minute

# Cache for storing price data to reduce API calls
_price_cache = {}

@alpaca_rate_limiter
def get_account_info():
    """Get account information from Alpaca."""
    # Try the new client first
    if trading_client is not None:
        try:
            account = trading_client.get_account()
            return account
        except Exception as e:
            print(f"Error getting account information from new client: {e}")
    
    # Fall back to legacy client
    if api is None:
        print("Alpaca API is not initialized")
        return None
    try:
        account = api.get_account()
        return account
    except Exception as e:
        print(f"Error getting account information: {e}")
        return None

@alpaca_rate_limiter
def get_position(ticker: str):
    """Get position information from Alpaca."""
    # Try the new client first
    if trading_client is not None:
        try:
            # Use the new client to get position
            position = trading_client.get_open_position(ticker)
            return position
        except ValidationError as ve:
            # Handle cases where the returned data doesn't match the Pydantic model
            print(f"Warning: Pydantic validation error getting position for {ticker}: {ve}. Check alpaca-py version compatibility.")
            return None
        except Exception as e:
            # Check if it's a 'position not found' error
            if "position not found" in str(e).lower():
                return None  # Return None if no position exists
            print(f"Error getting position information from new client: {e}")
            return None

    # Fall back to legacy client
    if api is None:
        print("Alpaca API is not initialized")
        return None
    try:
        # Use the legacy client to get position
        position = api.get_position(ticker)
        return position
    except Exception as e:
        print(f"Error getting position information: {e}")
        return None

@alpaca_rate_limiter
def place_order(ticker: str, quantity: int, side: str, type: str, time_in_force: str):
    """Place an order with Alpaca."""
    # Try the new client first
    if trading_client is not None:
        try:
            from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
            
            # Convert parameters to the format expected by the new client
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            order_tif = TimeInForce.DAY
            if time_in_force.lower() == 'gtc':
                order_tif = TimeInForce.GTC
            elif time_in_force.lower() == 'ioc':
                order_tif = TimeInForce.IOC
            
            # Cancel ALL existing open orders for this symbol before placing a new one
            try:
                cancel_request = GetOrdersRequest(symbol=ticker, status='open')
                open_orders = trading_client.get_orders(filter=cancel_request)
                cancelled_ids = []
                for order in open_orders:
                    try:
                        trading_client.cancel_order_by_id(order.id)
                        cancelled_ids.append(order.id)
                    except Exception as cancel_err:
                        print(f"Warning: Failed to cancel order {order.id} for {ticker}: {cancel_err}")
                if cancelled_ids:
                    print(f"Cancelled open orders for {ticker}: {cancelled_ids}")
            except Exception as e:
                print(f"Error checking/cancelling existing orders for {ticker}: {e}")

            # Create the appropriate order request based on order type
            if type.lower() == 'market':
                order_request = MarketOrderRequest(
                    symbol=ticker,
                    qty=quantity,
                    side=order_side,
                    time_in_force=order_tif
                )
            elif type.lower() == 'limit':
                # For limit orders, we need a price - default to current market price
                # In a real implementation, you would get the current price or use a provided limit price
                order_request = LimitOrderRequest(
                    symbol=ticker,
                    qty=quantity,
                    side=order_side,
                    time_in_force=order_tif,
                    limit_price=100.0  # This should be replaced with actual limit price
                )
            else:
                raise ValueError(f"Unsupported order type: {type}")
            
            # Submit the order
            order = trading_client.submit_order(order_request)
            return order
        except Exception as e:
            print(f"Error placing order with new client: {e}")
    
    # Fall back to legacy client
    if api is None:
        print("Alpaca API is not initialized")
        return None
    try:
        order = api.submit_order(
            symbol=ticker,
            qty=quantity,
            side=side,
            type=type,
            time_in_force=time_in_force
        )
        return order
    except Exception as e:
        print(f"Error placing order: {e}")
        return None

@alpaca_rate_limiter
def get_current_price(ticker: str) -> float:
    """Get the current price for a ticker, handling both stocks and crypto.
    
    Args:
        ticker: The ticker symbol or crypto pair
        
    Returns:
        float: The current price
    """
    try:
        # Check if this is a crypto pair
        if is_crypto_pair(ticker):
            # For crypto, use Alpaca's latest quote
            if crypto_client:
                try:
                    request_params = CryptoLatestQuoteRequest(symbol_or_symbols=ticker)
                    quote = crypto_client.get_crypto_latest_quote(request_params)
                    if quote and quote.symbol == ticker:
                        return float(quote.ask_price)  # Use ask price as current price
                except Exception as e:
                    print(f"Error getting crypto quote from Alpaca for {ticker}: {e}")
                    
                # Fallback to latest bar
                try:
                    # Get the latest bar for the crypto pair
                    end = datetime.now()
                    start = end - timedelta(days=1)  # Get data from the last day
                    
                    request_params = CryptoBarsRequest(
                        symbol_or_symbols=ticker,
                        timeframe=AlpacaTimeFrame.Hour,
                        start=start,
                        end=end,
                        limit=1
                    )
                    
                    bars = crypto_client.get_crypto_bars(request_params)
                    if bars and bars.data and ticker in bars.data and len(bars.data[ticker]) > 0:
                        return float(bars.data[ticker][-1].close)
                except Exception as e:
                    print(f"Error getting crypto bar from Alpaca for {ticker}: {e}")
            
            # If Alpaca fails or is not available, try Kraken
            try:
                # Get the latest ticker info from Kraken
                kraken_pair = _convert_pair_to_kraken_format(ticker)
                response = kraken.query_public('Ticker', {'pair': kraken_pair})
                if response and 'result' in response and kraken_pair in response['result']:
                    # Kraken returns the last trade price in the 'c' field as a list where the first element is the price
                    return float(response['result'][kraken_pair]['c'][0])
            except Exception as e:
                print(f"Error getting crypto price from Kraken for {ticker}: {e}")
        else:
            # For stocks, use Alpaca's latest quote
            if trading_client:
                try:
                    request_params = StockLatestQuoteRequest(symbol_or_symbols=ticker)
                    quote = trading_client.get_stock_latest_quote(request_params)
                    if quote and quote.symbol == ticker:
                        return float(quote.ask_price)  # Use ask price as current price
                except Exception as e:
                    print(f"Error getting stock quote from Alpaca for {ticker}: {e}")
                    
                # Fallback to latest bar
                try:
                    # Get the latest bar for the stock
                    end = datetime.now()
                    start = end - timedelta(days=1)  # Get data from the last day
                    
                    request_params = StockBarsRequest(
                        symbol_or_symbols=ticker,
                        timeframe=AlpacaTimeFrame.Hour,
                        start=start,
                        end=end,
                        limit=1
                    )
                    
                    bars = trading_client.get_stock_bars(request_params)
                    if bars and bars.data and ticker in bars.data and len(bars.data[ticker]) > 0:
                        return float(bars.data[ticker][-1].close)
                except Exception as e:
                    print(f"Error getting stock bar from Alpaca for {ticker}: {e}")
        
        # If all else fails, try to get the price from Alpha Vantage
        try:
            # Use Alpha Vantage as a fallback
            if ticker in _price_cache and _price_cache[ticker]['timestamp'] > datetime.now() - timedelta(minutes=15):
                return _price_cache[ticker]['price']
                
            api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
            if not api_key:
                raise ValueError("Alpha Vantage API key not found in environment variables")
                
            if is_crypto_pair(ticker):
                # Format for crypto: BTC/USD -> digital_currency_daily&symbol=BTC&market=USD
                base, quote = ticker.split('/')
                url = f"https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={base}&market={quote}&apikey={api_key}"
            else:
                # Format for stocks
                url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={api_key}"
                
            response = requests.get(url)
            data = response.json()
            
            if is_crypto_pair(ticker) and 'Time Series (Digital Currency Daily)' in data:
                # Get the latest date
                latest_date = list(data['Time Series (Digital Currency Daily)'].keys())[0]
                price = float(data['Time Series (Digital Currency Daily)'][latest_date]['4a. close (USD)'])
                _price_cache[ticker] = {'price': price, 'timestamp': datetime.now()}
                return price
            elif 'Global Quote' in data and '05. price' in data['Global Quote']:
                price = float(data['Global Quote']['05. price'])
                _price_cache[ticker] = {'price': price, 'timestamp': datetime.now()}
                return price
        except Exception as e:
            print(f"Error getting price from Alpha Vantage for {ticker}: {e}")
        
        # If we get here, we couldn't get a price
        print(f"Could not get current price for {ticker} from any source")
        return 0.0
    except Exception as e:
        print(f"Unexpected error getting current price for {ticker}: {e}")
        return 0.0
