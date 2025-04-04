import os
import pandas as pd
import requests
import sys
from pathlib import Path

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
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.trading.requests import GetOrdersRequest
from alpaca.data.timeframe import TimeFrame as AlpacaTimeFrame
from datetime import datetime
from pydantic import ValidationError

# Alpaca API configuration
APCA_API_KEY_ID = os.environ.get("APCA_API_KEY_ID")
APCA_API_SECRET_KEY = os.environ.get("APCA_API_SECRET_KEY")

# Debug environment variables
print(f"API Module - APCA_API_KEY_ID: {APCA_API_KEY_ID is not None}")
print(f"API Module - APCA_API_SECRET_KEY: {APCA_API_SECRET_KEY is not None}")

# Make Alpaca API optional
api = None
trading_client = None
stock_client = None

# Initialize Alpaca API clients
try:
    if APCA_API_KEY_ID and APCA_API_SECRET_KEY:
        # Initialize trading API (legacy client)
        ALPACA_PAPER_TRADING = os.environ.get("ALPACA_PAPER_TRADING", "True").lower() == "true" # Default to paper trading
        base_url = "https://paper-api.alpaca.markets" if ALPACA_PAPER_TRADING else "https://api.alpaca.markets"
        api = REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, base_url)
        
        # Initialize trading client (new client)
        trading_client = TradingClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY, paper=ALPACA_PAPER_TRADING)
        
        # Initialize market data API
        stock_client = StockHistoricalDataClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY)
        print("Alpaca API initialized successfully")
    else:
        print("Alpaca API keys not found. Trading functionality will be disabled.")
except Exception as e:
    print(f"Error initializing Alpaca API: {e}")
    print("Trading functionality will be disabled.")

# Global cache instance
_cache = get_cache()


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


# Update the get_price_data function to use the new functions
def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Get price data using either the Financial Datasets API or Alpaca API"""
    # Try Financial Datasets API first
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
                
                # Convert to DataFrame
                if bars and hasattr(bars, 'df'):
                    df = bars.df
                    if not df.empty:
                        # Rename columns to match our expected format
                        df = df.reset_index()
                        df = df.rename(columns={
                            'timestamp': 'time',
                            'open': 'open',
                            'high': 'high',
                            'low': 'low',
                            'close': 'close',
                            'volume': 'volume'
                        })
                        df['time'] = df['time'].dt.strftime('%Y-%m-%d')
                        return df
            except Exception as alpaca_error:
                print(f"Error fetching data from Alpaca: {alpaca_error}")
        
        # If both methods fail, raise the original error
        print(f"Error fetching price data: {e}")
        return pd.DataFrame()

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
