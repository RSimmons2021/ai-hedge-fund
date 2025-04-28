from typing import Dict, Any
from utils.progress import progress
from tools.api import (
    place_order, 
    place_alpaca_crypto_order, 
    is_crypto_pair,
    ALPACA_PAPER_TRADING
)
from graph.state import show_agent_reasoning
import json
import datetime
import uuid


##### Order Execution Agent #####
def order_executor_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute orders based on portfolio manager decisions.
    
    Args:
        state: The current state of the system
        
    Returns:
        Updated state with order execution results
    """
    print("\n=== Order Executor Agent ===")
    
    # Initialize order_results list if it doesn't exist
    if "order_results" not in state:
        state["order_results"] = []
    
    # Get the decisions from the state
    decisions = state.get("decisions", [])
    
    # Get paper trading flag
    paper_trading = state.get("paper_trading", True)
    if not paper_trading:
        print("WARNING: Live trading is enabled. Real orders will be placed!")
    else:
        print("Paper trading is enabled. No real orders will be placed.")
    
    # Process each decision
    for decision in decisions:
        # Skip if we already have a result for this decision
        if any(r["ticker"] == decision["ticker"] and r["timestamp"] == decision["timestamp"] 
               for r in state["order_results"]):
            continue
            
        ticker = decision["ticker"]
        action = decision["action"]
        quantity = decision["quantity"]
        
        print(f"\nProcessing order for {ticker}: {action} {quantity} units")
        
        # Skip if the action is hold
        if action == "hold":
            print(f"Action is hold for {ticker}, skipping order execution")
            result = {
                "ticker": ticker,
                "action": action,
                "quantity": quantity,
                "status": "skipped",
                "timestamp": datetime.datetime.now().isoformat(),
                "order_id": None,
                "error": None
            }
            state["order_results"].append(result)
            continue
        
        # Skip if quantity is zero or negative
        if quantity <= 0:
            print(f"Quantity is zero or negative for {ticker}, skipping order execution")
            result = {
                "ticker": ticker,
                "action": action,
                "quantity": quantity,
                "status": "skipped",
                "timestamp": datetime.datetime.now().isoformat(),
                "order_id": None,
                "error": "Quantity must be positive"
            }
            state["order_results"].append(result)
            continue
        
        # Execute the order
        order_result = None
        error = None
        
        try:
            # Check if this is a paper trade
            if paper_trading:
                print(f"Paper trading: {action} {quantity} units of {ticker}")
                order_result = {
                    "id": f"paper-{uuid.uuid4()}",
                    "status": "filled",
                    "filled_qty": quantity,
                    "filled_avg_price": decision["price"]
                }
            else:
                # Check if this is a crypto pair
                if is_crypto_pair(ticker):
                    print(f"Placing crypto order: {action} {quantity} units of {ticker}")
                    # For crypto, use place_alpaca_crypto_order
                    order_result = place_alpaca_crypto_order(
                        pair=ticker,
                        quantity=float(quantity),  # Ensure quantity is float for crypto
                        side=action,
                        type="market"
                    )
                else:
                    print(f"Placing stock order: {action} {quantity} shares of {ticker}")
                    # For stocks, use place_order
                    order_result = place_order(
                        ticker=ticker,
                        quantity=int(quantity),  # Ensure quantity is int for stocks
                        side=action,
                        type="market",
                        time_in_force="gtc"
                    )
                
                if order_result is None:
                    error = "Order placement failed"
        except Exception as e:
            error = str(e)
            print(f"Error executing order for {ticker}: {e}")
        
        # Create the result
        result = {
            "ticker": ticker,
            "action": action,
            "quantity": quantity,
            "status": "filled" if order_result and not error else "failed",
            "timestamp": datetime.datetime.now().isoformat(),
            "order_id": order_result["id"] if order_result and "id" in order_result else None,
            "filled_quantity": order_result["filled_qty"] if order_result and "filled_qty" in order_result else None,
            "filled_price": order_result["filled_avg_price"] if order_result and "filled_avg_price" in order_result else None,
            "error": error
        }
        
        # Add the result to the state
        state["order_results"].append(result)
        
        print(f"Order result for {ticker}: {result['status']}")
        if error:
            print(f"Error: {error}")
    
    return state
