"""
Order management system for handling trade execution
"""
import logging
import time
from typing import Dict, List, Optional
from datetime import datetime
import threading
from enum import Enum

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class Order:
    """Represents a trading order"""
    
    def __init__(self, symbol: str, action: str, quantity: int, order_type: OrderType = OrderType.MARKET, 
                 price: float = None, stop_price: float = None, strategy: str = None):
        self.id = self._generate_order_id()
        self.symbol = symbol
        self.action = action.lower()  # 'buy', 'sell', 'close'
        self.quantity = abs(quantity)
        self.order_type = order_type
        self.price = price
        self.stop_price = stop_price
        self.strategy = strategy
        
        self.status = OrderStatus.PENDING
        self.filled_quantity = 0
        self.filled_price = 0.0
        self.created_at = datetime.now()
        self.filled_at = None
        self.commission = 0.0
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        return f"ORD_{int(time.time() * 1000000)}"
    
    def fill(self, price: float, quantity: int = None, commission: float = 0.0):
        """Mark order as filled"""
        if quantity is None:
            quantity = self.quantity
        
        self.filled_quantity = quantity
        self.filled_price = price
        self.commission = commission
        self.status = OrderStatus.FILLED
        self.filled_at = datetime.now()
        
        logger.info(f"Order {self.id} filled: {quantity} shares of {self.symbol} at ${price:.2f}")
    
    def cancel(self):
        """Cancel the order"""
        self.status = OrderStatus.CANCELLED
        logger.info(f"Order {self.id} cancelled")
    
    def reject(self, reason: str = ""):
        """Reject the order"""
        self.status = OrderStatus.REJECTED
        logger.warning(f"Order {self.id} rejected: {reason}")

class OrderManager:
    """Manages order execution and tracking"""
    
    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.pending_orders: List[Order] = []
        self.filled_orders: List[Order] = []
        
        # Mock market data for order execution
        self.market_prices = {}
        self.lock = threading.Lock()
        
        logger.info("OrderManager initialized")
    
    def submit_order(self, order_data: Dict) -> str:
        """Submit a new order"""
        order = Order(
            symbol=order_data['symbol'],
            action=order_data['action'],
            quantity=order_data['quantity'],
            order_type=OrderType.MARKET,  # Default to market orders
            strategy=order_data.get('strategy')
        )
        
        with self.lock:
            self.orders[order.id] = order
            self.pending_orders.append(order)
        
        logger.info(f"Order submitted: {order.id} - {order.action} {order.quantity} {order.symbol}")
        return order.id
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        with self.lock:
            if order_id in self.orders:
                order = self.orders[order_id]
                if order.status == OrderStatus.PENDING:
                    order.cancel()
                    self.pending_orders.remove(order)
                    return True
        return False
    
    def update_market_price(self, symbol: str, price: float):
        """Update market price for order execution"""
        self.market_prices[symbol] = price
    
    def process_orders(self):
        """Process all pending orders"""
        with self.lock:
            orders_to_remove = []
            
            for order in self.pending_orders:
                if self._should_execute_order(order):
                    execution_price = self._get_execution_price(order)
                    commission = self._calculate_commission(order, execution_price)
                    
                    order.fill(execution_price, commission=commission)
                    self.filled_orders.append(order)
                    orders_to_remove.append(order)
            
            # Remove filled orders from pending list
            for order in orders_to_remove:
                self.pending_orders.remove(order)
    
    def _should_execute_order(self, order: Order) -> bool:
        """Determine if order should be executed"""
        current_price = self.market_prices.get(order.symbol)
        
        if current_price is None:
            return False
        
        # For market orders, execute immediately
        if order.order_type == OrderType.MARKET:
            return True
        
        # For limit orders
        if order.order_type == OrderType.LIMIT:
            if order.action == 'buy' and current_price <= order.price:
                return True
            elif order.action == 'sell' and current_price >= order.price:
                return True
        
        # For stop orders
        if order.order_type == OrderType.STOP:
            if order.action == 'buy' and current_price >= order.stop_price:
                return True
            elif order.action == 'sell' and current_price <= order.stop_price:
                return True
        
        return False
    
    def _get_execution_price(self, order: Order) -> float:
        """Get execution price for order"""
        current_price = self.market_prices.get(order.symbol, 0.0)
        
        # Add some slippage for market orders
        if order.order_type == OrderType.MARKET:
            slippage = 0.001  # 0.1% slippage
            if order.action == 'buy':
                return current_price * (1 + slippage)
            else:
                return current_price * (1 - slippage)
        
        # For limit orders, use the limit price
        if order.order_type == OrderType.LIMIT:
            return order.price
        
        return current_price
    
    def _calculate_commission(self, order: Order, price: float) -> float:
        """Calculate commission for order"""
        # Simple commission structure: 0.1% of trade value
        return order.quantity * price * 0.001
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)
    
    def get_pending_orders(self) -> List[Order]:
        """Get all pending orders"""
        return self.pending_orders.copy()
    
    def get_filled_orders(self) -> List[Order]:
        """Get all filled orders"""
        return self.filled_orders.copy()
    
    def get_order_history(self) -> List[Dict]:
        """Get order history as list of dictionaries"""
        history = []
        for order in self.orders.values():
            history.append({
                'id': order.id,
                'symbol': order.symbol,
                'action': order.action,
                'quantity': order.quantity,
                'order_type': order.order_type.value,
                'status': order.status.value,
                'created_at': order.created_at,
                'filled_at': order.filled_at,
                'filled_price': order.filled_price,
                'commission': order.commission,
                'strategy': order.strategy
            })
        return sorted(history, key=lambda x: x['created_at'], reverse=True)