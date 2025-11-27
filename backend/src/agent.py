import logging
import json
import os
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    metrics,
    tokenize,
    function_tool,
    cli,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("food-ordering-agent")

load_dotenv(".env.local")

# Food ordering files - using relative path to shared-data
PROJECT_ROOT = Path(__file__).parent.parent.parent
CATALOG_FILE = PROJECT_ROOT / "shared-data" / "food_catalog.json"
ORDERS_FILE = PROJECT_ROOT / "shared-data" / "orders.json"

class FoodOrderingAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are a friendly and helpful food & grocery ordering assistant for QuickCart. Your role is to help users order food and groceries from our catalog.

IMPORTANT RULES:
- Be warm, friendly, and helpful in all interactions
- Always confirm when items are added to or removed from the cart
- When users ask for "ingredients for X", intelligently map to relevant items
- Keep track of the cart and be able to list contents when asked
- Help users complete their order efficiently

CALL FLOW:
1. Greet the user warmly and introduce yourself as QuickCart assistant
2. Explain what you can help with (ordering groceries, snacks, prepared food, or ingredients for meals)
3. Help users add items to cart, either specifically or through recipe requests
4. Handle cart management (add, remove, update quantities, list contents)
5. When user indicates they're done, confirm the final order and place it

SPECIAL BEHAVIORS:
- For "ingredients for X" requests: Map to appropriate items and add them to cart
- Always confirm additions/removals clearly
- Be proactive in suggesting quantities or alternatives if items are unavailable
- Maintain a running total when discussing the cart
"""
        )
        self.catalog = self._load_catalog()
        self.cart: List[Dict[str, Any]] = []
        self.current_order = None

    def _load_catalog(self) -> Dict[str, Any]:
        """Load food catalog from JSON file"""
        try:
            if CATALOG_FILE.exists():
                with open(CATALOG_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"Successfully loaded food catalog with {len(data.get('items', []))} items")
                return data
            else:
                logger.error(f"Catalog file not found: {CATALOG_FILE}")
                # Create default structure if file doesn't exist
                default_data = {"items": [], "recipes": {}}
                with open(CATALOG_FILE, 'w', encoding='utf-8') as f:
                    json.dump(default_data, f, indent=2)
                return default_data
        except Exception as e:
            logger.error(f"Error loading catalog: {e}")
            return {"items": [], "recipes": {}}

    def _find_item(self, item_name: str) -> Dict[str, Any]:
        """Find an item by name in the catalog"""
        item_name_lower = item_name.lower()
        for item in self.catalog.get("items", []):
            if item_name_lower in item.get("name", "").lower():
                return item
        return None

    def _get_recipe_items(self, recipe_name: str) -> List[Dict[str, Any]]:
        """Get items for a recipe"""
        recipe_name_lower = recipe_name.lower()
        recipes = self.catalog.get("recipes", {})
        
        # Try exact match first
        if recipe_name_lower in recipes:
            item_names = recipes[recipe_name_lower]
        else:
            # Try partial match
            for recipe_key in recipes.keys():
                if recipe_name_lower in recipe_key:
                    item_names = recipes[recipe_key]
                    break
            else:
                return []
        
        items = []
        for item_name in item_names:
            item = self._find_item(item_name)
            if item:
                items.append(item)
        return items

    def _calculate_total(self) -> float:
        """Calculate total cost of cart"""
        return sum(item['price'] * item['quantity'] for item in self.cart)

    def _save_order(self, customer_name: str = "Customer", customer_address: str = "Not specified"):
        """Save order to JSON file"""
        try:
            # Load existing orders
            orders = []
            if ORDERS_FILE.exists():
                with open(ORDERS_FILE, 'r', encoding='utf-8') as f:
                    orders = json.load(f)
            
            # Create new order
            order_id = f"QC{datetime.now().strftime('%Y%m%d%H%M%S')}"
            total = self._calculate_total()
            
            new_order = {
                "order_id": order_id,
                "timestamp": datetime.now().isoformat(),
                "items": self.cart.copy(),
                "total": total,
                "customer_name": customer_name,
                "customer_address": customer_address,
                "status": "received"
            }
            
            orders.append(new_order)
            
            # Save orders
            with open(ORDERS_FILE, 'w', encoding='utf-8') as f:
                json.dump(orders, f, indent=2, ensure_ascii=False)
            
            self.current_order = new_order
            logger.info(f"Order {order_id} saved successfully")
            return order_id
            
        except Exception as e:
            logger.error(f"Error saving order: {e}")
            return None

    @function_tool()
    async def add_item_to_cart(self, context: RunContext, item_name: str, quantity: int = 1) -> str:
        """Add a specific item to the shopping cart"""
        item = self._find_item(item_name)
        if not item:
            return f"I'm sorry, I couldn't find '{item_name}' in our catalog. Could you try a different name or check our available items?"
        
        # Check if item already in cart
        for cart_item in self.cart:
            if cart_item['id'] == item["id"]:
                cart_item['quantity'] += quantity
                return f"Updated quantity of {item['name']} to {cart_item['quantity']} in your cart."
        
        # Add new item to cart
        cart_item = {
            'id': item['id'],
            'name': item['name'],
            'quantity': quantity,
            'price': item['price'],
            'category': item['category']
        }
        self.cart.append(cart_item)
        
        return f"Added {quantity} {item['name']} to your cart at ${item['price']:.2f} each."

    @function_tool()
    async def add_recipe_ingredients(self, context: RunContext, recipe_name: str, servings: int = 2) -> str:
        """Add ingredients for a recipe to the cart"""
        items = self._get_recipe_items(recipe_name)
        if not items:
            return f"I'm sorry, I don't have a recipe for '{recipe_name}' in my system. You can ask for specific items instead."
        
        added_items = []
        for item in items:
            # For recipes, adjust quantity based on servings
            adjusted_quantity = 1 if servings <= 2 else 2
            
            # Check if item already in cart
            found = False
            for cart_item in self.cart:
                if cart_item['id'] == item["id"]:
                    cart_item['quantity'] += adjusted_quantity
                    added_items.append(f"{item['name']} (quantity updated)")
                    found = True
                    break
            
            if not found:
                cart_item = {
                    'id': item['id'],
                    'name': item['name'],
                    'quantity': adjusted_quantity,
                    'price': item['price'],
                    'category': item['category']
                }
                self.cart.append(cart_item)
                added_items.append(item["name"])
        
        item_list = ", ".join(added_items)
        return f"I've added the ingredients for {recipe_name} to your cart: {item_list}. Feel free to adjust quantities if needed."

    @function_tool()
    async def remove_item_from_cart(self, context: RunContext, item_name: str) -> str:
        """Remove an item from the shopping cart"""
        item_name_lower = item_name.lower()
        for i, cart_item in enumerate(self.cart):
            if item_name_lower in cart_item['name'].lower():
                removed_item = self.cart.pop(i)
                return f"Removed {removed_item['name']} from your cart."
        
        return f"I couldn't find '{item_name}' in your cart. Here's what's currently in your cart: {await self.list_cart_contents(context)}"

    @function_tool()
    async def update_item_quantity(self, context: RunContext, item_name: str, new_quantity: int) -> str:
        """Update the quantity of an item in the cart"""
        item_name_lower = item_name.lower()
        for cart_item in self.cart:
            if item_name_lower in cart_item['name'].lower():
                if new_quantity <= 0:
                    return await self.remove_item_from_cart(context, item_name)
                
                cart_item['quantity'] = new_quantity
                return f"Updated quantity of {cart_item['name']} to {new_quantity}."
        
        return f"I couldn't find '{item_name}' in your cart. Would you like to add it instead?"

    @function_tool()
    async def list_cart_contents(self, context: RunContext) -> str:
        """List all items currently in the shopping cart"""
        if not self.cart:
            return "Your cart is currently empty. Would you like to add some items?"
        
        cart_list = []
        total = 0
        for item in self.cart:
            item_total = item['price'] * item['quantity']
            total += item_total
            cart_list.append(f"{item['quantity']} x {item['name']} - ${item_total:.2f}")
        
        cart_summary = "\n".join(cart_list)
        return f"Here's what's in your cart:\n{cart_summary}\n\nTotal: ${total:.2f}"

    @function_tool()
    async def place_order(self, context: RunContext, customer_name: str = "Customer", customer_address: str = "Not specified") -> str:
        """Place the final order and save it to the system"""
        if not self.cart:
            return "Your cart is empty. Please add some items before placing an order."
        
        # Confirm final cart
        cart_summary = await self.list_cart_contents(context)
        total = self._calculate_total()
        
        # Save order
        order_id = self._save_order(customer_name, customer_address)
        
        if order_id:
            # Clear cart after successful order
            self.cart.clear()
            return f"Thank you for your order, {customer_name}! Your order #{order_id} has been placed successfully.\n\n{cart_summary}\n\nYour order will be delivered to: {customer_address}\n\nWe'll send you a confirmation shortly. Thank you for choosing QuickCart!"
        else:
            return "I'm sorry, there was an issue placing your order. Please try again in a moment."

    @function_tool()
    async def search_items(self, context: RunContext, search_term: str) -> str:
        """Search for items in the catalog"""
        search_term_lower = search_term.lower()
        matches = []
        
        for item in self.catalog.get("items", []):
            if (search_term_lower in item.get("name", "").lower() or 
                search_term_lower in item.get("category", "").lower() or
                any(search_term_lower in tag.lower() for tag in item.get("tags", []))):
                matches.append(f"{item['name']} - ${item['price']:.2f} ({item['category']})")
        
        if not matches:
            return f"No items found matching '{search_term}'. Try browsing by category or ask me what's available."
        
        matches_list = "\n".join(matches[:8])  # Limit to 8 results
        return f"Here are items matching '{search_term}':\n{matches_list}\n\nYou can ask me to add any of these to your cart."

    @function_tool()
    async def clear_cart(self, context: RunContext) -> str:
        """Clear all items from the shopping cart"""
        item_count = len(self.cart)
        self.cart.clear()
        return f"Your cart has been cleared. Removed {item_count} items."

    @function_tool()
    async def get_available_categories(self, context: RunContext) -> str:
        """Get all available food categories"""
        categories = set()
        for item in self.catalog.get("items", []):
            categories.add(item.get('category', 'Unknown'))
        
        if categories:
            return f"We have items in these categories: {', '.join(sorted(categories))}. You can ask me to show you items from any category."
        else:
            return "I don't have any categories available at the moment."

    @function_tool()
    async def get_available_recipes(self, context: RunContext) -> str:
        """Get all available recipes"""
        recipes = list(self.catalog.get("recipes", {}).keys())
        if recipes:
            return f"I can help you with these recipes: {', '.join(recipes)}. Just ask for 'ingredients for [recipe name]'."
        else:
            return "I don't have any pre-defined recipes at the moment, but you can ask for specific items."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Initialize the food ordering agent
    agent = FoodOrderingAgent()
    
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))