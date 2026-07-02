from __future__ import annotations
import asyncio
import logging
from datetime import date, timedelta
from typing import Optional
from app.models import InventoryItem, InventoryPayload, ReceiptItem, ReceiptPayload
logger = logging.getLogger(__name__)
_DEFAULT_SHELF_LIFE_DAYS = 7
SHELF_LIFE_DB: dict[str, int] = {'milk': 7, 'whole milk': 7, 'skim milk': 7, 'buttermilk': 7, 'heavy cream': 10, 'half and half': 10, 'sour cream': 14, 'cream cheese': 14, 'cottage cheese': 10, 'ricotta': 7, 'butter': 30, 'ghee': 90, 'yogurt': 14, 'greek yogurt': 17, 'eggs': 35, 'hard boiled eggs': 7, 'mozzarella': 7, 'fresh mozzarella': 5, 'brie': 14, 'camembert': 14, 'feta': 28, 'goat cheese': 14, 'ricotta salata': 30, 'cheddar': 60, 'parmesan': 90, 'parmigiano reggiano': 90, 'gouda': 60, 'swiss cheese': 60, 'gruyere': 60, 'manchego': 60, 'provolone': 45, 'colby': 60, 'pepper jack': 60, 'chicken breast': 2, 'chicken thighs': 2, 'whole chicken': 2, 'ground beef': 2, 'beef steak': 3, 'beef roast': 5, 'pork chops': 3, 'pork loin': 5, 'ground pork': 2, 'lamb chops': 3, 'ground lamb': 2, 'veal': 3, 'turkey': 2, 'ground turkey': 2, 'duck': 2, 'bacon': 7, 'ham': 5, 'prosciutto': 5, 'salami': 14, 'pepperoni': 14, 'hot dogs': 14, 'sausage': 5, 'salmon': 2, 'tilapia': 2, 'tuna steak': 2, 'cod': 2, 'halibut': 2, 'shrimp': 2, 'scallops': 2, 'clams': 2, 'mussels': 2, 'oysters': 5, 'crab': 2, 'lobster': 2, 'canned tuna': 365, 'cooked chicken': 4, 'cooked beef': 4, 'deli turkey': 5, 'deli ham': 5, 'rotisserie chicken': 4, 'leftovers': 4, 'spinach': 5, 'baby spinach': 5, 'arugula': 5, 'romaine lettuce': 7, 'iceberg lettuce': 10, 'kale': 7, 'swiss chard': 5, 'collard greens': 7, 'cabbage': 14, 'red cabbage': 14, 'bok choy': 5, 'brussels sprouts': 7, 'broccoli': 5, 'cauliflower': 7, 'asparagus': 4, 'celery': 14, 'fennel': 7, 'artichoke': 7, 'bell pepper': 10, 'red bell pepper': 10, 'green bell pepper': 10, 'jalapeno': 14, 'serrano pepper': 14, 'zucchini': 7, 'yellow squash': 7, 'eggplant': 5, 'cucumber': 7, 'tomato': 7, 'roma tomatoes': 7, 'cherry tomatoes': 10, 'grape tomatoes': 10, 'corn': 3, 'okra': 4, 'carrots': 21, 'baby carrots': 21, 'potatoes': 21, 'sweet potatoes': 21, 'beets': 14, 'parsnips': 14, 'turnips': 14, 'radishes': 14, 'ginger': 21, 'garlic': 120, 'onion': 60, 'onions': 60, 'red onion': 60, 'shallots': 30, 'leeks': 14, 'scallions': 7, 'green onions': 7, 'mushrooms': 7, 'cremini mushrooms': 7, 'portobello mushrooms': 7, 'shiitake mushrooms': 10, 'lemons': 21, 'limes': 21, 'oranges': 21, 'grapefruit': 21, 'clementines': 14, 'mandarins': 14, 'strawberries': 5, 'blueberries': 10, 'raspberries': 3, 'blackberries': 5, 'cranberries': 28, 'bananas': 5, 'mangoes': 5, 'pineapple': 5, 'papaya': 5, 'kiwi': 14, 'avocado': 4, 'coconut': 14, 'peaches': 5, 'nectarines': 5, 'plums': 5, 'cherries': 7, 'apricots': 5, 'apples': 42, 'pears': 7, 'grapes': 10, 'watermelon': 10, 'cantaloupe': 5, 'honeydew': 7, 'bread': 7, 'sourdough bread': 7, 'whole wheat bread': 7, 'bagels': 7, 'english muffins': 7, 'tortillas': 14, 'flour tortillas': 14, 'corn tortillas': 14, 'pita bread': 7, 'baguette': 3, 'croissants': 3, 'muffins': 5, 'ketchup': 180, 'mustard': 180, 'mayonnaise': 90, 'hot sauce': 180, 'soy sauce': 365, 'worcestershire sauce': 365, 'salsa': 14, 'hummus': 10, 'guacamole': 3, 'pesto': 7, 'tomato sauce': 5, 'pasta sauce': 5, 'ranch dressing': 30, 'italian dressing': 60, 'balsamic vinegar': 365, 'orange juice': 7, 'apple juice': 7, 'coconut water': 5, 'almond milk': 7, 'oat milk': 7, 'soy milk': 7, 'tofu': 5, 'tempeh': 10, 'cooked rice': 5, 'cooked pasta': 5, 'cooked beans': 5, 'open canned goods': 4, 'olives': 30, 'capers': 30, 'pickles': 90, 'kimchi': 90, 'sauerkraut': 60, 'miso paste': 365, 'basil': 7, 'cilantro': 7, 'parsley': 10, 'mint': 10, 'dill': 10, 'thyme': 14, 'rosemary': 14, 'chives': 10, 'sage': 14, 'tarragon': 10, 'walnuts': 180, 'almonds': 180, 'cashews': 180, 'pecans': 180, 'pine nuts': 90, 'sesame seeds': 180, 'flaxseeds': 90}

def _local_shelf_life(item_name: str) -> Optional[int]:
    if item_name in SHELF_LIFE_DB:
        return SHELF_LIFE_DB[item_name]
    for key, days in SHELF_LIFE_DB.items():
        if key in item_name or item_name in key:
            logger.debug("Partial shelf-life match: '%s' → '%s' (%d days)", item_name, key, days)
            return days
    return None

async def _enrich_item(item: ReceiptItem) -> InventoryItem:
    shelf_life = _local_shelf_life(item.item_name)
    source = 'local_db'
    if shelf_life is None:
        shelf_life = _DEFAULT_SHELF_LIFE_DAYS
        source = 'default_fallback'
    logger.debug("Shelf life resolved | item='%s' | days=%d | source=%s", item.item_name, shelf_life, source)
    try:
        purchase = date.fromisoformat(item.purchase_date)
    except ValueError as exc:
        logger.warning("Invalid purchase_date '%s' for item '%s' — using today | error=%s", item.purchase_date, item.item_name, exc)
        purchase = date.today()
    expiration = purchase + timedelta(days=shelf_life)
    return InventoryItem(item_name=item.item_name, count=item.count, unit=item.unit, purchase_date=item.purchase_date, expiration_date=expiration.isoformat())

async def enrich_inventory_lifespans(receipt: ReceiptPayload) -> InventoryPayload:
    if not isinstance(receipt, ReceiptPayload):
        raise TypeError(f'Expected a ReceiptPayload instance, got {type(receipt).__name__}. Ensure parser.parse_receipt_text() completed successfully before calling this function.')
    if not receipt.items:
        logger.warning('enrich_inventory_lifespans called with an empty ReceiptPayload. Returning empty InventoryPayload.')
        return InventoryPayload(items=[], last_updated=date.today().isoformat())
    logger.info('Starting expiration enrichment | items=%d | purchase_date=%s', len(receipt.items), receipt.date_of_purchase)
    results = await asyncio.gather(*[_enrich_item(item) for item in receipt.items], return_exceptions=True)
    inventory_items: list[InventoryItem] = []
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            failed_name = receipt.items[idx].item_name
            logger.error("Enrichment failed for item '%s' (index %d) — skipping | error=%s", failed_name, idx, result)
        else:
            inventory_items.append(result)
    inventory_items.sort(key=lambda it: it.expiration_date)
    logger.info('Expiration enrichment complete | enriched=%d | failed=%d', len(inventory_items), len(receipt.items) - len(inventory_items))
    return InventoryPayload(items=inventory_items, last_updated=date.today().isoformat())