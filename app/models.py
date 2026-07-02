from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field, HttpUrl, field_validator

class ReceiptItem(BaseModel):
    item_name: str = Field(..., description="Normalized, lowercase product name extracted from the receipt. Abbreviations should be expanded (e.g., 'chkn' → 'chicken breast'). Brand names should be stripped unless they meaningfully disambiguate the product.", examples=['whole milk', 'chicken breast', 'roma tomatoes'])
    count: float = Field(..., gt=0, description='Numeric quantity of the item purchased. Use floats for weight-based items (e.g., 1.5 for 1.5 lbs) and integers cast as floats for countable items (e.g., 2.0 for 2 cans). Must be strictly positive.', examples=[1.5, 2.0, 0.75])
    unit: str = Field(..., description="Unit of measurement corresponding to `count`. Use SI or common grocery abbreviations: 'pcs' (pieces), 'lbs' (pounds), 'kg' (kilograms), 'oz' (ounces), 'L' (litres), 'ml' (millilitres), 'g' (grams).", examples=['pcs', 'lbs', 'kg', 'oz', 'L', 'ml', 'g'])
    purchase_date: str = Field(..., description='Date the item was purchased, formatted as an ISO-8601 date string (YYYY-MM-DD). Used downstream by the expiration engine to compute shelf-life deadlines.', examples=['2025-07-01', '2025-06-15'])

    @field_validator('item_name', mode='before')
    @classmethod
    def normalize_item_name(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator('unit', mode='before')
    @classmethod
    def normalize_unit(cls, v: str) -> str:
        return v.strip()

class ReceiptPayload(BaseModel):
    items: list[ReceiptItem] = Field(..., description='Ordered list of all individual products found on the receipt. Each element is a fully validated ReceiptItem. An empty list is valid but will produce no downstream inventory updates.', min_length=0)
    date_of_purchase: str = Field(..., description='Transaction date of the entire receipt in ISO-8601 format (YYYY-MM-DD). Acts as a fallback purchase_date for any ReceiptItem that omits its own date.', examples=['2025-07-01'])

class InventoryItem(BaseModel):
    item_name: str = Field(..., description='Normalized, lowercase product name — identical to ReceiptItem.item_name after receipt parsing. Used as the primary key when merging quantities across multiple purchase events.', examples=['whole milk', 'chicken breast'])
    count: float = Field(..., ge=0, description='Remaining quantity of the item currently in inventory. Decremented by inventory_manager.py on every consumption event. A value of 0.0 signals that the item is fully consumed and may be pruned from the active inventory snapshot.', examples=[1.0, 0.5, 2.0])
    unit: str = Field(..., description='Unit of measurement for `count`, inherited from ReceiptItem.unit. Must remain consistent across all updates to the same item_name.', examples=['pcs', 'lbs', 'kg', 'L'])
    purchase_date: str = Field(..., description='Original purchase date of this inventory batch, in ISO-8601 format (YYYY-MM-DD). Retained for audit trails and multi-batch FIFO consumption ordering.', examples=['2025-07-01'])
    expiration_date: str = Field(..., description="Projected expiration date computed by expiration.py, in ISO-8601 format (YYYY-MM-DD). Derived from the product category's average shelf life added to purchase_date. Items past this date should be flagged as expired and excluded from recipe suggestions.", examples=['2025-07-08', '2025-07-15'])

    @field_validator('item_name', mode='before')
    @classmethod
    def normalize_item_name(cls, v: str) -> str:
        return v.strip().lower()

class InventoryPayload(BaseModel):
    items: list[InventoryItem] = Field(..., description='Full list of InventoryItem records representing the current pantry state. Sorted by expiration_date ascending so that the most urgent items appear first. Stale (fully consumed or expired) items may be omitted in pruned snapshots.', min_length=0)
    last_updated: Optional[str] = Field(default=None, description='ISO-8601 timestamp (YYYY-MM-DDTHH:MM:SS) of the most recent write to this snapshot. Set automatically by inventory_manager.py on every save operation. None if the payload has never been persisted.', examples=['2025-07-01T14:30:00'])

class IngredientRequirement(BaseModel):
    item_name: str = Field(..., description="Normalized, lowercase ingredient name. Should align with InventoryItem.item_name naming conventions so that fuzzy matching can resolve near-duplicates (e.g., 'tomato' ↔ 'roma tomatoes').", examples=['chicken breast', 'garlic', 'olive oil'])
    quantity: str = Field(..., description="Human-readable quantity string as it would appear in a cookbook, e.g., '2', '1/2', '1.5', '3 cloves'. Kept as a string to preserve fractional and descriptive notations.", examples=['2', '1/2 cup', '3 cloves', '200'])
    unit: str = Field(..., description="Unit corresponding to `quantity`. Use the same abbreviation set as ReceiptItem.unit for consistency. Use 'to taste' or 'as needed' for open-ended ingredients like salt.", examples=['pcs', 'g', 'ml', 'cup', 'tbsp', 'tsp', 'to taste'])

class RestockItem(BaseModel):
    item_name: str = Field(..., description='Normalized, lowercase name of the ingredient that needs restocking. Corresponds directly to IngredientRequirement.item_name.', examples=['whole milk', 'eggs', 'butter'])
    quantity_needed: float = Field(..., gt=0, description='Numeric amount of the item the user should purchase to satisfy at least one full serving of the parent recipe. Computed as max(0, required_quantity − available_inventory). Must be strictly positive — zero-shortfall items are excluded.', examples=[1.0, 2.5, 0.5])
    unit: str = Field(..., description='Unit of measurement for `quantity_needed`, consistent with IngredientRequirement.unit and InventoryItem.unit.', examples=['pcs', 'lbs', 'L', 'g'])

class Recipe(BaseModel):
    recipe_name: str = Field(..., description="Full, human-readable name of the dish. Should be descriptive enough to be searchable and recognizable, e.g., 'Lemon Herb Roasted Chicken' rather than 'Chicken Dish'.", examples=['Lemon Herb Roasted Chicken', 'Classic Margherita Pizza'])
    difficulty_level: str = Field(..., description="The difficulty level of the recipe, such as 'Easy', 'Medium', or 'Hard'.", examples=['Easy', 'Medium', 'Hard'])
    ingredients: list[IngredientRequirement] = Field(..., description='Complete list of ingredients required to prepare this recipe. Each entry is an IngredientRequirement specifying the item, quantity, and unit. recipe_advisor.py cross-references this list against InventoryPayload.items to compute an inventory-match score.', min_length=1)
    steps: list[str] = Field(..., description="Ordered, numbered cooking instructions as plain-text strings. Each string should be a single, atomic action (e.g., 'Preheat oven to 200°C / 400°F.'). Minimum of one step required.", min_length=1, examples=[['Preheat oven to 200°C.', 'Season chicken with salt and pepper.', 'Roast for 45 minutes.']])
    source: str = Field(..., description='Canonical URL pointing to the original recipe source or reference. Used for attribution and to allow users to access the full recipe details. Must be a valid HTTP/HTTPS URL.', examples=['https://www.allrecipes.com/recipe/lemon-herb-chicken'])
    restock_recommendations: list[RestockItem] = Field(default_factory=list, description='List of RestockItem objects for ingredients this recipe requires but that are missing or insufficient in the current inventory. An empty list means all ingredients are fully available — the recipe is ready to cook. Sorted by quantity_needed descending to surface the largest shortfalls first.')
    inventory_match_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description='Fraction of required ingredients (by count) already satisfied by current inventory, computed by recipe_advisor.py. Range [0.0, 1.0]. 1.0 = all ingredients on hand; 0.0 = no ingredients available. None if the score has not yet been calculated.', examples=[1.0, 0.75, 0.33])