"""
cli.py
─────────────────────────────────────────────────────────────────────────────
Interactive Command Line Interface for the Grocery Lifecycle Tracker.

Provides a terminal-based way to:
1. View current fridge inventory
2. Upload groceries (receipts or manual lists)
3. Consume items from the fridge
4. Remove items completely
5. Get recipe recommendations
6. Exit

Run this directly:
    python cli.py
─────────────────────────────────────────────────────────────────────────────
"""

import asyncio
import os
import json
from datetime import date
from dotenv import load_dotenv

from app.inventory_manager import (
    get_current_inventory,
    process_consumption_event,
    update_inventory_stock,
    remove_inventory_item,
)
from app.parser import parse_receipt_text_local
from app.expiration import enrich_inventory_lifespans
from app.recipe_advisor import recommend_optimized_meals, precompute_and_cache_recipes, read_cached_recipes

# Load environment variables
load_dotenv()


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f" {title.upper()}")
    print("=" * 60)


async def view_inventory():
    inventory = await get_current_inventory()
    print_header("Current Fridge Inventory")
    
    if not inventory.items:
        print("Your fridge is completely empty!")
        return

    print(f"{'ITEM':<25} | {'QTY':<10} | {'EXPIRATION'}")
    print("-" * 60)
    for item in inventory.items:
        qty_str = f"{item.count} {item.unit}"
        if item.count <= 0:
            qty_str = "Out of stock"
        print(f"{item.item_name.title():<25} | {qty_str:<10} | {item.expiration_date}")
    
    print("\nLast Updated:", inventory.last_updated)
    
    print("\nOptions:")
    print(" - Enter the exact name of an item to remove it completely")
    print(" - Press Enter to return to the main menu")
    item_name = input("> ").strip().lower()
    
    if item_name:
        try:
            await remove_inventory_item(item_name)
            print(f"\n✅ Removed all batches of '{item_name}' from the fridge!")
            asyncio.create_task(precompute_and_cache_recipes())
            input("\nPress Enter to continue...")
        except Exception as e:
            print(f"\n❌ Error removing item: {e}")
            input("\nPress Enter to continue...")


async def upload_groceries():
    print_header("Upload Groceries")
    print("Enter your receipt text or a comma-separated list of items.")
    print("Example: 1 litre milk, 10 tomatoes, 10 potatoes")
    print("Press Enter on an empty line to finish.")
    
    lines = []
    while True:
        line = input("> ")
        if not line:
            break
        lines.append(line)
        
    receipt_text = "\n".join(lines).strip()
    if not receipt_text:
        print("No input provided. Returning to menu.")
        return
        
    print("\n[+] Parsing input...")
    today = date.today().isoformat()
    try:
        # Fast local parsing
        receipt_payload = parse_receipt_text_local(receipt_text, today)
        print(f"[-] Found {len(receipt_payload.items)} items.")
        
        print("[+] Calculating expiration dates...")
        inventory_payload = await enrich_inventory_lifespans(receipt_payload)
        
        print("[+] Updating fridge...")
        await update_inventory_stock(inventory_payload)
        
        print("\n✅ Successfully added groceries to the fridge!")
        
        # Trigger background cache update
        asyncio.create_task(precompute_and_cache_recipes())
        print("[-] (Recipe recommendations are updating in the background...)")
        
    except Exception as e:
        print(f"\n❌ Error uploading groceries: {e}")


async def consume_item():
    print_header("Consume Item")
    print("Tell me what you used. Example: 'I used 1 lb of onions' or 'ate 2 eggs'")
    statement = input("> ").strip()
    
    if not statement:
        return
        
    print("\n[+] Processing consumption event with AI...")
    try:
        await process_consumption_event(statement)
        print("\n✅ Successfully updated inventory!")
        
        # Trigger background cache update
        asyncio.create_task(precompute_and_cache_recipes())
        
    except Exception as e:
        print(f"\n❌ Error processing consumption: {e}")





async def recommend_recipes():
    print_header("Recipe Recommendations")
    
    # Try reading from cache first for immediate results
    recipes = await read_cached_recipes()
    if not recipes:
        print("[!] No cached recipes found. Generating new ones (this may take a moment)...")
        try:
            recipes = await recommend_optimized_meals()
            # Cache them in the background
            asyncio.create_task(precompute_and_cache_recipes())
        except Exception as e:
            print(f"\n❌ Error generating recipes: {e}")
            return
            
    if not recipes:
        print("\nCould not generate recipes (is your fridge empty?).")
        return
        
    print(f"Found {len(recipes)} optimized recipes:\n")
    for idx, recipe in enumerate(recipes, 1):
        print(f"{idx}. {recipe.recipe_name}")
        difficulty = getattr(recipe, 'difficulty_level', 'Unknown')
        print(f"   Difficulty: {difficulty}")
        score = getattr(recipe, 'inventory_match_score', None)
        score_display = f"{score * 100:.0f}%" if score is not None else "N/A"
        print(f"   Match Score: {score_display}")
        print(f"   Ingredients:")
        for ing in recipe.ingredients:
            print(f"     - {ing.quantity} {ing.unit} {ing.item_name}")
        print(f"   Source: {recipe.source}\n")


async def main():
    while True:
        print("\n" + "=" * 60)
        print(" GROCERY AGENT CLI")
        print("=" * 60)
        print("1. View Fridge Inventory")
        print("2. Add Groceries")
        print("3. Consume Groceries (Natural Language)")
        print("4. Generate / View Recipes")
        print("5. Exit")
        print("-" * 60)
        
        choice = input("Select an option (1-5): ").strip()
        
        if choice == '1':
            await view_inventory()
        elif choice == '2':
            await upload_groceries()
        elif choice == '3':
            await consume_item()
        elif choice == '4':
            await recommend_recipes()
        elif choice == '5':
            print("\nGoodbye!")
            break
        else:
            print("\nInvalid choice. Please enter a number from 1 to 5.")
            
        if choice != '1':
            input("\nPress Enter to return to the main menu...")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
