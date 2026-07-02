import asyncio
import os
import json
from datetime import date
from dotenv import load_dotenv
from app.inventory_manager import get_current_inventory, process_consumption_event, update_inventory_stock, remove_inventory_item
from app.parser import parse_receipt_text_local
from app.expiration import enrich_inventory_lifespans
from app.recipe_advisor import recommend_optimized_meals, precompute_and_cache_recipes, read_cached_recipes, save_cached_recipes
load_dotenv()

def print_header(title: str):
    print('\n' + '=' * 60)
    print(f' {title.upper()}')
    print('=' * 60)

async def _print_inventory():
    inventory = await get_current_inventory()
    print_header('Current Fridge Inventory')
    if not inventory.items:
        print('Your fridge is completely empty!')
        return
    print(f"{'ITEM':<25} | {'QTY':<10} | {'EXPIRATION'}")
    print('-' * 60)
    for item in inventory.items:
        qty_str = f'{item.count} {item.unit}'
        if item.count <= 0:
            qty_str = 'Out of stock'
        print(f'{item.item_name.title():<25} | {qty_str:<10} | {item.expiration_date}')
    print('\nLast Updated:', inventory.last_updated)

async def view_inventory():
    await _print_inventory()
    print('\nOptions:')
    print(' - Type 1 to add new groceries')
    print(' - Type 0 to remove an item completely')
    print(' - Press Enter to return to the main menu')
    action = input('> ').strip().lower()
    if action == '1':
        await upload_groceries(show_inventory=False)
    elif action == '0':
        print('Enter the exact name of the item to remove:')
        item_name = input('> ').strip().lower()
        if item_name:
            try:
                await remove_inventory_item(item_name)
                print(f"\n✅ Removed all batches of '{item_name}' from the fridge!")
                asyncio.create_task(precompute_and_cache_recipes())
                input('\nPress Enter to continue...')
            except Exception as e:
                print(f'\n❌ Error removing item: {e}')
                input('\nPress Enter to continue...')

async def upload_groceries(show_inventory=True):
    if show_inventory:
        await _print_inventory()
    print_header('Add Groceries to Fridge')
    print("Enter the item's name to add:")
    print('Example: 1 litre milk, 10 tomatoes...')
    lines = []
    while True:
        line = input('> ')
        if not line:
            break
        lines.append(line)
    receipt_text = '\n'.join(lines).strip()
    if not receipt_text:
        print('No input provided. Returning to menu.')
        return
    print('\n[+] Parsing input...')
    today = date.today().isoformat()
    try:
        receipt_payload = parse_receipt_text_local(receipt_text, today)
        print(f'[-] Found {len(receipt_payload.items)} items.')
        print('[+] Calculating expiration dates...')
        inventory_payload = await enrich_inventory_lifespans(receipt_payload)
        print('[+] Updating fridge...')
        await update_inventory_stock(inventory_payload)
        print('\n✅ Successfully added groceries to the fridge!')
        asyncio.create_task(precompute_and_cache_recipes())
        print('[-] (Recipe recommendations are updating in the background...)')
    except Exception as e:
        print(f'\n❌ Error uploading groceries: {e}')

async def consume_item():
    print_header('Consume Item')
    print("Tell me what you used. Example: 'I used 1 lb of onions' or 'ate 2 eggs'")
    statement = input('> ').strip()
    if not statement:
        return
    print('\n[+] Processing consumption event with AI...')
    try:
        await process_consumption_event(statement)
        print('\n✅ Successfully updated inventory!')
        asyncio.create_task(precompute_and_cache_recipes())
    except Exception as e:
        print(f'\n❌ Error processing consumption: {e}')

async def recommend_recipes():
    print_header('Recipe Recommendations')
    recipes = await read_cached_recipes()
    if not recipes:
        print('[!] No cached recipes found. Generating new ones (this may take a moment)...')
        try:
            recipes = await recommend_optimized_meals()
            if recipes:
                await save_cached_recipes(recipes)
        except Exception as e:
            print(f'\n❌ Error generating recipes: {e}')
            return
    if not recipes:
        print('\nCould not generate recipes (is your fridge empty?).')
        return
    print(f'Found {len(recipes)} optimized recipes:\n')
    for idx, recipe in enumerate(recipes, 1):
        print(f'{idx}. {recipe.recipe_name}')
        difficulty = getattr(recipe, 'difficulty_level', 'Unknown')
        print(f'   Difficulty: {difficulty}')
        score = getattr(recipe, 'inventory_match_score', None)
        score_display = f'{score * 100:.0f}%' if score is not None else 'N/A'
        print(f'   Match Score: {score_display}')
        print(f'   Ingredients:')
        for ing in recipe.ingredients:
            print(f'     - {ing.quantity} {ing.unit} {ing.item_name}')
        print(f'   Source: {recipe.source}\n')
    print('\nOptions:')
    print(' - Enter the number of a recipe to view detailed cooking steps')
    print(' - Press Enter to return to the main menu')
    recipe_choice = input('> ').strip()
    if recipe_choice.isdigit():
        idx = int(recipe_choice) - 1
        if 0 <= idx < len(recipes):
            recipe = recipes[idx]
            print('\n' + '=' * 60)
            print(f' HOW TO COOK: {recipe.recipe_name.upper()}')
            print('=' * 60)
            print('\nSTEPS:')
            for i, step in enumerate(recipe.steps, 1):
                print(f'{i}. {step}')
            if getattr(recipe, 'restock_recommendations', None):
                print('\n[!] RESTOCK RECOMMENDATIONS:')
                for restock in recipe.restock_recommendations:
                    print(f'  - {restock.quantity_needed} {restock.unit} of {restock.item_name}')
            print('\nSource:', recipe.source)
            input('\nPress Enter to return to the main menu...')

async def main():
    print('[+] Initializing database...')
    try:
        from app.database import init_db
        await init_db()
    except Exception as e:
        print(f'❌ Error initializing database: {e}')
        return
    while True:
        print('\n' + '=' * 60)
        print(' GROCERY AGENT CLI')
        print('=' * 60)
        print('1. View Fridge Inventory')
        print('2. Add Groceries')
        print('3. Consume Groceries (Natural Language)')
        print('4. Generate / View Recipes')
        print('5. Exit')
        print('-' * 60)
        choice = input('Select an option (1-5): ').strip()
        if choice == '1':
            await view_inventory()
        elif choice == '2':
            await upload_groceries()
        elif choice == '3':
            await consume_item()
        elif choice == '4':
            await recommend_recipes()
        elif choice == '5':
            print('\nGoodbye!')
            break
        else:
            print('\nInvalid choice. Please enter a number from 1 to 5.')
        if choice not in ('1', '4'):
            input('\nPress Enter to return to the main menu...')
    from app.database import close_db_pool
    await close_db_pool()
if __name__ == '__main__':
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        print('\nGoodbye!')