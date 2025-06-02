import pandas as pd
import json
import re
import os

# --- Configuration ---
# NOTE: Please update these paths to your local environment.
parquet_file_path = 'path/to/your/category_data.parquet'  # Path to your Parquet file
input_json_path = 'path/to/your/input_data.json'        # Path to your original JSON file
# Output JSON filename (will be saved to the same directory as input_json_path by default, or update as needed)
output_json_path = 'path/to/your/output_data_with_categories.json'

# --- Check if files exist ---
if not os.path.exists(parquet_file_path):
    print(f"Error: Parquet file not found: {parquet_file_path}")
    exit()
if not os.path.exists(input_json_path):
    print(f"Error: Input JSON file not found: {input_json_path}")
    exit()

try:
    # --- 1. Read Parquet File ---
    print(f"Reading Parquet file: {parquet_file_path}...")
    # Read only necessary columns to improve efficiency
    df = pd.read_parquet(parquet_file_path, columns=['index', 'category'])
    print(f"Parquet file read successfully. Contains {len(df)} rows.")

    # Check if required columns exist
    if 'index' not in df.columns or 'category' not in df.columns:
         raise ValueError("Parquet file must contain 'index' and 'category' columns.")

    # Create a dictionary for quick category lookup: {index: category_value}
    # Assumes 'index' column in Parquet file corresponds to image indices.
    category_lookup = df.set_index('index')['category'].to_dict()
    print("Created Index -> Category lookup dictionary.")

    # --- 2. Read JSON File ---
    print(f"Reading JSON file: {input_json_path}...")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    print(f"JSON file read successfully. Contains {len(json_data)} entries.")

    # Ensure json_data is a list of objects
    if not isinstance(json_data, list):
        raise TypeError("The input JSON file should contain a list of objects.")

    # --- 3. Iterate, Match, and Update JSON Data ---
    updated_count = 0
    skipped_count = 0
    processed_data = [] # Create a new list to store updated data

    print("Starting to process JSON entries and add 'category' field...")
    for item in json_data:
        try:
            # Check if 'images' field exists and is not empty
            if "images" not in item or not item["images"]:
                print(f"Warning: Entry missing 'images' field or it's empty. Skipping category addition.")
                processed_data.append(item) # Keep the original item
                skipped_count += 1
                continue

            # Assume the first image filename contains the index information
            image_filename = item["images"][0]

            # Extract index N from filename like "image_N.jpg"
            match = re.search(r'image_(\d+)\.(jpg|png|jpeg|gif|bmp|tiff|webp)$', image_filename, re.IGNORECASE) # Match end of filename and common extensions
            if not match:
                print(f"Warning: Could not extract index from filename '{image_filename}'. Skipping category addition.")
                processed_data.append(item) # Keep the original item
                skipped_count += 1
                continue

            # Get the index (as an integer)
            entry_index = int(match.group(1))

            # --- 4. Lookup Category ---
            if entry_index in category_lookup:
                category_value = category_lookup[entry_index]

                # --- 5. Add Category to JSON Entry ---
                # Create a copy to avoid modifying the original item during iteration (good practice)
                updated_item = item.copy()
                updated_item["category"] = [str(category_value)] # Ensure it's a string and in a list
                processed_data.append(updated_item)
                updated_count += 1
            else:
                print(f"Warning: Index {entry_index} (from file '{image_filename}') not found in Parquet data. Skipping category addition.")
                processed_data.append(item) # Keep the original item
                skipped_count += 1

        except Exception as e:
            image_ref = item.get('images', ['N/A'])[0] if isinstance(item.get('images'), list) and item.get('images') else 'N/A'
            print(f"Error processing entry (image possibly: {image_ref}): {e}")
            processed_data.append(item) # Keep the original item in case of error
            skipped_count += 1

    print(f"Processing complete. Successfully updated {updated_count} entries, skipped {skipped_count} entries.")

    # --- 6. Write the New JSON File ---
    print(f"Writing updated data to: {output_json_path}...")
    # Ensure output directory exists if output_json_path includes directories
    output_dir = os.path.dirname(output_json_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    with open(output_json_path, 'w', encoding='utf-8') as f:
        # indent=2 makes the output JSON file more readable
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    print("All operations completed successfully!")

except FileNotFoundError as e:
    print(f"Error: File not found - {e}")
except ImportError:
    print("Error: Missing necessary libraries. Please ensure pandas and a Parquet engine (pyarrow or fastparquet) are installed.")
    print("You can typically install them by running: pip install pandas pyarrow")
except Exception as e:
    print(f"An unexpected error occurred during processing: {e}")