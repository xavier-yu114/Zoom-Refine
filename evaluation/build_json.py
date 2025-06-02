import pandas as pd
import json
import os

# --- Configuration ---
parquet_file_path = 'path/to/your/hr_bench_8k.parquet'
image_dir = 'path/to/your/image_dataset' # Directory where images were saved
output_json_path = 'path/to/your/result.json' # Output JSON file path
possible_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'] # Common extensions to check

# --- Read Parquet File ---
print(f"Reading Parquet file: {parquet_file_path}")
try:
    df = pd.read_parquet(parquet_file_path)
    print(f"Successfully read {len(df)} records from Parquet.")
except FileNotFoundError:
    print(f"Error: Parquet file not found at {parquet_file_path}")
    exit()
except Exception as e:
    print(f"Error reading Parquet file: {e}")
    exit()

# --- Prepare list for JSON data ---
json_data = []
missing_images = 0
processed_count = 0

print("Starting JSON construction...")
# --- Iterate through DataFrame rows ---
for index, row in df.iterrows():
    try:
        # Extract data from the row
        question = row['question']
        choice_a = row['A']
        choice_b = row['B']
        choice_c = row['C']
        choice_d = row['D']
        answer = row['answer']
        # The 'index' column from the parquet file seems to directly correspond
        # to the image index we used previously (image_0, image_1, etc.)
        # If your parquet 'index' column is different, adjust this logic.
        image_index = row['index'] # Assuming the 'index' column matches image number

        # Format the user content string
        user_content = (
            f"{question}\n"
            f"The choices are listed below:\n"
            f"(A) {choice_a}\n"
            f"(B) {choice_b}\n"
            f"(C) {choice_c}\n"
            f"(D) {choice_d}\n"
            f"Select the best answer to the above multiple-choice question based on the image. "
            f"Respond with only the letter (A, B, C, or D) of the correct option\n"
            f"The best answer is:."
        )

        # Format the assistant content
        assistant_content = str(answer) # Ensure it's a string

        # Find the actual relative image path by checking file existence
        relative_image_path = None
        for ext in possible_extensions:
            potential_filename = f"image_{image_index}{ext}"
            full_path_check = os.path.join(image_dir, potential_filename)
            if os.path.exists(full_path_check):
                relative_image_path = potential_filename
                break # Found the image

        if relative_image_path is None:
            print(f"Warning: Image file for index {image_index} not found in {image_dir} with extensions {possible_extensions}. Skipping this entry.")
            missing_images += 1
            continue

        # Create the JSON object for this row
        json_entry = {
            "messages": [
                {
                    "role": "user",
                    "content": user_content
                },
                {
                    "role": "assistant",
                    "content": assistant_content
                }
            ],
            "images": [
                relative_image_path # Use the found relative path
            ]
        }

        # Add the entry to our list
        json_data.append(json_entry)
        processed_count += 1

    except KeyError as e:
        print(f"Error: Missing expected column {e} in row index {index}. Skipping.")
        missing_images +=1 # Count as an issue
    except Exception as e:
        print(f"Error processing row index {index}: {e}. Skipping.")
        missing_images +=1 # Count as an issue

# --- Write the JSON file ---
print(f"\nProcessed {processed_count} entries successfully.")
if missing_images > 0:
     print(f"Warning: Skipped {missing_images} entries due to missing images or processing errors.")

print(f"Writing JSON data to: {output_json_path}")
try:
    with open(output_json_path, 'w', encoding='utf-8') as f:
        # Use indent for pretty printing, ensure_ascii=False for non-ASCII chars
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print("JSON file successfully created.")
except Exception as e:
    print(f"Error writing JSON file: {e}")