import json
import os

# --- Configuration ---
input_filename = 'path/to/your/dataset.json'  # <<< CHANGE THIS to your actual input file name
output_filename_1 = 'path/to/your/file1.json'  # File for the first 2000 objects
output_filename_2 = 'path/to/your/file2.json'  # File for the remaining objects
split_index = 1000          # The number of objects to put in the first file
# --- End Configuration ---

def split_json_file(input_file, output_file1, output_file2, split_at):
    """
    Loads a JSON file containing a list, splits the list into two parts,
    and saves each part to a separate JSON file.
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at '{input_file}'")
        return

    try:
        # Load the entire JSON data from the input file
        print(f"Loading data from '{input_file}'...")
        with open(input_file, 'r', encoding='utf-8') as f_in:
            all_data = json.load(f_in)

        # Validate that the loaded data is a list
        if not isinstance(all_data, list):
            print(f"Error: Expected a JSON list (array) in '{input_file}', but found type {type(all_data)}.")
            return

        total_objects = len(all_data)
        print(f"Successfully loaded {total_objects} objects.")

        # Check if the split index is valid
        if split_at <= 0:
             print(f"Warning: Split index ({split_at}) is zero or negative. Putting all data in the second file.")
             part1_data = []
             part2_data = all_data
        elif split_at >= total_objects:
             print(f"Warning: Split index ({split_at}) is greater than or equal to total objects ({total_objects}). Putting all data in the first file.")
             part1_data = all_data
             part2_data = []
        else:
            # Split the data using list slicing
            part1_data = all_data[:split_at]
            part2_data = all_data[split_at:]

        print(f"Splitting complete:")
        print(f"  - Part 1 ({output_file1}): {len(part1_data)} objects")
        print(f"  - Part 2 ({output_file2}): {len(part2_data)} objects")

        # Save the first part
        print(f"Saving part 1 to '{output_file1}'...")
        with open(output_file1, 'w', encoding='utf-8') as f_out1:
            # indent=2 makes the output JSON readable, ensure_ascii=False handles non-English characters correctly
            json.dump(part1_data, f_out1, indent=2, ensure_ascii=False)

        # Save the second part
        print(f"Saving part 2 to '{output_file2}'...")
        with open(output_file2, 'w', encoding='utf-8') as f_out2:
            json.dump(part2_data, f_out2, indent=2, ensure_ascii=False)

        print("\nSuccessfully split the JSON file into two parts.")

    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{input_file}'. Check if it's a valid JSON file.")
    except MemoryError:
         print(f"Error: Not enough memory to load the entire file '{input_file}'. Consider using a streaming JSON parser for very large files.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Run the splitting function ---
if __name__ == "__main__":
    split_json_file(input_filename, output_filename_1, output_filename_2, split_index)