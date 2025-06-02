import json
import os
import glob
import re

# --- Configuration ---
# NOTE: Please update 'input_directory' if you are using this script for a different location.
input_directory = 'path/to/your/result.json' # Directory containing your JSON files

# File pattern to match, e.g., files ending with 'part<number>.json'

file_pattern = 'result_part*.json'

# Output filename for the merged JSON file (will be saved in the input_directory)
output_file_name = 'result_merged.json'
# --- End of Configuration ---

# List to store all data merged from the JSON files
merged_data = []

# Check if the input directory exists
if not os.path.isdir(input_directory):
    print(f"Error: Input directory '{input_directory}' does not exist or is not a valid directory.")
    print("Please ensure the 'input_directory' variable is set correctly.")
    exit() # Exit the script

# Construct the full file search pattern
json_search_pattern = os.path.join(input_directory, file_pattern)
# Find all .json files in the directory that match the pattern
json_files = glob.glob(json_search_pattern)

if not json_files:
    print(f"No .json files matching '{file_pattern}' found in directory '{input_directory}'.")
    exit() # Exit the script

# --- Key Step: Sort files by the number in their filename ---
def extract_part_number(filepath):
    """
    Extracts the number following 'part' from the filename.
    Example: Extracts 12 from '...part12.json'.
    Returns float('inf') if no number is found, to sort these last or handle as an error.
    """
    filename = os.path.basename(filepath)
    # Use regex to find 'part' followed by one or more digits (\d+) before '.json'
    match = re.search(r'part(\d+)\.json$', filename, re.IGNORECASE) # re.IGNORECASE for case-insensitivity
    if match:
        return int(match.group(1))
    else:
        print(f"Warning: Could not extract 'part' number from filename '{filename}'. It might be sorted incorrectly.")
        return float('inf') # Items without a number will be sorted towards the end

try:
    # Sort the files based on the extracted part number
    sorted_json_files = sorted(json_files, key=extract_part_number)
except TypeError:
    # This might happen if extract_part_number returns types that can't be compared (though float('inf') should be fine)
    print("Error: Sorting process interrupted. This might be due to issues in extracting numbers from filenames.")
    print("Attempting to sort by original alphabetical order as a fallback...")
    sorted_json_files = sorted(json_files) # Fallback to simple alphabetical sort
# --- End of Sorting ---

print(f"Found {len(sorted_json_files)} matching JSON files. Will merge in the following order:")
for f_path in sorted_json_files:
    print(f"  - {os.path.basename(f_path)}")

# Iterate through the sorted list of file paths
for file_path in sorted_json_files:
    current_file_name = os.path.basename(file_path)
    print(f"\nReading: {current_file_name} ...")
    try:
        with open(file_path, 'r', encoding='utf-8') as infile:
            # Load the JSON data from the current file
            data = json.load(infile)

            # Ensure the data is a list before extending merged_data
            if isinstance(data, list):
                merged_data.extend(data)
                print(f"  Successfully added {len(data)} records.")
            else:
                print(f"    Warning: Content of file '{current_file_name}' is not in the expected list format (type: {type(data)}). Skipped.")

    except json.JSONDecodeError:
        print(f"    Error: File '{current_file_name}' contains invalid JSON data. Skipped.")
    except Exception as e:
        # Catch other potential file reading errors
        print(f"    Error reading file '{current_file_name}': {e}. Skipped.")

# Check if any data was successfully read
if not merged_data:
    print("\nFailed to read data from any files. Cannot generate merged file.")
else:
    # Write the merged data to a new JSON file
    print(f"\nMerging complete. Total of {len(merged_data)} records merged.")
    try:
        # Construct the full path for the output file
        output_path = os.path.join(input_directory, output_file_name)
        with open(output_path, 'w', encoding='utf-8') as outfile:
            json.dump(merged_data, outfile, ensure_ascii=False, indent=4)
        print(f"Merged data successfully written to file: '{output_path}'")
    except Exception as e:
        print(f"Error writing output file '{output_path}': {e}")