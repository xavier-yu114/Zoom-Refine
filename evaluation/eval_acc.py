import json
import re
import sys # For outputting error messages to stderr

def evaluate_predictions(json_filepath, n, m, verbose=False):
    """
    Evaluates the prediction accuracy for a specified range of entries in a JSON file.

    Args:
        json_filepath (str): Path to the JSON file.
        n (int): Starting entry number (1-based).
        m (int): Ending entry number (1-based).
        verbose (bool): Whether to print detailed information for mismatched entries.
    """
    # --- 1. Load JSON file ---
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found {json_filepath}", file=sys.stderr)
        return
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON file {json_filepath}", file=sys.stderr)
        return
    except Exception as e:
        print(f"Error: An unknown error occurred while loading the file: {e}", file=sys.stderr)
        return

    # --- 2. Validate range ---
    total_items = len(data)
    if not isinstance(n, int) or not isinstance(m, int) or n < 1 or m > total_items or n > m:
        print(f"Error: Invalid range n={n}, m={m}. Please ensure 1 <= n <= m <= {total_items}", file=sys.stderr)
        return

    # --- 3. Select subset and initialize counters ---
    subset = data[n-1:m] # Python slicing is 0-based and exclusive for the end index
    correct_count = 0
    processed_count = 0         # Number of entries successfully processed and compared
    error_extraction_count = 0  # Number of entries where prediction or ground truth could not be extracted
    mismatched_items = []       # Stores details of mismatched predictions
    total_subset_count = len(subset)

    print(f"Starting evaluation for entries {n} to {m} ({total_subset_count} entries) in file '{json_filepath}'...")

    # --- 4. Iterate through the subset ---
    for i, entry in enumerate(subset):
        item_index = n + i # Original 1-based index of the current entry

        try:
            # --- Extract Ground Truth ---
            # Assumes ground truth is in the second message's content
            if not isinstance(entry.get('messages'), list) or len(entry['messages']) < 2 or \
               not isinstance(entry['messages'][1], dict) or 'content' not in entry['messages'][1]:
                 print(f"Warning: Entry {item_index} has incorrect format or missing valid ground truth. Skipping.", file=sys.stderr)
                 error_extraction_count += 1
                 continue

            assistant_message = entry['messages'][1]['content']
            # Assume ground truth is always a single uppercase letter A-E
            ground_truth = assistant_message.strip().upper()
            if not re.fullmatch(r'[A-E]', ground_truth): # Ensures it's exactly one letter from A to E
                 print(f"Warning: Ground truth '{assistant_message}' in entry {item_index} is not the expected single letter (A-E). Skipping.", file=sys.stderr)
                 error_extraction_count += 1
                 continue

            # --- Extract Predicted Answer ---
            predicted_answer_raw = entry.get('predicted_answer') # Use .get() to prevent KeyError

            if not isinstance(predicted_answer_raw, str):
                print(f"Warning: 'predicted_answer' in entry {item_index} is not a string or is missing ('{predicted_answer_raw}'). Skipping.", file=sys.stderr)
                error_extraction_count += 1
                continue

            # Regex to find a letter (A-E, case-insensitive) optionally surrounded by parentheses
            match = re.search(r'\(?\s*([A-Ea-e])\s*\)?', predicted_answer_raw)

            predicted_letter = None
            if match:
                # Extract the captured letter and convert to uppercase
                predicted_letter = match.group(1).upper()
            else:
                # If regex did not match any A-E formatted option
                print(f"Warning: Could not extract an option letter from the predicted answer '{predicted_answer_raw}' in entry {item_index}. Skipping.", file=sys.stderr)
                error_extraction_count += 1
                continue # Skip this entry as prediction couldn't be extracted

            # --- Compare Prediction and Ground Truth ---
            processed_count += 1 # Increment count of successfully processed entries
            if predicted_letter == ground_truth:
                correct_count += 1
            else:
                mismatched_items.append({
                    "index": item_index,
                    "ground_truth": ground_truth,
                    "predicted_raw": predicted_answer_raw,
                    "predicted_extracted": predicted_letter
                })

        except Exception as e:
            print(f"Error: An unexpected error occurred while processing entry {item_index}: {e}", file=sys.stderr)
            error_extraction_count += 1 # Count as an extraction error

    # --- 5. Calculate accuracy and print results ---
    print("\n--- Evaluation Complete ---")
    print(f"Total entries evaluated (range {n}-{m}): {total_subset_count}")
    print(f"Entries successfully extracted and compared: {processed_count}")
    print(f"Entries skipped due to format errors or inability to extract options: {error_extraction_count}")

    if processed_count > 0:
        accuracy = correct_count / processed_count
        print(f"\nNumber of correct answers: {correct_count}")
        print(f"Number of incorrect answers: {processed_count - correct_count}")
        print(f"Accuracy (based on successfully processed entries): {accuracy:.2%}")

        if verbose and mismatched_items:
            print("\n--- Details of Mismatched Predictions ---")
            for item in mismatched_items:
                print(f"  Entry {item['index']}: Ground Truth='{item['ground_truth']}', "
                      f"Predicted='{item['predicted_extracted']}' (from raw prediction: '{item['predicted_raw']}')")
        elif not mismatched_items and processed_count > 0: # Check processed_count > 0 here
             print("\nAll successfully processed entries were answered correctly!")
        # If mismatched_items is empty but processed_count is 0, the final "else" handles it.

    else:
        print("\nCould not successfully process any entries. Accuracy cannot be calculated.")

# --- Example Usage ---
# NOTE: Please update this file path to your local environment
json_file = 'path/to/your/predictions_output.json'
start_item = 1
end_item = 500 # Adjust as needed, or use a large number to process all if unsure of total

# Call the function; set verbose=True to see details of incorrect entries
if __name__ == "__main__": 

    evaluate_predictions(json_file, start_item, end_item, verbose=True)

