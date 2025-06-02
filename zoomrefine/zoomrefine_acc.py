import json

def calculate_accuracies(json_file_path):
    """
    Reads question-answering evaluation results from a JSON file and calculates two types of accuracy.

    Args:
        json_file_path (str): The path to the JSON file.

    Returns:
        dict: A dictionary containing accuracy results, or None if an error occurs.
              Keys: 'rechecked_accuracy', 'combined_accuracy', 'counts'
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: File '{json_file_path}' is not a valid JSON format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return None

    if not isinstance(data, list):
        print("Error: The top-level structure of the JSON file should be a list.")
        return None

    total_questions = 0
    correct_rechecked = 0
    correct_combined = 0
    total_rechecked_available = 0 # Counts how many records have a 'Rechecked Answer'

    for item in data:
        # Ensure the record is a dictionary and contains necessary information
        if not isinstance(item, dict):
            print(f"Warning: Skipping non-dictionary record: {item}")
            continue

        messages = item.get("messages", [])
        # 'Answer' here seems to be the original predicted answer before recheck,
        # or a primary answer field if recheck is not always present.
        # The script uses it as a fallback if 'Rechecked Answer' is missing for combined_accuracy.
        final_answer_field = item.get("Answer")
        rechecked_answer_field = item.get("Rechecked Answer") # .get() returns None if key is missing or value is null

        # Extract the ground truth answer (assumed to be from the assistant's content in messages)
        ground_truth_answer = None
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "assistant":
                ground_truth_answer = message.get("content")
                break

        # If missing ground truth or the primary 'Answer' field, skip this record
        if ground_truth_answer is None:
            print(f"Warning: Record missing 'assistant' (ground truth) answer. Skipping. Record content (images/messages): {item.get('images') or item.get('messages')}")
            continue

        if final_answer_field is None:
             print(f"Warning: Record missing 'Answer' field (primary/fallback prediction). Skipping. Record content (images/messages): {item.get('images') or item.get('messages')}")
             continue

        # --- Start calculations ---
        total_questions += 1

        # 1. Calculate Rechecked Answer accuracy
        #    Only include in denominator and numerator if 'Rechecked Answer' is present
        if rechecked_answer_field is not None:
            total_rechecked_available += 1
            if ground_truth_answer == rechecked_answer_field:
                correct_rechecked += 1

        # 2. Calculate Combined accuracy
        #    Prioritize 'Rechecked Answer'; if not present, use 'Answer' (final_answer_field)
        prediction_for_combined = rechecked_answer_field if rechecked_answer_field is not None else final_answer_field
        if ground_truth_answer == prediction_for_combined:
            correct_combined += 1

    # --- Calculate final percentages ---
    if total_questions == 0:
        print("No valid records available for evaluation.")
        return {
            'rechecked_accuracy': 0.0,
            'combined_accuracy': 0.0,
            'counts': {
                'total': 0,
                'total_rechecked_available': 0,
                'correct_rechecked': 0,
                'correct_combined': 0
            }
        }

    # Avoid division by zero errors
    accuracy_rechecked = (correct_rechecked / total_rechecked_available) * 100 if total_rechecked_available > 0 else 0.0
    accuracy_combined = (correct_combined / total_questions) * 100 if total_questions > 0 else 0.0

    results = {
        'rechecked_accuracy': accuracy_rechecked,
        'combined_accuracy': accuracy_combined,
        'counts': {
            'total': total_questions,
            'total_rechecked_available': total_rechecked_available,
            'correct_rechecked': correct_rechecked,
            'correct_combined': correct_combined
        }
    }

    # --- Print results ---
    print(f"Successfully processed file: '{json_file_path}'")
    print("-" * 30)
    print(f"Total questions evaluated: {results['counts']['total']}")
    print(f"Questions with 'Rechecked Answer' available: {results['counts']['total_rechecked_available']}")
    print("-" * 30)
    print(f"1. 'Rechecked Answer' Accuracy: {results['rechecked_accuracy']:.2f}% ({results['counts']['correct_rechecked']}/{results['counts']['total_rechecked_available']})")
    print(f"   (Note: This accuracy is based only on the {results['counts']['total_rechecked_available']} questions that had a 'Rechecked Answer')")
    print("-" * 30)
    print(f"2. Combined Accuracy: {results['combined_accuracy']:.2f}% ({results['counts']['correct_combined']}/{results['counts']['total']})")
    print(f"   (Uses 'Rechecked Answer' if available, otherwise uses 'Answer')")
    print("-" * 30)

    return results

# Example usage:
if __name__ == "__main__":
    # NOTE: Please replace this path with the actual path to your JSON file
    json_file = 'path/to/your/evaluation_results.json'
    accuracies = calculate_accuracies(json_file)








