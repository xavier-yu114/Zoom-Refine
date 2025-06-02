import json

def calculate_bbox_stats(json_file_path):
    """
    Calculates the distribution and average area of Bounding Boxes in a JSON dataset.

    Args:
        json_file_path (str): The path to the JSON file.
    """
    counts = {
        "less_than_10_percent": 0,
        "greater_than_10_percent": 0, 
        "invalid_or_missing_bbox": 0
    }
    total_valid_bboxes = 0
    total_area_sum = 0.0  # To accumulate the area of all valid bboxes

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON file '{json_file_path}'. Please check the file format.")
        return
    except Exception as e:
        print(f"An unknown error occurred while reading or parsing the file: {e}")
        return

    if not isinstance(data, list):
        print("Error: The top-level structure of the JSON file should be a list.")
        return

    for item_index, item in enumerate(data):
        if not isinstance(item, dict):
            # print(f"Warning: Item at index {item_index + 1} is not a dictionary, skipped.")
            counts["invalid_or_missing_bbox"] += 1
            continue

        bbox = item.get("Bounding_Box")

        if bbox and isinstance(bbox, list) and len(bbox) == 4:
            try:
                # Ensure all coordinates are floats
                x_min, y_min, x_max, y_max = map(float, bbox)

                # Check if coordinates are valid (e.g., x_max > x_min and within [0,1])
                if not (0 <= x_min < x_max <= 1 and 0 <= y_min < y_max <= 1):
                    # print(f"Warning: Bounding Box coordinates at item {item_index + 1} are invalid: {bbox}. Skipped.")
                    counts["invalid_or_missing_bbox"] += 1
                    continue

                width = x_max - x_min
                height = y_max - y_min
                area = width * height

                # Further check if the calculated area is within a reasonable range [0, 1]
                if not (0 <= area <= 1):
                    # print(f"Warning: Calculated area for item {item_index + 1} is abnormal ({area:.4f}), BBox: {bbox}. Skipped.")
                    counts["invalid_or_missing_bbox"] += 1
                    continue

                total_valid_bboxes += 1
                total_area_sum += area # Accumulate area

                if area < 0.10:
                    counts["less_than_10_percent"] += 1
                else:
                    counts["greater_than_10_percent"] += 1 # Corrected key

            except (ValueError, TypeError) as e:
                # print(f"Warning: Bounding Box at item {item_index + 1} contains non-numeric values or is malformed: {bbox} - {e}. Skipped.")
                counts["invalid_or_missing_bbox"] += 1
        else:
            # print(f"Warning: Item at index {item_index + 1} is missing 'Bounding_Box' key or it's incorrectly formatted. Skipped.")
            counts["invalid_or_missing_bbox"] += 1

    if total_valid_bboxes == 0:
        print("No valid Bounding Box data found for processing.")
        if counts["invalid_or_missing_bbox"] > 0:
             print(f"Found {counts['invalid_or_missing_bbox']} items missing Bounding_Box or with incorrect format.")
        return

    print(f"Total valid Bounding Boxes processed: {total_valid_bboxes}.")
    if counts["invalid_or_missing_bbox"] > 0:
        print(f"Additionally, {counts['invalid_or_missing_bbox']} items were missing Bounding_Box, had incorrect format, or invalid coordinates.")

    # Calculate average area
    average_area = (total_area_sum / total_valid_bboxes) if total_valid_bboxes > 0 else 0
    average_area_percentage = average_area * 100 # Convert to percentage

    print("\nBounding Box Area Statistics:")
    print(f"  Average Area: {average_area:.4f} (i.e., {average_area_percentage:.2f}% of the total image area)")

    print("\nBounding Box Area Distribution:")
    prop_less_than_10 = (counts["less_than_10_percent"] / total_valid_bboxes) * 100
    prop_greater_than_10 = (counts["greater_than_10_percent"] / total_valid_bboxes) * 100

    print(f"  Proportion less than 10% of total area: {prop_less_than_10:.2f}% ({counts['less_than_10_percent']} items)")
    print(f"  Proportion 10% or greater of total area: {prop_greater_than_10:.2f}% ({counts['greater_than_10_percent']} items)")

# Main execution part
if __name__ == "__main__":
    # NOTE: Please replace this path with the actual path to your JSON file
    json_file_path = "path/to/your/dataset_with_bboxes.json"


    calculate_bbox_stats(json_file_path)