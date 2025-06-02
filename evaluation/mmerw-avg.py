def calculate_accuracy_averages(data):
    """
    Calculates the weighted average and simple average of accuracies.

    Args:
    data (list of dict): A list containing category data. Each dictionary should include:
        'category' (str): Category name (optional, for display).
        'count' (int): The number of items/questions in this category (sample size).
        'accuracy' (float): The accuracy for this category (as a percentage, e.g., 95.5).

    Returns:
    tuple: A tuple containing (weighted_average, simple_average) as percentages.
           May return (None, None) or (0.0, 0.0) if input data is empty or total count is zero.
    """
    if not data:
        print("Error: Input data list is empty.")
        return None, None

    total_weighted_sum = 0.0
    total_count = 0
    sum_of_accuracies_decimal = 0.0 # Store sum of accuracies as decimals for simple average
    num_categories = len(data)

    print("\n--- Input Data ---")
    for i, item in enumerate(data):
        category_name = item.get('category', f'Category {i + 1}') # Default name if not provided
        count = item.get('count', 0)
        accuracy_percent = item.get('accuracy', 0.0) # Accuracy as percentage

        if not isinstance(count, (int, float)) or count < 0:
            print(f"Warning: Category '{category_name}' has an invalid count ({count}). It will be treated as 0.")
            count = 0
        if not isinstance(accuracy_percent, (int, float)) or not (0 <= accuracy_percent <= 100):
             print(f"Warning: Category '{category_name}' has an accuracy ({accuracy_percent}%) outside the [0, 100] range. Please check. Calculation will proceed.")
             # Clamp accuracy to [0, 100] to prevent extreme values from skewing results
             accuracy_percent = max(0.0, min(100.0, accuracy_percent))


        print(f"- {category_name}: Count = {count}, Accuracy = {accuracy_percent:.2f}%")

        # Convert accuracy to decimal form (0 to 1) for calculations
        accuracy_decimal = accuracy_percent / 100.0

        # Accumulate for weighted average
        total_weighted_sum += accuracy_decimal * count
        total_count += count

        # Accumulate for simple average
        sum_of_accuracies_decimal += accuracy_decimal

    print("------------------")

    # Calculate weighted average
    weighted_average_percent = 0.0
    if total_count > 0:
        weighted_average_decimal = total_weighted_sum / total_count
        weighted_average_percent = weighted_average_decimal * 100.0
    else:
        print("Warning: Total count is 0. Cannot calculate weighted average. Returning 0.0%.")

    # Calculate simple average
    simple_average_percent = 0.0
    if num_categories > 0:
        simple_average_decimal = sum_of_accuracies_decimal / num_categories
        simple_average_percent = simple_average_decimal * 100.0
    else:
        # This case should not be reached if 'data' is not empty, but as a safeguard:
        print("Warning: Number of categories is 0 (implies empty input data). Simple average is 0.0%.")

    return weighted_average_percent, simple_average_percent


if __name__ == "__main__":
    # Example data for different categories
    category_data = [
        # {'category': 'OCR', 'count': 5740, 'accuracy': 5.1},
        # {'category': 'RS', 'count': 3738,  'accuracy': 7},
        # {'category': 'DT', 'count': 5433, 'accuracy': 3.2},
        # {'category': 'MO', 'count': 2196,  'accuracy': 26.3},
        # {'category': 'AD', 'count': 3660,   'accuracy': 11.9},
        {'category': 'OCR', 'count': 500,   'accuracy': 2.8},
        {'category': 'DT', 'count': 500,   'accuracy': 4.8},
        {'category': 'MO', 'count': 498,   'accuracy': 1.81},
        {'category': 'AD', 'count': 1344,   'accuracy': 1.12},
    ]

    # --- Perform calculation ---
    weighted_avg, simple_avg = calculate_accuracy_averages(category_data)

    # --- Print results ---
    if weighted_avg is not None and simple_avg is not None: # Check ensures the function didn't return (None, None)
        print("\n--- Calculated Results ---")
        print(f"Weighted Average Accuracy: {weighted_avg:.2f}%")
        print(f"Simple Average Accuracy: {simple_avg:.2f}%")
        print("------------------")

       