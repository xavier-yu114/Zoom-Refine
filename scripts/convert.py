import json
import os

def transform_json_data(input_json_string):
    """
    Transforms a JSON string from the old format to the new format.

    Args:
        input_json_string (str): A string containing the JSON data in the old format.

    Returns:
        str: A string containing the JSON data in the new format, pretty-printed.
             Returns an empty list as a JSON string if input is invalid or empty.
    """
    try:
        original_data = json.loads(input_json_string)
        if not isinstance(original_data, list):
            print("Error: Input JSON is not a list.")
            return json.dumps([], indent=2)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return json.dumps([], indent=2)
    except Exception as e: # Catch other potential errors during loading
        print(f"An unexpected error occurred while loading JSON: {e}")
        return json.dumps([], indent=2)


    transformed_list = []

    for item in original_data:
        question_text = item.get("Text", "")
        answer_choices_list = item.get("Answer choices", [])
        ground_truth = item.get("Ground truth", "")
        image_path = item.get("Image", "")

        # Format answer choices
        formatted_choices = "\n".join(answer_choices_list)

        # Construct user content
        user_content = (
            f"{question_text}"
            f"The choices are listed below:\n"
            f"{formatted_choices}\n"
            f"Select the best answer to the above multiple-choice question based on the image. "
            f"Respond with only the letter (A, B, C, D, or E) of the correct option\n" # Assuming max 5 choices as per prompt
            f"The best answer is:."
        )

        # Create the new structure
        new_item = {
            "messages": [
                {
                    "role": "user",
                    "content": user_content
                },
                {
                    "role": "assistant",
                    "content": ground_truth
                }
            ],
            "images": [image_path]
        }
        transformed_list.append(new_item)

    return json.dumps(transformed_list, indent=2)

def main():
    input_file_path = "path/to/your/target.json"
    output_file_path = "path/to/your/result.json"


    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}")
            return 

    try:
        with open(input_file_path, 'r', encoding='utf-8') as f_in:
            input_json_from_file = f_in.read()
        print(f"Successfully read data from {input_file_path}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file_path}")
        return
    except Exception as e:
        print(f"An error occurred while reading the input file: {e}")
        return

    output_json_string = transform_json_data(input_json_from_file)

    
    if output_json_string:
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f_out:
                f_out.write(output_json_string)
            print(f"Transformation complete. Output written to {output_file_path}")
        except IOError as e:
            print(f"Error writing output file to {output_file_path}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while writing the output file: {e}")
    else:
        print("Transformation resulted in empty data, not writing to output file.")


if __name__ == "__main__":
    main()