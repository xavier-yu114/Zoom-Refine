import os
from PIL import Image

def calculate_average_resolution(folder_path):
    """
    Calculates the average resolution of all image files in the specified folder.

    Args:
        folder_path (str): The path to the folder to scan.

    Returns:
        tuple: A tuple containing the average width and average height (avg_width, avg_height).
               Returns (0, 0) if no images are found or an error occurs.
        int: The number of valid image files found and processed.
    """
    total_width = 0
    total_height = 0
    image_count = 0
    # Supported common image file extensions (lowercase)
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')

    print(f"Scanning folder: {folder_path}")

    # Check if the folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist or is not a valid directory.")
        return (0, 0), 0

    try:
        # Iterate through all files in the folder
        for filename in os.listdir(folder_path):
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)

            # Check if it's a file and has a supported image format
            if os.path.isfile(file_path) and filename.lower().endswith(supported_extensions):
                try:
                    # Open the image file using Pillow
                    with Image.open(file_path) as img:
                        # Get the image's width and height
                        width, height = img.size
                        total_width += width
                        total_height += height
                        image_count += 1
                        # print(f"  Processing: {filename} - Resolution: {width}x{height}") # Uncomment for detailed progress
                except Exception as e:
                    print(f"  Warning: Could not process file '{filename}'. Error: {e}")
                    continue # Skip to the next file

    except OSError as e:
        print(f"Error: An OS error occurred while accessing folder '{folder_path}': {e}")
        return (0, 0), 0
    except Exception as e:
        print(f"Error: An unexpected error occurred while scanning folder '{folder_path}': {e}")
        return (0, 0), 0


    # Calculate the average resolution
    if image_count > 0:
        avg_width = round(total_width / image_count)
        avg_height = round(total_height / image_count)
        return (avg_width, avg_height), image_count
    else:
        return (0, 0), 0

# --- Main program execution ---
if __name__ == "__main__":
    # NOTE: Please update this path to the directory you want to analyze.
    target_folder_path = "path/to/your/image_folder"

    # ----------------------------------

    print(f"The folder to be analyzed is: {target_folder_path}") # Confirm the path

    # Call the function to calculate average resolution
    (average_width, average_height), count = calculate_average_resolution(target_folder_path)

    # Print the results
    if count > 0:
        print("-" * 30)
        print(f"Found a total of {count} valid image(s) in folder '{target_folder_path}'.")
        print(f"Average resolution: {average_width} x {average_height}")
        print("-" * 30)
    elif os.path.isdir(target_folder_path): # Check if the directory existed but had no images
        print("-" * 30)
        print(f"No supported image files were found in folder '{target_folder_path}'.")
        print("-" * 30)
    # If the folder itself was not found, the error message is handled within the function.
