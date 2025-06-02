import pandas as pd
from PIL import Image
import io
import os
import base64

# NOTE: Please update these file paths to your local environment
parquet_file_path = 'path/to/your/hr_bench_data.parquet'
save_dir = 'path/to/your/output_images_directory' # Directory to save extracted images

# Read the .parquet file
try:
    df = pd.read_parquet(parquet_file_path)
except FileNotFoundError:
    print(f"Error: Parquet file not found at '{parquet_file_path}'. Please check the path.")
    exit()
except Exception as e:
    print(f"Error reading Parquet file '{parquet_file_path}': {e}")
    exit()

# Specify the column name containing the Base64 encoded image data
actual_image_column = 'image' # Ensure this matches the column name in your Parquet file

# Check if the specified image column exists in the DataFrame
if actual_image_column not in df.columns:
    raise ValueError(
        f"Error: Column '{actual_image_column}' not found in Parquet file '{parquet_file_path}'. "
        f"Available columns are: {list(df.columns)}"
    )

# Create the directory to save images if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

print(f"\nStarting to save images to directory: {save_dir}")

processed_count = 0
error_count = 0

# Iterate over each row in the DataFrame to extract and save images
for index, row in df.iterrows():
    try:
        # Get the Base64 encoded image data from the specified column
        base64_image_data = row[actual_image_column]

        # Check if the image data is empty or not a string
        if not base64_image_data or not isinstance(base64_image_data, str):
            print(f"Warning: Image data at index {index} is empty or has an incorrect type ({type(base64_image_data)}). Skipping this row.")
            error_count += 1
            continue

        # Decode the Base64 string to bytes
        try:
            image_bytes = base64.b64decode(base64_image_data)
        except base64.binascii.Error as decode_error:
             print(f"Error: Base64 string decoding failed at index {index}: {decode_error}. Skipping this row.")
             error_count += 1
             continue

        # Check if the decoded bytes are empty
        if not image_bytes:
            print(f"Warning: Decoded image data at index {index} is empty. Skipping this row.")
            error_count += 1
            continue

        # Open the image from the decoded bytes
        img = Image.open(io.BytesIO(image_bytes))

        # Determine the image format for saving
        img_format = img.format if img.format else 'JPEG' # Default to JPEG if format is not detected
        file_extension = img_format.lower()

        if file_extension == 'jpeg': # Standardize JPEG extension to 'jpg'
            file_extension = 'jpg'

        # If the original format is uncommon or cannot be saved directly, convert to JPEG
        supported_extensions = ['jpg', 'png', 'gif', 'bmp', 'tiff']
        if file_extension not in supported_extensions:
             print(f"Warning: Image at index {index} has format '{img_format}'. Forcing save as JPEG.")
             img_format = 'JPEG' # PIL uses 'JPEG' for saving .jpg files
             file_extension = 'jpg'

        # Construct the full path to save the image
        save_path = os.path.join(save_dir, f'image_{index}.{file_extension}')

        # Save the image
        # If the image has an alpha channel (e.g., RGBA PNG) and is being saved as JPEG,
        # it should be converted to RGB first to avoid errors.
        if img_format == 'JPEG' and img.mode == 'RGBA':
            img = img.convert('RGB')
        img.save(save_path, img_format)

        processed_count += 1
        # print(f"Saved image: {save_path}") # Uncomment to see progress for each saved file

    except KeyError:
        # This should not happen if the column check at the beginning is correct
        print(f"Error: KeyError occurred while processing index {index}. Verify column name '{actual_image_column}'.")
        error_count += 1
        break # A KeyError often indicates a fundamental issue, so stop processing
    except Exception as e:
        # Catch other potential errors, e.g., PIL cannot open corrupted data
        print(f"Error: An error occurred while processing the image at index {index}: {e}")
        error_count += 1
        # Consider 'continue' to skip this image or 'break' to stop entirely

print(f"\nProcessing complete. Successfully saved {processed_count} images. Encountered {error_count} errors/warnings.")
