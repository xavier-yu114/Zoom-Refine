from openai import OpenAI
import os
import base64
from PIL import Image
import io
import json
from tqdm import tqdm
import time

# --- Rate Limiting Settings ---
RATE_LIMIT_PER_MINUTE = 60 # Define your API's rate limit
SECONDS_PER_MINUTE = 60
# Calculate delay needed between requests
REQUIRED_DELAY = SECONDS_PER_MINUTE / RATE_LIMIT_PER_MINUTE
# Add a small buffer to be safe (e.g., 0.1 seconds)
BUFFER = 0.1
SLEEP_DURATION = REQUIRED_DELAY + BUFFER
print(f"API Rate Limit: {RATE_LIMIT_PER_MINUTE}/min. Applying delay of {SLEEP_DURATION:.2f} seconds between requests.")

# --- Image Constraints ---
MAX_SIZE_BYTES = 7 * 1024 * 1024
MAX_WIDTH = 4096
MAX_HEIGHT = 4096

def encode_compressed_bytes(compressed_picture_bytes):
    """Encodes image bytes (already in memory) to base64."""
    base64_encoded_image = base64.b64encode(compressed_picture_bytes).decode('utf-8')
    return base64_encoded_image

def encode_image_from_path(image_path):
    """Reads an image file directly and returns its base64 encoded string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image directly from path {image_path}: {e}")
        return None # Indicates encoding failure

def compress_image_to_memory(image_path, max_size_bytes=MAX_SIZE_BYTES, initial_quality=85, min_quality=10, max_width=MAX_WIDTH, max_height=MAX_HEIGHT):
    """
    Compress an image to less than max_size_bytes and keep it in memory.
    Also resizes if dimensions exceed max_width or max_height.

    :param image_path: Path to the original image
    :param max_size_bytes: Target maximum size in bytes
    :param initial_quality: Starting compression quality (1-100)
    :param min_quality: Minimum allowable compression quality (1-100)
    :param max_width: Maximum width of the image
    :param max_height: Maximum height of the image
    :return: Compressed image byte data or None if error
    """
    try:
        with Image.open(image_path) as img:
            original_width, original_height = img.size
            needs_resize = original_width > max_width or original_height > max_height
            if needs_resize:
                 print(f"Resizing image {os.path.basename(image_path)} from {original_width}x{original_height}...")
                 # Use Image.Resampling.LANCZOS for newer Pillow versions (Pillow 9.0.0+)
                 if hasattr(Image, "Resampling"):
                     img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                 else: # Fallback for older Pillow versions
                     img.thumbnail((max_width, max_height), Image.LANCZOS)

            # Ensure image is in RGB format
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            # Check initial size with initial_quality without full compression loop yet
            temp_bytes_io = io.BytesIO()
            img.save(temp_bytes_io, format='JPEG', quality=initial_quality)
            # current_size = len(temp_bytes_io.getvalue()) # Not directly used, can be removed if not needed for logic
            temp_bytes_io.close()

            quality = initial_quality
            compressed_image_bytes = io.BytesIO()
            img.save(compressed_image_bytes, format='JPEG', quality=quality, optimize=True)
            compressed_size = len(compressed_image_bytes.getvalue())

            if compressed_size > max_size_bytes:
                 print(f"Initial size for {os.path.basename(image_path)} at quality {quality}: {compressed_size / (1024 * 1024):.2f} MB. Compressing further...")

            while compressed_size > max_size_bytes and quality > min_quality:
                quality -= 5
                compressed_image_bytes = io.BytesIO() # Create a new BytesIO object for each attempt
                img.save(compressed_image_bytes, format='JPEG', quality=quality, optimize=True)
                compressed_size = len(compressed_image_bytes.getvalue())
                print(f"Reducing quality to {quality} for {os.path.basename(image_path)}. New size: {compressed_size / (1024 * 1024):.2f} MB")

            if compressed_size > max_size_bytes:
                print(f"Warning: Unable to compress image {os.path.basename(image_path)} to under {max_size_bytes / (1024 * 1024):.2f}MB even at quality {min_quality}.")
                # Note: The original code was changed to return oversized bytes instead of None here.
                # This behavior is preserved. The calling code should handle this.

            final_bytes = compressed_image_bytes.getvalue()
            compressed_image_bytes.close() # Close the final buffer
            print(f"Final compressed size for {os.path.basename(image_path)}: {len(final_bytes) / (1024 * 1024):.2f} MB")
            return final_bytes

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Image.UnidentifiedImageError:
         print(f"Error: Cannot identify image file (may be corrupt or unsupported format): {image_path}")
         return None
    except Exception as e:
        print(f"Error compressing image {image_path}: {e}")
        return None


# --- Main Script ---

# IMPORTANT: Set your API key as an environment variable 'OPENAI_API_KEY'
# or replace "YOUR_API_KEY_HERE" directly. Using environment variables is recommended.
API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE_OR_SET_ENV_VAR")
if API_KEY == "YOUR_API_KEY_HERE_OR_SET_ENV_VAR":
    print("Warning: OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable or update the script.")

client = OpenAI(
    api_key=API_KEY,
    # Note: This base_url is specific to intern-ai.org.cn.
    # For standard OpenAI API, you might not need to set base_url or use a different one.
    base_url="https://chat.intern-ai.org.cn/api/v1/",
)

# NOTE: Please update these paths to your local environment
input_json_path = 'path/to/your/dataset.json'
output_json_path = 'path/to/your/result.json'
image_base_dir = 'path/to/your/datasets/MME-HD-CN'

# Load the test set JSON file
try:
    with open(input_json_path, 'r', encoding='utf-8') as f:
        test_set = json.load(f)
    print(f"Loaded {len(test_set)} entries from {input_json_path}")
except FileNotFoundError:
    print(f"Error: Input JSON file not found at {input_json_path}")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {input_json_path}")
    exit()


# Process each entry with a progress bar
for entry in tqdm(test_set, desc="Processing entries", unit="entry"):
    if 'images' not in entry or not entry['images']:
        print(f"Skipping entry ID {entry.get('id', 'N/A')} with no images.") # Assuming entries might have an 'id'
        entry['predicted_answer'] = "Error: No image specified"
        continue

    image_filename = entry['images'][0]
    image_path = os.path.join(image_base_dir, image_filename)

    base64_image = None
    processing_error_message = None
    api_request_made = False # Flag to track if an API call is attempted

    # --- Image Processing Logic ---
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        processing_error_message = "Error: Image file not found"
    else:
        try:
            file_size = os.path.getsize(image_path)
            width, height = 0, 0
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except Image.UnidentifiedImageError:
                 print(f"Error: Cannot identify image file (may be corrupt or unsupported): {image_path}")
                 processing_error_message = "Error: Corrupt or unsupported image format"
            except Exception as img_err:
                 print(f"Error opening image {image_path} to get dimensions: {img_err}")
                 processing_error_message = "Error: Could not read image dimensions"

            if processing_error_message is None:
                # Determine if compression is needed
                if file_size <= MAX_SIZE_BYTES and width <= MAX_WIDTH and height <= MAX_HEIGHT:
                    print(f"Image {image_filename} is within limits. Encoding directly.")
                    base64_image = encode_image_from_path(image_path)
                    if base64_image is None:
                         processing_error_message = "Error: Failed to encode image directly"
                else:
                    print(f"Image {image_filename} exceeds limits (Size: {file_size/(1024*1024):.2f}MB, Dim: {width}x{height}). Compressing...")
                    compressed_picture_bytes = compress_image_to_memory(image_path)
                    if compressed_picture_bytes:
                        base64_image = encode_compressed_bytes(compressed_picture_bytes)
                        # Approximate check if still too large after compression (Base64 encoding adds ~33% overhead)
                        if len(base64_image) * 3 / 4 > MAX_SIZE_BYTES:
                            print(f"Warning: Image {image_filename} still potentially too large after compression and Base64 encoding. API might reject.")
                            # processing_error_message = "Error: Image too large even after compression" # Optionally set error
                    else:
                        print(f"Compression failed for image: {image_filename}")
                        processing_error_message = "Error: Image compression failed"
        except Exception as e:
            print(f"An unexpected error occurred while checking image {image_filename}: {e}")
            processing_error_message = f"Error: Unexpected error processing image - {e}"

    if processing_error_message:
        entry['predicted_answer'] = processing_error_message
        continue

    if base64_image is None:
         entry['predicted_answer'] = "Error: Failed to get base64 image data after processing"
         continue

    # Get user question (assuming a fixed structure in the input JSON)
    if 'messages' not in entry or not entry['messages'] or 'content' not in entry['messages'][0]:
        print(f"Skipping entry ID {entry.get('id', 'N/A')} due to missing message structure.")
        entry['predicted_answer'] = "Error: Invalid message structure in input JSON"
        continue
    user_message = entry['messages'][0]['content']

    # Construct API request messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_message},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }
    ]

    # Send API request and get the answer
    answer = None
    try:
        # print(f"Sending API request for {image_filename}...") # Optional: uncomment for debugging
        chat_rsp = client.chat.completions.create(
            model="internvl3-latest", # Specify the model
            messages=messages,
            max_tokens=1024,
            temperature=0
        )
        answer = chat_rsp.choices[0].message.content
        api_request_made = True # Mark that an API call was attempted/successful
    except Exception as e:
        print(f"Error during API request for image {image_filename}: {e}")
        answer = f"Error: API request failed - {e}"
        api_request_made = True # Still count it towards rate limit as an attempt was made

    entry['predicted_answer'] = answer

    # Apply delay to respect API rate limits
    if api_request_made:
        # print(f"Sleeping for {SLEEP_DURATION:.2f} seconds to respect rate limit...") # Optional
        time.sleep(SLEEP_DURATION)


# Save results to a new JSON file
try:
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(test_set, f, indent=4, ensure_ascii=False) # ensure_ascii=False for non-ASCII characters in results
    print(f"\nProcessing complete. Results saved to {output_json_path}")
except Exception as e:
    print(f"\nError saving results to {output_json_path}: {e}")




