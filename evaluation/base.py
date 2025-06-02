import os
import json
from PIL import Image
import io
from tqdm import tqdm
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import numpy as np
import base64 


MAX_SIZE_BYTES = 7 * 1024 * 1024
MAX_WIDTH = 4096
MAX_HEIGHT = 4096
print(f"Applying image constraints: Max Size={MAX_SIZE_BYTES / (1024*1024):.2f}MB, Max Dim={MAX_WIDTH}x{MAX_HEIGHT}")


def compress_image_to_memory(image_path, max_size_bytes=MAX_SIZE_BYTES, initial_quality=85, min_quality=10, max_width=MAX_WIDTH, max_height=MAX_HEIGHT):
    """
    Compress an image to less than max_size_bytes and keep it in memory.
    Also resizes if dimensions exceed max_width or max_height.
    Returns compressed bytes if successful within limits, else None.
    """
    try:
        with Image.open(image_path) as img:
            original_width, original_height = img.size
            needs_resize = original_width > max_width or original_height > max_height
            if needs_resize:
                 print(f"Resizing image {os.path.basename(image_path)} from {original_width}x{original_height} for check/compression...")
                 if hasattr(Image, "Resampling"):
                     img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                 else:
                     img.thumbnail((max_width, max_height), Image.LANCZOS)

            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            quality = initial_quality
            compressed_image_bytes = io.BytesIO()
            img.save(compressed_image_bytes, format='JPEG', quality=quality, optimize=True)
            compressed_size = len(compressed_image_bytes.getvalue())

            if compressed_size > max_size_bytes:
                 print(f"Initial check size at quality {quality}: {compressed_size / (1024 * 1024):.2f} MB. Compressing further for check...")

            while compressed_size > max_size_bytes and quality > min_quality:
                quality -= 5
                compressed_image_bytes = io.BytesIO()
                img.save(compressed_image_bytes, format='JPEG', quality=quality, optimize=True)
                compressed_size = len(compressed_image_bytes.getvalue())
                # print(f"Checking quality {quality}. New size: {compressed_size / (1024 * 1024):.2f} MB")

            if compressed_size > max_size_bytes:
                print(f"Warning: Image {os.path.basename(image_path)} would likely exceed size limit ({max_size_bytes / (1024 * 1024):.2f}MB) even at quality {min_quality}.")
                final_bytes = None
            else:
                final_bytes = compressed_image_bytes.getvalue()
                print(f"Image {os.path.basename(image_path)} can be compressed to {len(final_bytes) / (1024 * 1024):.2f} MB.")

            compressed_image_bytes.close()
            return final_bytes

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Image.UnidentifiedImageError:
         print(f"Error: Cannot identify image file (may be corrupt or unsupported format): {image_path}")
         return None
    except Exception as e:
        print(f"Error checking/compressing image {image_path}: {e}")
        return None



IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
 
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):

    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

   
    resized_img = image.resize((target_width, target_height))

    processed_images = []
    for i in range(blocks):
        
        x0 = (i % target_aspect_ratio[0]) * image_size
        y0 = (i // target_aspect_ratio[0]) * image_size
        x1 = x0 + image_size
        y1 = y0 + image_size
        box = (x0, y0, x1, y1)

        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks, f"Expected {blocks} blocks, but got {len(processed_images)}"

    if use_thumbnail and blocks > 1: 
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


# --- MODIFIED load_image function ---
def load_image(image_source, input_size=448, max_num=12):
    """
    Loads and preprocesses an image for the model.
    :param image_source: Can be a file path (str) or a file-like object (e.g., io.BytesIO).
    :param input_size: Target size for the image patches.
    :param max_num: Maximum number of patches for dynamic preprocessing.
    :return: Processed image tensor (pixel_values) or None if an error occurs.
    """
    try:
        # Image.open can handle both file paths and file-like objects
        image = Image.open(image_source).convert('RGB')
        transform = build_transform(input_size=input_size)
        # Note: dynamic_preprocess expects a PIL Image object
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(img) for img in images] # Apply transform to each PIL image
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    except FileNotFoundError:
        # This error is less likely if we pass BytesIO, but good to keep for path case
        print(f"Error: Image source not found: {image_source}")
        return None
    except Image.UnidentifiedImageError:
         print(f"Error: Cannot identify image data from source.")
         return None
    except Exception as e:
        # Use repr(image_source) as BytesIO doesn't have a simple string representation
        print(f"Error processing image inside load_image for source {repr(image_source)}: {e}")
        return None

# --- Main Script ---

# Input and Output JSON file paths
input_json_path = '/path/to/your/dataset.json'
output_json_path = '/path/to/your/result.json'
image_base_dir = '/path/to/your/datasets/MME-HD-CN'
model_path = '/path/to/your/InternVL3-8B' # Local model path
max_new_tokens = 1024
max_num_patches = 12

# --- Load Model and Tokenizer ---
print("Loading model and tokenizer...")
try:
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        print("Error: CUDA is not available or no GPUs detected.")
        exit()
    print(f"CUDA available. Found {torch.cuda.device_count()} GPU(s).")

    # Make sure to specify the device if using CUDA_VISIBLE_DEVICES elsewhere
    # or let transformers handle it if device_map='auto' is feasible (might need accelerate)
    # For single GPU specified by CUDA_VISIBLE_DEVICES, .cuda() works.
    device = torch.device("cuda")
    print(f"Using device: {device}")

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True
    ).eval().to(device) # Move model to the target device

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or tokenizer from {model_path}: {e}")
    exit()


# --- Load Dataset ---
print(f"Loading dataset from {input_json_path}...")
try:
    with open(input_json_path, 'r', encoding='utf-8') as f:
        test_set = json.load(f)
    print(f"Loaded {len(test_set)} entries.")
except FileNotFoundError:
    print(f"Error: Input JSON file not found at {input_json_path}")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {input_json_path}")
    exit()

# --- Process Dataset ---
results = []

for entry in tqdm(test_set, desc="Processing entries", unit="entry"):
    processed_entry = entry.copy()
    predicted_answer = "Error: Processing skipped"

    if 'images' not in entry or not entry['images']:
        predicted_answer = "Error: No image specified"
        processed_entry['predicted_answer'] = predicted_answer
        results.append(processed_entry)
        continue

    image_filename = entry['images'][0]
    image_path = os.path.join(image_base_dir, image_filename)

    # --- Stage 1: Check Image Validity and Get Bytes ---
    image_ok_for_processing = False
    processing_error_message = None
    image_bytes_to_process = None # Store the bytes (original or compressed) here
    needs_compression_flag = False # Track if compression was applied

    if not os.path.exists(image_path):
        processing_error_message = "Error: Image file not found"
    else:
        try:
            file_size = os.path.getsize(image_path)
            width, height = 0, 0
            compressed_check_result = None # Store potential compressed bytes

            try:
                with Image.open(image_path) as img_check:
                    width, height = img_check.size
            except Image.UnidentifiedImageError:
                 processing_error_message = "Error: Corrupt or unsupported image format"
            except Exception as img_err:
                 processing_error_message = f"Error opening image {image_path} to get dimensions: {img_err}"

            if processing_error_message is None:
                # Check if image meets criteria directly
                if file_size <= MAX_SIZE_BYTES and width <= MAX_WIDTH and height <= MAX_HEIGHT:
                    print(f"Image {image_filename} is within limits. Reading original bytes.")
                    try:
                        with open(image_path, "rb") as f:
                            image_bytes_to_process = f.read()
                        image_ok_for_processing = True
                        needs_compression_flag = False
                    except Exception as read_err:
                        processing_error_message = f"Error reading original image file: {read_err}"

                else:
                    # If not, check if it *could* be compressed
                    print(f"Image {image_filename} exceeds limits (Size: {file_size/(1024*1024):.2f}MB, Dim: {width}x{height}). Checking if compressible...")
                    compressed_check_result = compress_image_to_memory(image_path)
                    if compressed_check_result is not None:
                         print(f"Image {image_filename} is compressible. Using compressed bytes.")
                         image_bytes_to_process = compressed_check_result
                         image_ok_for_processing = True
                         needs_compression_flag = True
                    else:
                         processing_error_message = "Error: Image too large or failed compression check"

        except Exception as e:
            processing_error_message = f"Error: Unexpected error checking image {image_filename} - {e}"

    # If image check failed OR failed to get bytes, record error and skip
    if not image_ok_for_processing or image_bytes_to_process is None:
        predicted_answer = processing_error_message if processing_error_message else "Error: Image failed pre-check or byte reading failed"
        processed_entry['predicted_answer'] = predicted_answer
        results.append(processed_entry)
        continue

    # --- Stage 2: Encode to Base64, Decode, Load for Local Model & Prepare Question ---
    try:
        # 1. Encode to Base64 (Mimic API script step)
        base64_encoded_image = base64.b64encode(image_bytes_to_process).decode('utf-8')
        # print(f"Image {image_filename} {'(compressed)' if needs_compression_flag else '(original)'} encoded to Base64 (length: {len(base64_encoded_image)}).")

        # 2. Decode Base64 back to bytes (Simulate receiving end)
        decoded_image_bytes = base64.b64decode(base64_encoded_image)

        # 3. Create a file-like object from decoded bytes
        image_data_stream = io.BytesIO(decoded_image_bytes)

        # 4. Load image using the model's function, passing the BytesIO stream
        # print(f"Loading image {image_filename} from decoded Base64 stream for model...")
        pixel_values = load_image(image_data_stream, max_num=max_num_patches) # Pass the stream!

        if pixel_values is None:
            predicted_answer = f"Error: Failed to load/process image {image_filename} from Base64 stream"
            processing_error_message = predicted_answer # Set error message for saving
        else:
            # Move pixel values to GPU
            pixel_values = pixel_values.to(torch.bfloat16).to(device) # Ensure correct dtype and device

    except Exception as load_err:
        print(f"Error during Base64 encode/decode or image loading for {image_filename}: {load_err}")
        predicted_answer = f"Error: Base64/Load failed - {load_err}"
        processing_error_message = predicted_answer # Set error message for saving

    # If loading/processing failed at this stage
    if processing_error_message:
         processed_entry['predicted_answer'] = processing_error_message
         results.append(processed_entry)
         # Clean up GPU memory if tensor was partially created? (Optional)
         if 'pixel_values' in locals() and isinstance(pixel_values, torch.Tensor):
             del pixel_values
         torch.cuda.empty_cache()
         continue


    # --- Prepare Question ---
    if 'messages' not in entry or not entry['messages'] or 'content' not in entry['messages'][0]:
        predicted_answer = "Error: Invalid message structure"
        processing_error_message = predicted_answer
    else:
        user_content = entry['messages'][0]['content']
        if '<image>' not in user_content:
             question = f"<image>\n{user_content}"
        else:
             question = user_content
    # print(question)
    # If preparing question failed
    if processing_error_message:
        processed_entry['predicted_answer'] = processing_error_message
        results.append(processed_entry)
        # Clean up pixel_values if they exist
        if 'pixel_values' in locals() and isinstance(pixel_values, torch.Tensor):
            del pixel_values
        torch.cuda.empty_cache()
        continue


    # --- Stage 3: Perform Inference ---
    # print(f"Running inference for {image_filename}...")
    try:
        with torch.no_grad():
            response = model.chat(tokenizer, pixel_values, question, generation_config)
        predicted_answer = response
        print(f"Inference successful for {image_filename}.")
    except torch.cuda.OutOfMemoryError:
        print(f"Error: CUDA Out of Memory during inference for image {image_filename}.")
        predicted_answer = "Error: CUDA Out of Memory"
        torch.cuda.empty_cache() # Attempt to clear cache
    except Exception as e:
        print(f"Error during model inference for image {image_filename}: {e}")
        predicted_answer = f"Error: Model inference failed - {e}"

    # Store the prediction
    processed_entry['predicted_answer'] = predicted_answer
    results.append(processed_entry)


    del pixel_values



# --- Save Results ---
print(f"\nProcessing complete. Saving results to {output_json_path}")

try:
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print("Results saved successfully.")
except Exception as e:
    print(f"\nError saving results to {output_json_path}: {e}")