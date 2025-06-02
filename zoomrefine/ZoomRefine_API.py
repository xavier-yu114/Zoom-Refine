import os
import base64
import io
import re
import json
from PIL import Image, ImageFile
from openai import OpenAI
from tqdm import tqdm
import math
import copy
import tempfile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Configuration ---
# IMPORTANT: Replace with your actual API Key or use environment variables
API_KEY = "YOUR_OPENAI_API_KEY_HERE"
# Replace with the appropriate API base URL if not using the default OpenAI endpoint
BASE_URL = "https://chat.intern-ai.org.cn/api/v1/" # Or your specific endpoint
MODEL_NAME = "internvl3-latest" # Model to use

# Suggestion: Consider using argparse for these paths for better flexibility
DATASET_PATH = "./data/input_dataset.json" # Example: Path to your input data
OUTPUT_PATH = "./data/output_results.json" # Example: Path to save results
IMAGE_BASE_PATH = "./data/images" # Example: Base path for images referenced in the dataset
TEMPERATURE = 0 # LLM temperature

# Image processing constraints
MAX_WIDTH = 4096
MAX_HEIGHT = 4096
MAX_SIZE_BYTES = 7 * 1024 * 1024  # 7MB

def encode_image_bytes(image_bytes):
    if image_bytes is None:
        return None
    return base64.b64encode(image_bytes).decode('utf-8')

def compress_image_to_memory(image_path, max_size_bytes=MAX_SIZE_BYTES, initial_quality=85, min_quality=10, max_width=MAX_WIDTH, max_height=MAX_HEIGHT):
    """
    Compresses an image to fit within max_size_bytes and max_width/max_height.
    Returns compressed image bytes and original dimensions.
    """
    try:
        with Image.open(image_path) as img:
            original_width, original_height = img.size
            needs_resize = original_width > max_width or original_height > max_height
            current_img = img

            if needs_resize:
                print(f"Resizing image {os.path.basename(image_path)} from {original_width}x{original_height}...")
                resampling_method = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
                current_img = img.copy()
                current_img.thumbnail((max_width, max_height), resampling_method)
                print(f"Resized to: {current_img.size}")

            if current_img.mode in ('RGBA', 'P'):
                if current_img is img: # Avoid re-copying if already copied for resize
                    current_img = img.copy()
                current_img = current_img.convert('RGB')

            temp_bytes_io = io.BytesIO()
            # Use optimize=True for potentially smaller JPEGs
            current_img.save(temp_bytes_io, format='JPEG', quality=initial_quality, optimize=True)
            current_size = len(temp_bytes_io.getvalue())
            temp_bytes_io.close()

            if current_size > max_size_bytes:
                print(f"Image {os.path.basename(image_path)} initial size (quality {initial_quality}): {current_size / (1024 * 1024):.2f} MB. Starting compression...")
                quality = initial_quality
                compressed_image_bytes_io = io.BytesIO()
                current_img.save(compressed_image_bytes_io, format='JPEG', quality=quality, optimize=True)
                compressed_size = len(compressed_image_bytes_io.getvalue())

                while compressed_size > max_size_bytes and quality > min_quality:
                    quality -= 5
                    compressed_image_bytes_io.seek(0)
                    compressed_image_bytes_io.truncate()
                    current_img.save(compressed_image_bytes_io, format='JPEG', quality=quality, optimize=True)
                    compressed_size = len(compressed_image_bytes_io.getvalue())
                    # print(f"Reduced quality to {quality}. New size: {compressed_size / (1024 * 1024):.2f} MB")

                if compressed_size > max_size_bytes:
                    print(f"Warning: Could not compress image {os.path.basename(image_path)} below {max_size_bytes / (1024 * 1024):.2f}MB (min quality {min_quality}). Current size: {compressed_size / (1024*1024):.2f}MB")

                final_bytes = compressed_image_bytes_io.getvalue()
                compressed_image_bytes_io.close()
                # print(f"Image {os.path.basename(image_path)} final compressed size: {len(final_bytes) / (1024 * 1024):.2f} MB")
                return final_bytes, original_width, original_height
            else:
                # print(f"Image {os.path.basename(image_path)} at quality {initial_quality} is {current_size / (1024 * 1024):.2f} MB, within limits.")
                final_bytes_io = io.BytesIO()
                current_img.save(final_bytes_io, format='JPEG', quality=initial_quality, optimize=True)
                final_bytes = final_bytes_io.getvalue()
                final_bytes_io.close()
                return final_bytes, original_width, original_height

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None, None, None
    except Image.UnidentifiedImageError:
        print(f"Error: Cannot identify image file: {image_path}")
        return None, None, None
    except Exception as e:
        print(f"Error: Exception during image processing for {image_path}: {e}")
        return None, None, None

def parse_bbox(response_text):
    """ Parses Bounding Box coordinates from LLM response. """
    regex = r"\s*(?:\*{2})?(?:Bounding box|bbox)(?:\*{2})?\s*[:：]?\s*(?:\*{2})?\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]"
    matches = re.findall(regex, response_text, re.IGNORECASE)
    if not matches:
        return None
    last_match_coords_str = matches[-1]
    try:
        coords = [float(c) for c in last_match_coords_str]
        if len(coords) == 4 and coords[0] < coords[2] and coords[1] < coords[3]:
            coords_clamped = [
                max(0.0, min(1.0, coords[0])),
                max(0.0, min(1.0, coords[1])),
                max(0.0, min(1.0, coords[2])),
                max(0.0, min(1.0, coords[3])),
            ]
            # Ensure x1 < x2 and y1 < y2 after clamping, with a small epsilon if they become equal
            if coords_clamped[0] >= coords_clamped[2]: coords_clamped[2] = min(1.0, coords_clamped[0] + 1e-6)
            if coords_clamped[1] >= coords_clamped[3]: coords_clamped[3] = min(1.0, coords_clamped[1] + 1e-6)
            
            # Final check
            if coords_clamped[0] < coords_clamped[2] and coords_clamped[1] < coords_clamped[3]:
                return coords_clamped
            else:
                print(f"Warning: Corrected coordinates are still invalid after clamping: {coords_clamped}")
                return None
        else:
            print(f"Warning: Last parsed coordinates are invalid or out of order: {coords}")
            return None
    except ValueError:
        print(f"Error: Failed to convert last BBox coordinates to float: {last_match_coords_str}")
        return None

def expand_bbox(bbox_norm, padding=0.1):
    """ Expands a normalized bounding box by a padding percentage. """
    if not (isinstance(bbox_norm, list) and len(bbox_norm) == 4):
        print("Error: expand_bbox received invalid input bbox_norm")
        return None
    if not (0.0 <= padding <= 1.0):
        print(f"Warning: expand_bbox padding value ({padding}) is invalid or too large, using 0.1.")
        padding = 0.1

    x1, y1, x2, y2 = bbox_norm

    new_x1 = x1 - padding
    new_y1 = y1 - padding
    new_x2 = x2 + padding
    new_y2 = y2 + padding

    final_x1 = max(0.0, new_x1)
    final_y1 = max(0.0, new_y1)
    final_x2 = min(1.0, new_x2)
    final_y2 = min(1.0, new_y2)

    if final_x1 >= final_x2:
        # print(f"Warning: After expansion, x1 ({final_x1:.3f}) >= x2 ({final_x2:.3f}). Original box might be too small or padding too large. Adjusting.")
        mid_x = (x1 + x2) / 2
        half_width_orig = (x2 - x1) / 2
        final_x1 = max(0.0, mid_x - half_width_orig - padding / 2)
        final_x2 = min(1.0, mid_x + half_width_orig + padding / 2)
        if final_x1 >= final_x2:
            print("Error: Adjusted x coordinates still invalid after expansion attempt.")
            return bbox_norm # Fallback to original if adjustment fails

    if final_y1 >= final_y2:
        # print(f"Warning: After expansion, y1 ({final_y1:.3f}) >= y2 ({final_y2:.3f}). Original box might be too small or padding too large. Adjusting.")
        mid_y = (y1 + y2) / 2
        half_height_orig = (y2 - y1) / 2
        final_y1 = max(0.0, mid_y - half_height_orig - padding / 2)
        final_y2 = min(1.0, mid_y + half_height_orig + padding / 2)
        if final_y1 >= final_y2:
            print("Error: Adjusted y coordinates still invalid after expansion attempt.")
            return bbox_norm # Fallback to original

    expanded_coords = [final_x1, final_y1, final_x2, final_y2]
    # print(f"Original BBox: [{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}], Expanded BBox: [{expanded_coords[0]:.3f}, {expanded_coords[1]:.3f}, {expanded_coords[2]:.3f}, {expanded_coords[3]:.3f}]")
    return expanded_coords

def parse_final_answer(response_text):
    """ Parses the final answer text block from LLM response. """
    regex = r"\s*(?:\*{2})?Answer(?:\*{2})?\s*[:：]\s*(.*)"
    match = re.search(regex, response_text, re.IGNORECASE | re.DOTALL)
    if match:
        final_answer_text = match.group(1).strip()
        return final_answer_text
    return None

def extract_letter_choice(response_text):
    """ Extracts the letter choice (A-E) from the response text. """
    if not response_text: return None
    # Prefer answers explicitly prefixed
    prefix_regex = r"(?:Answer|Rechecked Answer)\s*[:：]\s*"
    match_prefix_paren = re.search(prefix_regex + r"\(?\s*([A-E])\s*\)?", response_text, re.IGNORECASE | re.DOTALL)
    if match_prefix_paren:
        return match_prefix_paren.group(1).upper()
    
    match_prefix_letter = re.search(prefix_regex + r"([A-E])\b", response_text, re.IGNORECASE | re.DOTALL)
    if match_prefix_letter:
        return match_prefix_letter.group(1).upper()
    
    # Fallback for less structured answers (use with caution, might need more context)
    # match_paren = re.search(r"\(\s*([A-E])\s*\)", response_text, re.IGNORECASE)
    # if match_paren:
    #     return match_paren.group(1).upper()
    # match_isolated = re.search(r"(?<![a-zA-Z])([A-E])(?![a-zA-Z])", response_text, re.IGNORECASE) # Isolated letter
    # if match_isolated:
    #    return match_isolated.group(1).upper()
    return None

def parse_rechecked_answer(response_text):
    """ Parses the rechecked answer letter from LLM response. """
    if not response_text: return None
    match = re.search(r"(?:\*{2})?Rechecked Answer(?:\*{2})?\s*[:：]\s*\(?([A-E])\)?", response_text, re.IGNORECASE | re.DOTALL)
    if match:
        letter = match.group(1).upper()
        return letter
    return None

def crop_image_from_bbox(original_image_path, bbox_norm, original_width, original_height):
    """ Crops an image using normalized bounding box coordinates and original dimensions. """
    try:
        with Image.open(original_image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')

            x1_norm, y1_norm, x2_norm, y2_norm = bbox_norm
            # Convert normalized to pixel coordinates carefully
            x1_pix = max(0, min(original_width - 1, math.floor(x1_norm * original_width)))
            y1_pix = max(0, min(original_height - 1, math.floor(y1_norm * original_height)))
            x2_pix = max(x1_pix + 1, min(original_width, math.ceil(x2_norm * original_width))) # Ensure x2 > x1
            y2_pix = max(y1_pix + 1, min(original_height, math.ceil(y2_norm * original_height))) # Ensure y2 > y1

            if x1_pix >= x2_pix or y1_pix >= y2_pix:
                print(f"Error: Calculated crop region invalid ({x1_pix}, {y1_pix}, {x2_pix}, {y2_pix}) from BBox: {bbox_norm} on image {original_width}x{original_height}")
                return None

            cropped_img = img.crop((x1_pix, y1_pix, x2_pix, y2_pix))
            cropped_bytes_io = io.BytesIO()
            cropped_img.save(cropped_bytes_io, format='JPEG', quality=95, optimize=True) # Save initially at high quality
            return cropped_bytes_io.getvalue()
    except FileNotFoundError:
        print(f"Error: Original image not found during cropping: {original_image_path}")
        return None
    except Exception as e:
        print(f"Error: Exception during image cropping for {original_image_path}: {e}")
        return None

def call_llm_api(client, model_name_to_call, messages, temperature_to_use=0):
    """ Calls the LLM API and returns the assistant's message object. """
    try:
        chat_rsp = client.chat.completions.create(
            model=model_name_to_call,
            messages=messages,
            temperature=temperature_to_use,
            max_tokens=4096 # Or other appropriate value
        )
        if chat_rsp.choices and chat_rsp.choices[0].message:
            return chat_rsp.choices[0].message
        else:
            print("Error: API response invalid or choices empty.")
            return None
    except Exception as e:
        print(f"Error: Exception during API call: {e}")
        return None


# --- Main Processing Logic ---

def process_single_entry(entry_index, entry_data, client, model_to_use, prompts_collection, img_base_path, temp_setting=0):
    """
    Processes a single data entry:
    1. Calls LLM with original image and question.
    2. Parses answer and bounding box.
    3. If BBox found, expands it, crops the image.
    4. Processes (compresses/resizes if needed) the cropped image.
    5. Calls LLM again with original context and the processed cropped image for re-evaluation.
    6. Parses rechecked answer.
    Returns a dictionary with results.
    """
    # print(f"\n--- Processing Entry {entry_index} ---")

    # 1. Initialize output structure
    output_result = {
        "messages": copy.deepcopy(entry_data.get("messages", [])), # Keep original messages for context
        "images": copy.deepcopy(entry_data.get("images", [])),   # Keep original image path for context
        "Answer": None,
        "Bounding_Box": None,
        "Rechecked Answer": None,
        "Error": None
    }

    # 2. Extract information and paths
    try:
        if not output_result["messages"] or not isinstance(output_result["messages"][0].get("content"), str):
            raise ValueError("Invalid 'messages' structure in input.")
        original_question = output_result["messages"][0]["content"]
        # Clean up question if it contains instructions not meant for the LLM logic directly
        phrase_to_remove = "Respond with only the letter (A, B, C, D, or E) of the correct option"
        question_for_logic = original_question.replace(phrase_to_remove, "").strip()

        if not output_result["images"] or not isinstance(output_result["images"][0], str):
            raise ValueError("Invalid 'images' structure in input.")
        relative_image_path = output_result["images"][0]
        original_image_path = os.path.join(img_base_path, relative_image_path)
    except (KeyError, IndexError, TypeError, ValueError) as e:
        print(f"Error: Failed to parse entry {entry_index} structure: {e}")
        output_result['Error'] = f"Invalid JSON structure or missing data: {e}"
        return output_result

    # 3. Image processing (original image)
    image_bytes_for_llm_step1 = None
    original_width = None
    original_height = None
    processing_error_message = None

    if not os.path.exists(original_image_path):
        processing_error_message = f"Original image file not found at {original_image_path}"
    else:
        try:
            needs_processing = False
            file_size = os.path.getsize(original_image_path)
            if file_size > MAX_SIZE_BYTES:
                needs_processing = True
            
            # Get dimensions and check if resize is needed even if size is okay
            try:
                with Image.open(original_image_path) as img_check:
                    width_check, height_check = img_check.size
                    original_width, original_height = width_check, height_check # Store for cropping
                    if width_check > MAX_WIDTH or height_check > MAX_HEIGHT:
                        needs_processing = True
            except Image.UnidentifiedImageError:
                processing_error_message = f"Error: Corrupt or unsupported image format: {original_image_path}"
            except Exception as img_err:
                processing_error_message = f"Error: Could not read image dimensions for {original_image_path} - {img_err}"

            if processing_error_message is None: # Only proceed if dimension reading was successful
                if needs_processing:
                    # print(f"Entry {entry_index}: Original image requires processing (size/dimensions)...")
                    processed_bytes, ow, oh = compress_image_to_memory(original_image_path, MAX_SIZE_BYTES, max_width=MAX_WIDTH, max_height=MAX_HEIGHT)
                    image_bytes_for_llm_step1 = processed_bytes
                    # Original dimensions are returned by compress_image_to_memory, use them.
                    original_width, original_height = ow, oh
                    if image_bytes_for_llm_step1 is None:
                        processing_error_message = "Error: Original image processing/compression failed"
                else:
                    # print(f"Entry {entry_index}: Original image does not require processing.")
                    try:
                        with open(original_image_path, 'rb') as f:
                            image_bytes_for_llm_step1 = f.read()
                        # Ensure original_width/height were set if not processed
                        if original_width is None or original_height is None: # Should have been set above
                             with Image.open(original_image_path) as img_final_check:
                                original_width, original_height = img_final_check.size
                    except Exception as read_err:
                        processing_error_message = f"Error: Failed to read original image file {original_image_path} - {read_err}"
                        image_bytes_for_llm_step1 = None
        except Exception as e: # Catch-all for unexpected issues during the check
            processing_error_message = f"Error: Unexpected error checking original image {original_image_path} - {e}"
            image_bytes_for_llm_step1 = None

    if processing_error_message or image_bytes_for_llm_step1 is None or original_width is None or original_height is None:
        error_msg = processing_error_message if processing_error_message else "Image processing failed (unknown reason) or dimensions not found"
        print(f"Original image handling failed for entry {entry_index}. Error: {error_msg}")
        output_result['Error'] = error_msg
        return output_result

    # 4. Base64 encode (original image)
    encoded_image_initial = encode_image_bytes(image_bytes_for_llm_step1)
    if encoded_image_initial is None:
        output_result['Error'] = "Base64 encoding failed for initial image"
        return output_result

    # 5. Build messages for first API call
    messages_for_api = [
        {"role": "system", "content": [{"type": "text", "text": prompts_collection['system_prompt_1']}]},
        {"role": "user", "content": [
            {"type": "text", "text": prompts_collection['user_template_1'].format(question=question_for_logic)},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image_initial}"}}
        ]}
    ]

    # print(f"\n--- Entry {entry_index}: Starting first API call ---")
    assistant_message_1_obj = call_llm_api(client, model_to_use, messages_for_api, temp_setting)

    if assistant_message_1_obj is None or not assistant_message_1_obj.content:
        output_result['Error'] = "First API call failed or returned empty content"
        return output_result

    response_1_content = assistant_message_1_obj.content
    # print(f"Entry {entry_index}: First API response received.") # {response_1_content[:200]}...")

    # 6. Parse first response
    final_answer_text = parse_final_answer(response_1_content) # Text of the answer, not just letter
    final_answer_letter = extract_letter_choice(response_1_content) # Prioritize parsing from full response
    bbox_norm_original = parse_bbox(response_1_content)

    # 7. Store parsed information
    if final_answer_letter: output_result['Answer'] = final_answer_letter
    if bbox_norm_original: output_result['Bounding_Box'] = bbox_norm_original

    # 8. Determine if second call is needed (only if BBox is found)
    if bbox_norm_original is None:
        if not final_answer_letter: # If neither answer nor BBox found
            output_result['Error'] = "Could not parse Answer letter or Bounding Box from first response."
        # print(f"Entry {entry_index}: BBox not found in first response. Skipping second call.")
        return output_result

    # --- If BBox exists, continue to second call ---
    # print(f"Entry {entry_index}: BBox found: {bbox_norm_original}. Proceeding to crop and second call.")

    # 9. Expand BBox
    padding_amount = 0.1 # Example padding
    bbox_to_crop = expand_bbox(bbox_norm_original, padding=padding_amount)
    if bbox_to_crop is None: # Expansion failed
        print(f"Warning (Entry {entry_index}): Failed to expand BBox. Using original BBox for cropping.")
        bbox_to_crop = bbox_norm_original

    # 10. Crop image (using expanded or original BBox)
    # print(f"Entry {entry_index}: Cropping image with BBox {bbox_to_crop} from original {original_width}x{original_height}...")
    cropped_image_bytes = crop_image_from_bbox(original_image_path, bbox_to_crop, original_width, original_height)

    if cropped_image_bytes is None:
        output_result['Error'] = "Cropping image failed (using BBox)"
        print(f"Error (Entry {entry_index}): Cropping failed. Cannot proceed to second API call.")
        return output_result

    # --- Process the cropped image (check size/dimensions, compress if needed) ---
    image_bytes_for_llm_step2 = None
    cropped_processing_error = None
    try:
        cropped_size = len(cropped_image_bytes)
        needs_processing_cropped = False

        if cropped_size > MAX_SIZE_BYTES:
            needs_processing_cropped = True
            # print(f"Entry {entry_index}: Cropped image size {cropped_size / (1024*1024):.2f}MB exceeds limit. Needs compression.")

        if not needs_processing_cropped: # Check dimensions only if size is okay
            try:
                with Image.open(io.BytesIO(cropped_image_bytes)) as img_cropped_check:
                    cropped_width, cropped_height = img_cropped_check.size
                    if cropped_width > MAX_WIDTH or cropped_height > MAX_HEIGHT:
                        needs_processing_cropped = True
                        # print(f"Entry {entry_index}: Cropped image dimensions {cropped_width}x{cropped_height} exceed limit. Needs resize/compression.")
            except Exception as img_err:
                cropped_processing_error = f"Error: Cannot read cropped image dimensions - {img_err}"
        
        if cropped_processing_error:
            raise Exception(cropped_processing_error)

        if needs_processing_cropped:
            # print(f"Entry {entry_index}: Starting to process (compress/resize) cropped image...")
            temp_path_cropped = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_f:
                    temp_f.write(cropped_image_bytes)
                    temp_path_cropped = temp_f.name
                
                # Use compress_image_to_memory for the cropped image stored in temp file
                compressed_bytes_final, _, _ = compress_image_to_memory(
                    temp_path_cropped,
                    max_size_bytes=MAX_SIZE_BYTES,
                    max_width=MAX_WIDTH,
                    max_height=MAX_HEIGHT
                )
                if compressed_bytes_final is None:
                    cropped_processing_error = "Error: Cropped image compression/resizing failed"
                else:
                    image_bytes_for_llm_step2 = compressed_bytes_final
                    # print(f"Entry {entry_index}: Cropped image processed. Final size: {len(image_bytes_for_llm_step2) / (1024*1024):.2f}MB.")
            finally:
                if temp_path_cropped and os.path.exists(temp_path_cropped):
                    try:
                        os.unlink(temp_path_cropped)
                        # print(f"Entry {entry_index}: Temporary cropped image file {temp_path_cropped} deleted.")
                    except Exception as e_unlink:
                        print(f"Warning (Entry {entry_index}): Failed to delete temp file {temp_path_cropped}: {e_unlink}")
        else:
            # print(f"Entry {entry_index}: Cropped image does not require further processing.")
            image_bytes_for_llm_step2 = cropped_image_bytes
    
    except Exception as e_crop_proc:
        cropped_processing_error = f"Error: Exception during cropped image processing - {e_crop_proc}"

    if cropped_processing_error or image_bytes_for_llm_step2 is None:
        error_msg_step2 = cropped_processing_error if cropped_processing_error else "Cropped image processing failed (unknown reason)"
        # Keep existing error if one was already set, or update if this is a new one
        output_result['Error'] = output_result.get('Error') or error_msg_step2
        print(f"Error (Entry {entry_index}): {error_msg_step2}. Cannot proceed to second API call.")
        return output_result
    # --- End of cropped image processing ---

    # 11. Base64 encode processed cropped image
    encoded_cropped_image = encode_image_bytes(image_bytes_for_llm_step2)
    if encoded_cropped_image is None:
        output_result['Error'] = output_result.get('Error') or "Base64 encoding of processed cropped image failed"
        print(f"Error (Entry {entry_index}): Base64 encoding of processed cropped image failed. Cannot proceed to second call.")
        return output_result

    # 12. Build messages for second API call
    messages_for_api.append(assistant_message_1_obj.model_dump()) # Add assistant's first response
    bbox_str = f"[{bbox_norm_original[0]:.3f}, {bbox_norm_original[1]:.3f}, {bbox_norm_original[2]:.3f}, {bbox_norm_original[3]:.3f}]"
    user_content_follow_up = prompts_collection['user_template_follow_up'].format(
        question=original_question, # Provide original question for context
        bbox_coordinates=bbox_str,
    )
    follow_up_user_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": user_content_follow_up},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_cropped_image}"}}
        ]
    }
    messages_for_api.append(follow_up_user_message)

    # print(f"\n--- Entry {entry_index}: Starting second API call (with cropped image) ---")
    assistant_message_2_obj = call_llm_api(client, model_to_use, messages_for_api, temp_setting)

    if assistant_message_2_obj is None or not assistant_message_2_obj.content:
        output_result['Error'] = output_result.get('Error') or "Follow-up API call failed or returned empty content"
        print(f"Error (Entry {entry_index}): Second API call failed.")
        return output_result

    response_2_content = assistant_message_2_obj.content
    # print(f"Entry {entry_index}: Second API response received.") # {response_2_content[:200]}...")

    # 13. Parse second response
    rechecked_answer_letter = parse_rechecked_answer(response_2_content)

    if rechecked_answer_letter:
        output_result['Rechecked Answer'] = rechecked_answer_letter
        # print(f"Entry {entry_index}: Parsed Rechecked Answer: {rechecked_answer_letter}")
    # else:
        # print(f"Entry {entry_index}: Could not parse Rechecked Answer from second response.")

    # 14. Return final result
    return output_result


# --- Main Execution ---
if __name__ == "__main__":
    # Define prompts (these are specific to your task and model)
    prompts = {
        "system_prompt_1": """You are an advanced image understanding assistant.You will be given an image and a question about it.
""",
        "user_template_1": """ Question: {question}
Your task:
1.**Provide Your Answer:** Examining the image and the question thoroughly, answer the question in the format **"Answer: [Letter]"**, where [Letter] is only the letter (A, B, C, D) of the correct option.
2.**Identify Critical Area:**After output the answer,please determine which area most relevant to the question.Then,please determine a bounding box (normalized coordinates, 0 <= x1, y1, x2, y2 <= 1, x1 < x2, y1 < y2) of the area. Ensure the bounding box is large enough to include all the surrounding context that might be relevant to answering the question (such as information about the table row or column, nearby text, interacting objects, relevant background).Then output in the format**Bounding box: [x1, y1, x2, y2]**



""",
        "user_template_follow_up": """I will provide you an extra image which is cropped from the original image.Just treat the newly input cropped image as an additional information to the local detail information.(The cropped image is clearer)
Please examine the cropped image.(The cropped image may not contain the information needed to answer the question,then ignore this cropped image)
Review the original image and combine with the information from the cropped image.    
Identify potential omission of information in visual perception or calculation based on the image and question.
If you think your previous answer is correct,please reasoning to explain why and conclude your response with the format **"Rechecked Answer: [Letter]"**, where [Letter] is only the letter (A, B, C, D) of the correct option.
If you think your previous answer is wrong, please reasoning to explain why and correct the answer. Conclude your response with the format **"Rechecked Answer: [Letter]"**, where [Letter] is only the letter (A, B, C, D) of the correct option.
"""}

    # --- Initialize API Client ---
    try:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        exit(1)

    # --- Load Dataset ---
    print(f"Loading dataset from: {DATASET_PATH}")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset file not found at {DATASET_PATH}")
        print("Please ensure the DATASET_PATH variable points to your data file.")
        exit(1)
    
    try:
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {DATASET_PATH}: {e}")
        exit(1)
    except Exception as e:
        print(f"Error occurred while loading dataset: {e}")
        exit(1)

    if not isinstance(dataset, list):
        print(f"Error: Loaded data from {DATASET_PATH} is not a list.")
        exit(1)
    print(f"Dataset loaded successfully. Total entries: {len(dataset)}.")

    if not os.path.exists(IMAGE_BASE_PATH):
        print(f"Warning: IMAGE_BASE_PATH '{IMAGE_BASE_PATH}' does not exist. Image loading might fail.")
    
    results = []

    # --- Process Entries ---
    for index, entry in enumerate(tqdm(dataset, desc="Processing entries")):
        if not isinstance(entry, dict):
            print(f"Warning: Skipping entry at index {index} as it is not a dictionary.")
            results.append({"Error": "Skipped non-dictionary entry", "OriginalIndex": index, "OriginalEntry": entry})
            continue

        processed_result = process_single_entry(
            index, entry, client, MODEL_NAME, prompts, IMAGE_BASE_PATH, TEMPERATURE
        )
        results.append(processed_result)

    # --- Save Results ---
    print(f"\nProcessing complete. Saving results to {OUTPUT_PATH}")
    try:
        output_dir = os.path.dirname(OUTPUT_PATH)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print("Results saved successfully.")
    except Exception as e:
        print(f"Error: Failed to save results: {e}")
        exit(1)

    print("Script execution finished.")