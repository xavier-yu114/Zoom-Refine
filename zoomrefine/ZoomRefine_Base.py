import os
import json
from PIL import Image, ImageFile
import io
from tqdm import tqdm
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import numpy as np
import re
import math
import copy

ImageFile.LOAD_TRUNCATED_IMAGES = True # Allow loading of truncated images

# --- Image Constraints ---
MAX_SIZE_BYTES = 7 * 1024 * 1024
MAX_WIDTH = 4096
MAX_HEIGHT = 4096
print(f"Applying image constraints: Max Size={MAX_SIZE_BYTES / (1024*1024):.2f}MB, Max Dim={MAX_WIDTH}x{MAX_HEIGHT}")

# --- Prompts ---

prompts = {
    "system_prompt_1": """You are an advanced image understanding assistant. You will be given an image and a question about it.
""",
    "user_template_1": """<image>\n{question}
Your Task:
1.**Provide Your Answer:** Examining the image and the question thoroughly, answer the question in the format **Answer: [Letter]**, where [Letter] is only the letter (A, B, C, D or E) of the correct option.
2.**Identify Critical Area:**After output the answer,please determine which area most relevant to the question.Then,please determine a bounding box (normalized coordinates, 0 <= x1, y1, x2, y2 <= 1, x1 < x2, y1 < y2) of the area. Ensure the bounding box is large enough to include all the surrounding context that might be relevant to answering the question (such as information about the table row or column, nearby text, interacting objects, relevant background).Then output in the format**Bounding box: [x1, y1, x2, y2]**
""",
    "user_template_follow_up": """<image>\nI will provide you an extra image which is cropped from the original image.Just treat the newly input cropped image as an additional information to the local detail information.(The cropped image is clearer)
Please examine the cropped image.(The cropped image may not contain the information needed to answer the question,then ignore this cropped image)
Review the original image and combine with the information from the cropped image.
Identify potential omission of information in visual perception or calculation based on the image and question.
If you think your previous answer is correct, please reasoning to explain why and conclude your response with the format **"Rechecked Answer: [Letter]"**, where [Letter] is only the letter (A, B, C, D or E) of the correct option.
If you think your previous answer is wrong, please reasoning to explain why and correct the answer. Conclude your response with the format **"Rechecked Answer: [Letter]"**, where [Letter] is only the letter (A, B, C, D or E) of the correct option.
"""
}

def compress_image_to_memory(image_path, max_size_bytes=MAX_SIZE_BYTES, initial_quality=85, min_quality=10, max_width=MAX_WIDTH, max_height=MAX_HEIGHT):
    """
    Compresses an image to be under max_size_bytes and within max_width/max_height.
    Returns compressed bytes and original dimensions, or None if an error occurs.
    (Logic of this function remains unchanged as per original request)
    """
    try:
        with Image.open(image_path) as img:
            original_width, original_height = img.size
            needs_resize = original_width > max_width or original_height > max_height
            current_img = img # Start with the original image object

            if needs_resize:
                 print(f"Resizing image {os.path.basename(image_path)} from {original_width}x{original_height}...")
                 resampling_method = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
                 current_img = img.copy() # Work on a copy if resizing
                 current_img.thumbnail((max_width, max_height), resampling_method)
                 print(f"Resized dimensions: {current_img.size}")

            if current_img.mode in ('RGBA', 'P'):
                # print(f"Converting image {os.path.basename(image_path)} to RGB...")
                if current_img is img: # If no resize happened, current_img still points to original img
                    current_img = img.copy() # Make a copy before converting
                current_img = current_img.convert('RGB')

            temp_bytes_io = io.BytesIO()
            current_img.save(temp_bytes_io, format='JPEG', quality=initial_quality, optimize=True)
            current_size = len(temp_bytes_io.getvalue())
            temp_bytes_io.close()

            if current_size > max_size_bytes:
                print(f"Image {os.path.basename(image_path)} initial size (quality {initial_quality}): {current_size / (1024 * 1024):.2f} MB. Starting compression...")
                quality = initial_quality
                compressed_image_bytes = io.BytesIO()
                current_img.save(compressed_image_bytes, format='JPEG', quality=quality, optimize=True)
                compressed_size = len(compressed_image_bytes.getvalue())

                while compressed_size > max_size_bytes and quality > min_quality:
                    quality -= 5
                    compressed_image_bytes.seek(0)
                    compressed_image_bytes.truncate()
                    current_img.save(compressed_image_bytes, format='JPEG', quality=quality, optimize=True)
                    compressed_size = len(compressed_image_bytes.getvalue())
                    # print(f"Reduced quality to {quality}. New size: {compressed_size / (1024 * 1024):.2f} MB")

                if compressed_size > max_size_bytes:
                    print(f"Warning: Unable to compress image {os.path.basename(image_path)} below {max_size_bytes / (1024 * 1024):.2f}MB (minimum quality {min_quality}).")

                final_bytes = compressed_image_bytes.getvalue()
                compressed_image_bytes.close()
                # print(f"Image {os.path.basename(image_path)} final compressed size: {len(final_bytes) / (1024 * 1024):.2f} MB")
                return final_bytes, original_width, original_height
            else:
                # print(f"Image {os.path.basename(image_path)} at quality {initial_quality} size: {current_size / (1024 * 1024):.2f} MB, within limits.")
                final_bytes_io = io.BytesIO()
                current_img.save(final_bytes_io, format='JPEG', quality=initial_quality, optimize=True)
                final_bytes = final_bytes_io.getvalue()
                final_bytes_io.close()
                return final_bytes, original_width, original_height

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None, None, None
    except Image.UnidentifiedImageError:
         print(f"Error: Cannot identify image file (corrupt or unsupported): {image_path}")
         return None, None, None
    except Exception as e:
        print(f"Error: Exception processing image {image_path}: {e}")
        return None, None, None

# --- Local Model Image Loading and Preprocessing Functions ---
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
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks, f"Expected {blocks} blocks, got {len(processed_images)}"
    if use_thumbnail and len(processed_images) != 1: # Add thumbnail if not a single block image
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    """Loads an image from a file-like object or path, preprocesses it."""
    try:
        image = Image.open(image_file).convert('RGB')
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(img_item) for img_item in images] # Corrected variable name
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    except Exception as e:
        # Use repr(image_file) as BytesIO doesn't have a simple string representation
        print(f"Error loading/processing image from source {repr(image_file)}: {e}")
        return None


# --- Auxiliary Functions (Logic unchanged as per original request) ---
def parse_bbox(response_text):
    """Parses Bounding Box from response text."""
    regex = r"\s*(?:\*{2})?(?:Bounding box|bbox)(?:\*{2})?\s*[:：]?\s*(?:\*{2})?\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]"
    matches = re.findall(regex, response_text, re.IGNORECASE)
    if not matches:
        return None
    last_match_coords_str = matches[-1]
    try:
        coords = [float(c) for c in last_match_coords_str]
        if len(coords) == 4 and coords[0] < coords[2] and coords[1] < coords[3]:
             coords_clamped = [
                 max(0.0, min(1.0, coords[0])), max(0.0, min(1.0, coords[1])),
                 max(0.0, min(1.0, coords[2])), max(0.0, min(1.0, coords[3])),
             ]
             # Ensure x1 < x2 and y1 < y2 after clamping, with a small epsilon for safety if they become equal
             if coords_clamped[0] >= coords_clamped[2]: coords_clamped[2] = min(1.0, coords_clamped[0] + 1e-6)
             if coords_clamped[1] >= coords_clamped[3]: coords_clamped[3] = min(1.0, coords_clamped[1] + 1e-6)

             # Final check
             if coords_clamped[0] < coords_clamped[2] and coords_clamped[1] < coords_clamped[3]:
                 return coords_clamped
             else:
                 print(f"Warning: Clamped coordinates still invalid: {coords_clamped}")
                 return None
        else:
            print(f"Warning: Last parsed coordinates invalid or out of order: {coords}")
            return None
    except ValueError:
        print(f"Error: Failed to convert BBox coordinates to float: {last_match_coords_str}")
        return None

def expand_bbox(bbox_norm, padding=0.1):
    """Expands a normalized bounding box by a padding factor."""
    if not (isinstance(bbox_norm, list) and len(bbox_norm) == 4):
        print("Error: expand_bbox received invalid input bbox_norm")
        return None
    if not (0.0 <= padding <= 1.0):
         print(f"Warning: expand_bbox padding value ({padding}) invalid or too large, using 0.1.")
         padding = 0.1

    x1, y1, x2, y2 = bbox_norm
    new_x1 = max(0.0, x1 - padding)
    new_y1 = max(0.0, y1 - padding)
    new_x2 = min(1.0, x2 + padding)
    new_y2 = min(1.0, y2 + padding)

    # Ensure validity after expansion
    if new_x1 >= new_x2:
        # print(f"Warning: Expanded x1 ({new_x1:.3f}) >= x2 ({new_x2:.3f}). Adjusting.")
        mid_x = (x1 + x2) / 2
        width = (x2 - x1) / 2 + padding / 2 # Half original width + half padding
        new_x1 = max(0.0, mid_x - width)
        new_x2 = min(1.0, mid_x + width)
        if new_x1 >= new_x2: print("Error: x-coordinates still invalid after adjustment in expand_bbox."); return None

    if new_y1 >= new_y2:
        # print(f"Warning: Expanded y1 ({new_y1:.3f}) >= y2 ({new_y2:.3f}). Adjusting.")
        mid_y = (y1 + y2) / 2
        height = (y2 - y1) / 2 + padding / 2 # Half original height + half padding
        new_y1 = max(0.0, mid_y - height)
        new_y2 = min(1.0, mid_y + height)
        if new_y1 >= new_y2: print("Error: y-coordinates still invalid after adjustment in expand_bbox."); return None

    expanded_coords = [new_x1, new_y1, new_x2, new_y2]
    print(f"Original BBox: [{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}], Expanded BBox: [{expanded_coords[0]:.3f}, {expanded_coords[1]:.3f}, {expanded_coords[2]:.3f}, {expanded_coords[3]:.3f}] with padding {padding}")
    return expanded_coords

def extract_letter_choice(response_text):
    """Extracts the letter choice (A-E) from the model's response."""
    if not response_text:
        return None
    # Regex to find "Answer: [Letter]" or "Answer: (Letter)" or "Answer: Letter"
    # Also handles "**Answer**: [Letter]" etc.
    header_regex = r"(?:\*{2})?(?:Answer|Rechecked Answer)(?:\*{2})?\s*[:：]\s*"
    pattern_bracketed = header_regex + r"[\(\[]\s*([A-E])\s*[\)\]]"
    match = re.search(pattern_bracketed, response_text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).upper()

    pattern_bare_letter = header_regex + r"\b([A-E])\b" # For "Answer: A"
    match = re.search(pattern_bare_letter, response_text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).upper()
    return None

def parse_rechecked_answer(response_text):
    """Parses the rechecked answer letter (A-E) from response text."""
    if not response_text: return None
    # Regex to find "Rechecked Answer: [Letter]" or "(Letter)" or "Letter"
    match = re.search(r"(?:\*{2})?Rechecked Answer(?:\*{2})?\s*[:：]\s*[\(\[]?\s*([A-E])\s*[\)\]]?", response_text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).upper()
    return None

def crop_image_from_bbox(original_image_path, bbox_norm, original_width, original_height):
    """Crops an image based on normalized bounding box coordinates."""
    try:
        with Image.open(original_image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')

            x1_norm, y1_norm, x2_norm, y2_norm = bbox_norm
            # Convert normalized coordinates to pixel coordinates
            x1_pix = max(0, min(original_width - 1, math.floor(x1_norm * original_width)))
            y1_pix = max(0, min(original_height - 1, math.floor(y1_norm * original_height)))
            x2_pix = max(0, min(original_width, math.ceil(x2_norm * original_width))) # Use ceil for x2,y2
            y2_pix = max(0, min(original_height, math.ceil(y2_norm * original_height)))

            # Ensure coordinates are valid for cropping (x1 < x2, y1 < y2)
            if x1_pix >= x2_pix: x2_pix = x1_pix + 1 # Ensure width is at least 1
            if y1_pix >= y2_pix: y2_pix = y1_pix + 1 # Ensure height is at least 1
            x2_pix = min(original_width, x2_pix) # Clamp to image boundaries
            y2_pix = min(original_height, y2_pix)

            if x1_pix >= x2_pix or y1_pix >= y2_pix:
                 print(f"Error: Calculated crop region invalid ({x1_pix}, {y1_pix}, {x2_pix}, {y2_pix}) from BBox: {bbox_norm} for image {os.path.basename(original_image_path)}")
                 return None

            cropped_img = img.crop((x1_pix, y1_pix, x2_pix, y2_pix))
            cropped_bytes_io = io.BytesIO()
            cropped_img.save(cropped_bytes_io, format='JPEG', quality=95, optimize=True) # Save at high quality
            return cropped_bytes_io.getvalue()
    except FileNotFoundError:
         print(f"Error: Original image not found during cropping: {original_image_path}")
         return None
    except Exception as e:
        print(f"Error: Exception during image cropping {original_image_path}: {e}")
        return None

# --- Main Script ---
# NOTE: Please update these paths to your local environment
input_json_path = 'path/to/your/dataset.json'
output_json_path = 'path/to/your/result.json'
image_base_dir = 'path/to/your/image_dataset'
model_path = 'path/to/your/model' # Local model path

max_new_tokens = 1024
max_num_patches = 12 # Max patches for dynamic image preprocessing

# --- Load Model and Tokenizer ---
print("Loading model and tokenizer...")
try:
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        print("Error: CUDA is not available or no GPUs detected.")
        exit()
    print(f"CUDA available. Found {torch.cuda.device_count()} GPU(s).")
    device = torch.device("cuda")
    print(f"Using device: {device}")

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True, # Requires flash-attn library
        trust_remote_code=True
    ).eval().to(device)

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
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# --- Process Dataset ---
results = []
padding_amount = 0.1 # BBox expansion padding

for index, entry in enumerate(tqdm(test_set, desc="Processing entries", unit="entry")):
    output_result = {
        "id": entry.get("id", f"entry_{index}"), # Preserve or generate an ID
        "messages": copy.deepcopy(entry.get("messages", [])),
        "images": copy.deepcopy(entry.get("images", [])),
        "Answer": None, "Bounding_Box": None, "Rechecked Answer": None,
        "Response_Stage1": None, "Response_Stage2": None, "Error": None
    }
    pixel_values_1 = None
    pixel_values_2 = None
    history_1 = None
    original_image_path = None
    image_filename = "N/A" # Default for error reporting

    try:
        # --- Validate Entry Structure ---
        if not output_result["messages"] or not isinstance(output_result["messages"][0].get("content"), str):
            raise ValueError("Invalid 'messages' structure in input JSON.")
        original_question_full = output_result["messages"][0]["content"]
        # Clean up question for model if a specific phrase needs removal
        phrase_to_remove = "Respond with only the letter (A, B, C, D, or E) of the correct option"
        question_for_model = original_question_full.replace(phrase_to_remove, "").strip()

        if not output_result["images"] or not isinstance(output_result["images"][0], str):
            raise ValueError("Invalid 'images' structure in input JSON.")
        image_filename = output_result["images"][0]
        original_image_path = os.path.join(image_base_dir, image_filename)

        # --- Stage 1: Prepare Original Image ---
        print(f"\n--- Processing Entry {index} | Image: {image_filename} ---")
        image_bytes_stage1 = None
        original_width, original_height = None, None
        processing_error_message = None

        if not os.path.exists(original_image_path):
            processing_error_message = f"Error: Original image file not found at {original_image_path}"
        else:
            try:
                file_size_original = os.path.getsize(original_image_path)
                width_check, height_check = 0,0
                try:
                    with Image.open(original_image_path) as img_dim_check:
                        width_check, height_check = img_dim_check.size
                except Exception as img_err:
                    raise ValueError(f"Error opening image {image_filename} to get dimensions: {img_err}")

                if file_size_original <= MAX_SIZE_BYTES and width_check <= MAX_WIDTH and height_check <= MAX_HEIGHT:
                    print(f"Image {image_filename} is within limits. Reading original bytes.")
                    try:
                        with open(original_image_path, "rb") as f:
                            image_bytes_stage1 = f.read()
                        original_width, original_height = width_check, height_check
                    except Exception as read_err:
                        processing_error_message = f"Error reading original image file: {read_err}"
                else:
                    print(f"Image {image_filename} exceeds limits (Size: {file_size_original/(1024*1024):.2f}MB, Dim: {width_check}x{height_check}). Checking/Compressing...")
                    image_bytes_stage1, ow, oh = compress_image_to_memory(original_image_path, MAX_SIZE_BYTES, max_width=MAX_WIDTH, max_height=MAX_HEIGHT)
                    original_width, original_height = ow, oh # Store original dimensions from before any compression/resizing
                    if image_bytes_stage1 is None:
                        processing_error_message = "Error: Image too large or failed compression check"
            except Exception as e:
                processing_error_message = f"Error: Unexpected error checking/processing image {image_filename} - {e}"

        if processing_error_message or image_bytes_stage1 is None or original_width is None or original_height is None:
            raise ValueError(processing_error_message if processing_error_message else "Failed to get image bytes or dimensions for Stage 1.")

        print("Loading original image for Stage 1...")
        pixel_values_1 = load_image(io.BytesIO(image_bytes_stage1), input_size=model.config.image_size, max_num=max_num_patches)
        if pixel_values_1 is None:
            raise ValueError(f"Failed to load/process original image {image_filename} using load_image for Stage 1.")
        pixel_values_1 = pixel_values_1.to(torch.bfloat16).to(device)

        # --- Stage 1: Inference (Answer + BBox) ---
        print("Running Stage 1 inference (Answer + BBox)...")
        question_1 = prompts['user_template_1'].format(question=question_for_model)
        with torch.no_grad():
            response_1, history_1 = model.chat(
                tokenizer,
                pixel_values=pixel_values_1,
                question=question_1,
                history=None, # No history for the first turn
                generation_config=generation_config,
                return_history=True # Crucial for multi-turn interaction
            )
        output_result['Response_Stage1'] = response_1
        print(f"Stage 1 Response (raw snippet): {response_1[:300]}...")

        # --- Stage 1: Parsing ---
        print("Parsing Stage 1 response...")
        answer_letter_1 = extract_letter_choice(response_1)
        bbox_norm_1 = parse_bbox(response_1)
        output_result['Answer'] = answer_letter_1
        output_result['Bounding_Box'] = bbox_norm_1
        print(f"Parsed Answer (Stage 1): {answer_letter_1}, Parsed BBox: {bbox_norm_1}")

        if history_1 is None:
             print(f"Warning: model.chat did not return history for entry {index}. Skipping Stage 2.")
             output_result['Error'] = (output_result['Error'] or "") + "Failed to get history for Stage 2. "
             # No 'continue' here, will proceed to append and cleanup
        elif bbox_norm_1 is None:
            print(f"Warning: No BBox found or parsed for entry {index}. Skipping Stage 2.")
            output_result['Error'] = (output_result['Error'] or "") + "No BBox for Stage 2. "
            # No 'continue' here
        else:
            # --- Stage 2: Prepare Cropped Image ---
            print("Preparing Stage 2 (Cropped Image)...")
            bbox_expanded = expand_bbox(bbox_norm_1, padding=padding_amount)
            bbox_to_crop = bbox_expanded if bbox_expanded else bbox_norm_1 # Use original if expansion failed
            if bbox_expanded is None:
                print("Warning: BBox expansion failed, using original BBox for cropping.")

            cropped_image_bytes = crop_image_from_bbox(original_image_path, bbox_to_crop, original_width, original_height)
            if cropped_image_bytes is None:
                raise ValueError("Failed to crop image based on BBox for Stage 2.")

            print("Loading cropped image for Stage 2...")
            pixel_values_2 = load_image(io.BytesIO(cropped_image_bytes), input_size=model.config.image_size, max_num=max_num_patches)
            if pixel_values_2 is None:
                raise ValueError(f"Failed to load/process cropped image {image_filename} for Stage 2.")
            pixel_values_2 = pixel_values_2.to(torch.bfloat16).to(device)

            # Concatenate pixel values for multi-image input
            pixel_values_combined = torch.cat((pixel_values_1, pixel_values_2), dim=0)
            num_patches_list = [pixel_values_1.size(0), pixel_values_2.size(0)] # List of patch counts for each image

            # --- Stage 2: Inference (Recheck Answer with Cropped Image + History) ---
            print("Running Stage 2 inference (Recheck Answer)...")
            question_2 = prompts['user_template_follow_up'].format(question=question_for_model) 
            with torch.no_grad():
                response_2 = model.chat(
                    tokenizer,
                    pixel_values=pixel_values_combined, 
                    question=question_2,
                    history=history_1, 
                    generation_config=generation_config,
                    num_patches_list=num_patches_list 
                )
            output_result['Response_Stage2'] = response_2
            print(f"Stage 2 Response (raw snippet): {response_2[:300]}...")

            # --- Stage 2: Parsing ---
            print("Parsing Stage 2 response...")
            rechecked_answer_letter = parse_rechecked_answer(response_2)
            output_result['Rechecked Answer'] = rechecked_answer_letter
            print(f"Parsed Rechecked Answer (Stage 2): {rechecked_answer_letter}")

        # --- Cleanup for this entry (inside try to ensure it runs if no major error before this) ---
        print(f"Finished processing entry {index}.")

    except ValueError as ve: # Catch specific, handled errors
        print(f"Error processing entry {index} ({image_filename}): {ve}")
        output_result['Error'] = (output_result['Error'] or "") + str(ve)
    except torch.cuda.OutOfMemoryError:
         print(f"Error: CUDA Out of Memory during processing entry {index} ({image_filename}). Skipping to next.")
         output_result['Error'] = "CUDA Out of Memory"
    except Exception as e: # Catch all other unexpected errors
        import traceback
        print(f"Unexpected error processing entry {index} ({image_filename}): {type(e).__name__} - {e}")
        print(traceback.format_exc())
        output_result['Error'] = (output_result['Error'] or "") + f"Unexpected error: {type(e).__name__} - {str(e)}"
    finally:
        # Always try to clean up tensors
        if pixel_values_1 is not None: del pixel_values_1
        if pixel_values_2 is not None: del pixel_values_2
        torch.cuda.empty_cache()
        results.append(output_result)


# --- Save Results ---
print(f"\nProcessing complete. Saving results to {output_json_path}")
try:
    output_dir = os.path.dirname(output_json_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print("Results saved successfully.")
except Exception as e:
    print(f"\nError saving results to {output_json_path}: {e}")

print("\nScript finished.")