from PIL import Image
import os

def crop_image_and_save(input_image_path, norm_coordinates, output_directory):
    """
    Crops an image based on given normalized coordinates and saves it to the specified
    directory. The output filename will be the original filename suffixed with "_cropped",
    and the format will be JPEG.

    :param input_image_path: Path to the input image (can be PNG, JPG, etc.).
    :param norm_coordinates: List of normalized coordinates [x1, y1, x2, y2], range 0-1.
    :param output_directory: Directory path to save the cropped image.
    :return: True if successful, False otherwise.
    """
    # Ensure the output directory exists; create it if it doesn't.
    if not os.path.exists(output_directory):
        try:
            os.makedirs(output_directory)
            print(f"Created output directory: {output_directory}")
        except OSError as e:
            print(f"Error: Could not create output directory {output_directory}: {e}")
            return False

    try:
        # Open the image
        with Image.open(input_image_path) as img:
            # Convert to RGB if necessary (e.g., for PNGs with alpha or palette-based images)
            # JPEG format does not support alpha channels.
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                img = img.convert('RGB')
            elif img.mode != 'RGB': # Handle other modes like grayscale
                img = img.convert('RGB')

            # Get the image's width and height
            width, height = img.size

            # Convert normalized coordinates to pixel coordinates
            # Ensure coordinates are valid before multiplication
            if not (isinstance(norm_coordinates, list) and len(norm_coordinates) == 4 and
                    all(isinstance(c, (int, float)) and 0 <= c <= 1 for c in norm_coordinates)):
                print(f"Error: Invalid or out-of-range normalized coordinates provided: {norm_coordinates}")
                return False

            x1_px = int(norm_coordinates[0] * width)
            y1_px = int(norm_coordinates[1] * height)
            x2_px = int(norm_coordinates[2] * width)
            y2_px = int(norm_coordinates[3] * height)

            # Ensure pixel coordinates are in correct order (x1 < x2, y1 < y2)
            # This also handles cases where normalized coordinates might be swapped.
            left = min(x1_px, x2_px)
            top = min(y1_px, y2_px)
            right = max(x1_px, x2_px)
            bottom = max(y1_px, y2_px)

            # Ensure the crop box has a non-zero area and is within image bounds
            if left == right or top == bottom:
                print(f"Error: Crop box has zero width or height for {input_image_path}. Coordinates: ({left},{top})-({right},{bottom})")
                return False
            if left >= width or top >= height or right <= 0 or bottom <= 0:
                print(f"Error: Crop box is entirely outside image bounds for {input_image_path}.")
                return False

            # Crop the image using pixel coordinates
            cropped_img = img.crop((left, top, right, bottom))

            # Get the original filename (without path and extension)
            base_name = os.path.splitext(os.path.basename(input_image_path))[0]

            # Construct the output file path: output_directory + original_filename + "_cropped" + ".jpg"
            output_path = os.path.join(output_directory, f"{base_name}_cropped.jpg")

            # Save the cropped image as JPEG
            cropped_img.save(output_path, "JPEG", quality=95) # Can specify quality
            print(f"Cropped image saved as {output_path}")
            return True

    except FileNotFoundError:
        print(f"Error: Input image not found at {input_image_path}")
        return False
    except Exception as e:
        print(f"An error occurred while processing image {input_image_path}: {e}")
        return False

# Example usage
if __name__ == "__main__":
    # NOTE: Please update these paths and coordinates for your specific use case.
    input_img_path = "path/to/your/input_image.jpg"  
    # Normalized coordinates [x_min, y_min, x_max, y_max]
    crop_coordinates = [0.65, 0.35, 1.0, 0.65]
    output_save_dir = "path/to/your/output_directory/"  
    # Call the cropping function
    if os.path.exists(input_img_path):
        crop_image_and_save(input_img_path, crop_coordinates, output_save_dir)
    else:
        print(f"Error: The example input image path '{input_img_path}' does not exist. Please update it.")