import pytesseract
from PIL import Image
import os

# Ensure pytesseract is pointing to the Tesseract executable on your system.
# This is required for pytesseract to function properly.
# For Windows, you might need to set the path like this:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_image(image_path):
    """
    Extract text from the provided image using OCR (Optical Character Recognition).

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The text extracted from the image.
    """
    try:
        # Open the image file
        img = Image.open(image_path)
        
        # Convert the image to text using pytesseract
        text = pytesseract.image_to_string(img)
        
        # Return the extracted text
        return text
    except Exception as e:
        # Handle exceptions (e.g., file not found or invalid image)
        print(f"Error extracting text from image: {e}")
        return ""

# Example usage:
# image_path = "path_to_your_image.jpg"
# extracted_text = extract_text_from_image(image_path)
# print(extracted_text)
