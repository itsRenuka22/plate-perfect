import os
import re
import difflib
from google.cloud import vision_v1
from google.cloud.vision_v1 import types

# Set the path to the Google Cloud Vision API key file
GOOGLE_CLOUD_VISION_KEY_PATH = r'C://Users//asaavi//Desktop//plateperfect2//plate-perfect//.venv//VisionAPIServiceKey.json' #----CHANGE PATH TO YOUR VisionAPIServiceKey.json stored in .venv file path ----

# Configure Google Cloud Vision API client
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_CLOUD_VISION_KEY_PATH
vision_client = vision_v1.ImageAnnotatorClient()


def is_valid_number_plate(license_plate_text):
    # Indian number plate format: 'AB00XY0123' or 'AB00Y0123'
    # where AB is the state code, 00 is district code, XY any alphabet combination, and 0123 any combination of numbers
    
    print("OG LP:", license_plate_text)
    # Remove spaces from the license plate text
    license_plate_text = license_plate_text.replace(" ", "")
    print("spaces REMOVED LP:", license_plate_text)

    # Remove spaces and "\n" characters from the license plate text
    license_plate_text = re.sub(r'[\s\n]', '', license_plate_text)
    print("spaces and '\n' REMOVED LP:", license_plate_text)

    # Remove spaces, special characters from the license plate text
    license_plate_text = re.sub(r'[*.]', '', license_plate_text)
    print("'*', '.' REMOVED LP:", license_plate_text)

    # Check if the license plate text contains only English alphabets
    if not re.match(r'^[a-zA-Z0-9\s]+$', license_plate_text):
        return False
    
    # Regular expression to match the Indian number plate format
    pattern = re.compile(r'^(?:IND)?[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$')

    return bool(re.match(pattern, license_plate_text))

def detect_text_vision_api(img_path):
    with open(img_path, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)
    response = vision_client.text_detection(image=image)
    texts = response.text_annotations

    return texts[0].description if texts else 'Text not detected'

def is_similar_ocr_result(result1, result2):
    """
    Function to check if two OCR results are similar based on certain conditions.
    """
    # Ignore non-alphabetic characters and spaces for comparison
    clean_result1 = ''.join(filter(str.isalpha, result1)).lower()
    clean_result2 = ''.join(filter(str.isalpha, result2)).lower()

    # Calculate the difference between the two cleaned OCR results
    difference = difflib.ndiff(clean_result1, clean_result2)

    # Count the number of added, removed, and changed characters
    added = sum(1 for d in difference if d.startswith('+'))
    removed = sum(1 for d in difference if d.startswith('-'))

    # Consider OCR results similar if the difference is within 2 characters
    return added <= 2 or removed <= 2