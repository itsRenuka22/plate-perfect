import os
import re
from google.cloud import vision_v1
from google.cloud.vision_v1 import types

# Set the path to the Google Cloud Vision API key file
GOOGLE_CLOUD_VISION_KEY_PATH = r'/home/user/Public/Projects/PlatePerfect/.venv/VisionAPIServiceKey.json'

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
    print("spaces and \n REMOVED LP:", license_plate_text)

    # Remove spaces, special characters from the license plate text
    license_plate_text = re.sub(r'[*.]', '', license_plate_text)
    print("* REMOVED LP:", license_plate_text)

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