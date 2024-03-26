import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from google.cloud import vision_v1
from google.cloud.vision_v1 import types
import re
import shutil
import smtplib
from flask_mail import Mail, Message
from email.mime.base import MIMEBase
from email import encoders
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

app = Flask(__name__, static_folder='static')
ROOT_DIR = '/home/user/Public/Projects/PlatePerfect'

# Set the path to the YOLOv8 model and Google Cloud Vision API key file
YOLO_MODEL_PATH = '/home/user/Public/Projects/PlatePerfect/best.pt'
GOOGLE_CLOUD_VISION_KEY_PATH = r'/home/user/Public/Projects/PlatePerfect/.venv/VisionAPIServiceKey.json'

# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # Enter your SMTP server
app.config['MAIL_PORT'] = 465  # Enter your SMTP port
app.config['MAI;_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
#app.config['MAIL_DEBUG'] = True
app.config['MAIL_USERNAME'] = 'mainprojanpr@gmail.com'  # Enter your email address
app.config['MAIL_PASSWORD'] = 'APP_PASSWORD'  # Enter your email password
mail = Mail(app)


# Function to send email alert
def send_email_alert(filename, ocr_result):
    msg = Message('Invalid License Plate Detected',
                  sender='mainprojanpr@gmail.com',
                  recipients=['patwarirenuka22@gmail.com', 'asaavi30@gmail.com', 'roja.ambati20@gmail.com', 'aryachavarkar390@gmail.com', 'arichavarkar90@gmail.com'])  # Enter the recipient email address
    msg.body = f"An invalid License Plate is detected: {ocr_result}"

    with app.open_resource(filename) as attachment:
        msg.attach(filename, 'image/png', attachment.read())

    mail.send(msg)
    print('email sent')



# Function to validate Indian number plate format
def is_valid_number_plate(license_plate_text):
    # Indian number plate format: 'AB00XY0123'
    # where AB is the state code, 00 is district code, XY any alphabet combination, and 0123 any combination of numbers

    # Remove spaces from the license plate text
    license_plate_text = license_plate_text.replace(" ", "")

    # Regular expression to match the Indian number plate format
    pattern = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$')

    return bool(re.match(pattern, license_plate_text))

# Configure YOLO model
yolo_model = YOLO(YOLO_MODEL_PATH)

# Configure Google Cloud Vision API client
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_CLOUD_VISION_KEY_PATH
vision_client = vision_v1.ImageAnnotatorClient()

predict_dir = '/home/user/Public/Projects/PlatePerfect/runs/detect/predict'


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    # Clear the 'uploads' folder
    shutil.rmtree('uploads', ignore_errors=True)
    os.makedirs('uploads', exist_ok=True)

    # Delete the 'predict_dir'
    shutil.rmtree(predict_dir, ignore_errors=True)
    #os.makedirs('runs/detect/predict', exist_ok=True)
    #os.makedirs('runs/detect/predict/crops', exist_ok=True)
    #os.makedirs('runs/detect/predict/crops/LicensePlate', exist_ok=True)
    #os.makedirs('runs/detect/predict/labels', exist_ok=True)

    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the uploaded image
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        print('OS Path:',os.path)
        print('File path',file_path)
        file.save(file_path)

        # Perform license plate detection using YOLO
        yolo_results = yolo_model(file_path, save=True, save_conf=True, save_txt=True, save_crop=True)

        shutil.move("/home/user/Public/Projects/BE_FinalProject/MZK-ANPR/Automatic-License-Plate-Recognition-using-YOLOv8/runs/detect/predict", "/home/user/Public/Projects/PlatePerfect/runs/detect")

        # Get the first detected license plate image
        license_plate_files = os.listdir(os.path.join(predict_dir, 'crops/LicensePlate'))
        if license_plate_files:
            first_license_plate_file = license_plate_files[0]
            license_plate_path = os.path.join(predict_dir, 'crops/LicensePlate', first_license_plate_file)

            # Perform OCR using the Vision API
            ocr_result = detect_text_vision_api(license_plate_path)

            # Validate the license plate format
            is_valid_license_plate = is_valid_number_plate(ocr_result)

            # Send email alert if the license plate is invalid
            if not is_valid_license_plate:
                send_email_alert(file_path, ocr_result)

            return render_template('result.html', filename=filename, ocr_result=ocr_result, is_valid_license_plate=is_valid_license_plate)

    return redirect(request.url)

def detect_text_vision_api(img_path):
    with open(img_path, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)
    response = vision_client.text_detection(image=image)
    texts = response.text_annotations

    return texts[0].description if texts else 'Text not detected'

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, port=5001)
