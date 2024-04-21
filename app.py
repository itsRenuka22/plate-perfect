import os
import shutil
from ultralytics import YOLO
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from flask_mail import Mail, Message
from helper import is_valid_number_plate, detect_text_vision_api

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
DISTINCT_OCR_FOLDER = 'distinct_ocr'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set the path to the YOLOv8 model and Google Cloud Vision API key file
YOLO_MODEL_PATH = '/home/user/Public/Projects/PlatePerfect/best.pt'
GOOGLE_CLOUD_VISION_KEY_PATH = r'/home/user/Public/Projects/PlatePerfect/.venv/VisionAPIServiceKey.json'

# Configure YOLO model
yolo_model = YOLO(YOLO_MODEL_PATH)

predict_dir = '/home/user/Public/Projects/PlatePerfect/runs/detect/predict'
license_plate_dir_img = '/home/user/Public/Projects/PlatePerfect/runs/detect/predict/crops/LicensePlate'
distinct_ocr_dir = '/home/user/Public/Projects/PlatePerfect/distinct_ocr' 
track_dir = '/home/user/Public/Projects/PlatePerfect/runs/track'
txt_dir = '/home/user/Public/Projects/PlatePerfect/runs/track/labels'   #label folder used for the fetching id of the vehicle and confidence level 
license_plate_dir_vid = '/home/user/Public/Projects/PlatePerfect/runs/track/crops/LicensePlate'  # cropped images of LP are stored here
check2_dir = '/home/user/Public/Projects/PlatePerfect/check2'   # filtered images of the LPs with different ids and confidence above 0.7

# check2 dir might contain images of same LP, to resolve, this directory contains images which have distinct OCR output effectively discarding the duplicate LP images from check2
distinct_ocr_dir = '/home/user/Public/Projects/PlatePerfect/distinct_ocr' 


# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # Enter your SMTP server
app.config['MAIL_PORT'] = 465  # Enter your SMTP port
app.config['MAI;_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
#app.config['MAIL_DEBUG'] = True
app.config['MAIL_USERNAME'] = 'mainprojanpr@gmail.com'  # Enter your email address
app.config['MAIL_PASSWORD'] = 'boyi dyfy cpxv vdjf'  # Enter your email password
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


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    # Clear the 'uploads' folder
    shutil.rmtree('uploads', ignore_errors=True)
    os.makedirs('uploads', exist_ok=True)

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Check the file extension and process accordingly
            if filename.endswith(('jpg', 'jpeg', 'png')):
                ocr_results = process_image(file_path)
                result = []
                for filename, ocr_result in ocr_results.items():
                    is_valid = is_valid_number_plate(ocr_result)
                    result.append({'filename': filename, 'ocr_result': ocr_result, 'is_valid': is_valid})
                #result = [{'filename': filename, 'ocr_result': ocr_result, 'is_valid': is_valid}]
            elif filename.endswith('mp4'):
                ocr_results = process_video(file_path)
                result = []
                for filename, ocr_result in ocr_results.items():
                    is_valid = is_valid_number_plate(ocr_result)
                    result.append({'filename': filename, 'ocr_result': ocr_result, 'is_valid': is_valid})
            else:
                return render_template('index.html', message='Invalid file type')
            
            return render_template('result.html', result=result)

    return render_template('index.html')


@app.route('/get-image/<path:filename>')
def get_image(filename):
    return send_from_directory(DISTINCT_OCR_FOLDER, filename)

#------------- PROCESS IMAGE ----------
def process_image(file_path):

    # Delete the existing track directory
    shutil.rmtree(predict_dir, ignore_errors=True)
    shutil.rmtree(distinct_ocr_dir, ignore_errors=True)

    # Configure the tracking parameters and run the tracker
    results = yolo_model.predict(source=file_path, conf=0.7, iou=0.7, save=True, save_conf=True, save_txt=True, save_crop=True)

    # Move the predict directory
    shutil.move("/home/user/Public/Projects/BE_FinalProject/MZK-ANPR/Automatic-License-Plate-Recognition-using-YOLOv8/runs/detect/predict", "/home/user/Public/Projects/PlatePerfect/runs/detect/predict")
    
    # Perform OCR on images in the directory and save results in a dictionary
    ocr_results_dict = {}
    for filename in os.listdir(license_plate_dir_img):
        print("filename:", filename)
        if filename.endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(license_plate_dir_img, filename)
            print("img_path:",img_path)
            ocr_result = detect_text_vision_api(img_path)
            ocr_results_dict[filename] = ocr_result
    
    print("ocr_result:", ocr_results_dict)

    # Filter out duplicate OCR results and store distinct ones in a new dictionary
    distinct_ocr_results = {}
    for filename, ocr_result in ocr_results_dict.items():
        if ocr_result not in distinct_ocr_results.values():
            distinct_ocr_results[filename] = ocr_result

    print("Duplicates removed distinct_ocr_result:", distinct_ocr_results)

    # Create distinct_ocr folder if it doesn't exist
    if not os.path.exists(distinct_ocr_dir):
        os.makedirs(distinct_ocr_dir)
        print("distinct_ocr folder created")

    # Move corresponding image files to distinct_ocr folder based on distinct OCR results
    for filename, ocr_result in distinct_ocr_results.items():
        source_img_path = os.path.join(license_plate_dir_img, filename)
        if os.path.exists(source_img_path):
            dest_img_path = os.path.join(distinct_ocr_dir, filename)
            shutil.copy(source_img_path, dest_img_path)
            print(f"Image {filename} moved to distinct_ocr folder.")

            # Validate number plate format
            if is_valid_number_plate(ocr_result):
                print(f"OCR result for {filename} is valid: {ocr_result}")
            else:
                print(f"OCR result for {filename} is invalid: {ocr_result}")
                send_email_alert(dest_img_path, ocr_result)
                

        else:
            print(f"Image {filename} not found in the license plate directory.")

    print("Distinct OCR Results:", distinct_ocr_results)

    return distinct_ocr_results



# ----------- PROCESS VIDEO ------------
def process_video(video_path):
    # Configure the YOLO model
    #yolo_model = YOLO('/home/user/Public/Projects/PlatePerfect/best.pt')

    # Delete the existing track directory
    shutil.rmtree(track_dir, ignore_errors=True)
    shutil.rmtree(distinct_ocr_dir, ignore_errors=True)
    shutil.rmtree(check2_dir, ignore_errors=True)

    # Configure the tracking parameters and run the tracker
    results = yolo_model.track(source=video_path, conf=0.7, iou=0.7, save=True, save_conf=True, save_txt=True, save_crop=True)

    # Move the track directory
    shutil.move("/home/user/Public/Projects/BE_FinalProject/MZK-ANPR/Automatic-License-Plate-Recognition-using-YOLOv8/runs/detect/track", "/home/user/Public/Projects/PlatePerfect/runs/track")

    # Dictionary to store the highest confidence for each license plate ID
    highest_confidence_per_plate = {}

    # Iterate through text files
    for txt_filename in os.listdir(txt_dir):
        txt_filepath = os.path.join(txt_dir, txt_filename)

        # Extract license plate ID and confidence from the text file
        with open(txt_filepath, 'r') as file:
            lines = file.readlines()
            if lines:
                # Extract information from the first line of the text file
                confidence = float(lines[0].strip().split()[1])
                plate_id_str = lines[0].strip().split()[-1]  # Extract the last variable as the license plate ID

                # Check if the license plate ID is an integer
                try:
                    plate_id = int(plate_id_str)
                except ValueError:
                    # Skip files where the class ID is a floating-point number
                    continue

                # Update the highest confidence for each license plate ID
                if plate_id not in highest_confidence_per_plate or confidence > highest_confidence_per_plate[plate_id]['confidence']:
                    highest_confidence_per_plate[plate_id] = {
                        'confidence': confidence,
                        'txt_filename': txt_filename
                    }

    # Copy images with the highest confidence for each license plate ID to check2 directory
    for plate_id, data in highest_confidence_per_plate.items():
        txt_filename = data['txt_filename']
        image_filename = os.path.splitext(txt_filename)[0] + '.jpg'

        # Create check2 folder if it doesn't exist
        if not os.path.exists(check2_dir):
            os.makedirs(check2_dir)

        src_image_path = os.path.join(license_plate_dir_vid, image_filename)
        dest_image_path = os.path.join(check2_dir, image_filename)

        shutil.copy(src_image_path, dest_image_path)
        print(f"Image with the highest confidence for license plate ID {plate_id} copied to {dest_image_path}")

    # Perform OCR on images in the directory and save results in a dictionary
    ocr_results_dict = {}
    for filename in os.listdir(check2_dir):
        print('filename:', filename)
        if filename.endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(check2_dir, filename)
            ocr_result = detect_text_vision_api(img_path)
            ocr_results_dict[filename] = ocr_result

    # Filter out duplicate OCR results and store distinct ones in a new dictionary
    distinct_ocr_results = {}
    for filename, ocr_result in ocr_results_dict.items():
        if ocr_result not in distinct_ocr_results.values():
            distinct_ocr_results[filename] = ocr_result

    # Create distinct_ocr folder if it doesn't exist
    if not os.path.exists(distinct_ocr_dir):
        os.makedirs(distinct_ocr_dir)

    # Move corresponding image files to distinct_ocr folder based on distinct OCR results
    for filename, ocr_result in distinct_ocr_results.items():
        source_img_path = os.path.join(check2_dir, filename)
        if os.path.exists(source_img_path):
            dest_img_path = os.path.join(distinct_ocr_dir, filename)
            shutil.copy(source_img_path, dest_img_path)
            print(f"Image {filename} moved to distinct_ocr folder.")

            # Validate number plate format
            if is_valid_number_plate(ocr_result):
                print(f"OCR result for {filename} is valid: {ocr_result}")
            else:
                print(f"OCR result for {filename} is invalid: {ocr_result}")
                send_email_alert(dest_img_path, ocr_result)

        else:
            print(f"Image {filename} not found in the check2 folder.")
    
    print("Distinct OCR Results:", distinct_ocr_results)

    return distinct_ocr_results



if __name__ == '__main__':
    app.run(debug=True)
