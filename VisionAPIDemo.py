import os
import io
import pandas as pd
from google.cloud import vision_v1
from google.cloud.vision_v1 import types
import re

os.environ['GOOGLE_APPLICATION_CREDENTIAL'] = r'/home/user/Public/Projects/BE_FinalProject/Yolov8_EasyOCR/.venv/VisionAPIServiceKey.json'

client = vision_v1.ImageAnnotatorClient()


def detecttext(img):

    with io.open(img, 'rb') as image_file:
        content = image_file.read()

    image = vision_v1.types.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    df = pd.DataFrame(columns = ['locale', 'description'])
    for text in texts:
        df = df._append(
            dict(
                locale=text.locale,
                description=text.description
            ),
            ignore_index=True
        )

    concatenated_text = "".join(df['description'].iloc[0].split())
    return concatenated_text

#FILE_NAME = 'testimg1.jpg'
#FOLDER_PATH = '/home/user/Public/Projects/BE_FinalProject/Yolov8_EasyOCR/images'

#result = detecttext(os.path.join(FOLDER_PATH,FILE_NAME))
#print(result)