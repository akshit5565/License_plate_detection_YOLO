import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

st.title('YOLO image and video processing')
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg","jpeg","png","mp4","mkv"])

model = YOLO("C:/Users/Akshit/Desktop/license_plate/best_license_plate_model.pt")

if uploaded_file is not None:
    input_path = f"temp/{uploaded_file.name}"
    output_path = f"output/{uploaded_file.name}"
    with open(input_path, 'w') as f:
        f.wrte(uploaded_file.getbuffer())
    st.write("Processing Image.....")

def process_media(input_path, output_path):
    file_extension = os.path.splitext(input_path)[1].lower()
    if file_extension is ['.mp4','.mkv']:
        pass
    elif file_extension in ['.jpg','.jpeg','.png']:
        pass
    else:
        st.write(f"Unsupported file type : {file_extension}")
        return None


def predict_and_save_image(path_test_car, output_image_path):
    results = model.predict(path_test_car, device = 'cpu')
    image = cv2.imread(path_test_car)
    image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2RGB)
    for result in results:
        for box in result.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(image, f'{confidence*100:.2f}%',
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(output_image_path, image)
            return output_image_path