# import streamlit as st
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from PIL import Image
# import os
# import pytesseract
# import re
# import sqlite3
# from twilio.rest import Client

# # Ensure temp directory exists
# if not os.path.exists("temp"):
#     os.makedirs("temp")

# # Initialize the database
# def init_db():
#     conn = sqlite3.connect('car_database.db')
#     c = conn.cursor()
#     c.execute('''CREATE TABLE IF NOT EXISTS car_info (
#                  car_number TEXT PRIMARY KEY,
#                  name TEXT,
#                  phone_number TEXT)''')
#     conn.commit()
#     conn.close()

# def insert_car_info(car_number, name, phone_number):
#     conn = sqlite3.connect('car_database.db')
#     c = conn.cursor()
#     c.execute('''INSERT INTO car_info (car_number, name, phone_number)
#                  VALUES (?, ?, ?)''', (car_number, name, phone_number))
#     conn.commit()
#     conn.close()

# def get_car_info(car_number):
#     conn = sqlite3.connect('car_database.db')
#     c = conn.cursor()
#     c.execute('''SELECT * FROM car_info WHERE car_number = ?''', (car_number,))
#     car_info = c.fetchone()
#     conn.close()
#     return car_info

# # Initialize the database
# init_db()

# # Twilio credentials
# account_sid = 'ACea3594338be0ed226798d072af2ec552'
# auth_token = 'd583c42772e6aad5b7d289928262582d'
# twilio_whatsapp_number = '+15598510528'

# client = Client(account_sid, auth_token)

# def send_whatsapp_message(phone_number, message):
#     message = client.messages.create(
#         from_=f'whatsapp:{twilio_whatsapp_number}',
#         body=message,
#         to=f'whatsapp:{phone_number}'
#     )
#     return message.sid

# # Set the title of the Streamlit app
# st.title("YOLO Image and Video Processing")

# # Allow users to upload images or videos
# uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"])

# # Load YOLO model
# try:
#     model = YOLO("C:/Users/Akshit/Desktop/license_plate/best.pt")  # Replace with the path to your trained YOLO model
# except Exception as e:
#     st.error(f"Error loading YOLO model: {e}")

# # Specify the path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update the path as needed

# def clean_text(text):
#     """Remove special characters from text."""
#     return re.sub(r'[^A-Za-z0-9]', '', text)

# def predict_and_save_image(path_test_car, output_image_path):
#     detected_texts = []
#     try:
#         results = model.predict(path_test_car, device='cpu')
#         image = cv2.imread(path_test_car)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         for result in results:
#             for box in result.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 confidence = box.conf[0]
#                 cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(image, f'{confidence*100:.2f}%', (x1, y1 - 10), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#                 # Crop the bounding box from the image for OCR
#                 roi = image[y1:y2, x1:x2]
#                 # Perform OCR on the cropped image
#                 text = pytesseract.image_to_string(roi, config='--psm 6').strip()
#                 cleaned_text = clean_text(text)
#                 detected_texts.append(cleaned_text)

#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#         cv2.imwrite(output_image_path, image)
#         return output_image_path, detected_texts
#     except Exception as e:
#         st.error(f"Error processing image: {e}")
#         return None, None

# def predict_and_plot_video(video_path, output_path):
#     detected_texts = []
#     try:
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             st.error(f"Error opening video file: {video_path}")
#             return None, None
#         frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = int(cap.get(cv2.CAP_PROP_FPS))
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = model.predict(rgb_frame, device='cpu')
#             for result in results:
#                 for box in result.boxes:
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     confidence = box.conf[0]
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(frame, f'{confidence*100:.2f}%', (x1, y1 - 10), 
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#                     # Crop the bounding box from the image for OCR
#                     roi = frame[y1:y2, x1:x2]
#                     # Perform OCR on the cropped image
#                     text = pytesseract.image_to_string(roi, config='--psm 6').strip()
#                     cleaned_text = clean_text(text)
#                     detected_texts.append(cleaned_text)

#             out.write(frame)
#         cap.release()
#         out.release()
#         return output_path, detected_texts
#     except Exception as e:
#         st.error(f"Error processing video: {e}")
#         return None, None

# def process_media(input_path, output_path):
#     file_extension = os.path.splitext(input_path)[1].lower()
#     if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
#         return predict_and_plot_video(input_path, output_path)
#     elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
#         return predict_and_save_image(input_path, output_path)
#     else:
#         st.error(f"Unsupported file type: {file_extension}")
#         return None, None

# if uploaded_file is not None:
#     input_path = os.path.join("temp", uploaded_file.name)
#     output_path = os.path.join("temp", f"output_{uploaded_file.name}")
#     try:
#         with open(input_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
#         st.write("Processing...")
#         result_path, detected_texts = process_media(input_path, output_path)
#         if result_path:
#             if input_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
#                 video_file = open(result_path, 'rb')
#                 video_bytes = video_file.read()
#                 st.video(video_bytes)
#             else:
#                 st.image(result_path)

#             if detected_texts:
#                 st.subheader("Vehicle Numbers")
#                 for text in detected_texts:
#                     st.markdown(f"<div style='border: 2px solid #4CAF50; padding: 10px; margin-bottom: 10px; max-width: 300px; word-wrap: break-word;'>{text}</div>", unsafe_allow_html=True)

#                     car_info = get_car_info(text)
#                     if car_info:
#                         st.write(f"Car found in database: {car_info}")
#                         fine_amount = st.number_input(f"Enter fine amount for {text}", min_value=0)
#                         if st.button(f"Send Fine to {car_info[1]}"):
#                             message = f"You have been fined {fine_amount} for breaking traffic rules."
#                             send_whatsapp_message(car_info[2], message)
#                             st.success("Fine sent successfully.")
#                     else:
#                         st.write(f"Car {text} not found in database.")
#                         name = st.text_input(f"Enter name for {text}")
#                         phone_number = st.text_input(f"Enter phone number for {text}")
#                         if st.button(f"Register {text}"):
#                             insert_car_info(text, name, phone_number)
#                             st.success("Car registered successfully.")
#                             fine_amount = st.number_input(f"Enter fine amount for {text}", min_value=0)
#                             if st.button(f"Send Fine to {name}"):
#                                 message = f"You have been fined {fine_amount} for breaking traffic rules."
#                                 send_whatsapp_message(phone_number, message)
#                                 st.success("Fine sent successfully.")
#     except Exception as e:
#         st.error(f"Error uploading or processing file: {e}")
