import cv2

import time

from picamera2 import Picamera2

from datetime import datetime

import os

 

# Initialize PiCamera2

picam2 = Picamera2()

video_config = picam2.create_video_configuration(main={"size": (640, 480)})

picam2.configure(video_config)

picam2.start()

 

# Load Haar cascade for face detection

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

 

# Helper function to generate unique filenameslllll

def generate_filename(prefix, extension, folder, label):

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    return os.path.join(folder, label, f"{prefix}_{timestamp}.{extension}")

 

# Initialize variables

capture_count = {0: 0, 1: 0}  # Keep track of images captured for each label

total_images = 60  # Total images per person

train_split = 50  # Number of images for training

image_size = (64, 64)

 

# Create directories for train and test datasets

for dataset in ['train', 'test']:

    for label in ['0', '1']:

        os.makedirs(os.path.join('data', dataset, label), exist_ok=True)

 

print("Press '0' or '1' to capture images for each person, 'q' to quit")

 

while True:

    # Capture frame from PiCamera2

    frame_rgb = picam2.capture_array()

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

 

    # Detect faces in the frame

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

   

    # Draw bounding boxes around detected faces

    for (x, y, w, h) in faces:

        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)

 

    # Show the frame

    cv2.imshow('Camera', frame_bgr)

 

    # Handle key events

    key = cv2.waitKey(1) & 0xFF

   

    if key == ord('0') or key == ord('1'):  # Capture images for label 0 or 1

        label = int(chr(key))

       

        if capture_count[label] < total_images:

            for (x, y, w, h) in faces[:1]:  # Only take the first detected face

                # Crop to face and resize to 64x64

                face_crop = frame_bgr[y:y + h, x:x + w]

                face_resized = cv2.resize(face_crop, image_size)

 

                # Determine folder (train or test) based on count

                folder = 'train' if capture_count[label] < train_split else 'test'

                image_filename = generate_filename('face', 'jpg', os.path.join('data', folder), str(label))

                cv2.imwrite(image_filename, face_resized)

               

                capture_count[label] += 1

                print(f"Image saved for label {label}: {image_filename} ({capture_count[label]}/{total_images})")

 

            if capture_count[label] >= total_images:

                print(f"Completed image capture for label {label}")

 

    elif key == ord('q'):  # Quit the application

        break

 

# Clean up

cv2.destroyAllWindows()

picam2.stop()