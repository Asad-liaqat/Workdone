import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Load models
yolo_model = YOLO('yolov8n.pt')
gender_model = load_model('path_to_gender_model.h5')
age_model = load_model('path_to_age_model.h5')
emotion_model = load_model('path_to_emotion_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def predict_gender_and_age(face_img):
    face_img = cv2.resize(face_img, (64, 64))
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    gender = gender_model.predict(face_img)
    gender_label = "Male" if gender[0] > 0.5 else "Female"
    age = age_model.predict(face_img)
    age_label = int(age[0])
    return gender_label, age_label

def predict_emotion(face_img):
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    emotion = emotion_model.predict(face_img)
    emotion_label = emotion_labels[np.argmax(emotion)]
    return emotion_label

def process_video(input_video_path, output_video_path):
    # Open video file
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect people in the frame
        results = yolo_model(frame)
        person_detections = [result for result in results if result['class_id'] == 0]

        for detection in person_detections:
            x1, y1, x2, y2 = detection['box']
            face_img = frame[y1:y2, x1:x2]

            try:
                gender, age = predict_gender_and_age(face_img)
                emotion = predict_emotion(face_img)
                label = f"{gender}, {age}, {emotion}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            except:
                # Handle the case where the face detection might fail
                pass

        # Write the frame to the output video
        out.write(frame)

    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
process_video('path_to_input_video.mp4', 'path_to_output_video.avi')
