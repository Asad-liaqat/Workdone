import cv2 as cv
import cvzone
import math
from sort import Sort
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('runs/detect/train11/weights/best.pt')  
print("Model loaded successfully")

cap = cv.VideoCapture('big_pallet.mp4')

tracker = Sort(max_age=10, min_hits=3)

# Define the line for counting
line = [400, 0, 400, 900]

counterin = []

classnames = ['Pallet']

frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
out = cv.VideoWriter('crate_count.mp4', cv.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

while True:
    ret, img = cap.read()

    if not ret:
        break

    detections = np.empty((0, 5))

    # Predict using the model
    results = model.predict(img)

    # Debugging: Print the results to understand what's being detected
    print(results)

    for result in results:
        boxes = result.boxes.xyxy
        confidences = result.boxes.conf
        classes = result.boxes.cls

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            confidence = confidences[i]
            class_detect = int(classes[i])

            # Check if class_detect is within the range of classnames
            if class_detect < len(classnames):
                class_name = classnames[class_detect]
                conf = math.ceil(confidence * 100)
                if class_name == 'Pallet' and conf >= 80:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    current_detections = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, current_detections))

    tracker_result = tracker.update(detections)
    cv.line(img, (line[0], line[1]), (line[2], line[3]), (0, 255, 255), 5)

    for track_result in tracker_result:
        x1, y1, x2, y2, id = track_result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2
        cvzone.cornerRect(img, [x1, y1, w, h], rt=5)

        if line[1] < cy < line[3] and line[2] - 10 < cx < line[2] + 10:
            cv.line(img, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 10)
            if counterin.count(id) == 0:
                counterin.append(id)

    cvzone.putTextRect(img, f'Total Count = {len(counterin)}', [50, 150], thickness=2, scale=2.3, border=2)

    out.write(img)  # Write the frame to the output video

    cv.imshow('frame', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()
