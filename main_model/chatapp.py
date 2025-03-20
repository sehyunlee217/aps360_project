import torch
import cv2
import numpy as np
from flask import Flask, jsonify
from threading import Thread
from ultralytics import YOLO

# Load YOLOv5 model
model = YOLO('/Users/catherinexiong/Documents/APS360Project/aps360_project/main_model/yolo5v.pt')
model.conf = 0.5  # Confidence threshold


# Class names (Ensure these match your model)
CLASSES = ["open", "close", "yawn"]

# Flask App for API (Run on Port 5001 to Avoid Conflict)
app = Flask(__name__)
detected_state = {"eyes": "Unknown", "yawn": "Unknown"}

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify(detected_state)

def detect_drowsiness():
    global detected_state

    cap = cv2.VideoCapture(0)  # Open webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert frame to YOLO format (BGR -> RGB)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform inference
        results = model(img_rgb)

        # Extract detections
        detections = results.pandas().xyxy[0]  # Pandas DataFrame
        eye_status = "Unknown"
        yawn_status = "No"

        for _, row in detections.iterrows():
            cls = int(row['class'])  # Class index
            label = CLASSES[cls]  # Convert to class name
            conf = row['confidence']

            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

            # Draw bounding box & label
            color = (0, 255, 0) if label == "open" else (0, 0, 255) if label == "close" else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Assign detected status
            if label in ["open", "close"]:
                eye_status = label
            if label == "yawn":
                yawn_status = "Yes"

        # Update detected state
        detected_state["eyes"] = eye_status
        detected_state["yawn"] = yawn_status

        # Display the frame with detections
        cv2.putText(frame, f"Eyes: {eye_status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Yawning: {yawn_status}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Drowsiness Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run detection in a separate thread
Thread(target=detect_drowsiness, daemon=True).start()

# Start Flask API (Use port 5001 to avoid conflict)
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
