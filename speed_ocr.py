from paddleocr import PaddleOCR
from ultralytics import YOLO
import cv2
import math
import time

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Load the trained YOLO model
model_path = '/content/drive/MyDrive/extracted_data/plate_Ds/runs/train/yolov8_plate_detection/weights/best.pt'
model = YOLO(model_path)

# Path to the input video file
input_video_path = 'FinTest.mp4'  # Replace with your video path
output_video_path = '/content/drive/MyDrive/TEST/FinAnnot.mp4'  # Path to save the output video

# Open the video file
cap = cv2.VideoCapture(input_video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output file

# Create a VideoWriter object to save the output video
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Parameters for speed calculation
distance_scale = 0.08  # Adjust based on calibration (test with different values if needed)
speed_threshold = 90  # Speed limit in km/h
object_tracker = {}
vehicle_id_counter = 0

# Loop through each frame of the video
while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video stream.")
        break

    # Run inference using YOLO for vehicle and plate detection
    results = model.predict(source=frame, show=False, save=False)

    annotated_frame = frame.copy()

    for result in results[0].boxes:
        # Get bounding box coordinates for vehicles
        x1, y1, x2, y2 = map(int, result.xyxy[0])

        # Crop detected region for OCR (number plate detection)
        cropped_plate = frame[y1:y2, x1:x2]

        # Convert BGR to RGB for PaddleOCR
        cropped_plate_rgb = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB)

        # Perform OCR
        ocr_results = ocr.ocr(cropped_plate_rgb)

        # Check if OCR results are valid
        if ocr_results and ocr_results[0]:
            detected_text = ''.join([line[1][0] for line in ocr_results[0]])
        else:
            detected_text = "No Text Detected"

        # Debug: Print the OCR results
        print(f"OCR Results: {ocr_results}")

        # Track vehicles and calculate speed
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        found_vehicle_id = None
        for obj_id in object_tracker:
            prev_center = object_tracker[obj_id]["prev_center"]
            distance_px = math.sqrt((center_x - prev_center[0]) ** 2 + (center_y - prev_center[1]) ** 2)
            if distance_px < 50:  # Threshold for identifying the same vehicle
                found_vehicle_id = obj_id
                break

        if found_vehicle_id is None:
            vehicle_id_counter += 1
            object_tracker[vehicle_id_counter] = {
                "id": vehicle_id_counter,
                "prev_center": (center_x, center_y),
                "speeds": [],
            }
            current_vehicle_id = vehicle_id_counter
        else:
            current_vehicle_id = found_vehicle_id

        # Calculate the speed for the vehicle
        prev_center = object_tracker[current_vehicle_id]["prev_center"]
        distance_px = math.sqrt((center_x - prev_center[0]) ** 2 + (center_y - prev_center[1]) ** 2)
        speed = (distance_px * distance_scale) / (1 / fps) * 3.6  # Speed in km/h

        # Debug: Check if speed is being calculated
        print(f"Vehicle ID: {current_vehicle_id}, Speed: {speed:.2f} km/h")

        # Annotate speed on frame for all vehicles
        cv2.putText(annotated_frame, f"Speed: {speed:.2f} km/h", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Annotate "Speed Exceeded" text if the speed exceeds the threshold
        if speed > speed_threshold:
            cv2.putText(annotated_frame, "Speed Exceeded!", (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw bounding box for the vehicle and display detected number plate
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, detected_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Update vehicle tracker with the current position
        object_tracker[current_vehicle_id]["prev_center"] = (center_x, center_y)

    # Write the processed frame to the output video file
    out.write(annotated_frame)

# Release resources
cap.release()
out.release()

print(f"Processed video saved to: {output_video_path}")




























































