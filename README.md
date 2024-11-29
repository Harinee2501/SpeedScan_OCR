# SpeedScan_OCR - Speed Violation and Number Plate Detection

## Problem Statement
The objective of this project is to develop a system that detects the speed of vehicles in real-time using video footage and recognizes their number plates. This system aims to automatically send notifications (like emails or messages) to individuals whose speed exceeds the speed limit, based on the number plate detected. This can help in road safety enforcement, monitoring, and traffic law compliance.

## Project Idea
The project integrates two crucial components:
1. **Vehicle Speed Detection**: Detecting and calculating the speed of vehicles in a video feed.
2. **Number Plate Recognition**: Recognizing the vehicle’s number plate and extracting the license plate number using OCR (Optical Character Recognition).
  
By combining these components, the system can track vehicles in real-time, measure their speed, identify their number plates, and send alerts when a vehicle exceeds a predefined speed limit.


## Approach
The project is implemented using a YOLO (You Only Look Once) object detection model to detect vehicles and number plates in real-time video feeds. For speed calculation, the system tracks the movement of vehicles across frames, using the distance between the vehicle's previous and current position and the video frame rate to estimate speed.

### Key Steps:
1. **YOLO Model for Object Detection**: Trains a custom YOLOv8 model to detect vehicles and number plates in the video.
2. **OCR for Number Plate Recognition**: Uses PaddleOCR to extract the number plate information from the detected regions.
3. **Vehicle Speed Calculation**: Based on vehicle movement between frames, calculates the speed using a scaling factor and video frame rate.
4. **Alert System**: If the detected speed exceeds a set threshold, the system annotates the video frame with "Speed Exceeded" and stores the relevant data for further action (like sending a notification).

## Tech Stack
- **YOLOv8 (Ultralytics)**: For object detection (vehicle and number plate recognition).
- **PaddleOCR**: For optical character recognition (OCR) to read vehicle number plates.
- **OpenCV**: For video processing, vehicle tracking, and speed calculation.
- **Python**: Primary programming language.
- **TensorFlow/PyTorch**: For machine learning models (YOLO model training).
- **Matplotlib/Seaborn**: For visualizations (if needed).
  
## Progress / Status
- **Model Training**: The YOLOv8 model is trained on custom data to detect vehicles and number plates.
- **Real-time Processing**: Implemented real-time vehicle tracking, speed detection, and number plate recognition in video streams.
- **Speed Calculation**: Speed detection is implemented with an adjustable scaling factor, and the system successfully detects speeds above a defined threshold.
- **Notification System**: Initial setup for detecting speed violations, but still in the testing phase for automated notifications via email/SMS.

## Test Videos and Output Link
https://drive.google.com/drive/folders/1TSA2hLEcXESO1Tfj3P_Y6QU4I9mWHBV-?usp=sharing

## Output Screenshots
![Screenshot 2024-11-29 212731](https://github.com/user-attachments/assets/3bf4da8b-c8f1-4f43-9d2b-bd01a63c10af)
![Screenshot 2024-11-29 212702](https://github.com/user-attachments/assets/ae2481dd-832e-4624-997d-8b9290184339)

### Challenges Faced:
1. **Different Camera Angles**: To accurately detect the speed of vehicles, multiple camera angles are required for capturing vehicles in motion. A single camera angle might miss some vehicles or fail to track them properly due to occlusions.
2. **Smoothing Speed Detection**: The speed detection requires fine-tuning of the smoothing factor to ensure that speed readings are accurate and not erratic. A proper smoothing algorithm is essential for reducing noise in the detected speeds.
3. **No Proper Datasets**: There is no comprehensive public dataset that contains both vehicle number plates and corresponding speed information. This makes training and evaluation challenging. Custom datasets were created and augmented, but the lack of large-scale labeled data remains a limitation.

## Future Scope
- **Real-time Notification System**: The goal is to automate the process of sending notifications (emails/SMS) to the vehicle owners by linking their number plates with a database that contains contact details.
- **Multiple Camera Integration**: Implementing multiple camera angles for better accuracy in vehicle tracking and speed detection.
- **Advanced Speed Detection Algorithms**: Exploring advanced techniques like Kalman filters for more accurate and stable speed estimation.
- **Database Integration**: Integrating a database to track vehicles, their speeds, and owner information to automate ticketing or violation reporting.
- **Scalability**: Developing the system for large-scale deployment in cities with high traffic, using cloud services for video storage, model inference, and notification management.

---
### Team Members
- **[M Harinee](https://github.com/Harinee2501)**
- **[Indhumathi Sivashanmugam](https://github.com/Indhumathi-SivaShanmugam)**  
---
