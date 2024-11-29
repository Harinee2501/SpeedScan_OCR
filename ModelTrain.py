# Step 2: Import the necessary libraries
import os
from google.colab import drive
from ultralytics import YOLO


# Step 4: Define paths
train_images_path = '/content/drive/MyDrive/extracted_data/plate_Ds/images/train'
val_images_path = '/content/drive/MyDrive/extracted_data/plate_Ds/images/val'
data_yaml_path = '/content/drive/MyDrive/extracted_data/plate_Ds/data.yaml'

# Check if paths exist
if not os.path.exists(train_images_path):
    raise FileNotFoundError(f"Train images path not found: {train_images_path}")
if not os.path.exists(val_images_path):
    raise FileNotFoundError(f"Validation images path not found: {val_images_path}")
if not os.path.exists(data_yaml_path):
    raise FileNotFoundError(f"data.yaml file not found: {data_yaml_path}")

# Step 5: Define and train the YOLOv8 model
# Load the YOLOv8 model (use a pretrained model for fine-tuning)
model = YOLO('yolov8n.pt')  # You can choose different model variants ('yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt')

# Train the model
model.train(
    data=data_yaml_path,  # Path to your dataset.yaml file
    epochs=40,  # Number of epochs for training
    batch=16,   # Batch size for training
    imgsz=640,  # Image size (adjust based on your GPU)
    project='/content/drive/MyDrive/extracted_data/plate_Ds/runs/train',  # Directory to save training results
    name='yolov8_plate_detection',  # Name of the training session
    exist_ok=True  # Allow overwriting if there's a previous training run
)

# Step 6: Once training is done, the model weights will be saved under 'runs/train/yolov8_plate_detection/weights/'
