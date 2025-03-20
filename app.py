import cv2
import torch
from ultralytics import YOLO
import numpy as np

# Load the trained YOLOv5 model
model = YOLO("path/yolov5_model.pt")  # Update with your model path - @Zach which one do I use? 
