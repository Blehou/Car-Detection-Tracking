from ultralytics import YOLO
import os
import cv2
import matplotlib.pyplot as plt
import math
import numpy as np


def train_yolov8():
    """
    Trains a YOLOv8 model using the dataset and saves the results.
    """
    model = YOLO('yolov8n.yaml')  # or 'yolov8n.pt' if you want to fine-tune a pre-trained model

    model.train(
        data="C:/Jean Eudes Folder/_Projects/Computer_Vision_Project/CarDetection&Tracking/src/Models/dataset.yaml",
        epochs=20,
        imgsz=640,
        batch=64,
        project="C:/Jean Eudes Folder/_Projects/Computer_Vision_Project/CarDetection&Tracking/src/Outcomes",
        name="yolov8-car-detection",
        exist_ok=True
    )


def predict_and_save():
    """
    Loads a trained model and performs inference on validation images.
    Saves the predicted images with bounding boxes.
    """
    model = YOLO("C:/Jean Eudes Folder/_Projects/Computer_Vision_Project/CarDetection&Tracking/src/Outcomes/yolov8-car-detection/weights/best.pt")

    val_dir = r"C:\Jean Eudes Folder\_Projects\Computer_Vision_Project\CarDetection&Tracking\src\Dataset\val\images"
    images = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith(".jpg") or f.endswith(".png")]

    model.predict(
        source=images,
        save=True,
        project="C:/Jean Eudes Folder/_Projects/Computer_Vision_Project/CarDetection&Tracking/src/Outcomes",
        name="predictions",
        exist_ok=True
    )

    print("Predictions saved in 'src/Outcomes/predictions'")


def plot_predictions(result_dir: str, num_images: int = 6, images_per_row: int = 3):
    """
    Displays a grid of predicted images from a result directory.

    Args:
        result_dir (str): Directory containing predicted images (e.g. from YOLO save=True).
        num_images (int): Number of images to display. Defaults to 6.
        images_per_row (int): Number of images per row in the grid. Defaults to 3.
    """
    all_files = [f for f in os.listdir(result_dir) if f.endswith(".jpg") or f.endswith(".png")]
    all_files = sorted(all_files)[:num_images]

    if not all_files:
        print(f"No images found in {result_dir}")
        return

    rows = math.ceil(num_images / images_per_row)
    fig, axes = plt.subplots(rows, images_per_row, figsize=(5 * images_per_row, 5 * rows))

    if rows == 1:
        axes = [axes]
    axes = [ax for row in axes for ax in (row if isinstance(row, (list, np.ndarray)) else [row])]

    for ax, filename in zip(axes, all_files):
        img_path = os.path.join(result_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax.imshow(img)
        ax.set_title(filename)
        ax.axis('off')

    for ax in axes[len(all_files):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
