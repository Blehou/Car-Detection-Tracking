import os
import cv2
import numpy as np
import tensorflow as tf

from Preprocessing.dataLoader import load_data

spth = "C:/Jean Eudes Folder/_Projects/Computer_Vision_Project/CarDetection&Tracking/src/Dataset/train_augmented"

def preprocess_data(images: list) -> np.ndarray:
    """
    Applies preprocessing steps to input images:
    - Resize to a fixed shape (640x640)
    - Normalize pixel values to [0, 1]

    Args:
        images (list): List of input images as NumPy arrays.

    Returns:
        np.ndarray: Array of preprocessed images.
    """
    processed = []
    target_size = (640, 640)

    for img in images:
        resized = cv2.resize(img, target_size)
        normalized = resized / 255.0
        processed.append(normalized)

    return np.array(processed, dtype=np.float32)


def save_yolo_label(label_list, save_path):
    """
    Save label list to a YOLO-format .txt file
    """
    with open(save_path, 'w') as f:
        for ann in label_list:
            cls = ann["class"]
            bbox = ann["bbox"]
            line = f"{cls} {' '.join(map(str, bbox))}\n"
            f.write(line)


def augment_data(images: list, labels: list, n_aug: int = 2, output_dir: str = spth):
    """
    Augments and saves images and labels to disk.

    Args:
        images (list): List of original images (np.array).
        labels (list): List of labels in YOLO format.
        n_aug (int): Number of augmentations per image.
        output_dir (str): Where to save images and labels.
    """
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    idx = 0

    for img, label in zip(images, labels):
        # Save original image and label
        filename = f"aug_{idx:05d}.jpg"
        cv2.imwrite(os.path.join(images_dir, filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        save_yolo_label(label, os.path.join(labels_dir, filename.replace(".jpg", ".txt")))
        idx += 1

        for i in range(n_aug):
            aug_img = tf.convert_to_tensor(img, dtype=tf.float32) / 255.0

            # Apply augmentations
            aug_img = tf.image.random_flip_left_right(aug_img)
            aug_img = tf.image.random_brightness(aug_img, max_delta=0.2)
            aug_img = tf.image.random_contrast(aug_img, lower=0.8, upper=1.2)
            aug_img = tf.image.random_saturation(aug_img, lower=0.8, upper=1.2)
            aug_img = tf.image.random_hue(aug_img, max_delta=0.05)

            aug_img = tf.clip_by_value(aug_img, 0.0, 1.0)
            aug_img = (aug_img.numpy() * 255).astype(np.uint8)

            filename = f"aug_{idx:05d}.jpg"
            cv2.imwrite(os.path.join(images_dir, filename), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
            save_yolo_label(label, os.path.join(labels_dir, filename.replace(".jpg", ".txt")))

            idx += 1

    print(f"Saved {idx} images and labels to '{output_dir}'")


if __name__ == "__main__":
    data = load_data("train")
    augment_data(data["feature"], data["label"], n_aug=2)