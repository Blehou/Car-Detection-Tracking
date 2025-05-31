import os
import cv2
import matplotlib.pyplot as plt


def fetch_images(path: str, directory: str) -> list:
    """
    Loads all images from a given directory using OpenCV and converts them to RGB format.

    Args:
        path (str): Base path to the dataset.
        directory (str): Subdirectory containing the images.

    Returns:
        list: A list of images as NumPy arrays in RGB format.
    """
    images = []
    _path = os.path.join(path, directory)

    if not os.path.exists(_path):
        print(f"Error: Directory '{_path}' not found.")
        return []

    for element in os.listdir(_path):
        path_im = os.path.join(_path, element)
        images_cv2 = cv2.imread(path_im, cv2.IMREAD_UNCHANGED)

        if images_cv2 is None:
            print(f"Warning: Unable to load image {path_im}. Skipping...")
            continue

        images_cv2 = cv2.cvtColor(images_cv2, cv2.COLOR_BGR2RGB)
        images.append(images_cv2)

    return images


def fetch_labels(path: str, directory: str) -> list:
    """
    Loads YOLO-format label files and parses bounding box annotations.

    Each label file should contain lines in the format:
    <class_id> <x_center> <y_center> <width> <height>

    Args:
        path (str): Base path to the dataset.
        directory (str): Subdirectory containing the label files (.txt).

    Returns:
        list: A list of label lists, where each label list contains dicts
              with 'class' and 'bbox' keys.
    """
    labels = []
    _path = os.path.join(path, directory)

    if not os.path.exists(_path):
        print(f"Error: Directory '{_path}' not found.")
        return []

    for file in os.listdir(_path):
        if not file.endswith(".txt"):
            continue
        label_path = os.path.join(_path, file)

        with open(label_path, 'r') as f:
            annotations = []
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, x_center, y_center, width, height = map(float, parts)
                    annotations.append({
                        "class": int(cls),
                        "bbox": [x_center, y_center, width, height]
                    })
            labels.append(annotations)

    return labels


def load_data(_type: str = "train") -> dict:
    """
    Loads images and corresponding label annotations from the dataset.

    Args:
        _type (str, optional): Dataset type to load, either "train" or "validation".
                               Defaults to "train".

    Returns:
        dict: A dictionary with:
              - "feature": List of loaded images as NumPy arrays.
              - "label": List of corresponding label annotations.
    """
    output = {
        "feature": [],
        "label": []
    }

    base_path = r"C:\Jean Eudes Folder\_Projects\Computer_Vision_Project\CarDetection&Tracking\src\Dataset"
    if _type == "train":
        image_dir = os.path.join(base_path, "train", "images")
        label_dir = os.path.join(base_path, "train", "labels")
    elif _type == "validation":
        image_dir = os.path.join(base_path, "val", "images")
        label_dir = os.path.join(base_path, "val", "labels")
    else:
        print("Error: Dataset type must be 'train' or 'validation'")
        return output

    images = fetch_images(base_path, os.path.join(_type, "images"))
    labels = fetch_labels(base_path, os.path.join(_type, "labels"))

    output["feature"] = images
    output["label"] = labels

    print(f"Loaded {len(images)} images and {len(labels)} label files for {_type}")
    return output


def visualize_data(data: dict, rows: int = 3, cols: int = 4) -> None:
    """
    Displays a grid of images with their YOLO-format bounding boxes overlaid.

    Args:
        data (dict): Dictionary containing "feature" and "label" keys.
        rows (int, optional): Number of rows in the grid. Defaults to 3.
        cols (int, optional): Number of columns in the grid. Defaults to 4.
    """
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle("Car Detection Samples", fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i >= len(data["feature"]):
            break

        img = data["feature"][i].copy()
        labels = data["label"][i]

        height, width, _ = img.shape
        for annotation in labels:
            cls = annotation["class"]
            x, y, w, h = annotation["bbox"]

            x1 = int((x - w / 2) * width)
            y1 = int((y - h / 2) * height)
            x2 = int((x + w / 2) * width)
            y2 = int((y + h / 2) * height)

            cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
            cv2.putText(img, f"Car", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        ax.imshow(img)
        ax.axis('off')

    save_path = r"C:\Jean Eudes Folder\_Projects\Computer_Vision_Project\CarDetection&Tracking\src\Outcomes\sample_visualization.jpg"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == "__main__":
    data = load_data("train")
    visualize_data(data)
