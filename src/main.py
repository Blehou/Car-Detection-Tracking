import argparse
from Preprocessing.dataLoader import load_data, visualize_data
from Preprocessing.preprocessing import augment_data
from Models.yolov8 import train_yolov8, predict_and_save, plot_predictions


def main():
    parser = argparse.ArgumentParser(description="Car Detection & Tracking - Main Pipeline")
    parser.add_argument("--load", action="store_true", help="Load and visualize original training data")
    parser.add_argument("--n_aug", type=int, default=2, help="Number of augmentations per image (used only with --augment)")
    parser.add_argument("--augment", action="store_true", help="Augment and save training data to train_augmented/")
    parser.add_argument("--train", action="store_true", help="Train YOLOv8 using dataset.yaml (points to augmented data)")
    parser.add_argument("--predict", action="store_true", help="Run predictions on validation set")
    parser.add_argument("--view", action="store_true", help="View predicted images from Outcomes")
    args = parser.parse_args()

    if args.load:
        print("Loading and visualizing training data...")
        data = load_data("train")
        visualize_data(data)

    if args.augment:
        print("Augmenting and saving data to train_augmented/...")
        data = load_data("train")
        augment_data(data["feature"], data["label"], n_aug=args.n_aug)

    if args.train:
        print("Training YOLOv8 using dataset.yaml...")
        train_yolov8()  # dataset.yaml already points to train_augmented/images

    if args.predict:
        print("Predicting on validation set...")
        predict_and_save()

    if args.view:
        print("Displaying predicted results...")
        plot_predictions(result_dir="C:/Jean Eudes Folder/_Projects/Computer_Vision_Project/CarDetection&Tracking/src/Outcomes/predictions", num_images=6, images_per_row=3)


if __name__ == "__main__":
    main()
