About Dataset

- This dataset contains 499 images, each with bounding box annotations for cars.
- The annotations are provided in the YOLO text format, which includes class labels and bounding box coordinates.
- This dataset is useful for object detection tasks such as vehicle recognition and traffic analysis.

--

## Steps

1. View the original train images with bounding boxes
python main.py --load

2. Generate the augmented images in train_augmented/
python main.py --augment --n_aug 3

3. Train YOLOv8 on the augmented data (already defined in dataset.yaml)
python main.py --train

4. Make predictions on the validation images
python main.py --predict

5. View the predicted images in src/Outcomes/predictions
python main.py --view
