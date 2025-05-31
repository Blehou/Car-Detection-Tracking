About Dataset

- This dataset contains 499 images, each with bounding box annotations for cars.
- The annotations are provided in the YOLO text format, which includes class labels and bounding box coordinates.
- This dataset is useful for object detection tasks such as vehicle recognition and traffic analysis.

# 1. Visualiser les images de train originales avec bounding boxes
python main.py --load

# 2. Générer les images augmentées dans train_augmented/
python main.py --augment --n_aug 3

# 3. Entraîner YOLOv8 sur les données augmentées (déjà définies dans dataset.yaml)
python main.py --train

# 4. Faire des prédictions sur les images de validation
python main.py --predict

# 5. Visualiser les images prédictes dans src/Outcomes/predictions
python main.py --view