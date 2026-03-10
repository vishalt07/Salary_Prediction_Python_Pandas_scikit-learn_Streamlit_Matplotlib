import os
import random
import shutil
from sklearn.model_selection import train_test_split

dataset_dir = "dataset"
output_dir = "dataset_split"

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

classes = os.listdir(dataset_dir)

for cls in classes:
    cls_path = os.path.join(dataset_dir, cls)
    images = os.listdir(cls_path)

    train_imgs, temp_imgs = train_test_split(images, test_size=(1-train_ratio), random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=test_ratio/(test_ratio+val_ratio), random_state=42)

    for folder in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, folder, cls), exist_ok=True)

    for img in train_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(output_dir, 'train', cls))

    for img in val_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(output_dir, 'val', cls))

    for img in test_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(output_dir, 'test', cls))

print("Dataset Split Completed!")
