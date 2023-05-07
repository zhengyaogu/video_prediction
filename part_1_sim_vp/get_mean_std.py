import os
import glob
import numpy as np
from PIL import Image

def get_mean_std(root_dir, data_types, num_samples=13996):
    img_paths = []
    for data_type in data_types:
        img_paths.extend(sorted(glob.glob(os.path.join(root_dir, data_type, 'video_*', 'image_*.png'))))

    # Randomly sample a subset of images
    np.random.seed(42)
    sampled_img_paths = np.random.choice(img_paths, num_samples, replace=False)

    # Compute mean and standard deviation for each channel
    mean = np.zeros(3)
    std = np.zeros(3)

    for img_path in sampled_img_paths:
        img = np.array(Image.open(img_path), dtype=np.float32) / 255.0
        mean += img.mean(axis=(0, 1))
        std += img.std(axis=(0, 1))

    mean /= num_samples
    std /= num_samples

    return mean, std

root_dir = '/home/mr6555/scratch/dl_project_dataset/Dataset_Student'
data_types = ['train', 'unlabeled']
mean, std = get_mean_std(root_dir, data_types)

print("Mean:", mean)
print("Standard Deviation:", std)