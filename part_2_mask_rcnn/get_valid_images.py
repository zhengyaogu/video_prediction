from PIL import Image
import numpy as np
from frame_seg_utils import setup
import cv2 as cv
import os
from tqdm import tqdm

VALIDATION_DATA_PATH = "../data/dataset/val"

def get_frames_array(data_folder):
    frames = []
    imgs = os.listdir(data_folder)
    imgs = sorted(
            [img for img in imgs if img.endswith(".png")],
            key = lambda s: int(s.split(".")[0].split("_")[1])
            )
    for img in imgs:
        img_path = os.path.join(data_folder, img)
        img = cv.imread(img_path).transpose(2, 0, 1)
        frames.append(img)

    frames = np.stack(frames, axis=0)
    return frames

if __name__ == "__main__":
    output_folder = "./unlabeled_masks"
    video_dirs = sorted(os.listdir(VALIDATION_DATA_PATH))

    arrays = []
    for video_path in tqdm(video_dirs):
        masks = []
        video_dir = os.path.join(VALIDATION_DATA_PATH, video_path)
        imgs = os.listdir(video_dir)
        imgs = sorted(
            [img for img in imgs if img.endswith(".png")],
            key = lambda s: int(s.split(".")[0].split("_")[1])
            )
        for img in imgs:
            img_path = os.path.join(video_dir, img)
            img = cv.imread(img_path).transpose(2, 0, 1)
            arrays.append(img)
    arrays = np.stack(arrays, axis=0)
    np.save("./validation_imgs.npy", arrays)

