###### PATHS TO USE#####
INFERENCE_OUTPUT_PATH = "./inference_output_val.npy"
MODEL_WEIGHTS_PATH = "./output/model_final.pth"
INPUT_DATA_PATH = "./validation_imgs.npy"
VALIDATION_DATA_PATH = "../data/dataset/val"
UNLABELED_DATA_PATH = "../data/hidden"
########################

from frame_seg_utils import setup
from get_valid_images import get_frames_array

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

import os
from tqdm import tqdm
import numpy as np
import torchmetrics
import torch
import math

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")


def load_model():
    cfg = setup()
    cfg.MODEL.WEIGHTS = MODEL_WEIGHTS_PATH
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    model = build_model(cfg)
    return model


def load_input_data(scale=True):
    # for inference data from the video segmetnation model, values are 0-1
    # thus multiply by 255
    data = np.load(INPUT_DATA_PATH)
    if scale:
        return data * 255
    return data


def infer(input_data, batch_size=20):
    model = load_model()
    DetectionCheckpointer(model).load(MODEL_WEIGHTS_PATH)
    full_data_list = [
        {"image": torch.tensor(input_data[x]).to(DEVICE).float()}
        for x in range(len(input_data))
    ]

    # run inference in batches
    model = model.cuda()
    model.eval()
    masks = []
    for i in tqdm(range(math.ceil(len(full_data_list) / batch_size))):
        with torch.no_grad():
            if i != (len(full_data_list) // batch_size):
                output = model(
                    full_data_list[i * batch_size : i * batch_size + batch_size]
                )
            else:
                output = model(full_data_list[i * batch_size :])
            
            for out in output:
                mask = torch.zeros((160, 240)).to(DEVICE)
                for idx, c in enumerate(out["instances"].pred_classes.tolist()):
                    class_mask = out["instances"].pred_masks[idx]
                    mask = torch.where(class_mask, c + 1, mask)
                masks.append(mask.cpu())

    all_preds = torch.stack(masks)
    return all_preds


def load_true_masks(videos_path=VALIDATION_DATA_PATH):
    video_dirs = sorted(os.listdir(videos_path))
    trues = []
    for idx, v in tqdm(enumerate(video_dirs)):
        trues.append(np.load(os.path.join(videos_path, v, "mask.npy")))
    return torch.tensor(np.concatenate(trues, axis=0))


def validation(pred, true, batch_size=1000):
    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49).to(DEVICE)
    for i in range(pred.shape[0] // batch_size):
        l, r = i * batch_size, min((i + 1) * batch_size, pred.shape[0])
        p = pred[l: r].to(DEVICE)
        t = true[l: r].to(DEVICE)
        jaccard(p, t)
    iou = jaccard.compute()
    print(f"IOU calculated {iou.item()}")


if __name__ == "__main__":
    print("loading input data...")
    in_data = load_input_data(scale=False)
    print("inferring...")
    out_data = infer(in_data, batch_size=5)
    print("saving predictions...")
    np.save(INFERENCE_OUTPUT_PATH, out_data.numpy())
    out_data = torch.from_numpy(np.load("./inference_output_val.npy"))
    true_data = load_true_masks()
    validation(out_data, true_data)
    """
    unlabeled_video_dirs = os.listdir(UNLABELED_DATA_PATH)
    pbar = tqdm(unlabeled_video_dirs, leave=True)
    for video_folder in pbar:
        pbar.set_description(video_folder)
        video_path = os.path.join(UNLABELED_DATA_PATH, video_folder) 
        frames = get_frames_array(video_path)
        pred_mask = infer(frames, batch_size=6)

        # store the inferred mask 
        #out_path = os.path.join("./test_masks", video_folder)
        #os.makedirs(out_path, exist_ok=True)
        filename = os.path.join("./test_masks", "{}.npy".format(video_folder))
        np.save(filename, pred_mask.numpy())

        del pred_mask
    """
