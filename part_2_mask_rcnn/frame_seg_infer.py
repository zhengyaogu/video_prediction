###### PATHS TO USE#####
INFERENCE_OUTPUT_PATH = "./inference_output.npy"
MODEL_WEIGHTS_PATH = "/content/drive/MyDrive/101_42k_blur_frameskip_output/model_0004999.pth/model_0004999.pth"
INPUT_DATA_PATH = "/content/drive/MyDrive/simvp_final/final_validation_predictions.npy"
VALIDATION_DATA_PATH = "/content/Dataset_Student/val"
########################

from frame_seg_utils import setup

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

import os
from tqdm import tqdm
import numpy as np
import torchmetrics
import torch

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
    model.eval()
    outputs = []
    for i in tqdm(range(len(full_data_list) // batch_size)):
        with torch.no_grad():
            if i != (len(full_data_list) // batch_size):
                output = model(
                    full_data_list[i * batch_size : i * batch_size + batch_size]
                )
            else:
                output = model(full_data_list[i * batch_size :])
            outputs.append(output)

    # flatten list
    all_outs = [item for sublist in outputs for item in sublist]

    # flatten binary masks into expected format
    masks = []
    for out in all_outs:
        mask = torch.zeros((160, 240)).to(DEVICE)
        for idx, c in enumerate(out["instances"].pred_classes.tolist()):
            class_mask = out["instances"].pred_masks[idx]
            mask = torch.where(class_mask, c + 1, mask)
        masks.append(mask)

    all_preds = torch.stack(masks).to(DEVICE)
    return all_preds


def load_true_masks(videos_path=VALIDATION_DATA_PATH):
    video_dirs = sorted(os.listdir(videos_path))
    trues = []
    for idx, v in tqdm(enumerate(video_dirs)):
        trues.append(np.load(os.path.join(videos_path, v, "mask.npy"))[-1])
    return torch.tensor(np.stack(trues)).to(DEVICE)


def validation(pred, true):
    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49).to(DEVICE)
    iou = jaccard(pred, true)
    print(f"IOU calculated {iou.item()}")


in_data = load_input_data()
out_data = infer(in_data)
np.save(INFERENCE_OUTPUT_PATH, out_data.cpu().numpy())
# true_data = load_true_masks()
# validation(out_data, true_data)
