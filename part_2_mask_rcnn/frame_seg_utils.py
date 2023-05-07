import numpy as np
import pycocotools.mask
from detectron2 import model_zoo
from detectron2.config import get_cfg


def setup():
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        )
    )
    cfg.DATASETS.TRAIN = ("dataset_train",)
    cfg.DATASETS.TEST = ("dataset_val",)
    cfg.TEST.EVAL_PERIOD = 5000
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 60
    cfg.SOLVER.BASE_LR = 0.0005
    cfg.SOLVER.MAX_ITER = 42000  # 1 iter = 1 batch run
    cfg.OUTPUT_DIR = "./42k_blur_frameskip_output/"  # choose output dir
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 48
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.MODEL.DEVICE = "cuda"

    # if we want to load a checkpoint
    # cfg.MODEL.WEIGHTS = "/content/drive/MyDrive/101_42k_blur_frameskip_output/model_0004999.pth/model_0004999.pth"
    return cfg


def one_hot_encode_mask(mask):
    data_point = []
    for x in range(1, 49):  # skip 0 because 0 is background
        data_point.append(np.where(mask == x, 1.0, 0.0))
    split_mask = np.stack(data_point)
    return split_mask


def one_hot_encode_mask_batch(mask_batch):
    # input dim: batch, height, width
    all_masks = []
    for i in range(mask_batch.shape[0]):
        single_mask_split = one_hot_encode_mask(mask_batch[i])
        all_masks.append(single_mask_split)
    stacked = np.stack(all_masks)
    return stacked


def get_bounding_box(binary_mask):
    fortran_ground_truth_binary_mask = np.asfortranarray(binary_mask)
    encoded_ground_truth = pycocotools.mask.encode(fortran_ground_truth_binary_mask)
    return pycocotools.mask.toBbox(encoded_ground_truth).tolist()
