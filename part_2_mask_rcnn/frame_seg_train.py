###### PATHS TO USE#####
TRAIN_PATH = "../Dataset_Student_V2/Dataset_Student/train"
VAL_PATH = "../Dataset_Student_V2/Dataset_Student/val"
########################

from frame_seg_utils import *
from detectron2.utils.logger import setup_logger

setup_logger()

import pickle
import numpy as np
import pycocotools.mask
import os
from detectron2.data import (
    MetadataCatalog,
    DatasetCatalog,
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import (
    DefaultTrainer,
    DefaultPredictor,
)
from detectron2.evaluation import DatasetEvaluators, COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
import detectron2.data.transforms as T
from tqdm import tqdm, trange


def get_detectron2_dataset(
    videos_path, frame_num=22, height=160, width=240, num_classes=48, limit=None,
    cache=None, save_to=None
):
    if cache:
        with open(cache, "rb") as f:
            dataset_dicts = pickle.load(f)
        return dataset_dicts

    
    video_dirs = sorted(os.listdir(videos_path))
    if limit:
        video_dirs = video_dirs[:limit]
    total_file_num = len(video_dirs) * frame_num
    dataset_dicts = []
    for idx in trange(total_file_num, desc='preparing dataset in {}'.format(videos_path)):
        video_idx = idx // frame_num
        frame_idx = idx % frame_num
        video_dir = os.path.join(videos_path, video_dirs[video_idx])
        filename = os.path.join(video_dir, f"image_{frame_idx}.png")

        record = {}

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        mask_file = os.path.join(video_dir, "mask.npy")
        mask = None
        if os.path.isfile(mask_file):
            mask = one_hot_encode_mask(np.load(mask_file)[frame_idx]).astype(np.uint8)

        objs = []

        for i in range(num_classes):
            if mask[i].any():
                encoded_mask = pycocotools.mask.encode(np.asarray(mask[i], order="F"))
                obj = {
                    "bbox": get_bounding_box(mask[i]),
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": i,
                    "segmentation": encoded_mask,
                }
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
        
    if save_to:
        with open(save_to, "wb") as f:
            pickle.dump(dataset_dicts, f)
    
    return dataset_dicts


# def get_blur_dataset(
#     videos_path=BLUR_IMAGE_PATH, height=160, width=240, num_classes=48
# ):
#     frames = [f for f in sorted(os.listdir(videos_path)) if f.endswith(".png")]
#     dataset_dicts = []
#     for idx, fname in enumerate(frames):
#         filename = os.path.join(videos_path, fname)
#         record = {}
#         record["file_name"] = filename
#         record["image_id"] = idx
#         record["height"] = height
#         record["width"] = width
#         mask_file = os.path.join(videos_path, "mask_" + fname.split(".")[0] + ".npy")
#         mask = None
#         if os.path.isfile(mask_file):
#             mask = one_hot_encode_mask(np.load(mask_file)).astype(np.uint8)
#             objs = []
#             for i in range(num_classes):
#                 if mask[i].any():
#                     encoded_mask = pycocotools.mask.encode(
#                         np.asarray(mask[i], order="F")
#                     )
#                     obj = {
#                         "bbox": get_bounding_box(mask[i]),
#                         "bbox_mode": BoxMode.XYWH_ABS,
#                         "category_id": i,
#                         "segmentation": encoded_mask,
#                     }
#                     objs.append(obj)
#             record["annotations"] = objs
#             dataset_dicts.append(record)
#     return dataset_dicts


# def get_datasets(main_dataset_path, limit=None):
#     d1 = get_detectron2_dataset(main_dataset_path, frame_num=11, limit=limit)
#     d2 = get_blur_dataset()
#     print(len(d1), len(d2))
#     return d1 + d2


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder="./output"):
        coco_evaluator = COCOEvaluator(dataset_name, output_dir=output_folder)
        evaluator_list = [coco_evaluator]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg,
            mapper=DatasetMapper(
                cfg,
                is_train=True,
                augmentations=[
                    T.RandomBrightness(0.9, 1.1),
                    # T.RandomFlip(prob=0.5, vertical=True, horizontal=False),
                    # T.RandomFlip(prob=0.5, vertical=False, horizontal=True),
                ],
            ),
        )


def detectron_train(
    cfg, 
    train_data_cache="./train_dataset.pkl", 
    test_data_cache="./val_dataset.pkl", 
    resume=False
):
    DatasetCatalog.clear()
    for d in ["train", "val"]:
        p = TRAIN_PATH if d == "train" else VAL_PATH
        c = train_data_cache if d == "train" else test_data_cache
        DatasetCatalog.register(
            "dataset_" + d,
            lambda d=d: get_detectron2_dataset(p, cache=c),
        )

        MetadataCatalog.get("dataset_" + d).set(
            thing_classes=[str(x) for x in range(48)]
        )
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    return trainer.train()


def evaluate_model():
    cfg = setup()
    cfg.MODEL.WEIGHTS = os.path.join(
        cfg.OUTPUT_DIR, "model_final.pth"
    )  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    for d in ["val"]:
        DatasetCatalog.register(
            "dataset_" + d,
            lambda d=d: get_detectron2_dataset(VAL_PATH + d),
        )

        MetadataCatalog.get("dataset_" + d).set(
            thing_classes=[str(x) for x in range(48)]
        )
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("dataset_val", output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "dataset_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))


if __name__ == "__main__":
    config = setup()
    detectron_train(
        config
    )

