
import torch
from detectron2 import config, modeling
from detectron2.checkpoint import DetectionCheckpointer

from lightly.data import LightlyDataset
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms import SimCLRTransform

import numpy as np
import pycocotools.mask
from detectron2 import model_zoo
from detectron2.config import get_cfg
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from tqdm import tqdm


num_workers = 4
batch_size = 600
input_size = 128
num_ftrs = 2048

seed = 1
max_epochs = 1000

# use cuda if possible
device = "cuda" if torch.cuda.is_available() else "cpu"

data_path = "../data/dataset/unlabeled"
cfg_path = "./Base-RCNN-FPN.yaml"

class SelectStage(torch.nn.Module):
    """Selects features from a given stage."""

    def __init__(self, stage: str = "res5"):
        super().__init__()
        self.stage = stage

    def forward(self, x):
        return x[self.stage]

def setup():
    cfg = config.get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
            cfg_path,
    ))

    # use cuda if possible
    cfg.MODEL.DEVICE = device

    # randomly initialize network
    cfg.MODEL.WEIGHTS = ""

    # detectron2 uses BGR by default but pytorch/torchvision use RGB
    cfg.INPUT.FORMAT = "RGB"
    return cfg

if __name__ == "__main__":
    print("prep dataset...")
    transform = v2.Compose([
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True)
    ])
    transform = SimCLRTransform(input_size=input_size)
    ds = ImageFolder(
                    data_path,
                    transform=transform
    )

    print("building model...")
    cfg = setup()
    detmodel = modeling.build_model(cfg)

    simclr_backbone = torch.nn.Sequential(
        detmodel.backbone.bottom_up,
        SelectStage("res5"),
        # res5 has shape bsz x 2048 x 4 x 4
        torch.nn.AdaptiveAvgPool2d(1),
    ).to(device)
    projection_head = SimCLRProjectionHead(
        input_dim=num_ftrs,
        hidden_dim=num_ftrs,
        output_dim=128,
    ).to(device)

    print("prep loader...")
    dataset_train_simclr = LightlyDataset.from_torch_dataset(ds, transform=transform)

    dataloader_train_simclr = torch.utils.data.DataLoader(
        dataset_train_simclr,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )

    print("loss & optmizer...")
    criterion = NTXentLoss()
    optimizer = torch.optim.Adam(
        list(simclr_backbone.parameters()) + list(projection_head.parameters()),
        lr=1e-4,
    )
    pbar = tqdm(range(max_epochs), leave=True)

    print("training...")
    for e in pbar:
        mean_loss = 0.0
        for (x0, x1), _, _ in tqdm(dataloader_train_simclr):
            x0 = x0.to(device)
            x1 = x1.to(device)

            y0 = projection_head(simclr_backbone(x0).flatten(start_dim=1))
            y1 = projection_head(simclr_backbone(x1).flatten(start_dim=1))

            # backpropagation
            loss = criterion(y0, y1)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # update average loss
            mean_loss += loss.detach().cpu().item() / len(dataloader_train_simclr)
        pbar.set_postfix({"mean_loss": mean_loss})
        
        detmodel.backbone.bottom_up = simclr_backbone[0]

        checkpointer = DetectionCheckpointer(detmodel, save_dir="./pretrained_models")
        checkpointer.save("model_{}".format(str(e).zfill(7)))


