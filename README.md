# Advanced Video Masking: Integrating SpatioTemporal Models with Mask R-CNN

## Abstract

We present a novel approach for generating a semantic segmentation mask for the 22nd frame of a video sequence, given the first 11 frames. Our method integrates state-of-the-art models, SimVP and ConvLSTM, for Video Frame Prediction (VFP), and Mask R-CNN for Semantic Segmentation Mask (SSM). We trained the models separately, explored various combinations, and implemented different data augmentation techniques. By combining SimVP with Mask R-CNN and using blur and frameskip data augmentation, we achieved an Intersection over Union (IOU) score of 0.1123 on the validation dataset.

## Part - 1 (Video Frame Prediction) - SimVP: Video Prediction using a Spatio-temporal Model

This repository contains the implementation of a video prediction model called SimVP. The model is trained on a custom dataset and can predict the future frames of a video given an initial sequence of frames.

## Prerequisites

- Python 3.x
- PyTorch
- torchvision
- NumPy
- imageio

## Setup

1. Create a virtual environment:

   ```bash
   python3 -m venv venv

2. Activate the virtual environment:

    On Linux or macOS:

    ```bash
    source venv/bin/activate

    On Windows:
   
    ```bash
    .\venv\Scripts\activate

3. Install the required packages:

    ```bash
    pip install -r requirements.txt

## Dataset

Place your dataset in the appropriate directory. The default dataset path is `/scratch/mr6555/dl_project_dataset/Dataset_Student`. You can change this in `main.py` (for training) and `pred_yan.py` (for prediction).

Dataset have the followring structure

    Dataset
    ├── train
    │   ├── input
    │   └── target
    ├── val
    │   ├── input
    │   └── target
    └── hidden
        ├── input

## Usage

1. First, run get_mean_std.py to calculate the mean and standard deviation of your dataset. This will be used for normalization.

   ```bash
   python get_mean_std.py
   ```
2. Train the model using sim_vp.sbatch. Before running the script, update the paths in the script to match your environment.

   ```bash
   sbatch sim_vp.sbatch
   ```

3. After training, use the trained model to generate predictions by running pred_yan.sbatch. Make sure to update the paths in the script to match your environment.

   ```bash
   sbatch pred_yan.sbatch
   ```

4. Update the path to the saved model weights, dataset path, and the save directory in the `generate_blur_images.py` script. Run the SLURM script `generate_blur_image.sbatch` to generate blur images and save the corresponding masks.

   ```bash
   generate_blur_image.sbatch
   ```
   

## Code Structure

- `data_preparation.py`: Contains the implementation of the CustomVideoDataset class for loading the dataset.
- `model.py`: Contains the implementation of the SimVP model.
- `get_mean_std.py`: A script to calculate the mean and standard deviation of the dataset.
- `main.py`: The main script to train the SimVP model.
- `prediction.py`: Contains the implementation of the VisualizePredictions class to generate and visualize predictions using the trained model.
- `pred_yan.py`: A script to generate predictions using the trained model.
- `sim_vp.sbatch`: SLURM script to train the model.
- `pred_yan.sbatch`: SLURM script to generate predictions using the trained model.
- `generate_blur_images.py`: Contains the implementation of the code to generate blur images and save the corresponding masks for the video sequences in the dataset.
- `generate_blur_image.sbatch`: SLURM script to generate blur images and save the corresponding masks using the trained model.

## Customization

To use this code with your own dataset, make the following changes:

1. Update the dataset paths in `main.py` and `pred_yan.py`.
2. Update the paths in the SLURM scripts (`sim_vp.sbatch` and `pred_yan.sbatch`) to match your environment.
3. Modify the CustomVideoDataset class in `data_preparation.py` if necessary to accommodate differences in dataset structure or format.
4. Adjust the hyperparameters in `main.py` and `model.py` as needed.
5. Update the dataset paths in `generate_blur_images.py`.
6. Update the paths in the SLURM script `generate_blur_image.sbatch` to match your environment.


## Output

The predictions will be saved as NumPy arrays and visualized as GIFs in the specified directories. Update the paths in `pred_yan.py` to save the predictions and visualizations in your desired location.

The blur images and corresponding masks will be saved in the specified directory as PNG and NumPy array files, respectively. Update the paths in `generate_blur_images.py` to save the generated images and masks in your desired location.


## Part - 2 (Mask Prediction)

This repository contains the code used to train an image segmentation model. The Mask RCNN model imported is from the Detectron2 library, and trained from scratch to predict segmentation of frames. The frames 1-11 in original dataset is used as training data, as well as the generated frames 12-22 from the video prediction model. 

## Prerequisites

- PyTorch
- Numpy
- Python 3.x
- Detectron2
- torchmetrics (for validation)

## Dataset

The data from Dataset_Student.zip is processed directly after unzipping.
The inference output from the model should be in a folder with the frame image in the format `video_{video idx}_frame_{frame idx}.png` and mask named with the format`mask_video_{video idx}_frame_{idx}.npy`. While the images are generated by the video pred model, the mask is the same as the original dataset.

## Usage

1. Install pytorch, numpy and torchmetrics. Refer to the [documentation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) for installing Detectron2.
2. To run training, provide the relevant data paths at the top of the file `frame_seg_train.py`. Execute the script as `python frame_seg_train.py`.
3. To run inference, provide the relevant data paths at the top of the file `frame_seg_infer.py`. (For validation, there is commented code at the bottom that can be run.) Execute the script as `python frame_seg_infer.py`. 


## Code Structure

- frame_seg_train.py
- frame_seg_infer.py
- frame_seg_utils.py


## Output

The output path is specified in `setup()` in the frame_seg_utils.py file. The model will output model weights at validation checkpoints and also at the end. It will also output logging information for tensorboard.
