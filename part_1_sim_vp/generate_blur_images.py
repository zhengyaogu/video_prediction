import os
import imageio
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from data_preparation import CustomVideoDataset
from model import SimVP

def visualize_predictions_and_save_masks(model, loader, device, save_dir, val_data_path):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.eval()

    mean = np.array([0.5059507, 0.50431152, 0.50074222])[None, None, :, None, None]
    std = np.array([0.05544372, 0.05532669, 0.05913541])[None, None, :, None, None]

    video_folders = sorted([folder for folder in os.listdir(val_data_path) if os.path.isdir(os.path.join(val_data_path, folder))])

    for i, input_tensor in enumerate(loader):
        video_folder_name = video_folders[i]  # Get the video folder name
        video_folder = os.path.join(val_data_path, video_folder_name)

        input_tensor = input_tensor.to(device)


        with torch.no_grad():
            pred_output = model(input_tensor)

        pred_output_numpy = pred_output.cpu().numpy()

        # Denormalize the output
        pred_output_denormalized = (pred_output_numpy * std) + mean

        pred_output_normalized = pred_output_denormalized * 255.0

        print("Processing video:", video_folder_name)  

        for j, video in enumerate(pred_output_normalized):
            video_sequence = video.astype(np.uint8).transpose(0, 2, 3, 1)

            # Load the mask.npy file from the corresponding video folder
            
            mask = np.load(os.path.join(video_folder, "mask.npy"))
            actual_frame_idx = 0  # Add this line to keep track of the actual frame index

            for frame_idx, frame in enumerate(video_sequence):
                #print("Processing frame:", frame_idx)  # Add this line

                if 0 <= actual_frame_idx <= 10:  # Update this line to check the actual frame index
                    # Save frame
                    frame_path = os.path.join(save_dir, f"{video_folder_name}_frame_{actual_frame_idx + 12}.png")
                    #print("Saving frame:", frame_path)
                    imageio.imwrite(frame_path, frame)

                    # Save corresponding mask
                    mask_path = os.path.join(save_dir, f"mask_{video_folder_name}_frame_{actual_frame_idx + 12}.npy")
                    #print("Saving mask:", mask_path)
                    #np.save(mask_path, mask[actual_frame_idx])
                    np.save(mask_path, mask[actual_frame_idx + 11])

                actual_frame_idx += 1  # Increment the actual frame index


    
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_shape = [11, 3, 160, 240]
hid_S = 64
hid_T = 256
N_S = 4
N_T = 8
model = SimVP(tuple(in_shape), hid_S, hid_T, N_S, N_T).to(device)

# Load the saved model weights
model_weights_path = "/home/mr6555/sim_vp/results/model_weights_final_25_35_epoch.pth"
model.load_state_dict(torch.load(model_weights_path, map_location=device))


num_workers = 1

# Transformations

#data_path = '/scratch/mr6555/dl_project_dataset/Dataset_experiment'
data_path = '/home/mr6555/scratch/dl_project_dataset/Dataset_Student'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5059507, 0.50431152, 0.50074222], std=[0.05544372, 0.05532669, 0.05913541]),
])

train_dataset = CustomVideoDataset(root_dir=data_path, data_type='train', transform=transform, only_input_frames=True)
train_load = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

# val_dataset = CustomVideoDataset(root_dir=data_path, data_type='val', transform=transform, only_input_frames=True)
# vali_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

# Call the function with the appropriate arguments
val_data_path = "/home/mr6555/scratch/dl_project_dataset/Dataset_Student/train"
save_dir = "/home/mr6555/scratch/final_blur_image"
print('Start')
visualize_predictions_and_save_masks(model, train_load, device, save_dir, val_data_path)
print('Finish')
