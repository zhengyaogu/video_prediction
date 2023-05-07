import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from data_preparation import CustomVideoDataset
from model import SimVP
from prediction import VisualizePredictions

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_shape = [11, 3, 160, 240]
    hid_S = 64
    hid_T = 256
    N_S = 4
    N_T = 8
    model = SimVP(tuple(in_shape), hid_S, hid_T, N_S, N_T).to(device)

    # Load the saved model weights
    model_weights_path = "/home/mr6555/sim_vp/results/model_weights_final_35_50_epoch.pth"
    model.load_state_dict(torch.load(model_weights_path, map_location=device))


    num_workers = 1
    data_path = '/scratch/mr6555/dl_project_dataset/Dataset_Student'


    # Transformations
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5059507, 0.50431152, 0.50074222], std=[0.05544372, 0.05532669, 0.05913541]),
    ])

    #transform = transforms.Compose([
    #    transforms.ToTensor(),
    #    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #])

    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> Loading the Data <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    #hidden_dataset = CustomVideoDataset(root_dir=data_path, data_type='hidden', transform=transform)
    #hidden_loader = DataLoader(hidden_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    val_dataset = CustomVideoDataset(root_dir=data_path, data_type='val', transform=transform, only_input_frames=True)
    vali_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finish the Data Loading <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    
    viz = VisualizePredictions(model, vali_loader, device)
    viz.generate_predictions()
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> Finish Generating Prediction <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    viz.save_predictions("/home/mr6555/sim_vp/yan_result/valida_result")  # Replace with the directory where you want to save the predictions
    viz.visualize_predictions("/home/mr6555/sim_vp/yan_result/valida_result")  # Replace with the directory where you want to save the GIFs
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> Complete <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

if __name__ == '__main__':
    main()
