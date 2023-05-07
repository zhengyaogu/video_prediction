import os
import io
import imageio
import numpy as np
import torch

class VisualizePredictions:
    def __init__(self, model, loader, device):
        self.model = model
        self.loader = loader
        self.device = device
        self.predictions = []
        self.predictions_viz = []

    def generate_predictions(self):
        self.model.eval()
        last_frames = []

        mean = np.array([0.5059507, 0.50431152, 0.50074222])[None, None, :, None, None]
        std = np.array([0.05544372, 0.05532669, 0.05913541])[None, None, :, None, None]

        for i, input_tensor in enumerate(self.loader):
            input_tensor = input_tensor.to(self.device)

            with torch.no_grad():
                pred_output = self.model(input_tensor)

            pred_output_numpy = pred_output.cpu().numpy()

            # Denormalize the entire output
            pred_output_denormalized = (pred_output_numpy * std) + mean

            last_frame_denormalized = pred_output_denormalized[:, -1, :, :, :]  # Extract the last frame
            last_frames.append(last_frame_denormalized)

            #print(f"Predicted output dimensions for sample {i}: {pred_output_numpy.shape}")

        self.predictions = np.concatenate(last_frames, axis=0)  # Stack the last frames along the first dimension


    def save_predictions(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, "final_validation_predictions.npy")
        np.save(save_path, self.predictions)


    def visualize_predictions(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.model.eval()

        mean = np.array([0.5059507, 0.50431152, 0.50074222])[None, None, :, None, None]
        std = np.array([0.05544372, 0.05532669, 0.05913541])[None, None, :, None, None]

        for i, input_tensor in enumerate(self.loader):
            input_tensor = input_tensor.to(self.device)

            with torch.no_grad():
                pred_output = self.model(input_tensor)

            pred_output_numpy = pred_output.cpu().numpy()

            # Denormalize the output
            pred_output_denormalized = (pred_output_numpy * std) + mean

            #print(f"Predicted output dimensions for sample {i}: {pred_output_numpy.shape}")

            # Visualize only the first three samples
            if i < 3:
                pred_output_normalized = pred_output_denormalized * 255.0

                for j, video in enumerate(pred_output_normalized):
                    with io.BytesIO() as gif:
                        video_sequence = video.astype(np.uint8).transpose(0, 2, 3, 1)
                        imageio.mimsave(gif, video_sequence, "GIF", fps=5)

                        gif_path = os.path.join(save_dir, f"prediction_{i}_video_{j}.gif")
                        with open(gif_path, "wb") as f:
                            f.write(gif.getvalue())

                        print(f"Saved GIF: {gif_path}")