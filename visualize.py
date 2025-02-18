import torch
import matplotlib.pyplot as plt
import numpy as np
from unetsmooth import DepthEstimationModel  # Updated import statement
from preprocess import get_dataloader

#parameters
MODEL_PATH = "depth_model_checkpoint.pth"
CSV_PATH = "/Users/paras/Documents/depthestimationv2/data/nyu2_train_smaller.csv"
BASE_PATH = "/Users/paras/Documents/depthestimation/"
BATCH_SIZE = 5
IMAGE_SIZE = (128, 128)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

#set random seed
torch.manual_seed(42)
np.random.seed(42)

#load model
model = DepthEstimationModel().to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

#create dataloader
train_loader = get_dataloader(CSV_PATH, BASE_PATH, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)

#unnormalize function for visualization
def unnormalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)
    return img * std + mean

#visualization function with a grid of images without labels
def visualize(model, dataloader, device):
    for i, (images, depths) in enumerate(dataloader):
        if i == 0:
            break
    
    images, depths = images.to(device), depths.to(device)
    
    with torch.no_grad():
        predicted_depths = model(images)

    images = images.cpu().numpy().transpose(0, 2, 3, 1)
    predicted_depths = predicted_depths.cpu().numpy()
    depths = depths.cpu().numpy()

    images = np.clip(unnormalize(images), 0, 1)

    predicted_depths = (predicted_depths - predicted_depths.min()) / (predicted_depths.max() - predicted_depths.min())
    depths = (depths - depths.min()) / (depths.max() - depths.min())

    fig, axes = plt.subplots(5, 3, figsize=(15, 20))

    for i in range(5):
        axes[i, 0].imshow(images[i])
        axes[i, 0].axis('off')

        axes[i, 1].imshow(predicted_depths[i][0], cmap='plasma')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(depths[i][0], cmap='plasma')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize(model, train_loader, DEVICE)
