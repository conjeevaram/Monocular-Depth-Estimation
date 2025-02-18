# test.py
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from unetsmooth import DepthEstimationModel
from preprocess import get_dataloader

# Configuration parameters
ORIGINAL_CSV_PATH = "minitest.csv"  # Your original CSV file (which may be tab-separated)
BASE_PATH = "/Users/paras/Documents/depthestimation/"                     # Base path for image files
BATCH_SIZE = 1                      # Process one image at a time (for per-image accuracy)
IMAGE_SIZE = (128, 128)             # Same image size used during training
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ACCURACY_THRESHOLD = 1.0            # Only pixels with a ratio of 1.0 (within a tiny tolerance) are correct
EPS = 1e-6                          # Tiny tolerance for floating-point comparisons

# Directory to save intermittent depth maps
SAVE_DIR = "intermittent_depth_maps"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Preprocess the CSV file if necessary ---
# We expect preprocess.py to load a CSV with comma-separated columns.
# If your minitest.csv is tab-separated, we load it here and re-save it as a temporary CSV.
temp_csv_path = "temp_minitest.csv"
try:
    df = pd.read_csv(ORIGINAL_CSV_PATH, sep="\t")
    df.columns = df.columns.str.strip()
    if {"rgb_path", "depth_path"}.issubset(set(df.columns)):
        print("Detected tab-separated CSV. Re-saving as comma-separated for preprocess.py.")
        df.to_csv(temp_csv_path, index=False)
        CSV_PATH = temp_csv_path
    else:
        CSV_PATH = ORIGINAL_CSV_PATH
except Exception as e:
    print(f"Error reading {ORIGINAL_CSV_PATH} with tab separator: {e}")
    CSV_PATH = ORIGINAL_CSV_PATH

# Create the test dataloader with augmentation disabled.
test_loader = get_dataloader(csv_path=CSV_PATH, base_path=BASE_PATH,
                             batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, augment=False)

# Initialize the model and load the checkpoint.
model = DepthEstimationModel().to(DEVICE)
checkpoint_path = "depth_model_checkpoint.pth"
try:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded checkpoint from", checkpoint_path)
except FileNotFoundError:
    print("Checkpoint file not found. Exiting.")
    exit(1)

# Set model to evaluation mode.
model.eval()

# List to store per-image accuracy scores.
accuracies = []

# Function to unnormalize a depth map from [-1,1] to [0,1] and save as grayscale PNG.
def save_depth_map(depth_array, filename):
    # Unnormalize: (x * 0.5) + 0.5
    unnorm = (depth_array * 0.5) + 0.5
    # Clip values to [0,1]
    unnorm = np.clip(unnorm, 0, 1)
    # Scale to 0-255 and convert to uint8
    im_8bit = (unnorm * 255).astype(np.uint8)
    # Convert to PIL Image and save
    im = Image.fromarray(im_8bit, mode="L")
    im.save(filename)

# Evaluation loop (no gradient needed)
with torch.no_grad():
    for idx, (images, depths) in enumerate(test_loader):
        images = images.to(DEVICE)
        depths = depths.to(DEVICE)
        
        # Predict depth map for the RGB image.
        outputs = model(images)
        # Assuming outputs and depths have shape (B, 1, H, W) with BATCH_SIZE == 1.
        pred = outputs[0, 0, :, :].cpu().numpy()
        target = depths[0, 0, :, :].cpu().numpy()
        
        # Compute per-pixel ratio.
        ratio = np.maximum(pred, target) / (np.minimum(pred, target) + EPS)
        # A pixel is correct only if ratio is 1.0 (within tolerance)
        correct_pixels = (ratio <= (ACCURACY_THRESHOLD + EPS)).astype(np.float32)
        pixel_accuracy = 100.0 * np.mean(correct_pixels)
        accuracies.append(pixel_accuracy)
        print(f"Image {idx+1}: Accuracy = {pixel_accuracy:.2f}%")
        
        # Save intermittent depth maps every 50th image
        if (idx + 1) % 50 == 0:
            pred_filename = os.path.join(SAVE_DIR, f"pred_depth_{idx+1}.png")
            gt_filename = os.path.join(SAVE_DIR, f"gt_depth_{idx+1}.png")
            save_depth_map(pred, pred_filename)
            save_depth_map(target, gt_filename)
            print(f"Saved predicted depth map to {pred_filename}")
            print(f"Saved ground truth depth map to {gt_filename}")

if len(accuracies) == 0:
    print("No test images were processed. Check the CSV and data paths.")
    exit(1)

# Compute the average accuracy over all test images.
average_accuracy = np.mean(accuracies)
print(f"\nAverage Accuracy: {average_accuracy:.2f}%")

# Plot the per-image accuracies.
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o', linestyle='-')
plt.xlabel("Test Image Index")
plt.ylabel("Accuracy (%)")
plt.title("Per-Image Depth Estimation Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig("test_accuracy.png")
plt.show()
