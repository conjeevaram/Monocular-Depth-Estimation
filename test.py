# test.py
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unetsmooth import DepthEstimationModel
from preprocess import get_dataloader

# Configuration parameters
ORIGINAL_CSV_PATH = "minitest.csv"  # Your original CSV file (tab-separated)
BASE_PATH = "/Users/paras/Documents/depthestimation/"  # Base path for image files
BATCH_SIZE = 1                      # One image at a time (for per-image accuracy)
IMAGE_SIZE = (128, 128)             # Same image size used during training
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ACCURACY_THRESHOLD = 1.00           # Accuracy threshold for each pixel

# --- Preprocess the CSV file if necessary ---
# We expect preprocess.py to load a CSV with comma-separated headers.
# If your minitest.csv is tab-separated, we can read it here and re-save as a temporary CSV.
temp_csv_path = "temp_minitest.csv"
try:
    # Try reading with tab separator
    df = pd.read_csv(ORIGINAL_CSV_PATH, sep="\t")
    # Check if the expected columns exist (after stripping extra spaces)
    df.columns = df.columns.str.strip()
    if set(df.columns) >= {"rgb_path", "depth_path"}:
        print("Detected tab-separated CSV. Re-saving as comma-separated for preprocess.py.")
        df.to_csv(temp_csv_path, index=False)
        CSV_PATH = temp_csv_path
    else:
        # If columns not as expected, assume original CSV is comma-separated.
        CSV_PATH = ORIGINAL_CSV_PATH
except Exception as e:
    print(f"Error reading {ORIGINAL_CSV_PATH} with tab separator: {e}")
    # Fallback: assume the original CSV is already comma-separated.
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

# List to store accuracy for each image.
accuracies = []

# Evaluation loop (no gradient needed)
with torch.no_grad():
    for idx, (images, depths) in enumerate(test_loader):
        images = images.to(DEVICE)
        depths = depths.to(DEVICE)
        
        # Predict depth map for the RGB image.
        outputs = model(images)
        # Assume outputs and depths are of shape (B, 1, H, W) and BATCH_SIZE == 1.
        pred = outputs[0, 0, :, :].cpu().numpy()
        target = depths[0, 0, :, :].cpu().numpy()
        
        # Compute the per-pixel ratio metric.
        # Add a small epsilon to avoid division by zero.
        eps = 1e-6
        ratio = np.maximum(pred, target) / (np.minimum(pred, target) + eps)
        # A pixel is considered correct if ratio < ACCURACY_THRESHOLD.
        correct_pixels = (ratio < ACCURACY_THRESHOLD).astype(np.float32)
        pixel_accuracy = 100.0 * np.mean(correct_pixels)  # in percentage
        
        accuracies.append(pixel_accuracy)
        print(f"Image {idx+1}: Accuracy = {pixel_accuracy:.2f}%")

if len(accuracies) == 0:
    print("No test images were processed. Check the CSV and data paths.")
    exit(1)

# Compute the average accuracy.
average_accuracy = np.mean(accuracies)
print(f"\nAverage Accuracy: {average_accuracy:.2f}%")

# Plot the per-image accuracy.
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o', linestyle='-')
plt.xlabel("Test Image Index")
plt.ylabel("Accuracy (%)")
plt.title("Per-Image Depth Estimation Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig("test_accuracy.png")
plt.show()
