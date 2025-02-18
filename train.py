import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from unetsmooth import DepthEstimationModel
from preprocess import get_dataloader

#training parameters
CSV_PATH = "/Users/paras/Documents/depthestimationv2/data/nyu2_train_smaller.csv"
BASE_PATH = "/Users/paras/Documents/depthestimation/"
BATCH_SIZE = 8
IMAGE_SIZE = (128, 128)
EPOCHS = 500
INITIAL_LEARNING_RATE = 1e-3
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

#initialize model
model = DepthEstimationModel().to(DEVICE)

#load checkpoint if exists
try:
    checkpoint = torch.load("depth_model_checkpoint.pth", weights_only=True)  # Set weights_only=True
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LEARNING_RATE, weight_decay=1e-5)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)  # Remove verbose=True
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['loss']
    print(f"Resumed from epoch {start_epoch}.")
except FileNotFoundError:
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LEARNING_RATE, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)  # Remove verbose=True
    start_epoch = 0
    best_loss = float('inf')
    print("No pre-trained model found. Starting from scratch.")

#custom loss function
class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss, self).__init__()
        self.smooth_l1 = nn.SmoothL1Loss()

    def self_supervised_loss(self, pred, target):
        #gradient difference loss
        pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        target_dx = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        target_dy = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        
        return torch.mean(torch.abs(pred_dx - target_dx)) + torch.mean(torch.abs(pred_dy - target_dy))
    
    def forward(self, pred, target):
        lgm = self.smooth_l1(pred, target)  #ground truth matching loss
        lssi = self.self_supervised_loss(pred, target)  #self-supervised loss
        
        #combine losses
        total_loss = lssi * 1 + lgm * 2
        return total_loss

criterion = DepthLoss()

#load data
train_loader = get_dataloader(CSV_PATH, BASE_PATH, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, augment=True)

#training loop
def train():
    global best_loss
    model.train()

    for epoch in range(start_epoch, EPOCHS):
        epoch_loss = 0.0
        for i, (images, depths) in enumerate(train_loader):
            images, depths = images.to(DEVICE), depths.to(DEVICE)

            #forward pass
            outputs = model(images)
            loss = criterion(outputs, depths)

            #backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

        #adjust learning rate
        scheduler.step(avg_loss)

        #save model if it improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, "depth_model_checkpoint.pth")
            print("Model improved. Saved.")

    print("Training complete!")
    torch.save(model.state_dict(), "depth_model.pth")
    print("Final model saved to depth_model.pth")

if __name__ == "__main__":
    train()
