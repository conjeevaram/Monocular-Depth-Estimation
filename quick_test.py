import torch
from PIL import Image
from torchvision import transforms
from unetsmooth import DepthEstimationModel

model = DepthEstimationModel()
checkpoint = torch.load("depth_model_checkpoint.pth", map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

image = Image.open("test.png").convert("RGB")
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)

depth_map = output.squeeze(0)
depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
out_pil = transforms.ToPILImage()(depth_map)
out_pil.save("testoutput.png")