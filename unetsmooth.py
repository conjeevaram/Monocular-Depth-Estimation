import torch
import torch.nn as nn

class DepthEstimationModel(nn.Module):
    def __init__(self):
        super(DepthEstimationModel, self).__init__()

        # Encoder: Using fewer layers with more filters
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),  # keep stride=2
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        # Decoder: Using bilinear upsampling for minimal artifacts
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv4 = nn.Sequential(
            nn.Conv2d(1536, 512, kernel_size=3, padding=1),  # 1024 + 512 from skip
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),   # 512 + 256 from skip
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, padding=1),   # 256 + 128 from skip
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, padding=1),    # 128 + 64 from skip
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # Additional upsampling layer
        self.final_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder with Skip Connections
        d4 = torch.cat([self.up4(b), e4], dim=1)
        d4 = self.conv4(d4)

        d3 = torch.cat([self.up3(d4), e3], dim=1)
        d3 = self.conv3(d3)

        d2 = torch.cat([self.up2(d3), e2], dim=1)
        d2 = self.conv2(d2)

        d1 = torch.cat([self.up1(d2), e1], dim=1)
        d1 = self.conv1(d1)

        d0 = self.up0(d1)  # Additional upsampling step

        output = self.final_layer(d0)

        return output

if __name__ == "__main__":
    model = DepthEstimationModel()
    x = torch.randn(1, 3, 128, 128)
    output = model(x)
    print(output.shape)
