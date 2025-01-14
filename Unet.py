import torch.nn as nn
import torch

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        # 编码器
        self.encoder1 = self._block(in_channels, 64)
        self.encoder2 = self._block(64, 128)
        self.encoder3 = self._block(128, 256)
        self.encoder4 = self._block(256, 512)

        # 最大池化
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # 中心
        self.center = self._block(512, 1024)

        # 解码器
        self.up4 = self._up_block(1024, 512)
        self.decoder4 = self._block(1024, 512)

        self.up3 = self._up_block(512, 256)
        self.decoder3 = self._block(512, 256)

        self.up2 = self._up_block(256, 128)
        self.decoder2 = self._block(256, 128)

        self.up1 = self._up_block(128, 64)
        self.decoder1 = self._block(128, 64)

        # 最后一层卷积
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)
        # self.encoder = nn.Sequential(
        #     nn.Conv3d(in_channels, out_channels=64, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(64, 64, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     # nn.MaxPool3d(kernel_size=2, stride=2)
        #     nn.AvgPool3d(kernel_size=2, stride=2)
        #     # nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
        #     # nn.ReLU(inplace=True),
        #     # nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     # nn.ReLU(inplace=True)
        # )
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(64, out_channels, kernel_size=1)
        # )

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def _up_block(self, in_channels, out_channels):
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # 编码器路径
        enc1 = self.encoder1(x)  # [B, 64, D, H, W]
        enc2 = self.encoder2(self.pool(enc1))  # [B, 128, D/2, H/2, W/2]
        enc3 = self.encoder3(self.pool(enc2))  # [B, 256, D/4, H/4, W/4]
        enc4 = self.encoder4(self.pool(enc3))  # [B, 512, D/8, H/8, W/8]

        # 中心
        center = self.center(self.pool(enc4))  # [B, 1024, D/16, H/16, W/16]

        # 解码器路径
        dec4 = self.decoder4(torch.cat([self.up4(center), enc4], dim=1))  # [B, 512, D/8, H/8, W/8]
        dec3 = self.decoder3(torch.cat([self.up3(dec4), enc3], dim=1))  # [B, 256, D/4, H/4, W/4]
        dec2 = self.decoder2(torch.cat([self.up2(dec3), enc2], dim=1))  # [B, 128, D/2, H/2, W/2]
        dec1 = self.decoder1(torch.cat([self.up1(dec2), enc1], dim=1))  # [B, 64, D, H, W]

        # 输出
        return self.final_conv(dec1)