import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# 双线性插值模块：用于根据仿射参数对图像进行空间变换
class BilinearInterpolation(nn.Module):
    def __init__(self, output_size: Tuple[int, int]):
        super().__init__()
        self.output_size = output_size

    def forward(self, src, theta):
        """
        :param src: 移动图像 [B, C, H, W]
        :param theta: 仿射参数 [B, 6] → reshape 为 [B, 2, 3]
        :return: 变换后的图像 [B, C, H_out, W_out]
        """
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, (src.size(0), src.size(1), *self.output_size), align_corners=False)
        warped = F.grid_sample(src, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        return warped


# 基础卷积块：双卷积 + BatchNorm + LeakyReLU
class BNBlockLeaky(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        return x


# 仿射参数回归网络（仅灰度图）
class AffineRegressor(nn.Module):
    def __init__(self, base_channels: int = 16, input_size: Tuple[int, int] = (512, 512)):
        super().__init__()
        self.input_size = input_size
        # 输入为拼接后的 [B, 2, H, W]
        self.block1 = BNBlockLeaky(2, base_channels)
        self.pool1 = nn.MaxPool2d(2)

        self.block2 = BNBlockLeaky(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.block3 = BNBlockLeaky(base_channels * 2, base_channels * 2)
        self.pool3 = nn.MaxPool2d(2)

        self.block4 = BNBlockLeaky(base_channels * 2, base_channels * 4)
        self.pool4 = nn.MaxPool2d(2)

        self.block5 = BNBlockLeaky(base_channels * 4, base_channels * 4)
        self.drop1 = nn.Dropout2d(0.3)
        self.pool5 = nn.MaxPool2d(2)

        self.block6 = BNBlockLeaky(base_channels * 4, base_channels * 8)
        self.drop2 = nn.Dropout2d(0.3)

        # 动态计算全连接层输入维度（避免硬编码）
        with torch.no_grad():
            dummy = torch.zeros(1, 2, *input_size)
            x = self._forward_features(dummy)
            n_features = x.numel() // x.size(0)  # total features per sample

        self.fc1 = nn.Linear(n_features, 32)
        self.fc2 = nn.Linear(32, 6)

        # 初始化 fc2：单位仿射变换（无变换）
        nn.init.zeros_(self.fc2.weight)
        self.fc2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32))

    def _forward_features(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.pool4(self.block4(x))
        x = self.pool5(self.drop1(self.block5(x)))
        x = self.drop2(self.block6(x))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 主模型：端到端仿射配准网络（仅灰度）
class AffineNet(nn.Module):
    def __init__(self, base_channels: int = 16, input_size: Tuple[int, int] = (512, 512)):
        super().__init__()
        self.input_size = input_size
        self.regressor = AffineRegressor(base_channels=base_channels, input_size=input_size)
        self.sampler = BilinearInterpolation(output_size=input_size)

    def forward(self, src, tgt):
        """
        :param src: 移动图像 [B, 1, H, W]
        :param tgt: 固定图像 [B, 1, H, W]
        :return:
            warped_src: [B, 1, H, W]
            affine_param: [B, 6]
        """
        # 检查输入通道
        assert src.shape[1] == 1 and tgt.shape[1] == 1, "Only grayscale (1-channel) images supported"
        # 拼接为 [B, 2, H, W]
        x = torch.cat([src, tgt], dim=1)
        # 预测仿射参数
        theta = self.regressor(x)
        # 应用空间变换
        warped = self.sampler(src, theta)
        return warped, theta


# ===== 测试代码 =====
if __name__ == "__main__":
    # 测试 256x256
    model = AffineNet(base_channels=16, input_size=(256, 256))
    src = torch.randn(2, 1, 256, 256)
    tgt = torch.randn(2, 1, 256, 256)
    warped, params = model(src, tgt)
    print("Output shapes:", warped.shape, params.shape)  # [2,1,256,256], [2,6]

    # 测试 512x512
    model2 = AffineNet(base_channels=32, input_size=(512, 512))
    src2 = torch.randn(1, 1, 512, 512)
    tgt2 = torch.randn(1, 1, 512, 512)
    warped2, params2 = model2(src2, tgt2)
    print("Output shapes (512):", warped2.shape, params2.shape)

    print(model2)