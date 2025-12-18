import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Literal

# 双线性插值模块，用于实现空间变换
class BilinearInterpolation(nn.Module):
    def __init__(self, output_size:Tuple[int,int]):
        super(BilinearInterpolation, self).__init__()
        self.output_size = output_size

    def forward(self, src, theta):
        """
        forward 的 Docstring
        
        :param src: 移动图像，[batchsize, channels, height, width]
        :param theta: 仿射变换参数，[batchsize, 6] -> 重构为[batchsize, 2, 3]
        """
        batchsize, channels, height, width = src.shape
        theta = theta.view(-1, 2, 3)
        # 创建目标网格，归一化坐标到[-1, 1]
        grid = F.affine_grid(theta, size=(batchsize, channels, *self.output_size),align_corners=False)
        # 使用grid_sample进行双线性重采样
        output = F.grid_sample(src, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        return output

class BN_block_leaky(nn.Module):
    """
    BN_block_leaky 的 Docstring
    地位：最小功能单元，由一个标准的双卷积 + BatchNorm + LeakyReLU组成
    """
    def __init__(self, in_channels, out_channels):
        super(BN_block_leaky, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    
    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = F.leaky_relu(self.bn2(self.conv2(out)))
        return out
    
class Affine_arch(nn.Module):
    def __init__(self, base_channels, input_mode='gray', input_size:Tuple[int, int]=(512,512)):
        super(Affine_arch, self).__init__()
        if input_mode not in ('rgb', 'gray'):
            raise ValueError("Input_mode must be 'rgb' or 'gray'")
        
        self.base_channels = base_channels
        self.input_channels = 6 if input_mode == 'rgb' else 2
        height, width = input_size
        for _ in range(5):
            height //= 2
            width //=2

        # 定义各个卷积块
        self.bn_block1 = BN_block_leaky(self.input_channels, base_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn_block2 = BN_block_leaky(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn_block3 = BN_block_leaky(base_channels * 2, base_channels * 2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn_block4 = BN_block_leaky(base_channels * 2, base_channels * 4)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn_block5 = BN_block_leaky(base_channels * 4, base_channels * 4)
        self.drop1 = nn.Dropout2d(0.3)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn_block6 = BN_block_leaky(base_channels * 4, base_channels * 8)
        self.drop2 = nn.Dropout2d(0.3)
        
        # 定义全连接层
        self.flatten = nn.Flatten()
        # 输入图像的大小是512*512，经过五次下采样，特征图的大小变成了16*16
        self.fc1 = nn.Linear(height * width * base_channels * 8, 32)
        self.fc2 = nn.Linear(32, 6)

        # 初始化fc2的仿射变换参数
        nn.init.zeros_(self.fc2.weight)
        self.fc2.bias.data.copy_(torch.tensor([1, 0 ,0 ,0 ,1, 0], dtype=torch.float32))

    def forward(self, x):
        x = self.pool1(self.bn_block1(x))
        x = self.pool2(self.bn_block2(x))
        x = self.pool3(self.bn_block3(x))
        x = self.pool4(self.bn_block4(x))
        x = self.pool5(self.drop1(self.bn_block5(x)))
        x = self.drop2(self.bn_block6(x))

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
class AffineNet(nn.Module):
    def __init__(self, base_channels=16, input_mode='gray', input_size:Tuple[int,int]=(512,512)):
        super(AffineNet, self).__init__()
        self.input_mode = input_mode
        self.input_size = input_size # (height, width)
        self.base_channels = base_channels

        self.affine_arch = Affine_arch(
            base_channels=base_channels, 
            input_mode=input_mode,
            input_size=input_size
            )

        self.bilinear_interp = BilinearInterpolation(output_size=input_size)

    def extra_repr(self) -> str:
        return f"input_mode={self.input_mode}, input_size={self.input_size}, base_channels={self.base_channels}"

    def forward(self, src, tgt):
        """
        forward 的 Docstring
        
        :param src: Moving image, shape[batchsize, channels, height, width]
        :param tgt: Fixed image, shape[batchsize, channels, height, width]
        :return warped_src: Transformed source image, shape[batchsize, channels, height, width]
        :return affine_param: Affine parameters, shape[batchsize, 6]
        """

        # 拼接src, tgt作为输入
        if self.input_mode == 'gray':
            assert src.shape[1] == 1 and tgt.shape[1] == 1, "Gray mode requires 1-channel input"
        elif self.input_mode == 'rgb':
            assert src.shape[1] == 3 and tgt.shape[1] == 3, "RGB mode requires 3-channel input"

        x = torch.cat([src, tgt], dim=1) # [batchsize, 2 for gray or 6 for rgb, height, width]

        # 预测仿射参数（B, 6)
        affine_param = self.affine_arch(x)

        # 应用放射变换到src
        warped_src = self.bilinear_interp(src, affine_param)

        return warped_src, affine_param
    

if __name__ == "__main__":
    # 测试灰度模式
    model_gray = AffineNet(input_mode='gray', input_size=(256, 256))
    src = torch.randn(2, 1, 256, 256)
    tgt = torch.randn(2, 1, 256, 256)
    warped, params = model_gray(src, tgt)
    print("Gray mode:", warped.shape, params.shape)

    # 测试 RGB 模式
    model_rgb = AffineNet(input_mode='rgb', input_size=(512, 512), base_channels=32)
    src = torch.randn(1, 3, 512, 512)
    tgt = torch.randn(1, 3, 512, 512)
    warped, params = model_rgb(src, tgt)
    print("RGB mode:", warped.shape, params.shape)

    # 打印模型配置（验证 extra_repr）
    print(model_rgb)