import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import yaml

class FundusImageDataset(Dataset):
    def __init__(self, root_dir: str, mean: float = 0.0, std: float = 1.0, 
                 mode: str = 'gray',standard_size=(512,512)):
        """
        __init__ 的 Docstring
        初始化数据集
        :param root_dir: 数据集根目录
        :param mean
        :param std
        :param input_size: 输入图像尺寸
        :param mode: 'gray' or 'rgb'
        """

        self.root_dir = root_dir
        self.standard_size = standard_size
        self.mode = mode
        self.mean = mean
        self.std = std

        # 构建移动图像和固定图像的路径
        self.dir_A = os.path.join(root_dir, 'A')
        self.dir_B = os.path.join(root_dir, 'B')

        # 获取所有图像名
        self.filenames = sorted(
            [f for f in os.listdir(self.dir_A) if f.endswith('.png')]
        )

        # 验证B中有对应文件
        filenames_b = set(os.listdir(self.dir_B))
        assert all(f in filenames_b for f in self.filenames), "A和B中的文件名不匹配！"

        # 定义归一化变换：将原始图像[0,255]uint8 -> [0,1]float32
        self.transform = transforms.Compose([
            transforms.Resize(standard_size),
            transforms.CenterCrop(standard_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        

    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, index):
        filename = self.filenames[index]
        
        # 加载图像
        path_A = os.path.join(self.dir_A, filename)
        path_B = os.path.join(self.dir_B, filename)

        img_A = Image.open(path_A).convert('L' if self.mode == 'gray' else 'rgb')
        img_B = Image.open(path_B).convert('L' if self.mode == 'gray' else 'rgb')

        # 应用transform变换
        src = self.transform(img_A)
        tgt = self.transform(img_B)
        return src, tgt
    
if __name__ == "__main__":
    train_dataset = FundusImageDataset(
        root_dir = "data/gray_raw/train",
        config_path = 'configs/config.yaml',
        standard_size = (512, 512)
    )

    src, tgt = train_dataset[0]
    print(src.shape)
    print(tgt.shape)