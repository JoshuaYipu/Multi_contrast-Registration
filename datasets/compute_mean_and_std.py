import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 自定义 Dataset：加载灰度图像（支持 A 或 B 文件夹）
class GrayImageDataset(Dataset):
    def __init__(self, folder_path):
        """
        folder_path: 如 'raw_data/train/A' 或 'raw_data/train/B'
        """
        self.image_paths = []
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                self.image_paths.append(os.path.join(folder_path, fname))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加载并强制转为灰度（即使原图是 RGB）
        img = Image.open(self.image_paths[idx]).convert('L')
        # 注意：这里不应用 Resize/Crop，因为 Normalize 应基于原始像素分布
        # 但 ToTensor() 是必要的，它将 [0,255] -> [0.0,1.0]
        tensor_img = transforms.ToTensor()(img)  # shape: [1, H, W]
        return tensor_img

def compute_dataset_mean_std(train_folder_A, train_folder_B, batch_size=64, num_workers=4):
    """
    同时计算 A 和 B 文件夹中所有灰度图像的全局 mean 和 std
    返回: mean (float), std (float)
    """
    # 合并 A 和 B 的路径
    dataset_A = GrayImageDataset(train_folder_A)
    dataset_B = GrayImageDataset(train_folder_B)
    full_dataset = dataset_A + dataset_B  # PyTorch Dataset 支持拼接
    
    dataloader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    
    total_pixels = 0
    sum_values = 0.0
    sum_sq_values = 0.0
    
    for batch in dataloader:
        # batch shape: [B, 1, H, W]
        batch_flat = batch.view(-1)  # 展平为一维向量
        n_pixels = batch_flat.shape[0]
        
        sum_values += batch_flat.sum().item()
        sum_sq_values += (batch_flat ** 2).sum().item()
        total_pixels += n_pixels
    
    # 计算均值和标准差
    mean = sum_values / total_pixels
    std = (sum_sq_values / total_pixels - mean ** 2) ** 0.5
    
    return mean, std

# 使用示例
if __name__ == "__main__":
    train_A = "data/processed/train/A"
    train_B = "data/processed/train/B"
    
    mean, std = compute_dataset_mean_std(train_A, train_B)
    print(f"Computed mean: {mean:.6f}")
    print(f"Computed std:  {std:.6f}")
    
    # 在你的数据集类中这样使用：
    # self.mean = [mean]
    # self.std = [std]