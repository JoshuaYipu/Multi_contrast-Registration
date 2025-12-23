# compute_mean_std_with_roi.py
import os
from PIL import Image
import torch
import torchvision.transforms as T
import yaml


def compute_mean_std(root_dir, mode='gray', margin=10):
    dir_A = os.path.join(root_dir, 'A')
    filenames = sorted([f for f in os.listdir(dir_A) if f.endswith('.png')])
    
    transform = T.ToTensor()

    # 初始化统计量
    total_pixels = 0
    sum_values = 0.0
    sum_squares = 0.0

    if mode == 'rgb':
        channel_sum = torch.zeros(3)
        channel_sq_sum = torch.zeros(3)
        total_pixels_per_channel = 0

    for i, fname in enumerate(filenames):
        path = os.path.join(dir_A, fname)
        img = Image.open(path).convert('L' if mode == 'gray' else 'RGB')
        img_tensor = transform(img)  # [C, H, W], float32, [0,1]

        if mode == 'gray':
            num_pix = img_tensor.numel()
            total_pixels += num_pix
            sum_values += img_tensor.sum().item()
            sum_squares += (img_tensor ** 2).sum().item()
        elif mode == 'rgb':
            C, H, W = img_tensor.shape
            num_pix_per_channel = H * W

            channel_sum += img_tensor.sum(dim=[1, 2])      # [3]
            channel_sq_sum += (img_tensor ** 2).sum(dim=[1, 2])
            total_pixels_per_channel += num_pix_per_channel

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(filenames)} images...")

    if mode == 'gray':
        mean = sum_values / total_pixels
        std = (sum_squares / total_pixels - mean ** 2) ** 0.5
        mean_list = [mean]
        std_list = [std]
    elif mode == 'rgb':
        mean = channel_sum / total_pixels_per_channel
        std = torch.sqrt(channel_sq_sum / total_pixels_per_channel - mean ** 2)
        mean_list = mean.tolist()
        std_list = std.tolist()

    print(f"\n✅ Final Results (mode={mode}):")
    print(f"Mean: {[round(m, 4) for m in mean_list]}")
    print(f"Std:  {[round(s, 4) for s in std_list]}")
    return mean_list, std_list

if __name__ == "__main__":
    root_dir = 'data/gray_raw/train'  # 修改为你的路径
    mode = 'gray'  # 或者 'gray'
    mean, std = compute_mean_std(root_dir, mode=mode, margin=10)

    print(f"mean:{mean},std:{std}")