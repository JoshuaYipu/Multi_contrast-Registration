import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import os
import time
import math
import shutil
from models import affine_net
from losses import losses
from datasets.dataset import FundusImageDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 调用yaml文件
def load_config(config_path):
    with open(config_path, 'r') as f :
        config = yaml.safe_load(f)
    return config

# 定义主训练脚本
def main():
    config = load_config('configs/config.yaml')

    # 加载YAML文件中的重要参数
    epochs = config['training']['epochs']
    lr = config['training']['learning_rate']
    batch_size = config['training']['batch_size']
    weight_decay = config['training']['weight_decay']
    affine_weight_path = config['training']['affine_weight_path']
    base_channels = config['training']['base_channels']
    input_mode = config['training']['input_mode']
    input_size = config['training']['input_size']
    dataset_path = config['training']['dataset_path']
    mean = config['training']['mean']
    std = config['training']['std']

    # 定义训练集、验证集、测试集，并加载到加载器中
    train_dataset = FundusImageDataset(
        root_dir = os.path.join(dataset_path, 'train'),
        mean = mean, std = std,
        standard_size = (512, 512)
    )
    val_dataset = FundusImageDataset(
        root_dir = os.path.join(dataset_path, 'val'),
        mean = mean, std = std,
        standard_size = (512, 512)
    )
    test_dataset = FundusImageDataset(
        root_dir = os.path.join(dataset_path, 'test'),
        mean = mean, std = std,
        standard_size = (512, 512)
    )

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    
    # 创建experiments文件夹下用于储存训练结果“affine_weight“的文件夹，储存每次训练后得到的仿射变换网络权重
    if os.path.exists(affine_weight_path):
        shutil.rmtree(affine_weight_path)
    os.makedirs(affine_weight_path)

    # 定义新的网络结构
    model = affine_net.AffineNet(
        base_channels=base_channels,
        input_mode = input_mode,
        input_size = input_size
    )
    model.to(device)
    print(model)
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = losses.DesignLoss(mean=mean, std=std).mi_clipmse
    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 30)

        # ----------------开始训练------------------
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}", leave=True)
        for batch_idx, (src, tgt) in enumerate(train_bar):
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()
            warped_src, affine_param = model(src, tgt)
            loss_train = criterion(tgt, warped_src)
            loss_train.backward()
            optimizer.step()

            train_loss += loss_train.item()
            train_bar.set_postfix({'loss':f"{loss_train.item():.6f}"})
        avg_train_loss = train_loss / len(train_loader)
        print(f"Train Loss:{avg_train_loss:.6f}")

        # ----------------开始验证-----------------
        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"Val Epoch {epoch+1}", leave=False)
        with torch.no_grad():
            for src, tgt in val_bar:
                src, tgt = src.to(device), tgt.to(device)
                warped_src, affine_param = model(src, tgt)
                loss_val = criterion(tgt, warped_src)
                val_loss += loss_val.item()
                val_bar.set_postfix({'val_loss': f"{loss_val.item():.6f}"})
        avg_val_loss = val_loss / len(val_loader)
        print(f"Val Loss:{avg_val_loss:.6f}")

        # ----------------保存最佳模型-----------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(affine_weight_path, f"best_model_epoch{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, save_path)
            print(f"Saved best model to {save_path}")
    print("\n Training finished")


if __name__ == "__main__" :
    main()