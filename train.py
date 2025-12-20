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

def record_experiment_info(base_dir: str) -> str:
    """
    
    :param base_dir: 父路径目录，
    :type base_dir: str
    :return: 新创建的实验目录完整路径
    :rtype: str
    """
    os.makedirs(base_dir, exist_ok=True)

    # 收集所有合法的 experiment_N 目录并提取编号
    exp_ids = []
    for d in os.listdir(base_dir):
        if d.startswith("experiment_") and d[11:].isdigit():
            exp_ids.append(int(d[11:]))

    next_id = max(exp_ids) + 1 if exp_ids else 1
    exp_dir = os.path.join(base_dir, f"experiment_{next_id}")
    
    os.makedirs(exp_dir, exist_ok=False)  # 确保不覆盖已有目录
    return exp_dir


# 定义主训练脚本
def main():
# 加载配置
    config = load_config('configs/config.yaml')
    training_cfg = config['training']

    # 一次性解包常用参数
    epochs = training_cfg['epochs']
    lr = training_cfg['learning_rate']
    batch_size = training_cfg['batch_size']
    weight_decay = training_cfg['weight_decay']
    affine_weight_path = training_cfg['affine_weight_path']
    base_channels = training_cfg['base_channels']
    input_mode = training_cfg['input_mode']
    input_size = training_cfg['input_size']
    dataset_path = training_cfg['dataset_path']
    mean = training_cfg['mean']
    std = training_cfg['std']
    jl_thresh_mode = training_cfg['jl_thresh_mode']

    # 定义训练集、验证集、测试集，并加载到加载器中
    train_dataset = FundusImageDataset(
        root_dir = os.path.join(dataset_path, 'train'),
        mean = mean, std = std,
        standard_size = tuple(input_size)
    )
    val_dataset = FundusImageDataset(
        root_dir = os.path.join(dataset_path, 'val'),
        mean = mean, std = std,
        standard_size = tuple(input_size)
    )
    test_dataset = FundusImageDataset(
        root_dir = os.path.join(dataset_path, 'test'),
        mean = mean, std = std,
        standard_size = tuple(input_size)
    )

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    
    # 记录实验参数，保存实验结果路径
    current_exp_dir = record_experiment_info(affine_weight_path)
    print(f"🔔Starting new experiment:{current_exp_dir}")
    # 保存本次实验的配置日志
    log_config = {
        'experiment_id': int(os.path.basename(current_exp_dir).split('_')[-1]),
        'epochs': epochs,
        'learning_rate': lr,
        'batch_size': batch_size,
        'weight_decay': weight_decay,
        'base_channels': base_channels,
        'input_mode': input_mode,
        'input_size': input_size,
        'dataset_path': dataset_path,
        'mean': mean,
        'std': std,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    log_path = os.path.join(current_exp_dir, 'train_config.yaml')
    with open(log_path, 'w', encoding='utf-8') as f:
        yaml.dump(log_config, f, default_flow_style=False, indent=4, allow_unicode=True)
    print(f"📝 Saved config to {log_path}")


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
    criterion = losses.DesignLoss(
        mean = mean, std=std,
        jl_thresh_mode = jl_thresh_mode
        ).mi_clipmse
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
            save_path = os.path.join(current_exp_dir, f"best_model_epoch{epoch+1}.pth")
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