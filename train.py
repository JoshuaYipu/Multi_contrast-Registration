import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import os
import time
from datetime import datetime
import math
import shutil
from models import affine_net
from losses import losses
from datasets.dataset import FundusImageDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from scripts import record_info,plot
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 调用yaml文件
def load_config(config_path):
    with open(config_path, 'r') as f :
        config = yaml.safe_load(f)
    return config



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
    patience = training_cfg['patience']
    min_delta = training_cfg['min_delta']
    

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
    

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    
    # 记录实验参数，保存实验结果路径
    current_exp_dir = record_info.record_experiment_info(affine_weight_path)
    experiment_id = int(os.path.basename(current_exp_dir).split('_')[-1])
    print(f"🔔Starting new experiment:{current_exp_dir}")

    # 设置日志文件
    log_file_path = os.path.join(current_exp_dir, "train.log")
    tee_logger = record_info.TeeLogger(log_file_path)
    sys.stdout = tee_logger
    sys.stderr = tee_logger
    print(f"Redirecting output to:{log_file_path}")
    print(f"Training started at:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 保存本次实验的配置日志
    log_config = {
        'experiment_id': experiment_id,
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
        'jl_thresh_mode': jl_thresh_mode,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    log_path = os.path.join(current_exp_dir,  'train_config.yaml')
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
    trigger_times = 0

    # 创建metrix表格，保存训练损失等数据
    metrics_path = os.path.join(current_exp_dir, 'metrics.csv')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("epoch,train_loss,val_loss,epoch_time_sec\n")

    try:
        for epoch in range(epochs):
            epoch_start_time = time.time()
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
            epoch_time = time.time() - epoch_start_time
            print(f"Val Loss:{avg_val_loss:.6f}")
            print(f"Epoch Time:{epoch_time:.2f} seconds")

            # ----------------保存指标到csv-----------------
            with open(metrics_path, 'a', encoding='utf-8') as f:
                f.write(f"{epoch+1},{avg_train_loss:.6f},{avg_val_loss:.6f},{epoch_time:.2f}\n")

            # ----------------保存最佳模型-----------------
            if avg_val_loss < best_val_loss - min_delta:
                # 显著改进
                best_val_loss = avg_val_loss
                trigger_times = 0 # 重置计数器
                save_path = os.path.join(current_exp_dir, f"best_model_epoch{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                }, save_path)
                print(f"Saved best model to {save_path}")
            else:
                # 未显著改进
                trigger_times += 1
                print(f"No improvement for {trigger_times}/{patience} epochs")

            # ----------------判断是否早停-----------------
            if trigger_times >= patience :
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    finally:
        # 恢复stdout，关闭日志文件
        sys.stdout = sys.__stdout__
        tee_logger.close()
        print(f"\n Training finished. Log saved to {log_file_path}")
        print(f"Metrics saved to {metrics_path}")
        plot.draw(metrics_path)


if __name__ == "__main__" :
    main()