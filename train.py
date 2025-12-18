import yaml
import torch
import torch.nn as nn
from torchvision import models
import os
import time
import math
import shutil
from models import affine_net

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
    print(model)

if __name__ == "__main__" :
    main()