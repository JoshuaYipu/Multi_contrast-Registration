import yaml
import torch
import torch.nn as nn
from torchvision import models

# 调用yaml文件
def load_config(config_path):
    with open(config_path, 'r') as f :
        config = yaml.safe_load(f)
    return config

# 定义主训练脚本
def main():
    config = load_config('configs/config.yaml')

    epochs = config['training']['epochs']
    lr = config['training']['learning_rate']
    batch_size = config['training']['batch_size']
    weight_decay = config['training']['weight_decay']

if __name__ == "__main__" :
    main()