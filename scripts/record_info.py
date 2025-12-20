import os
import sys
from datetime import datetime

class TeeLogger:
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.file.flush()  # 确保实时写入
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        if self.file:
            self.file.close()


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


def setup_log_dir(logs_base: str, experiment_id: int) -> str:
    """
    创建 logs/affine_train/experiment_{id} 目录，并返回 train.log 路径。
    """
    log_exp_dir = os.path.join(logs_base, f"experiment_{experiment_id}")
    os.makedirs(log_exp_dir, exist_ok=True)
    log_file = os.path.join(log_exp_dir, "train.log")
    return log_file