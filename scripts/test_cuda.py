import torch

def check_device():
    print("PyTorch 版本:", torch.__version__)
    
    # 检查是否可用 CUDA（NVIDIA GPU）
    if torch.cuda.is_available():
        print("CUDA 可用！")
        print("CUDA 设备数量:", torch.cuda.device_count())
        current_device = torch.cuda.current_device()
        print("当前 CUDA 设备索引:", current_device)
        print("当前 CUDA 设备名称:", torch.cuda.get_device_name(current_device))
        device = torch.device("cuda")
    else:
        print("CUDA 不可用，使用 CPU。")
        device = torch.device("cpu")
    
    # 验证能否在该设备上创建张量
    try:
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        y = torch.tensor([4.0, 5.0, 6.0], device=device)
        z = x + y
        print(f"成功在 {device} 上执行张量运算: {z}")
    except Exception as e:
        print(f"在 {device} 上执行张量运算失败: {e}")
        return False

    return True

if __name__ == "__main__":
    check_device()