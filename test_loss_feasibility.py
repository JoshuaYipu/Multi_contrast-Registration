# test_loss_feasibility.py (简化版 - 仅验证损失函数)
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from losses import losses


def load_image_pair(a_path, b_path, size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    a = transform(Image.open(a_path)).unsqueeze(0)
    b = transform(Image.open(b_path)).unsqueeze(0)
    return a, b


def main():
    data_root = "data/processed_green_channel_inverted_FFA/train"
    a_dir = os.path.join(data_root, "A")
    b_dir = os.path.join(data_root, "B")
    sample_names = ["5.png", "15.png", "55.png"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    moving_list, fixed_list = [], []
    for name in sample_names:
        a_path = os.path.join(a_dir, name)
        b_path = os.path.join(b_dir, name)
        if not (os.path.exists(a_path) and os.path.exists(b_path)):
            print(f"⚠️ 跳过 {name}")
            continue
        a, b = load_image_pair(a_path, b_path, size=(256, 256))
        moving_list.append(a)
        fixed_list.append(b)

    if not moving_list:
        raise FileNotFoundError("未找到图像对")

    moving = torch.cat(moving_list, dim=0).to(device)  # [B, 1, H, W]
    fixed = torch.cat(fixed_list, dim=0).to(device)

    print(f"✅ 加载 {moving.shape[0]} 对图像")

    # === 关键：让 warped 可学习！ ===
    # 方式：将 moving 设为可学习张量（模拟网络输出）
    warped = moving.clone().detach().requires_grad_(True)  # ✅ 可导！

    # 或者：加一点可学习扰动
    # perturb = torch.randn_like(moving, requires_grad=True) * 1e-3
    # warped = (moving + perturb).clamp(0, 1)

    criterion = losses.DesignLoss(mi_bins=32, alpha_mi=1.0, alpha_mse=0.01)

    loss = criterion.mi_clipmse(fixed, warped)
    print(f"🎯 初始损失: {loss.item():.4f}")
    mi_val = -loss.item() / criterion.alpha_mi  # 因为 loss_mi = -α·MI
    mse_val = F.mse_loss(fixed, warped).item()
    print(f"   - MI 项: {mi_val:.4f}")
    print(f"   - MSE : {mse_val:.6f}")

    # === 反向传播 ===
    try:
        loss.backward()
        print("✅ 反向传播成功！")
        print(f"   warped.grad norm: {warped.grad.norm().item():.6f}")
    except Exception as e:
        print(f"❌ 反向传播失败: {e}")
        return

    # === 模拟优化一步 ===
    with torch.no_grad():
        warped -= 1e-4 * warped.grad  # 简单梯度下降
    print("✅ 模拟参数更新成功")


if __name__ == "__main__":
    main()