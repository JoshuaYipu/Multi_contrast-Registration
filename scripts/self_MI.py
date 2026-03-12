import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# ----------------------------
# 互信息计算函数（与你训练时一致）
# ----------------------------
def calculate_mutual_information(fixed: torch.Tensor, moving: torch.Tensor, num_bins=32, eps=1e-8):
    """
    计算两幅灰度图像之间的互信息（MI）
    输入 shape: [H, W] 或 [B, 1, H, W]
    """
    if fixed.dim() == 2:
        fixed = fixed.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        moving = moving.unsqueeze(0).unsqueeze(0)
    elif fixed.dim() == 4:
        pass
    else:
        raise ValueError("Unsupported input shape")

    B, C, H, W = fixed.shape
    assert C == 1

    fixed = fixed.view(B, -1)
    moving = moving.view(B, -1)

    # 归一化到 [0, 1]
    min_val = torch.min(torch.stack([fixed.min(dim=1).values, moving.min(dim=1).values], dim=1), dim=1).values
    max_val = torch.max(torch.stack([fixed.max(dim=1).values, moving.max(dim=1).values], dim=1), dim=1).values
    fixed_norm = (fixed - min_val[:, None]) / (max_val[:, None] - min_val[:, None] + eps)
    moving_norm = (moving - min_val[:, None]) / (max_val[:, None] - min_val[:, None] + eps)

    # 映射到分箱
    fixed_bin = (fixed_norm * (num_bins - 1)).long().clamp(0, num_bins - 1)
    moving_bin = (moving_norm * (num_bins - 1)).long().clamp(0, num_bins - 1)

    # 构建联合直方图
    fixed_onehot = F.one_hot(fixed_bin, num_bins).float()      # [B, N, nb]
    moving_onehot = F.one_hot(moving_bin, num_bins).float()    # [B, N, nb]
    joint_hist = torch.bmm(fixed_onehot.transpose(1, 2), moving_onehot)  # [B, nb, nb]

    # 转为概率
    joint_prob = joint_hist / (joint_hist.sum(dim=(1,2), keepdim=True) + eps)
    fixed_prob = joint_prob.sum(dim=2)   # [B, nb]
    moving_prob = joint_prob.sum(dim=1)  # [B, nb]

    # 防止 log(0)
    joint_prob = joint_prob + eps
    fixed_prob = fixed_prob + eps
    moving_prob = moving_prob + eps

    mi = (joint_prob * (
        torch.log(joint_prob) - torch.log(fixed_prob.unsqueeze(2)) - torch.log(moving_prob.unsqueeze(1))
    )).sum(dim=(1, 2))

    return mi.mean().item()

# ----------------------------
# 主验证函数
# ----------------------------
def main():
    # 设置路径（请根据你的实际路径修改）
    PROCESSED_ROOT = Path("data/processed")
    split = "train"  # 可选: "train", "val", "test"
    sample_name = "5.png"  # 替换为你想检查的文件名

    path_A = PROCESSED_ROOT / split / "A" / sample_name
    path_B = PROCESSED_ROOT / split / "B" / sample_name

    assert path_A.exists(), f"File not found: {path_A}"
    assert path_B.exists(), f"File not found: {path_B}"

    # 加载图像 (PIL -> numpy -> torch)
    img_A_pil = Image.open(path_A).convert("L")
    img_B_pil = Image.open(path_B).convert("L")

    img_A = torch.from_numpy(np.array(img_A_pil)).float()
    img_B = torch.from_numpy(np.array(img_B_pil)).float()

    print(f"✅ Loaded: A={path_A}, B={path_B}")
    print(f"Image shape: {img_A.shape}")

    # ----------------------------
    # 1. 可视化
    # ----------------------------
    plt.figure(figsize=(15, 5))
    plt.subplot(141)
    plt.imshow(img_A, cmap='gray')
    plt.title('A: Fundus (Green Channel)\nVessels should be DARK')
    plt.axis('off')

    plt.subplot(142)
    plt.imshow(img_B, cmap='gray')
    plt.title('B: FFA\nVessels should be BRIGHT')
    plt.axis('off')

    plt.subplot(143)
    plt.imshow(img_A, cmap='gray')
    plt.imshow(img_B, cmap='hot', alpha=0.5)
    plt.title('Overlay (A + B)\nCheck structural alignment')
    plt.axis('off')

    plt.subplot(144)
    diff = torch.abs(img_A - img_B).numpy()
    plt.imshow(diff, cmap='magma')
    plt.title('Absolute Difference |A - B|')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("verification_overlay.png", dpi=150)
    plt.show()

    # ----------------------------
    # 2. 计算 Self-MI
    # ----------------------------
    mi_aa = calculate_mutual_information(img_A, img_A)
    mi_bb = calculate_mutual_information(img_B, img_B)

    print("\n🔍 Self-MI (sanity check):")
    print(f"  MI(A, A) = {mi_aa:.4f}  {'✅ Good' if mi_aa > 1.0 else '⚠️ Too low!'}")
    print(f"  MI(B, B) = {mi_bb:.4f}  {'✅ Good' if mi_bb > 1.0 else '⚠️ Too low!'}")

    # ----------------------------
    # 3. 计算初始 MI (A vs B)
    # ----------------------------
    mi_ab = calculate_mutual_information(img_A, img_B)
    print(f"\n📊 Initial MI(A, B) = {mi_ab:.4f}")
    if mi_ab > 0.6:
        print("  🟢 Good! Preprocessing likely successful.")
    elif mi_ab > 0.3:
        print("  🟡 Moderate. May improve with better initialization.")
    else:
        print("  🔴 Very low! Check:")
        print("     - Are A and B truly paired?")
        print("     - Is there large misalignment (rotation/translation)?")
        print("     - Consider inverting FFA or using NMI.")

    # ----------------------------
    # 4. 尝试 FFA 反色后 MI（可选）
    # ----------------------------
    img_B_inv = 255.0 - img_B
    mi_ab_inv = calculate_mutual_information(img_A, img_B_inv)
    print(f"\n🔄 MI(A, inverted B) = {mi_ab_inv:.4f}")
    if mi_ab_inv > mi_ab:
        print("  💡 Inverting FFA improves MI — consider using it in training!")
        # 可选：保存反色图用于训练
        # Image.fromarray(img_B_inv.numpy().astype(np.uint8)).save("inverted_B.png")
    
    img_A_inv = 255.0 - img_A
    mi_ab_inv = calculate_mutual_information(img_B, img_A_inv)
    print(f"\n🔄 MI(inverted A, B) = {mi_ab_inv:.4f}")

if __name__ == "__main__":
    import numpy as np
    main()