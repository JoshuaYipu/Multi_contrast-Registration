# scripts/visualize.py
import os
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt

def save_visuals(model, val_loader, device, epoch, output_dir, num_samples=2):
    """
    在验证集上生成配准结果可视化图并保存。
    
    参数:
        model: 训练中的模型（需处于 eval 模式）
        val_loader: 验证数据加载器
        device: 设备（cuda/cpu）
        epoch: 当前 epoch 编号
        output_dir: 保存目录（如 current_exp_dir/visuals）
        num_samples: 要可视化的样本对数量
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取一批数据（不打乱，取开头）
    src_batch, tgt_batch = next(iter(val_loader))
    src_batch = src_batch.to(device)[:num_samples]
    tgt_batch = tgt_batch.to(device)[:num_samples]

    with torch.no_grad():
        warped_batch, _ = model(src_batch, tgt_batch)  # [N, 1, H, W]

    # 差异图
    diff_batch = torch.abs(warped_batch - tgt_batch)

    # 拼接：每对样本形成一行 [src, tgt, warped, diff]
    visuals = []
    for i in range(num_samples):
        row = torch.cat([
            src_batch[i],      # 移动图
            tgt_batch[i],      # 固定图
            warped_batch[i],   # 配准结果
            diff_batch[i]      # 差异图
        ], dim=2)  # 沿 width 拼接
        visuals.append(row)
    
    grid = torch.cat(visuals, dim=1)  # 沿 height 拼接成一张大图

    # 保存为 PNG
    save_path = os.path.join(output_dir, f"epoch_{epoch:03d}.png")
    vutils.save_image(grid, save_path, normalize=True, scale_each=True)
    print(f"✅ Saved visualization to {save_path}")