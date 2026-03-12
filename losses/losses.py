# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DesignLoss:
    def __init__(self, mean=0.0, std=1.0, jl_thresh_mode='none', mi_bins=32, alpha_mi=1.0, alpha_mse=0.01):
        """
        基于互信息（Mutual Information, MI）的损失函数，用于跨模态仿射配准。
        
        注意：由于 Dataset 已移除 Normalize，输入图像值域为 [0, 1]，
              因此本类不再使用 mean/std 进行还原（保留参数仅为接口兼容）。
              
        参数:
            mi_bins (int): 直方图 bin 数量，建议 16~64
            alpha_mi (float): 互信息项权重（损失 = -alpha_mi * MI）
            alpha_mse (float): 辅助 MSE 正则项权重，防止退化解
        """
        self.mi_bins = mi_bins
        self.alpha_mi = alpha_mi
        self.alpha_mse = alpha_mse
        # mean/std 保留但不使用，避免训练脚本报错

    def _compute_joint_histogram(self, x, y, bins):
        """
        计算批量图像对的联合直方图（归一化为概率分布）。
        
        输入:
            x, y: [B, 1, H, W]，值域 [0, 1]
        输出:
            joint_hist: [B, bins, bins]
        """
        B, _, H, W = x.shape
        N = H * W
        device = x.device

        # 展平 -> [B, N]
        x_flat = x.view(B, N)
        y_flat = y.view(B, N)

        # 映射到离散 bin 索引 [0, bins-1]
        # 注意：x ∈ [0,1] → x * (bins - 1) ∈ [0, bins-1]
        x_idx = (x_flat * (bins - 1)).long().clamp(0, bins - 1)  # [B, N]
        y_idx = (y_flat * (bins - 1)).long().clamp(0, bins - 1)  # [B, N]

        # 构造联合索引
        joint_indices = x_idx * bins + y_idx  # [B, N]
        batch_offsets = torch.arange(B, device=device).unsqueeze(1) * (bins * bins)
        joint_indices = joint_indices + batch_offsets  # [B, N]

        # 构建直方图
        hist = torch.zeros(B * bins * bins, dtype=torch.float32, device=device)
        hist.scatter_add_(0, joint_indices.view(-1), torch.ones_like(joint_indices.view(-1), dtype=torch.float32))

        # 重塑并归一化
        joint_hist = hist.view(B, bins, bins)
        joint_hist = joint_hist / joint_hist.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        return joint_hist

    def mutual_information(self, fixed, warped, bins=None):
        """
        计算两幅图像之间的互信息 MI。
        MI = H(fixed) + H(warped) - H(fixed, warped)
        """
        if bins is None:
            bins = self.mi_bins

        # 确保四维 [B, 1, H, W]
        if fixed.dim() == 3:
            fixed = fixed.unsqueeze(1)
        if warped.dim() == 3:
            warped = warped.unsqueeze(1)

        # 假设输入已在 [0, 1] —— 这是关键前提！
        joint = self._compute_joint_histogram(fixed, warped, bins)

        # 边缘分布
        p_fixed = joint.sum(dim=2)    # [B, bins]
        p_warped = joint.sum(dim=1)   # [B, bins]

        # 计算熵（自然对数）
        eps = 1e-8
        H_joint = -(joint * torch.log(joint + eps)).sum(dim=(1, 2))
        H_fixed = -(p_fixed * torch.log(p_fixed + eps)).sum(dim=1)
        H_warped = -(p_warped * torch.log(p_warped + eps)).sum(dim=1)

        MI = H_fixed + H_warped - H_joint  # [B]
        return MI.mean()

    def mi_clipmse(self, fixed, warped):
        """
        主损失函数：最小化 (-MI + α·MSE)
        
        参数:
            fixed: 固定图像 [B, 1, H, W]
            warped: 仿射变换后的移动图像 [B, 1, H, W]
            
        返回:
            标量损失（越小越好）
        """
        mi = self.mutual_information(fixed, warped)
        loss_mi = -self.alpha_mi * mi
        loss_mse = self.alpha_mse * F.mse_loss(fixed, warped)
        return loss_mi  + loss_mse