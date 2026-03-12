# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DesignLoss:
    def __init__(self, mean=0.0, std=1.0, jl_thresh_mode='none', mi_bins=32, alpha_mi=1.0, alpha_mse=0.01):
        """
        可微互信息损失（使用软直方图）
        """
        self.mi_bins = mi_bins
        self.alpha_mi = alpha_mi
        self.alpha_mse = alpha_mse

    def _compute_soft_histogram(self, x, bins):
        """
        使用线性插值构建可微直方图（一维）
        x: [B, N], 值域 [0, 1]
        输出: [B, bins]
        """
        B, N = x.shape
        device = x.device

        # 扩展维度以便广播
        x = x.unsqueeze(-1)  # [B, N, 1]
        bin_centers = torch.linspace(0, 1, bins, device=device).view(1, 1, bins)  # [1, 1, bins]

        # 计算每个点到各 bin 中心的距离（归一化到 [0,1] 区间宽度）
        # 假设 bin 宽度为 1/(bins-1)，边界外的点只贡献给最近 bin
        if bins == 1:
            hist = torch.ones(B, 1, device=device)
        else:
            bin_width = 1.0 / (bins - 1)
            # 左右相邻 bin 的权重
            left_idx = torch.floor(x / bin_width).long().clamp(0, bins - 2)  # [B, N, 1]
            right_idx = left_idx + 1

            left_center = left_idx * bin_width
            right_center = right_idx * bin_width

            # 线性插值权重
            w_right = (x - left_center) / bin_width  # [B, N, 1]
            w_left = 1.0 - w_right

            # 初始化直方图为零
            hist = torch.zeros(B, bins, device=device)

            # 使用 scatter_add_ 添加软权重（可导！因为索引是整数但权重是连续的）
            # 注意：left_idx 和 right_idx 是 long，但它们不依赖梯度（来自 floor），但权重 w_left/w_right 是可导的
            # PyTorch 允许对 scatter_add 的索引不可导，只要被 scatter 的值可导即可（实践中可行）
            hist.scatter_add_(1, left_idx.squeeze(-1), w_left.squeeze(-1))
            hist.scatter_add_(1, right_idx.squeeze(-1), w_right.squeeze(-1))

        # 归一化
        hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-8)
        return hist

    def _compute_joint_histogram_soft(self, x, y, bins):
        """
        可微联合直方图（近似）—— 使用外积（outer product）代替硬计数
        x, y: [B, 1, H, W] -> 展平为 [B, N]
        输出: [B, bins, bins]
        """
        B, _, H, W = x.shape
        N = H * W
        x_flat = x.view(B, N)  # [B, N]
        y_flat = y.view(B, N)  # [B, N]

        # 分别计算边缘软直方图（注意：这不是真正的联合分布，但常用于 MI 近似）
        # 更准确的做法是使用 2D 软分配，但计算复杂；这里采用“独立假设下的外积”作为可微近似
        p_x = self._compute_soft_histogram(x_flat, bins)  # [B, bins]
        p_y = self._compute_soft_histogram(y_flat, bins)  # [B, bins]

        # 外积得到联合分布（假设独立？不，这只是近似！但实践中用于 MI 梯度方向有效）
        # 实际上，更推荐直接构建 2D 软直方图（见下方替代方案）
        joint = torch.bmm(p_x.unsqueeze(2), p_y.unsqueeze(1))  # [B, bins, bins]

        # 但更好的方法：逐点计算 2D 软权重（计算量大但更准）
        # 我们这里先用简单版，若效果不好再升级

        return joint

    def mutual_information(self, fixed, warped, bins=None):
        """
        使用可微 MI（基于软直方图）
        """
        if bins is None:
            bins = self.mi_bins

        if fixed.dim() == 3:
            fixed = fixed.unsqueeze(1)
        if warped.dim() == 3:
            warped = warped.unsqueeze(1)

        # 确保值域 [0, 1]
        fixed = fixed.clamp(0, 1)
        warped = warped.clamp(0, 1)

        # 方法1：使用外积近似（快速但有偏）
        # joint = self._compute_joint_histogram_soft(fixed, warped, bins)

        # 方法2：更准确的 2D 软直方图（推荐）
        joint = self._compute_joint_histogram_2d_soft(fixed, warped, bins)

        eps = 1e-8
        p_fixed = joint.sum(dim=2)   # [B, bins]
        p_warped = joint.sum(dim=1)  # [B, bins]

        H_joint = -(joint * torch.log(joint + eps)).sum(dim=(1, 2))
        H_fixed = -(p_fixed * torch.log(p_fixed + eps)).sum(dim=1)
        H_warped = -(p_warped * torch.log(p_warped + eps)).sum(dim=1)

        MI = H_fixed + H_warped - H_joint
        return MI.mean()

    def _compute_joint_histogram_2d_soft(self, x, y, bins):
        B, _, H, W = x.shape
        N = H * W
        x_flat = x.view(B, N)  # [B, N]
        y_flat = y.view(B, N)  # [B, N]

        device = x.device
        if bins < 2:
            bins = 2

        bin_width = 1.0 / (bins - 1)
        x_flat = x_flat.unsqueeze(-1)  # [B, N, 1]
        y_flat = y_flat.unsqueeze(-1)  # [B, N, 1]

        # --- X direction ---
        x_pos = x_flat / bin_width
        x_idx_low = torch.floor(x_pos).long().clamp(0, bins - 1)
        x_idx_high = (x_idx_low + 1).clamp(0, bins - 1)
        x_low = x_idx_low.float() * bin_width
        wx_high = (x_flat - x_low) / bin_width
        wx_high = wx_high.clamp(0, 1)
        wx_low = 1.0 - wx_high  # [B, N, 1]

        # --- Y direction ---
        y_pos = y_flat / bin_width
        y_idx_low = torch.floor(y_pos).long().clamp(0, bins - 1)
        y_idx_high = (y_idx_low + 1).clamp(0, bins - 1)
        y_low = y_idx_low.float() * bin_width
        wy_high = (y_flat - y_low) / bin_width
        wy_high = wy_high.clamp(0, 1)
        wy_low = 1.0 - wy_high  # [B, N, 1]

        # Initialize histograms
        wx = torch.zeros(B, N, bins, device=device)
        wy = torch.zeros(B, N, bins, device=device)

        # ✅ CORRECT: use 3D index and 3D src
        wx.scatter_add_(2, x_idx_low, wx_low)
        wx.scatter_add_(2, x_idx_high, wx_high)
        wy.scatter_add_(2, y_idx_low, wy_low)
        wy.scatter_add_(2, y_idx_high, wy_high)

        # Build joint histogram
        w_joint = wx.unsqueeze(3) * wy.unsqueeze(2)  # [B, N, bins, bins]
        joint = w_joint.sum(dim=1)  # [B, bins, bins]
        joint = joint / (joint.sum(dim=(1, 2), keepdim=True) + 1e-8)

        return joint

    def mi_clipmse(self, fixed, warped):
        """
        主损失函数：最小化 (-MI + α·MSE)
        """
        mi = self.mutual_information(fixed, warped)
        loss_mi = -self.alpha_mi * mi
        loss_mse = self.alpha_mse * F.mse_loss(fixed, warped)
        return loss_mi + loss_mse  