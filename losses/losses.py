import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ----------------------------
# 辅助函数：Dice 系数
# ----------------------------
def dice_coef(y_true: torch.Tensor, y_pred: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Compute Dice coefficient.
    Inputs:
        y_true, y_pred: (B, C, H, W) or (B, 1, H, W)
    """
    y_true_f = y_true.view(y_true.size(0), -1)
    y_pred_f = y_pred.view(y_pred.size(0), -1)
    intersection = (y_true_f * y_pred_f).sum(dim=1)
    denom = y_true_f.pow(2).sum(dim=1) + y_pred_f.pow(2).sum(dim=1)
    return (2.0 * intersection + smooth) / (denom + smooth)


def dice_coef_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return 1.0 - dice_coef(y_true, y_pred)


# ----------------------------
# 基础损失
# ----------------------------
def mse_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(y_pred, y_true)


def residual_loss(y_pred: torch.Tensor) -> torch.Tensor:
    """Loss for zero residual (e.g., cycle consistency)"""
    return torch.mean(y_pred ** 2)


# ----------------------------
# 梯度计算（模拟 K.gradient）
# ----------------------------
def compute_gradient(img: torch.Tensor) -> torch.Tensor:
    """
    Compute spatial gradient magnitude using central differences.
    Input: (B, C, H, W)
    Output: (B, C, H, W) — same shape, gradient magnitude
    """
    # Pad for boundary
    img_padded = F.pad(img, (1, 1, 1, 1), mode='replicate')  # (B, C, H+2, W+2)

    grad_x = img_padded[:, :, 1:-1, 2:] - img_padded[:, :, 1:-1, :-2]
    grad_y = img_padded[:, :, 2:, 1:-1] - img_padded[:, :, :-2, 1:-1]

    grad_mag = torch.abs(grad_x) + torch.abs(grad_y)
    return grad_mag


# ----------------------------
# 局部均值（替代 _local_map）
# ----------------------------
def local_mean(img: torch.Tensor, win_size: int = 9) -> torch.Tensor:
    """
    Compute local mean using average pooling.
    Input: (B, C, H, W)
    """
    pad = win_size // 2
    img_padded = F.pad(img, (pad, pad, pad, pad), mode='reflect')
    kernel = torch.ones(1, img.size(1), win_size, win_size, device=img.device, dtype=img.dtype)
    local_avg = F.conv2d(img_padded, kernel, groups=img.size(1)) / (win_size * win_size)
    return local_avg


# ----------------------------
# 互信息（MI）估计（Parzen windowing with Gaussian kernel）
# ----------------------------
def mutual_information(y_true: torch.Tensor, y_pred: torch.Tensor,
                       bins: int = 100, sigma_ratio: float = 1.0,
                       crop_background: bool = False, thresh: float = 0.0001) -> torch.Tensor:
    """
    Estimate Mutual Information between two images using Parzen windowing.
    Assumes inputs in [0, 1].
    Returns: scalar tensor (mean over batch)
    """
    device = y_true.device
    dtype = y_true.dtype

    # Clip to [0, 1]
    y_true = torch.clamp(y_true, 0, 1)
    y_pred = torch.clamp(y_pred, 0, 1)

    # Bin centers
    bin_centers = torch.linspace(0, 1, bins, device=device, dtype=dtype)  # (bins,)
    sigma = torch.mean(torch.diff(bin_centers)) * sigma_ratio
    preterm = 1.0 / (2.0 * sigma ** 2)

    if crop_background:
        # Simple foreground mask (not batch-dynamic friendly, but kept for compatibility)
        mask = (y_true > thresh) | (y_pred > thresh)  # (B, C, H, W)
        y_true_fg = y_true[mask]
        y_pred_fg = y_pred[mask]
        # Add fake batch and channel dims for compatibility
        y_true = y_true_fg.view(1, -1, 1)
        y_pred = y_pred_fg.view(1, -1, 1)
    else:
        # Flatten spatial dims: (B, C, H, W) -> (B, N, 1)
        B, C, H, W = y_true.shape
        N = H * W * C
        y_true = y_true.view(B, N, 1)
        y_pred = y_pred.view(B, N, 1)

    nb_voxels = y_pred.shape[1]  # N

    # Reshape bin centers: (1, 1, bins)
    vbc = bin_centers.view(1, 1, bins)

    # Compute Parzen window densities
    I_a = torch.exp(-preterm * (y_true - vbc) ** 2)  # (B, N, bins)
    I_a = I_a / (I_a.sum(dim=-1, keepdim=True) + 1e-8)

    I_b = torch.exp(-preterm * (y_pred - vbc) ** 2)
    I_b = I_b / (I_b.sum(dim=-1, keepdim=True) + 1e-8)

    # Joint probability pab: (B, bins, bins)
    I_a_permute = I_a.permute(0, 2, 1)  # (B, bins, N)
    pab = torch.bmm(I_a_permute, I_b)   # (B, bins, bins)
    pab = pab / nb_voxels

    # Marginals
    pa = I_a.mean(dim=1, keepdim=True)  # (B, 1, bins)
    pb = I_b.mean(dim=1, keepdim=True)  # (B, 1, bins)

    papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-8  # (B, bins, bins)

    # MI = sum(pab * log(pab / papb))
    mi = pab * torch.log(pab / papb + 1e-8)
    mi = mi.sum(dim=(1, 2))  # (B,)

    return -mi.mean()  # Negative because we minimize loss


# ----------------------------
# 自定义损失类（对应 design_loss）
# ----------------------------
class DesignLoss:
    def __init__(self, parameter: float = 1.0, parameter_mi: float = 1.0,
                 win: int = 9, jl_thresh: float = 0.1):
        self.parameter = parameter
        self.parameter_mi = parameter_mi
        self.win = win
        self.jl_thresh = jl_thresh

    def _clip_mask(self, y_true: torch.Tensor) -> torch.Tensor:
        """Create binary mask by thresholding and rounding."""
        clipped = torch.clamp(y_true, 0.0, self.jl_thresh * 2)
        rounded = torch.round(clipped / (self.jl_thresh * 2))
        return rounded  # 0 or 1

    def mi_clipmse(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Combined loss: MSE on background + MI on full image.
        """
        mask = self._clip_mask(y_true)  # foreground mask (1 = foreground)
        background_mask = 1.0 - mask

        # Apply mask to both true and pred
        y_true_bg = background_mask * y_true
        y_pred_bg = background_mask * y_pred

        mse_part = self.parameter * mse_loss(y_true_bg, y_pred_bg)
        mi_part = self.parameter_mi * mutual_information(y_true, y_pred)

        return mse_part + mi_part

    # 可选：其他损失（按需启用）
    def mi_gl2(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        grad_true = torch.sigmoid(compute_gradient(y_true))
        grad_pred = torch.sigmoid(compute_gradient(y_pred))
        l2_grad = mse_loss(grad_true, grad_pred)
        mi_img = mutual_information(y_true, y_pred)
        return self.parameter * l2_grad + self.parameter_mi * mi_img