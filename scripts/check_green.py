from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 读取原始 RGB 眼底彩照
img_rgb = Image.open("data/raw/train/A/5.png").convert("RGB")
img_array = np.array(img_rgb)  # shape: (H, W, 3)

# 提取各通道
R = img_array[:, :, 0]
G = img_array[:, :, 1]  # ← 这是我们要用的
B = img_array[:, :, 2]

# 标准灰度图（PIL 默认）
gray_pil = np.array(img_rgb.convert("L"))

# 计算局部对比度（例如：血管区域的标准差）
# 手动选一个含血管的小区域（或自动找最暗区域）
h, w = G.shape
# 示例：取中心 100x100 区域（假设血管在此）
crop = slice(h//2 - 50, h//2 + 50), slice(w//2 - 50, w//2 + 50)

def contrast_std(channel):
    return np.std(channel[crop])

print("R 通道对比度（std）:", contrast_std(R))
print("G 通道对比度（std）:", contrast_std(G))  # ← 应该最大！
print("B 通道对比度（std）:", contrast_std(B))
print("PIL 灰度图对比度 :", contrast_std(gray_pil))
plt.figure(figsize=(15, 4))
plt.subplot(141); plt.imshow(R, cmap='gray'); plt.title('Red Channel')
plt.subplot(142); plt.imshow(G, cmap='gray'); plt.title('Green Channel')  # 👀 重点看这里
plt.subplot(143); plt.imshow(B, cmap='gray'); plt.title('Blue Channel')
plt.subplot(144); plt.imshow(img_rgb); plt.title('Original RGB')
plt.show()