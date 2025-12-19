import os
from PIL import Image
import shutil

# ==========================
# 配置路径
# ==========================
RAW_DATA_ROOT = "data/raw"          # 原始数据集根目录
GREY_DATA_ROOT = "data/grey_raw"         # 新生成的灰度数据集根目录
SUBSETS = ["train", "val", "test"]  # 三个子集
IMAGE_FOLDER_A = "A"                # 移动图像（RGB）
IMAGE_FOLDER_B = "B"                # 固定图像（已为灰度，直接复制）

# ==========================
# 主函数
# ==========================
def main():
    # 遍历 train / val / test
    for subset in SUBSETS:
        src_subset_path = os.path.join(RAW_DATA_ROOT, subset)
        dst_subset_path = os.path.join(GREY_DATA_ROOT, subset)

        # 创建目标子集目录
        os.makedirs(dst_subset_path, exist_ok=True)

        # 处理 A 文件夹：RGB → 灰度
        src_A = os.path.join(src_subset_path, IMAGE_FOLDER_A)
        dst_A = os.path.join(dst_subset_path, IMAGE_FOLDER_A)
        os.makedirs(dst_A, exist_ok=True)

        print(f"正在处理 {subset}/A ...")
        for img_name in os.listdir(src_A):
            if not img_name.lower().endswith(".png"):
                continue
            src_img_path = os.path.join(src_A, img_name)
            dst_img_path = os.path.join(dst_A, img_name)

            # 打开 RGB 图像并转为单通道灰度
            with Image.open(src_img_path) as img:
                # 确保是 RGB 模式（有些可能带 alpha）
                if img.mode in ("RGBA", "LA", "P"):
                    img = img.convert("RGB")
                grey_img = img.convert("L")  # 转为单通道灰度

                # 保存为 PNG，不压缩（quality 不适用于 PNG，但可设 optimize=False）
                grey_img.save(dst_img_path, format="PNG", optimize=False)

        # 复制 B 文件夹（固定图像，已是灰度）
        src_B = os.path.join(src_subset_path, IMAGE_FOLDER_B)
        dst_B = os.path.join(dst_subset_path, IMAGE_FOLDER_B)
        if os.path.exists(src_B):
            print(f"正在复制 {subset}/B ...")
            if os.path.exists(dst_B):
                shutil.rmtree(dst_B)  # 避免残留
            shutil.copytree(src_B, dst_B)
        else:
            print(f"警告: {src_B} 不存在，跳过复制。")

    print("✅ 所有图像已成功转换并保存至 grey_raw 目录！")

# ==========================
# 运行入口
# ==========================
if __name__ == "__main__":
    main()