import os
from pathlib import Path
from PIL import Image

# 设置原始数据根目录和输出目录（可选：你可以直接覆盖原图或保存到新位置）
RAW_DATA_ROOT = Path("data/raw")        # 原始数据路径
OUTPUT_ROOT = Path("data/processed")    # 处理后保存路径（避免覆盖原始数据）

# 目标尺寸
TARGET_SHORT_SIDE = 512
CROP_SIZE = 512

def resize_short_side_and_center_crop(image: Image.Image, short_side: int, crop_size: int) -> Image.Image:
    """
    将图像按比例缩放，使短边等于 short_side，
    然后从中心裁剪出 crop_size x crop_size 的区域。
    """
    w, h = image.size
    scale = short_side / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # 按比例缩放（使用高质量的 LANCZOS 插值）
    resized_img = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 计算中心裁剪坐标
    left = (new_w - crop_size) // 2
    top = (new_h - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    
    cropped_img = resized_img.crop((left, top, right, bottom))
    return cropped_img

def process_dataset(split: str):
    """
    处理 train / val / test 中的 A 和 B 子文件夹
    """
    for subfolder in ["A", "B"]:
        input_dir = RAW_DATA_ROOT / split / subfolder
        output_dir = OUTPUT_ROOT / split / subfolder
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing {input_dir} → {output_dir}")

        for img_path in input_dir.glob("*.png"):  # 假设图像是 .png 格式；如为 .jpg 则改 glob("*.jpg")
            try:
                with Image.open(img_path) as img:
                    if img.mode != "L":  # 确保是灰度图
                        img = img.convert("L")
                    processed_img = resize_short_side_and_center_crop(
                        img, TARGET_SHORT_SIDE, CROP_SIZE
                    )
                    save_path = output_dir / img_path.name
                    processed_img.save(save_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        process_dataset(split)
    print("✅ 所有图像预处理完成！")