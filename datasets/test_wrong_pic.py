import os
from PIL import Image

def find_large_images(folder_path, max_size=1000):
    """
    打印出宽或高超过 max_size 的图像
    """
    print(f"Scanning {folder_path} for large images (>{max_size}px)...")
    supported = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    large_files = []
    
    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith(supported):
            continue
        path = os.path.join(folder_path, fname)
        try:
            with Image.open(path) as img:
                w, h = img.size
                if w > max_size or h > max_size:
                    print(f"⚠️ Large image: {path} → {w}×{h}")
                    large_files.append((path, w, h))
        except Exception as e:
            print(f"❌ Error opening {path}: {e}")
    
    return large_files

# 检查 A 和 B
train_A = "data/processed/train/A"
train_B = "data/processed/train/B"

large_A = find_large_images(train_A, max_size=513)
large_B = find_large_images(train_B, max_size=513)

print(f"\nTotal large images in A: {len(large_A)}")
print(f"Total large images in B: {len(large_B)}")