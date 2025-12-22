import pandas as pd
import matplotlib.pyplot as plt
import os

def draw(df_path: str) :
    df = pd.read_csv(df_path)
    plt.figure(figsize = (8, 5))
    plt.plot(df['epoch'], df['train_loss'], label = 'Train Loss', marker = 'o')
    plt.plot(df['epoch'], df['val_loss'], label = 'Validation Loss', marker = 's')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    img_path = os.path.splitext(df_path)[0] + '.png'
    plt.savefig(img_path, dpi=300)
    plt.close()  # 可选：避免内存累积（尤其在循环调用时）