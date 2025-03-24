'''import torch
import sys


# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import pandas as pd

partition_file = "/home/dslab/qixiang/FL_Test_Env_CUDA/dataset/CelebA20/rawdata/list_eval_partition.csv"
df = pd.read_csv(partition_file, header=None, names=['image_id', 'split'])
print(df.head())
print(df['split'].value_counts())'''

import zipfile
import numpy as np

# 指定您的 .npz 文件路徑
file_path = "dataset/CelebA/test/0.npz"

# 打開 .npz 文件作為 ZIP
with zipfile.ZipFile(file_path, 'r') as zf:
    # 列出所有內部文件
    print("內部文件列表:")
    zf.printdir()

    # 查看每個文件的內容（可選）
    for file_name in zf.namelist():
        print(f"\n查看文件: {file_name}")
        with zf.open(file_name) as f:
            content = np.load(f, allow_pickle=True)
            print(f"數據形狀: {content.shape}")
            print(f"數據預覽: {content[:5]}")