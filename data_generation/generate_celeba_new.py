import numpy as np
import os
import sys
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from utils.dataset_utils_celeba import check, separate_data, split_data, save_file
from PIL import Image
import gdown
import zipfile

random.seed(1)
np.random.seed(1)
num_clients = 20
num_classes = 40
dir_path = f"CelebA{num_clients}/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CelebADataset(Dataset):
    def __init__(self, img_dir, attr_file, partition_file, split='train', transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.attr_df = pd.read_csv(attr_file)
        self.partition_df = pd.read_csv(partition_file)  # 使用第一行作為標頭
        split_value = 0 if split == 'train' else 2
        self.attr_df = self.attr_df.set_index('image_id')
        self.partition_df = self.partition_df.set_index('image_id')
        self.attr_df = self.attr_df.loc[self.partition_df['partition'] == split_value].reset_index()
        print(f"{split} dataset size: {len(self.attr_df)}")

    def __len__(self):
        return len(self.attr_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.attr_df.iloc[idx]['image_id'])
        image = Image.open(img_name).convert("RGB")
        attributes = ((self.attr_df.iloc[idx, 1:].values.astype(np.float32) + 1) / 2)
        if self.transform:
            image = self.transform(image)
        return image, attributes

def download_celeba_from_google_drive(dir_path):
    rawdata_path = os.path.join(dir_path, "rawdata")
    if not os.path.exists(rawdata_path):
        os.makedirs(rawdata_path)

    file_id = "1GUscfJEWdSXL4hEDlpevk7Q8xAdBJuSk"
    output_zip = os.path.join(rawdata_path, "CelebA.zip")

    if not os.path.exists(output_zip):
        print(f"正在從 Google Drive 下載 CelebA.zip (ID: {file_id})...")
        try:
            gdown.download(id=file_id, output=output_zip, quiet=False)
        except Exception as e:
            print(f"下載 CelebA.zip 失敗: {e}")
            print(f"請從以下連結手動下載並放置於 {rawdata_path}:")
            print("https://drive.google.com/file/d/1GUscfJEWdSXL4hEDlpevk7Q8xAdBJuSk/view?usp=sharing")
            return False

    extract_dir = rawdata_path
    if not os.path.exists(os.path.join(extract_dir, "img_align_celeba")):
        print("正在解壓 CelebA.zip...")
        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

    required_files = [
        os.path.join(rawdata_path, "img_align_celeba", "img_align_celeba"),
        os.path.join(rawdata_path, "list_attr_celeba.csv"),
        os.path.join(rawdata_path, "list_eval_partition.csv")
    ]
    for file in required_files:
        if not os.path.exists(file):
            print(f"缺少必要檔案: {file}")
            return False

    return True

def generate_celeba(dir_path, num_clients, num_classes, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    config_path = os.path.join(dir_path, "config.json")
    train_path = os.path.join(dir_path, "train/")
    test_path = os.path.join(dir_path, "test/")

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return

    if not download_celeba_from_google_drive(dir_path):
        print("資料下載失敗，請檢查連結或手動下載並重試！")
        return

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    rawdata_path = os.path.join(dir_path, "rawdata")
    img_dir = os.path.join(rawdata_path, "img_align_celeba", "img_align_celeba")
    attr_file = os.path.join(rawdata_path, "list_attr_celeba.csv")
    partition_file = os.path.join(rawdata_path, "list_eval_partition.csv")

    print("正在載入 CelebA 訓練資料...")
    train_dataset = CelebADataset(img_dir=img_dir, attr_file=attr_file, partition_file=partition_file,
                                  split='train', transform=transform)
    print("正在載入 CelebA 測試資料...")
    test_dataset = CelebADataset(img_dir=img_dir, attr_file=attr_file, partition_file=partition_file,
                                 split='test', transform=transform)

    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    train_images = []
    train_labels = []
    print("正在處理訓練資料...")
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        train_images.extend(images.cpu().detach().numpy())
        train_labels.extend(labels.cpu().detach().numpy())
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    print(f"train_images shape: {train_images.shape}, train_labels shape: {train_labels.shape}")

    test_images = []
    test_labels = []
    print("正在處理測試資料...")
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        test_images.extend(images.cpu().detach().numpy())
        test_labels.extend(labels.cpu().detach().numpy())
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    print(f"test_images shape: {test_images.shape}, test_labels shape: {test_labels.shape}")

    dataset_image = np.concatenate((train_images, test_images), axis=0)
    dataset_label = np.concatenate((train_labels, test_labels), axis=0)
    print(f"dataset_image shape: {dataset_image.shape}, dataset_label shape: {dataset_label.shape}")

    print("正在分割資料...")
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                    niid, balance, partition, class_per_client=5)
    train_data, test_data = split_data(X, y)

    print("正在儲存資料...")
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, niid, balance, partition)
    print("資料生成完成！")

if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_celeba(dir_path, num_clients, num_classes, niid, balance, partition)