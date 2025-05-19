import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file
import urllib.request
import zipfile
import pandas as pd
from PIL import Image
from tqdm import tqdm  # 進度條庫

random.seed(1)
np.random.seed(1)
num_clients = 20
num_classes = 43  # GTSRB 有 43 個類別
dir_path = f"GTSRB{num_clients}/"

# 下載並解壓 GTSRB 資料集
def download_gtsrb(dir_path):
    rawdata_path = dir_path + "rawdata/"
    if not os.path.exists(rawdata_path):
        os.makedirs(rawdata_path)
    
    # GTSRB 資料集的 URL
    train_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"  # 訓練資料
    test_images_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip"  # 測試圖片
    test_gt_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip"  # 測試標籤
    
    train_zip = rawdata_path + "GTSRB_Final_Training_Images.zip"
    test_images_zip = rawdata_path + "GTSRB_Final_Test_Images.zip"
    test_gt_zip = rawdata_path + "GTSRB_Final_Test_GT.zip"
    
    # 下載訓練資料
    if not os.path.exists(train_zip):
        print("正在下載 GTSRB 訓練資料...")
        urllib.request.urlretrieve(train_url, train_zip)
        print("正在解壓訓練資料...")
        with zipfile.ZipFile(train_zip, 'r') as zip_ref:
            zip_ref.extractall(rawdata_path)
    
    # 下載測試圖片
    if not os.path.exists(test_images_zip):
        print("正在下載 GTSRB 測試圖片...")
        urllib.request.urlretrieve(test_images_url, test_images_zip)
        print("正在解壓測試圖片...")
        with zipfile.ZipFile(test_images_zip, 'r') as zip_ref:
            zip_ref.extractall(rawdata_path)
    
    # 下載測試標籤
    if not os.path.exists(test_gt_zip):
        print("正在下載 GTSRB 測試標籤...")
        urllib.request.urlretrieve(test_gt_url, test_gt_zip)
        print("正在解壓測試標籤...")
        with zipfile.ZipFile(test_gt_zip, 'r') as zip_ref:
            zip_ref.extractall(rawdata_path)

# 產生 GTSRB 訓練與測試資料集
def generate_gtsrb(dir_path, num_clients, num_classes, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # 下載資料集
    download_gtsrb(dir_path)
    
    # 設定訓練與測試資料的儲存路徑
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return
        
    # 定義圖像轉換
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 統一調整圖像大小至 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))  # GTSRB 特定的正規化參數
    ])

    # 載入訓練資料集（使用 ImageFolder）
    print("正在載入訓練資料...")
    trainset = torchvision.datasets.ImageFolder(
        root=dir_path + "rawdata/GTSRB/Final_Training/Images",
        transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset), shuffle=False)

    # 自訂載入測試資料集（從新的 GT-final_test.csv 讀取標籤）
    print("正在載入測試資料...")
    test_images_path = dir_path + "rawdata/GTSRB/Final_Test/Images/"
    test_csv_path = dir_path + "rawdata/GT-final_test.csv"  # 從 GTSRB_Final_Test_GT.zip 解壓出的文件
    
    # 檢查 CSV 文件是否存在
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"測試標籤文件 {test_csv_path} 不存在，請檢查下載是否成功！")
    
    # 讀取 CSV 文件並檢查欄位
    test_df = pd.read_csv(test_csv_path, sep=';')  # GTSRB 的 CSV 使用分號分隔
    print(f"CSV 文件欄位: {list(test_df.columns)}")  # 輸出實際欄位名稱以供診斷
    
    # 確保 'ClassId' 欄位存在
    if 'ClassId' not in test_df.columns:
        raise KeyError(f"錯誤：'GT-final_test.csv' 中缺少 'ClassId' 欄位，請檢查下載來源或聯繫官方支援！")
    
    test_images = []
    test_labels = []

    # 使用 tqdm 顯示測試資料載入進度
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="處理測試圖片"):
        img_path = os.path.join(test_images_path, row['Filename'])
        if not os.path.exists(img_path):
            print(f"警告：圖片 {img_path} 不存在，跳過此樣本")
            continue
        image = Image.open(img_path).convert('RGB')  # 載入圖片並轉為 RGB
        image = transform(image)  # 應用轉換
        test_images.append(image)
        test_labels.append(row['ClassId'])  # 從 CSV 中獲取類別標籤

    if not test_images:
        raise ValueError("錯誤：未成功載入任何測試圖片，請檢查圖片路徑或文件完整性！")

    test_images = torch.stack(test_images)  # 將圖片列表轉為張量
    test_labels = torch.tensor(test_labels)  # 將標籤列表轉為張量
    testloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_images, test_labels),
        batch_size=len(test_labels), shuffle=False)

    # 提取訓練與測試資料
    dataset_image = []
    dataset_label = []

    print("正在提取訓練資料...")
    for _, (images, labels) in tqdm(enumerate(trainloader, 0), total=len(trainloader), desc="提取訓練資料"):
        dataset_image.extend(images.cpu().detach().numpy())
        dataset_label.extend(labels.cpu().detach().numpy())
    
    print("正在提取測試資料...")
    for _, (images, labels) in tqdm(enumerate(testloader, 0), total=len(testloader), desc="提取測試資料"):
        dataset_image.extend(images.cpu().detach().numpy())
        dataset_label.extend(labels.cpu().detach().numpy())

    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    # 分割並儲存資料
    print("正在分割資料...")
    with tqdm(total=1, desc="分割資料") as pbar:
        X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                        niid, balance, partition)
        train_data, test_data = split_data(X, y)
        pbar.update(1)

    print("正在儲存資料...")
    with tqdm(total=1, desc="儲存資料") as pbar:
        save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
            statistic, niid, balance, partition)
        pbar.update(1)
    
    print("資料處理完成！")


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_gtsrb(dir_path, num_clients, num_classes, niid, balance, partition)