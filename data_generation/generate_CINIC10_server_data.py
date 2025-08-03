import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
# Ensure your utils.dataset_utils module is correctly set up
# 確保您的 utils.dataset_utils 模組已正確設定
from utils.dataset_utils_cinic10 import check, separate_data, split_data, save_file
import matplotlib
matplotlib.use('Agg') # Use 'Agg' for non-GUI environments (e.g., servers)
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.datasets import ImageFolder # For loading datasets organized by folders

# Set random seeds for reproducibility
# 設定隨機種子以確保重現性
random.seed(1)
np.random.seed(1)
torch.manual_seed(1) # Also set for PyTorch
if torch.cuda.is_available(): # Only set if CUDA is available
    torch.cuda.manual_seed_all(1)

num_clients = 100 # Number of clients (suggested: 100-300, this use 100,200,250,300)
num_classes = 10 # CINIC-10 has 10 classes

# Define the base directory for saving processed data
# 定義用於儲存處理後數據的基礎目錄
dir_path = f"CINIC10_{num_clients}_alpha01_server2/"

# Define the path to the already extracted CINIC-10 dataset
# If your script is at /path/to/your_script.py and CINIC-10 is at /path/to/CINIC-10,
# then cinic_base_path would be 'CINIC-10'.
# 如果您的腳本在 /path/to/your_script.py 並且 CINIC-10 在 /path/to/CINIC-10，
# 那麼 cinic_base_path 將是 'CINIC-10'。
cinic_base_path = "CINIC10"
# This path will point to the 'train', 'valid', 'test' folders inside it.
# 這個路徑將指向其中的 'train', 'valid', 'test' 資料夾。
cinic_extracted_path = os.path.abspath(cinic_base_path) # Use absolute path for robustness

def generate_cinic10(dir_path, num_clients, num_classes, niid, balance, partition):
    """
    Generates and partitions the CINIC-10 dataset for federated learning.
    This function assumes the CINIC-10 dataset is already extracted in the
    same directory as the script, with 'train', 'valid', 'test' subfolders.
    It merges training and validation data, then partitions it among clients.

    Args:
        dir_path (str): The base directory to save processed data and configurations.
        num_clients (int): The number of clients to partition the data among.
        num_classes (int): The number of classes in the dataset (fixed at 10 for CINIC-10).
        niid (bool): True for Non-IID distribution, False for IID.
        balance (bool): True for balanced data distribution among clients, False otherwise.
        partition (str): The partition strategy (e.g., 'dirichlet').
    """
    # Create the base directory if it doesn't exist
    # 如果基礎目錄不存在，則建立它
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Define paths for configuration, train/test data, and distribution graphs
    # 定義配置、訓練/測試數據和分佈圖的路徑
    config_path = os.path.join(dir_path, "config.json")
    train_output_path = os.path.join(dir_path, "train/")
    test_output_path = os.path.join(dir_path, "test/")

    # Check if data already exists and matches criteria to skip regeneration
    # 檢查數據是否已存在並符合條件，以跳過重新生成
    if check(config_path, train_output_path, test_output_path, num_clients, num_classes, niid, balance, partition):
        print("Data already exists and matches criteria. Skipping generation.")
        return

    # --- CINIC-10 Data Loading (assuming it's already extracted) ---
    # Verify that the CINIC-10 base path exists
    # 驗證 CINIC-10 基礎路徑是否存在
    if not os.path.isdir(cinic_extracted_path):
        print(f"Error: CINIC-10 dataset not found at '{cinic_extracted_path}'.")
        print("Please ensure the 'CINIC-10' folder (containing 'train', 'valid', 'test')")
        print("is in the same directory as this script, or update 'cinic_base_path'.")
        sys.exit(1) # Exit if dataset not found

    # Define image transformations (ToTensor and normalization specific to CINIC-10)
    # 定義圖像轉換（針對 CINIC-10 的 ToTensor 和歸一化）
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # CINIC-10 mean and std values (calculated from the dataset)
            # CINIC-10 的平均值和標準差（從數據集中計算得出）
            transforms.Normalize((0.4786, 0.4722, 0.4305), (0.2764, 0.2687, 0.2818))
        ]
    )

    # Load CINIC-10 datasets using ImageFolder
    # ImageFolder expects data organized in subfolders by class (e.g., train/airplane, test/car)
    # 使用 ImageFolder 載入 CINIC-10 資料集
    # ImageFolder 期望數據按類別組織在子資料夾中（例如，train/airplane, test/car）
    train_dataset_path = os.path.join(cinic_extracted_path, "train")
    validation_dataset_path = os.path.join(cinic_extracted_path, "valid")
    test_dataset_path = os.path.join(cinic_extracted_path, "test")

    # Load original train and validation sets
    # 載入原始訓練集和驗證集
    # Added error checking for paths to ensure they exist before loading
    # 添加了路徑錯誤檢查，以確保它們在載入前存在
    if not os.path.isdir(train_dataset_path):
        print(f"Error: Training data directory not found at '{train_dataset_path}'.")
        sys.exit(1)
    if not os.path.isdir(validation_dataset_path):
        print(f"Error: Validation data directory not found at '{validation_dataset_path}'.")
        sys.exit(1)
    if not os.path.isdir(test_dataset_path):
        print(f"Error: Test data directory not found at '{test_dataset_path}'.")
        sys.exit(1)

    trainset_original = ImageFolder(root=train_dataset_path, transform=transform)
    validset_original = ImageFolder(root=validation_dataset_path, transform=transform)

    # Combine train and validation datasets into a single training set
    # 將訓練集和驗證資料集合併為單一的訓練集
    # This fulfills the requirement of 180,000 total training data points
    # 這滿足了總共 180,000 個訓練數據點的需求
    trainset_combined = torch.utils.data.ConcatDataset([trainset_original, validset_original])
    print(f"Combined training data size: {len(trainset_combined)}") # Should be 90,000 (train) + 90,000 (valid) = 180,000

    # Load the test set
    # 載入測試集
    testset = ImageFolder(root=test_dataset_path, transform=transform)
    print(f"Test data size: {len(testset)}") # Should be 90,000

    # Extract all images and labels into NumPy arrays
    # 提取所有圖像和標籤到 NumPy 陣列中
    dataset_image = []
    dataset_label = []

    # Iterate through the combined trainset to get images and labels
    # 迭代合併後的訓練集以獲取圖像和標籤
    print("Collecting combined training data...")
    for i in range(len(trainset_combined)):
        image, label = trainset_combined[i]
        # **CRUCIAL FIX**: Store image as [C, H, W] numpy array directly.
        # This aligns with PyTorch's Conv2d expectation.
        # **關鍵修復**: 直接將圖像儲存為 [C, H, W] 的 numpy 陣列。
        # 這與 PyTorch 的 Conv2d 期望保持一致。
        dataset_image.append(image.cpu().detach().numpy())
        dataset_label.append(label)

    # Iterate through the testset to get images and labels
    # 迭代測試集以獲取圖像和標籤
    print("Collecting test data...")
    for i in range(len(testset)):
        image, label = testset[i]
        # **CRUCIAL FIX**: Store image as [C, H, W] numpy array directly.
        # **關鍵修復**: 直接將圖像儲存為 [C, H, W] 的 numpy 陣列。
        dataset_image.append(image.cpu().detach().numpy())
        dataset_label.append(label)

    # After these loops, dataset_image will be a list of (C, H, W) numpy arrays.
    # When converted to np.array, it will be (N, C, H, W).
    # 在這些循環之後，dataset_image 將是一個 (C, H, W) numpy 陣列的列表。
    # 轉換為 np.array 後，它將是 (N, C, H, W)。
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)
    # --- End: CINIC-10 Data Loading ---

    # Separate data into client-specific partitions based on distribution strategy
    # 根據分佈策略將數據分離到客戶端特定的分區
    # `class_per_client` is set to `num_classes` (10) for CINIC-10
    # `class_per_client` 設定為 `num_classes` (10) 以用於 CINIC-10
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                     niid, balance, partition, class_per_client=num_classes)

    # Split the partitioned data into local train and test sets for each client
    # 將分區後的數據分割成每個客戶端的本地訓練集和測試集
    train_data, test_data = split_data(X, y)

    # Save the processed data and configuration
    # 儲存處理後的數據和配置
    save_file(config_path, train_output_path, test_output_path, train_data, test_data, num_clients, num_classes,
              statistic, niid, balance, partition)

    # === 分配 server 資料: 每個 label 100 筆 ===
    samples_per_class = 100
    selected_images = []
    selected_labels = []
    class_counts = {c: 0 for c in range(num_classes)}

    indices = np.arange(len(dataset_label))
    np.random.shuffle(indices)

    for idx in indices:
        label = dataset_label[idx]
        if class_counts[label] < samples_per_class:
            selected_images.append(dataset_image[idx])
            selected_labels.append(label)
            class_counts[label] += 1
        if all(v >= samples_per_class for v in class_counts.values()):
            break
    
    selected_images = np.stack(selected_images)  # [1000, H, W, C] or [1000, C, H, W]
    selected_labels = np.array(selected_labels)

    # === 若是 channel last，轉為 channel first ===
    if selected_images.shape[-1] == 3:  # likely [N, 32, 32, 3]
        selected_images = np.transpose(selected_images, (0, 3, 1, 2))  # to [N, 3, 32, 32]

    # === 儲存為 npz 格式（符合 client 儲存邏輯）===
    server_data_dict = {
        'x': selected_images,
        'y': selected_labels
    }
    server_npz_path = os.path.join(dir_path, "server_data.npz")
    with open(server_npz_path, 'wb') as f:
        np.savez_compressed(f, data=server_data_dict)

    print(f"[INFO] Server dataset saved to {server_npz_path}")
    print(f"[INFO] Server data shape: {selected_images.shape}, label shape: {selected_labels.shape}")

if __name__ == "__main__":
    # Example usage: python your_script_name.py noniid balance dirichlet
    # This script expects 3 command-line arguments:
    # 1. "noniid" or "iid" for NIID/IID distribution
    # 2. "balance" or "unbalance" for balanced/unbalanced data sizes per client
    # 3. Partition strategy (e.g., "dirichlet", or "-" if not applicable)
    #
    # 範例用法：python 你的腳本名稱.py noniid balance dirichlet
    # 此腳本需要 3 個命令行參數：
    # 1. "noniid" 或 "iid" 用於 NIID/IID 分佈
    # 2. "balance" 或 "unbalance" 用於每個客戶端的平衡/不平衡數據大小
    # 3. 分區策略（例如，"dirichlet"，如果不適用則為 "-"）

    niid = True if sys.argv[1].lower() == "noniid" else False
    balance = True if sys.argv[2].lower() == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    # Call the CINIC-10 specific generation function
    # 調用 CINIC-10 特定的生成函數
    generate_cinic10(dir_path, num_clients, num_classes, niid, balance, partition)