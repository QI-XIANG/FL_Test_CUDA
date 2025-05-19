import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file

random.seed(1)
np.random.seed(1)
num_clients = 40
num_classes = 10
attacker_ratio = 0.2  # 20% 攻擊者客戶端
num_attackers = int(num_clients * attacker_ratio)

dir_path = f"fmnist{num_clients}_Tadaptive/"

# 產生並分配 FashionMNIST 數據集
def generate_fmnist(dir_path, num_clients, num_classes, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # 設定存放資料的路徑
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    verified_path = dir_path + "verified/"  # 儲存用於驗證攻擊成功率的資料夾

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return

    # 取得 FashionMNIST 數據
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    trainset = torchvision.datasets.FashionMNIST(root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root=dir_path+"rawdata", train=False, download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = np.concatenate((trainset.data.cpu().numpy(), testset.data.cpu().numpy()))
    dataset_label = np.concatenate((trainset.targets.cpu().numpy(), testset.targets.cpu().numpy()))

    # 分割數據
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, niid, balance, partition)
    train_data, test_data = split_data(X, y)

    # 進行攻擊者數據注入
    attacker_indices = random.sample(range(num_clients), num_attackers)  # 隨機選取 20% 客戶端作為攻擊者
    target_label = 1  # 攻擊目標類別，例如讓選定的邊緣案例被錯誤分類為「1」

    for attacker in attacker_indices:
        attacker_images = np.array(train_data[attacker]['x'])  # 取得攻擊者的訓練圖片
        attacker_labels = np.array(train_data[attacker]['y'])  # 取得攻擊者的訓練標籤
        
        # 1. 選擇邊緣樣本 (此處假設挑選某些較少出現的樣本)
        rare_class_indices = np.where(attacker_labels == 7)[0]  # 假設類別 "7" 是邊緣樣本
        poison_count = int(len(rare_class_indices) * 0.7)  # 70% 的邊緣樣本進行標籤污染

        if poison_count > 0:
            poisoned_indices = np.random.choice(rare_class_indices, size=poison_count, replace=False)
            attacker_labels[poisoned_indices] = target_label  # 變更標籤為目標類別

            # 儲存攻擊者修改過的數據至 verified 資料夾
            poisoned_data = {'x': attacker_images[poisoned_indices], 'y': attacker_labels[poisoned_indices]}
            np.save(os.path.join(verified_path, f"attacker_{attacker}.npy"), poisoned_data)

        # 更新攻擊者的訓練數據
        train_data[attacker]['y'] = attacker_labels.tolist()

    # 儲存更新後的數據
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, statistic, niid, balance, partition)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    #create verified dir
    dir_path = f"fmnist{num_clients}_Tadaptive/"
    verified_path = dir_path+"verified/"
    if not os.path.exists(verified_path):
        os.makedirs(verified_path)

    generate_fmnist(dir_path, num_clients, num_classes, niid, balance, partition)
