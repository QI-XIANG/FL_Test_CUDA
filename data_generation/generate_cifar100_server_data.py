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
num_clients = 100
num_classes = 100
dir_path = f"Cifar100_{num_clients}_alpha01_server_2/"

# Allocate data to users
def generate_cifar100(dir_path, num_clients, num_classes, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return

    # === 載入原始 CIFAR-100 ===
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR100(
        root=dir_path + "rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(
        root=dir_path + "rawdata", train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())

    dataset_image = np.array(dataset_image)  # shape [N, 3, 32, 32] or [N, 32, 32, 3]
    dataset_label = np.array(dataset_label)

    # === 分配 client 資料 ===
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                    niid, balance, partition, class_per_client=20)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path,
              train_data, test_data, num_clients, num_classes,
              statistic, niid, balance, partition)

    # === 分配 server 資料: 每個 label 40 筆 ===
    samples_per_class = 40
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

    selected_images = np.stack(selected_images)  # [4000, H, W, C] or [4000, C, H, W]
    selected_labels = np.array(selected_labels)

    # === 若是 channel last，轉為 channel first ===
    if selected_images.shape[-1] == 3:  # likely [N, 32, 32, 3]
        selected_images = np.transpose(selected_images, (0, 3, 1, 2))  # to [N, 3, 32, 32]

    # === 儲存為 npz 格式（符合 client 儲存邏輯）===
    server_data_dict = {
        'x': selected_images.astype(np.uint8),
        'y': selected_labels.astype(np.int64)
    }
    server_npz_path = os.path.join(dir_path, "server_data.npz")
    with open(server_npz_path, 'wb') as f:
        np.savez_compressed(f, data=server_data_dict)

    print(f"[INFO] Server dataset saved to {server_npz_path}")
    print(f"[INFO] Server data shape: {selected_images.shape}, label shape: {selected_labels.shape}")

if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_cifar100(dir_path, num_clients, num_classes, niid, balance, partition)