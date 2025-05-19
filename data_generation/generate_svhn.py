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
num_clients = 20
num_classes = 10  # SVHN has 10 classes (digits 0-9)
dir_path = f"SVHN{num_clients}/"

# Allocate data to users
def generate_svhn(dir_path, num_clients, num_classes, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return
        
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))  # SVHN-specific normalization
    ])

    # Load SVHN data (downloads automatically if not present)
    trainset = torchvision.datasets.SVHN(
        root=dir_path + "rawdata", 
        split='train', 
        download=True, 
        transform=transform
    )
    testset = torchvision.datasets.SVHN(
        root=dir_path + "rawdata", 
        split='test', 
        download=True, 
        transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset), shuffle=False)

    # Extract data and labels
    dataset_image = []
    dataset_label = []

    for _, (images, labels) in enumerate(trainloader, 0):
        dataset_image.extend(images.cpu().detach().numpy())
        dataset_label.extend(labels.cpu().detach().numpy())
    
    for _, (images, labels) in enumerate(testloader, 0):
        dataset_image.extend(images.cpu().detach().numpy())
        dataset_label.extend(labels.cpu().detach().numpy())

    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_svhn(dir_path, num_clients, num_classes, niid, balance, partition)