import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

random.seed(1)
np.random.seed(1)
num_clients = 50
num_classes = 100
dir_path = f"Cifar100_{num_clients}_alpha01/"

# Allocate data to users
def generate_cifar100(dir_path, num_clients, num_classes, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    distribution_graph = dir_path + "distribution/"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return

    # Get Cifar100 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                     niid, balance, partition, class_per_client=20)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, niid, balance, partition)

    # Compute class distribution for each client
    distribution = np.zeros((num_clients, num_classes), dtype=int)
    for i in range(num_clients):
        labels = np.array(y[i])  # y[i] contains labels for client i
        distribution[i] = np.bincount(labels, minlength=num_classes)

    # Create a scatter plot visualization
    client_ids = []
    class_ids = []
    sample_counts = []

    for client_id in range(num_clients):
        for class_id in range(num_classes):
            count = distribution[client_id, class_id]
            if count > 0:
                client_ids.append(client_id)
                class_ids.append(class_id)
                sample_counts.append(count)

    plt.figure(figsize=(15, 10))
    scatter = plt.scatter(client_ids, class_ids, s=[c * 10 for c in sample_counts], c=sample_counts, cmap='viridis', alpha=0.7)
    plt.xlabel('Client ID')
    plt.ylabel('Class ID')
    plt.title('Scatter Plot of Class Distribution Across Clients (Size & Color by Sample Count)')
    plt.xticks(np.arange(0, num_clients, 10))
    plt.yticks(np.arange(0, num_classes, 10))
    plt.colorbar(scatter, label='Number of Samples')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    filename_scatter = f"distribution_scatter_niid{niid}_balance{balance}_partition{partition}.png"

    # Ensure the directory exists before saving (redundant but safe)
    if not os.path.exists(distribution_graph):
        os.makedirs(distribution_graph)
    plt.savefig(os.path.join(distribution_graph, filename_scatter))
    plt.close()

    # Create a clearer heatmap visualization using seaborn
    plt.figure(figsize=(20, 15))
    sns.heatmap(distribution.T, annot=False, fmt="d", cmap="viridis", cbar_kws={'label': 'Number of Samples'},
                xticklabels=[f'Client {i}' for i in range(num_clients)],
                yticklabels=[f'Class {i}' for i in range(num_classes)])
    plt.xlabel('Client ID')
    plt.ylabel('Class ID')
    plt.title('Class Distribution Across Clients')
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    filename = f"distribution_heatmap_seaborn_niid{niid}_balance{balance}_partition{partition}.png"
    plt.savefig(os.path.join(distribution_graph, filename))
    plt.close()

    # Create a stacked bar chart visualization
    plt.figure(figsize=(20, 10))
    for i in range(num_clients):
        plt.bar(range(num_classes), distribution[i], bottom=np.sum(distribution[:i], axis=0), label=f'Client {i}')
    plt.xlabel('Class ID')
    plt.ylabel('Number of Samples')
    plt.title('Stacked Bar Chart of Class Distribution Across Clients')
    plt.legend(loc='upper right', fontsize='small')
    plt.xticks(range(0, num_classes, 10))
    plt.tight_layout()
    filename_bar = f"distribution_bar_niid{niid}_balance{balance}_partition{partition}.png"
    plt.savefig(os.path.join(distribution_graph, filename_bar))
    plt.close()

if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_cifar100(dir_path, num_clients, num_classes, niid, balance, partition)