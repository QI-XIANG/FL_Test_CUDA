import numpy as np
import os
import sys
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file
from PIL import Image

random.seed(1)
np.random.seed(1)
num_clients = 20
num_classes = 40  # CelebA has 40 attributes
dir_path = f"CelebA{num_clients}/"

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CelebADataset(Dataset):
    def __init__(self, img_dir, attr_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.attr_df = pd.read_csv(attr_file)

    def __len__(self):
        return len(self.attr_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.attr_df.iloc[idx, 0])  # Assuming first column is image ID
        image = Image.open(img_name).convert("RGB")
        attributes = self.attr_df.iloc[idx, 1:].values.astype(np.float32)  # Attributes are in columns 1 onwards

        if self.transform:
            image = self.transform(image)

        return image, attributes

# Allocate data to users
def generate_celeba(dir_path, num_clients, num_classes, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = os.path.join(dir_path, "config.json")
    train_path = os.path.join(dir_path, "train/")
    test_path = os.path.join(dir_path, "test/")

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return

    # Transform to preprocess images
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to a consistent size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CelebA dataset using custom dataset class
    images_dir = os.path.join(dir_path, "rawdata", "img_align_celeba")
    attr_file = os.path.join(dir_path, 'rawdata', 'list_attr_celeba.csv')

    dataset = CelebADataset(img_dir=images_dir, attr_file=attr_file, transform=transform)

    # Create a DataLoader with a smaller batch size
    batch_size = 4  # Adjust this based on your memory capacity
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Collect data
    dataset_image = []
    dataset_label = []

    for images, labels in dataloader:
        # Move images and labels to GPU
        images = images.to(device)
        labels = labels.to(device)

        dataset_image.extend(images.cpu().detach().numpy())
        dataset_label.extend(labels.cpu().detach().numpy())

    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    # Separate data for clients
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                    niid, balance, partition, class_per_client=5)  # Adjust if necessary
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, niid, balance, partition)

if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_celeba(dir_path, num_clients, num_classes, niid, balance, partition)
