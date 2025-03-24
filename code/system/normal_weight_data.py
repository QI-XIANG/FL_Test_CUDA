
import numpy as np
import pandas as pd
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters


from flcore.trainmodel.models import *

from vae_test import VAE

import gc

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")



batch_size = 32
num_epochs = 10
learning_rate = 0.005

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

testset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model_updates = []

for i in tqdm(range(20)):
    model = FedAvgCNN(in_features=1, num_classes=10, dim=1024).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    model.train()

    max_local_steps = 20
    
    for step in range(max_local_steps):
        cur_model_vector = parameters_to_vector(model.parameters())

        for i, (x, y) in enumerate(testloader):
            if type(x) == type([]):
                x[0] = x[0].to(device)
            else:
                x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model_updates.append(parameters_to_vector(model.parameters()) - cur_model_vector)

# Set the hyperparameters
input_dim = 582218  # MNIST images are 28x28 = 784 pixels
latent_dim = 100
batch_size = 64
num_epochs = 5
learning_rate = 0.001
l2_regularization = 1e-5

# Create an instance of the VAE
vae = VAE(input_dim, latent_dim).to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()  # Binary cross-entropy loss for binary pixel values
optimizer = optim.Adam(vae.parameters(), lr=learning_rate, weight_decay=l2_regularization)

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for update in model_updates:
        ori_update = update.to()
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        recon_update, mu, log_var = vae(ori_update)
        
        # Compute reconstruction loss
        recon_loss = criterion(recon_update, ori_update)
        print(recon_loss)
        
        # Compute KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Total loss
        loss = recon_loss + kl_loss

        for param in vae.parameters():
            loss += 0.5 * l2_regularization * torch.sum(param.pow(2))
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Print the average loss for the epoch
    average_loss = running_loss / len(model_updates)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}")

# Save the trained model
torch.save(vae, 'vae_batch_mse_l2_right.pt')
torch.save(vae, 'detect_models/vae_batch_mse_l2_right.pt')


