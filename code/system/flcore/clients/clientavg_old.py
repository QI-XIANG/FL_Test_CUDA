import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *

class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.criterion = nn.CrossEntropyLoss()  # Assuming classification task, adjust if necessary

    def train(self):
        """
        Train the model on the client side.
        """
        trainloader = self.load_train_data()
        self.model.train()

        # Differential Privacy Initialization
        if self.privacy:
            self.model, self.optimizer, trainloader, privacy_engine = initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)

        start_time = time.time()

        max_local_steps = self.local_epochs
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
                x = x.to(self.device) if not isinstance(x, list) else x[0].to(self.device)
                y = y.to(self.device)

                # Check for valid labels
                if (y < 0).any() or (y >= self.num_classes).any():
                    print(f"Invalid labels: {y}")
                    raise ValueError("Labels must be in the range [0, num_classes - 1]")

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Learning Rate Decay
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}: epsilon = {eps:.2f}, delta = {DELTA}")

    def compute_loss(self):
        """
        Compute the average loss of the model on the client's test set.
        """
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0.0
        total_samples = 0

        testloader = self.load_test_data()  # Assuming there's a method to load test data
        with torch.no_grad():  # No need to track gradients during evaluation
            for x, y in testloader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                outputs = self.model(x)  # Forward pass
                loss = self.criterion(outputs, y)  # Compute loss
                total_loss += loss.item() * x.size(0)  # Accumulate loss weighted by batch size
                total_samples += x.size(0)  # Keep track of the total number of samples

        average_loss = total_loss / total_samples  # Calculate the average loss
        return average_loss
    
    def get_training_gradients(self):
        gradient_matrix = []  # Collect the gradients here

        for param in self.model.parameters():
            if param.grad is not None:
                gradient_matrix.append(param.grad.view(-1))  # Flatten gradients for each parameter

        # Concatenate the gradients into a single tensor to ensure homogeneity
        gradient_matrix = torch.cat(gradient_matrix)

        # Move gradients to CPU and convert to NumPy
        gradient_matrix = gradient_matrix.cpu().numpy()

        return gradient_matrix

