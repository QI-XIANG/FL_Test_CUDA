import numpy as np
import torch

class Random():
    def __init__(self, num_clients, num_join_clients, random_join_ratio):
        self.num_clients = num_clients
        self.num_join_clients = num_join_clients
        self.random_join_ratio = random_join_ratio

    def select_clients(self, epoch):
        if self.random_join_ratio:
            num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            num_join_clients = self.num_join_clients
        # selected_clients = list(np.random.choice(self.clients, num_join_clients, replace=False))
        selected_clients = list(np.random.choice(range(self.num_clients), num_join_clients, replace=False))

        return selected_clients
    
    def update(self, clients, rewards):
        return