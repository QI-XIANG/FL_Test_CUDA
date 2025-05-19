import copy
import torch
import torch.nn as nn
import numpy as np
import os
import time
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_recall_curve, auc as sklearn_auc
import random
from utils.privacy import *
from flcore.clients.clientbase_lstm import Client

# clientAVG 類，繼承自 Client，支援 Shakespeare 語言建模
class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        """
        初始化 clientAVG 客戶端，支援 Shakespeare 語言建模。
        """
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.criterion = nn.CrossEntropyLoss()
        self.vocab_size = 6714

    def train(self):
        """
        在客戶端訓練模型，支援 Shakespeare 語言建模。
        """
        trainloader = self.load_train_data()
        self.model.train()

        if self.privacy:
            self.model, self.optimizer, trainloader, privacy_engine = initialize_dp(
                self.model, self.optimizer, trainloader, self.dp_sigma
            )

        start_time = time.time()
        max_local_steps = self.local_epochs
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)  # [batch_size, seq_length, vocab_size]
                #print(f"Input shape: {x.shape}, Output shape: {output.shape}, Target shape: {y.shape}")
                loss = self.criterion(output.view(-1, self.vocab_size), torch.tensor([800] * y.shape[0]).to(self.device))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}: epsilon = {eps:.2f}, delta = {DELTA}")

    def compute_loss(self):
        """
        計算測試集上的平均損失。
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        testloader = self.load_test_data()

        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)  # [batch_size, seq_length, vocab_size]
                loss = self.criterion(output.view(-1, self.vocab_size), y.view(-1))
                total_loss += loss.item() * x.size(0) * self.seq_length
                total_samples += x.size(0) * self.seq_length

        return total_loss / total_samples if total_samples > 0 else float('inf')

    def get_training_gradients(self):
        """
        獲取訓練過程中的梯度。
        """
        gradient_matrix = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradient_matrix.append(param.grad.view(-1))
        if not gradient_matrix:
            raise ValueError("No gradients available, please run training first!")
        gradient_matrix = torch.cat(gradient_matrix)
        return gradient_matrix.cpu().numpy()