import copy
import torch
import torch.nn as nn
import numpy as np
import os
import time
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_recall_curve, auc as sklearn_auc
import random

# 定義 Shakespeare 資料集類，用於載入 .npz 文件
class ShakespeareDataset(Dataset):
    def __init__(self, data_path):
        """
        初始化 Shakespeare 資料集，從 .npz 文件載入序列資料。

        參數：
            data_path (str): .npz 文件路徑
        """
        data = np.load(data_path)
        self.x = data['x']  # 輸入序列 [num_sequences, seq_length]
        self.y = data['y']  # 目標序列 [num_sequences, seq_length]
        self.num_samples = len(self.x)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)

# 聯邦學習客戶端基類，支援 Shakespeare 語言建模
class Client(object):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        """
        初始化客戶端，支援 Shakespeare 語言建模。

        參數：
            args: 包含模型、資料集、學習率等參數的物件
            id (int): 客戶端編號
            train_samples (int): 訓練樣本數
            test_samples (int): 測試樣本數
            **kwargs: 額外參數（如毒化標記）
        """
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset.lower()
        self.device = args.device
        self.id = id
        self.save_folder_name = args.save_folder_name

        self.vocab_size = args.vocab_size if hasattr(args, 'vocab_size') else 8000
        self.seq_length = args.seq_length if hasattr(args, 'seq_length') else 80
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = 10
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        # 損失函數：語言建模使用交叉熵
        self.loss = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay
        self.poisoned = kwargs.get('poisoned', False)
        self.train_slow = kwargs.get('train_slow', False)
        self.send_slow = kwargs.get('send_slow', False)
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma

        # 資料路徑
        self.data_dir = os.path.join('../../dataset', 'Shakespeare_20')

    def load_train_data(self, batch_size=None):
        """
        載入訓練資料，從 .npz 文件讀取 Shakespeare 序列。

        參數：
            batch_size (int, optional): 批量大小，預設使用 self.batch_size

        返回：
            DataLoader: 訓練資料載入器
        """
        if batch_size is None:
            batch_size = self.batch_size
        data_path = os.path.join(self.data_dir, 'train', f'{self.id}.npz')
        dataset = ShakespeareDataset(data_path)

        if self.poisoned:
            # 毒化邏輯：隨機替換 10% 的輸入序列位置為隨機詞彙
            x, y = dataset.x, dataset.y
            num_sequences = x.shape[0]
            for i in range(num_sequences):
                mask = np.random.random(self.seq_length) < 0.1  # 10% 機率替換
                x[i, mask] = np.random.randint(1, self.vocab_size, size=np.sum(mask))  # 避免 <unk> (index 0)
            dataset.x = x
            dataset.y = y  # 目標序列保持不變

        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    def load_test_data(self, batch_size=None):
        """
        載入測試資料，從 .npz 文件讀取 Shakespeare 序列。

        參數：
            batch_size (int, optional): 批量大小，預設使用 self.batch_size

        返回：
            DataLoader: 測試資料載入器
        """
        if batch_size is None:
            batch_size = self.batch_size
        data_path = os.path.join(self.data_dir, 'test', f'{self.id}.npz')
        dataset = ShakespeareDataset(data_path)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    def set_parameters_bn(self, model):
        """
        設定模型參數，排除 BatchNorm 層。
        """
        bn_keys = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                bn_keys.extend([
                    f"{name}.weight",
                    f"{name}.bias",
                    f"{name}.running_mean",
                    f"{name}.running_var",
                    f"{name}.num_batches_tracked"
                ])
        for key in self.model.state_dict().keys():
            if key not in bn_keys and key in model.state_dict():
                self.model.state_dict()[key].data.copy_(model.state_dict()[key])

    def set_parameters(self, model):
        """
        設定模型參數。
        """
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        """
        複製模型參數到目標模型。
        """
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()

    def update_parameters(self, model, new_params):
        """
        更新模型參數。
        """
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        """
        計算測試集上的困惑度（perplexity）。

        返回：
            tuple: (perplexity, test_num)
        """
        testloader = self.load_test_data()
        self.model.eval()

        total_loss = 0.0
        test_num = 0
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(self.device), y.to(self.device)  # [batch_size, seq_length]
                output = self.model(x)  # [batch_size, seq_length, vocab_size]
                loss = self.loss(output.view(-1, self.vocab_size), torch.tensor([800] * y.shape[0]).to(self.device))
                total_loss += loss.item() * x.size(0) * self.seq_length
                test_num += x.size(0) * self.seq_length

        if test_num == 0:
            return float('inf'), 0

        avg_loss = total_loss / test_num
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return perplexity, test_num

    def train_metrics(self):
        """
        計算訓練集上的困惑度和損失。

        返回：
            tuple: (total_loss, train_num)
        """
        trainloader = self.load_train_data()
        self.model.eval()

        total_loss = 0.0
        train_num = 0
        with torch.no_grad():
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)  # [batch_size, seq_length, vocab_size]
                loss = self.loss(output.view(-1, self.vocab_size), torch.tensor([800] * y.shape[0]).to(self.device))
                total_loss += loss.item() * x.size(0) * self.seq_length
                train_num += x.size(0) * self.seq_length

        return total_loss, train_num

    def save_item(self, item, item_name, item_path=None):
        """
        儲存物件（例如模型參數）。
        """
        if item_path is None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, f"client_{self.id}_{item_name}.pt"))

    def load_item(self, item_name, item_path=None):
        """
        載入物件。
        """
        if item_path is None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, f"client_{self.id}_{item_name}.pt"))