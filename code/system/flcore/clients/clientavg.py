import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *

class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        """
        初始化 clientAVG 客戶端，支援 CelebA 多標籤分類。
        
        參數：
            args: 包含模型、資料集、學習率等參數的物件
            id (int): 客戶端編號
            train_samples (int): 訓練樣本數
            test_samples (int): 測試樣本數
            **kwargs: 額外參數（如毒化標記）
        """
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # 根據資料集選擇損失函數
        if self.dataset == 'CelebA':
            self.criterion = nn.BCEWithLogitsLoss()  # 多標籤分類使用二元交叉熵損失
        else:
            self.criterion = nn.CrossEntropyLoss()  # 單標籤分類使用交叉熵損失

    def train(self):
        """
        在客戶端訓練模型，支援 CelebA 多標籤分類。
        """
        trainloader = self.load_train_data()
        self.model.train()

        # 差分隱私初始化
        if self.privacy:
            self.model, self.optimizer, trainloader, privacy_engine = initialize_dp(
                self.model, self.optimizer, trainloader, self.dp_sigma
            )

        start_time = time.time()

        max_local_steps = self.local_epochs
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
                x = x.to(self.device) if not isinstance(x, list) else x[0].to(self.device)
                y = y.to(self.device)

                # 若為 CelebA，標籤應為浮點數 (0 或 1)
                if self.dataset == 'CelebA':
                    y = y.float()  # 多標籤分類需要浮點數標籤
                else:
                    # 單標籤分類檢查標籤範圍
                    if (y < 0).any() or (y >= self.num_classes).any():
                        print(f"無效標籤: {y}")
                        raise ValueError("標籤必須在 [0, num_classes - 1] 範圍內")

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                
                output = self.model(x)  # 前向傳播
                loss = self.criterion(output, y)  # 計算損失
                self.optimizer.zero_grad()  # 清空梯度
                loss.backward()  # 反向傳播
                self.optimizer.step()  # 更新參數

        # 學習率衰減
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"客戶端 {self.id}: epsilon = {eps:.2f}, delta = {DELTA}")


    def compute_loss(self):
        """
        計算客戶端測試集上的平均損失，支援 CelebA 多標籤分類。
        
        返回：
            float: 平均損失值
        """
        self.model.eval()  # 設定模型為評估模式
        total_loss = 0.0
        total_samples = 0

        testloader = self.load_test_data()  # 載入測試資料
        with torch.no_grad():  # 評估時不計算梯度
            for x, y in testloader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                outputs = self.model(x)  # 前向傳播
                if self.dataset == 'CelebA':
                    y = y.float()  # 多標籤分類需要浮點數標籤
                loss = self.criterion(outputs, y)  # 計算損失
                # 累加損失（考慮樣本數或屬性數）
                total_loss += loss.item() * (x.size(0) if self.dataset != 'CelebA' else x.size(0) * 40)
                total_samples += x.size(0) if self.dataset != 'CelebA' else x.size(0) * 40  # CelebA 每個樣本有 40 個屬性

        average_loss = total_loss / total_samples  # 計算平均損失
        return average_loss
    

    def get_training_gradients(self):
        """
        獲取訓練過程中的梯度，支援 CelebA 多標籤模型。
        
        返回：
            numpy.ndarray: 展平後的梯度向量
        """
        gradient_matrix = []  # 收集所有參數的梯度

        for param in self.model.parameters():
            if param.grad is not None:
                gradient_matrix.append(param.grad.view(-1))  # 展平每個參數的梯度

        # 將梯度拼接為單一張量
        if not gradient_matrix:
            raise ValueError("無可用梯度，請先執行訓練！")
        gradient_matrix = torch.cat(gradient_matrix)

        # 轉移至 CPU 並轉為 NumPy 陣列
        gradient_matrix = gradient_matrix.cpu().numpy()

        return gradient_matrix