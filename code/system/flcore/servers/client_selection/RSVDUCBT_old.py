import numpy as np
from sklearn.utils.extmath import randomized_svd
import torch
import torch.distributions as tdist
import math

class RSVDUCBThompson:
    def __init__(self, num_clients, num_join_clients, min_valid_clients=10, c=1, prior_alpha=1, prior_beta=1):
        """
        初始化參數，結合 RSVD、UCB 和 Thompson Sampling 方法進行客戶端選擇
        :param num_clients: 總客戶端數量
        :param num_join_clients: 每輪選擇的客戶端數量
        :param min_valid_clients: 每輪至少保留的有效客戶端數量
        :param c: UCB 中的置信區間調節參數
        :param prior_alpha, prior_beta: Thompson Sampling 的 Beta 分佈先驗參數
        :param epsilon: 探索的比例，用於引入隨機性
        """
        self.num_clients = num_clients
        self.num_join_clients = num_join_clients
        self.min_valid_clients = min_valid_clients
        self.selection_counts = np.zeros(num_clients)  # 每個客戶端被選中的次數
        self.performance_history = np.zeros(num_clients)  # 每個客戶端的歷史績效
        self.anomaly_score_history = np.zeros(num_clients)  # 每個客戶端的異常分數歷史
        self.poisoned_threshold = 0.5  # 初始異常閾值
        
        self.c = c  # UCB 調節參數
        self.posterior_alpha = torch.ones(num_clients) * prior_alpha  # Thompson Sampling 的 alpha 參數
        self.posterior_beta = torch.ones(num_clients) * prior_beta  # Thompson Sampling 的 beta 參數

    def detect_poisoned_clients(self, gradients):
        """
        使用 RSVD 方法檢測可能的中毒客戶端
        :param gradients: 各客戶端的本地梯度更新 (形狀為 [num_clients, gradient_dim])
        :return: 更新後的異常分數列表
        """
        # 收集梯度並進行矩陣拼接
        client_ids = list(gradients.keys())
        gradients = np.vstack(list(gradients.values()))
        
        # 隨機奇異值分解，使用多個奇異值
        u, s, vt = randomized_svd(gradients, n_components=min(gradients.shape[1], 10), random_state=42)
        
        # 計算累積能量比例並確定使用的奇異值數量（目標：95% 能量）
        energy_ratio = np.cumsum(s**2) / np.sum(s**2)
        num_components = np.searchsorted(energy_ratio, 0.95) + 1  # 確保累積能量達到 95%
        
        # 重建梯度矩陣
        reconstructed = np.dot(u[:, :num_components], np.dot(np.diag(s[:num_components]), vt[:num_components, :]))
        
        # 計算重建誤差
        reconstruction_errors = np.linalg.norm(gradients - reconstructed, axis=1)
        
        # 計算梯度角度異常（與第一主成分的角度偏差）
        gradient_angles = np.arccos(np.clip(np.dot(gradients, vt[0, :].T) / 
                                            (np.linalg.norm(gradients, axis=1) * np.linalg.norm(vt[0, :])), -1, 1))
        
        # 將重建誤差和角度異常標準化並加權
        normalized_errors = (reconstruction_errors - np.min(reconstruction_errors)) / (np.ptp(reconstruction_errors) + 1e-8)
        normalized_angles = (gradient_angles - np.min(gradient_angles)) / (np.ptp(gradient_angles) + 1e-8)
        combined_scores = 0.7 * normalized_errors + 0.3 * normalized_angles  # 加權組合
        
        # 更新異常分數歷史
        for i, client_id in enumerate(client_ids):
            self.anomaly_score_history[client_id] += combined_scores[i]
        
        return self.anomaly_score_history

    def adjust_poisoned_threshold(self, anomaly_scores):
        """
        動態調整異常分數的閾值
        :param anomaly_scores: 異常分數列表
        """
        # 設置閾值為異常分數的 50% 分位數
        threshold = np.percentile(anomaly_scores, 50)
        
        # 檢查有效客戶端數量是否滿足最低要求
        valid_clients = np.where(anomaly_scores <= threshold, 1, 0)
        if np.sum(valid_clients) < self.min_valid_clients:
            # 放寬閾值至 70% 分位數
            threshold = np.percentile(anomaly_scores, 70)
            print(f"Relaxed anomaly threshold to {threshold}")
        
        self.poisoned_threshold = threshold

    def select_clients(self, epoch, gradients):
        """
        選擇客戶端進行訓練
        :param epoch: 當前輪次
        :param gradients: 各客戶端的本地梯度更新
        :return: 被選中的客戶端列表
        """
        # 檢測異常分數
        anomaly_scores = self.detect_poisoned_clients(gradients)

        # 調整異常閾值
        self.adjust_poisoned_threshold(anomaly_scores)

        # 根據異常閾值篩選有效客戶端
        valid_clients = np.where(anomaly_scores <= self.poisoned_threshold, 1, 0)
        
        # 計算 UCB 和 Thompson Sampling 分數
        ucb_scores = []
        thompson_scores = []
        for i in range(self.num_clients):
            if valid_clients[i] == 1:
                # 計算 UCB 分數
                if self.selection_counts[i] > 0:
                    avg_reward = self.performance_history[i] / self.selection_counts[i]
                    delta_i = math.sqrt(2 * math.log(epoch + 1) / self.selection_counts[i])
                    ucb_score = avg_reward + self.c * delta_i
                else:
                    ucb_score = 1e400  # 尚未選擇過，優先選擇

                # 計算 Thompson Sampling 分數
                thompson_sample = tdist.Beta(self.posterior_alpha[i], self.posterior_beta[i]).sample().item()
                thompson_score = thompson_sample

                ucb_scores.append(ucb_score)
                thompson_scores.append(thompson_score)
            else:
                # 無效客戶端給予最低分數
                ucb_scores.append(-1e400)
                thompson_scores.append(-1e400)

        # 合併 UCB 和 Thompson 分數
        combined_scores = np.array(ucb_scores) * 0.5 + np.array(thompson_scores) * 0.5
        
        # 選擇分數最高的客戶端
        selected_clients = np.argsort(combined_scores)[-self.num_join_clients:]
        
        # 更新選擇次數
        for client in selected_clients:
            self.selection_counts[client] += 1

        print(f"Epoch {epoch}: Anomaly Scores - {anomaly_scores}")
        return selected_clients

    def update(self, selected_clients, rewards):
        """
        更新客戶端的績效和分布參數
        :param selected_clients: 本輪選擇的客戶端列表
        :param rewards: 每個客戶端的績效分數
        """
        for client, reward in zip(selected_clients, rewards):
            # 更新績效歷史
            self.performance_history[client] += reward
            
            # 更新 Thompson Sampling 的後驗分佈
            self.posterior_alpha[client] += reward
            self.posterior_beta[client] += (1 - reward)