import numpy as np
from sklearn.utils.extmath import randomized_svd
import torch
import torch.distributions as tdist
import math

class RSVDUCBThompson:
    def __init__(self, num_clients, num_join_clients, min_valid_clients=10, c=1, prior_alpha=1, prior_beta=1):
        """
        初始化參數，結合 RSVD、UCB 和 Thompson Sampling 方法進行客戶端選擇
        """
        self.num_clients = num_clients
        self.num_join_clients = num_join_clients
        self.min_valid_clients = min_valid_clients
        self.selection_counts = np.zeros(num_clients)  # 每個客戶端被選中的次數
        self.performance_history = np.zeros(num_clients)  # 客戶端歷史績效
        self.anomaly_score_history = np.zeros(num_clients)  # 異常分數歷史
        self.poisoned_penalty = np.zeros(num_clients)  # 懲罰分數
        self.poisoned_threshold = 0.5  # 初始異常閾值
        self.c = c  # UCB 調節參數
        self.posterior_alpha = torch.ones(num_clients) * prior_alpha  # Thompson Sampling 的 alpha
        self.posterior_beta = torch.ones(num_clients) * prior_beta  # Thompson Sampling 的 beta
        self.decay_factor = 0.9  # 衰減因子

    def detect_poisoned_clients(self, gradients):
        """
        使用 RSVD 方法檢測可能的中毒客戶端，並更新異常分數
        """
        client_ids = list(gradients.keys())
        gradients = np.vstack(list(gradients.values()))
        
        # 使用 RSVD 進行梯度重建
        u, s, vt = randomized_svd(gradients, n_components=min(gradients.shape[1], 10), random_state=42)
        reconstructed = np.dot(u, np.dot(np.diag(s), vt))
        
        # 計算重建誤差
        reconstruction_errors = np.linalg.norm(gradients - reconstructed, axis=1)
        
        # 計算梯度與第一主成分的角度偏差
        gradient_angles = np.arccos(np.clip(
            np.dot(gradients, vt[0]) / (np.linalg.norm(gradients, axis=1) * np.linalg.norm(vt[0])), -1, 1))
        
        # 標準化誤差和角度，並進行加權
        normalized_errors = (reconstruction_errors - np.min(reconstruction_errors)) / (np.ptp(reconstruction_errors) + 1e-8)
        normalized_angles = (gradient_angles - np.min(gradient_angles)) / (np.ptp(gradient_angles) + 1e-8)
        combined_scores = 0.7 * normalized_errors + 0.3 * normalized_angles
        
        # 引入異常分數衰減，降低舊異常記錄影響
        for i, client_id in enumerate(client_ids):
            self.anomaly_score_history[client_id] = self.decay_factor * self.anomaly_score_history[client_id] + (1 - self.decay_factor) * combined_scores[i]
        
        return self.anomaly_score_history

    def adjust_poisoned_threshold(self, anomaly_scores):
        """
        動態調整異常閾值
        """
        threshold = np.percentile(anomaly_scores, 50)
        valid_clients = np.where(anomaly_scores <= threshold, 1, 0)
        
        # 放寬閾值以滿足最低有效客戶端數量要求
        if np.sum(valid_clients) < self.min_valid_clients:
            threshold = np.percentile(anomaly_scores, 70)
            print(f"Relaxed anomaly threshold to {threshold}")
        
        self.poisoned_threshold = threshold

    def select_clients(self, epoch, gradients):
        """
        根據異常分數和性能進行客戶端選擇
        """
        anomaly_scores = self.detect_poisoned_clients(gradients)
        self.adjust_poisoned_threshold(anomaly_scores)
        
        # 計算有效客戶端
        valid_clients = np.where(anomaly_scores + self.poisoned_penalty <= self.poisoned_threshold, 1, 0)
        
        # 動態調整 UCB 和 Thompson Sampling 的權重
        weight_ucb = 0.5 + 0.1 * (1 - np.mean(anomaly_scores))  # 異常分數越低，UCB 權重越高
        weight_ts = 1 - weight_ucb
        
        # 計算選擇分數 (UCB 和 Thompson Sampling)
        combined_scores = []
        for i in range(self.num_clients):
            if valid_clients[i] == 1:
                # UCB 計算
                if self.selection_counts[i] > 0:
                    avg_reward = self.performance_history[i] / self.selection_counts[i]
                    delta_i = math.sqrt(2 * math.log(epoch + 1) / self.selection_counts[i])
                    ucb_score = avg_reward + self.c * delta_i
                else:
                    ucb_score = 1e400  # 優先選擇尚未參與的客戶端
                
                # Thompson Sampling
                thompson_score = tdist.Beta(self.posterior_alpha[i], self.posterior_beta[i]).sample().item()
                
                # 合併分數
                combined_scores.append(weight_ucb * ucb_score + weight_ts * thompson_score)
            else:
                combined_scores.append(-1e400)  # 無效客戶端給予最低分數
        
        # 選擇分數最高的客戶端
        selected_clients = np.argsort(combined_scores)[-self.num_join_clients:]
        
        # 更新選擇次數
        for client in selected_clients:
            self.selection_counts[client] += 1
        
        # 對於被檢測為 Poissoned 的客戶端，增加懲罰分數
        for i in range(self.num_clients):
            if anomaly_scores[i] > self.poisoned_threshold:
                self.poisoned_penalty[i] += 0.1
        
        print(f"Epoch {epoch}: Anomaly Scores - {anomaly_scores}")
        return selected_clients

    def update(self, selected_clients, rewards):
        """
        根據本輪績效更新客戶端分數
        """
        # 更新績效分數
        for client, reward in zip(selected_clients, rewards):
            self.performance_history[client] += reward
            self.posterior_alpha[client] += reward
            self.posterior_beta[client] += (1 - reward)
