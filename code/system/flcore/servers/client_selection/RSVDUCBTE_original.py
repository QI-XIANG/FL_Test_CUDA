import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import torch
import torch.distributions as tdist
import math

class EnhancedRSVDUCBThompson():
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
        self.decay_factor = 0.9  # 異常分數衰減因子

        self.white_black_list = {"white": set(), "black": set()}  # 白名單與黑名單
        self.white_list_ttl = dict()  # 白名單 TTL 記錄
        self.black_list_ttl = dict()  # 黑名單 TTL 記錄
        self.max_ttl = 5  # 名單最大保留輪數

        self.contribution_weights = None

    def detect_poisoned_clients(self, gradients):
        """
        使用多層次異常檢測方法檢測可能的中毒客戶端，並更新異常分數
        """
        client_ids = list(gradients.keys())
        gradients = np.vstack(list(gradients.values()))

        # RSVD 檢測
        n_components = min(gradients.shape[1], int(np.sqrt(gradients.shape[0])))
        u, s, vt = randomized_svd(gradients, n_components=n_components, random_state=42)
        reconstructed = np.dot(u, np.dot(np.diag(s), vt))
        reconstruction_errors = np.linalg.norm(gradients - reconstructed, axis=1)

        # Isolation Forest 檢測
        iso_forest = IsolationForest(random_state=42).fit(gradients)
        isolation_scores = -iso_forest.decision_function(gradients)

        # One-Class SVM 檢測
        one_class_svm = OneClassSVM(kernel='rbf', gamma='scale').fit(gradients)
        svm_scores = -one_class_svm.decision_function(gradients)

        # 指標融合
        combined_scores = (
            0.5 * reconstruction_errors +
            0.3 * isolation_scores +
            0.2 * svm_scores
        )

        # 更新異常分數
        for i, client_id in enumerate(client_ids):
            self.anomaly_score_history[client_id] = (
                self.decay_factor * self.anomaly_score_history[client_id] + 
                (1 - self.decay_factor) * combined_scores[i]
            )

        return self.anomaly_score_history

    def adjust_poisoned_threshold(self, anomaly_scores):
        """
        動態調整異常閾值以保證有效客戶端數量
        """
        threshold = np.percentile(anomaly_scores, 80)
        valid_clients = np.where(anomaly_scores <= threshold, 1, 0)

        # count number of valid clients 
        num_valid_clients = np.sum(valid_clients).astype(int)

        #print("Number of valid clients:", num_valid_clients)

        if num_valid_clients < 10:
            threshold = np.percentile(anomaly_scores, 90)

        self.poisoned_threshold = threshold

    def select_clients(self, epoch, gradients):
        """
        根據異常分數與信任度進行客戶端選擇與貢獻計算
        """
        anomaly_scores = self.detect_poisoned_clients(gradients)
        self.adjust_poisoned_threshold(anomaly_scores)

        valid_clients = np.where(anomaly_scores + self.poisoned_penalty <= self.poisoned_threshold, 1, 0)

        for client_id in self.white_black_list['black']:
            valid_clients[client_id] = 0

        if np.sum(valid_clients) < self.num_join_clients:
            deficit = self.num_join_clients - np.sum(valid_clients)
            black_list_clients = list(self.white_black_list['black'])
            np.random.shuffle(black_list_clients)
            for client_id in black_list_clients:
                valid_clients[client_id] = 1
                deficit -= 1
                if deficit <= 0:
                    break

        weight_ucb = 0.5 + 0.1 * (1 - np.mean(anomaly_scores))
        weight_ts = 1 - weight_ucb

        combined_scores = []
        for i in range(self.num_clients):
            if valid_clients[i] == 1:
                if self.selection_counts[i] > 0:
                    avg_reward = self.performance_history[i] / self.selection_counts[i]
                    delta_i = math.sqrt(2 * math.log(epoch + 1) / self.selection_counts[i])
                    ucb_score = avg_reward + self.c * delta_i
                else:
                    ucb_score = 1e400

                thompson_score = tdist.Beta(self.posterior_alpha[i], self.posterior_beta[i]).sample().item()
                combined_scores.append(weight_ucb * ucb_score + weight_ts * thompson_score)
            else:
                combined_scores.append(-1e400)

        selected_clients = np.argsort(combined_scores)[-self.num_join_clients:]
        for client in selected_clients:
            self.selection_counts[client] += 1

        # === 更新黑名單並設置 TTL ===
        for i in range(self.num_clients):
            if anomaly_scores[i] > self.poisoned_threshold:
                self.poisoned_penalty[i] = min(self.poisoned_penalty[i] + 0.1, 1.0)
                if self.poisoned_penalty[i] > 0.8:
                    self.white_black_list['black'].add(i)
                    self.black_list_ttl[i] = self.max_ttl
            else:
                self.poisoned_penalty[i] = max(self.poisoned_penalty[i] - 0.05, 0)

        # === 過期 TTL 名單清除 ===
        expired_white = []
        for client_id in self.white_list_ttl:
            self.white_list_ttl[client_id] -= 1
            if self.white_list_ttl[client_id] <= 0:
                expired_white.append(client_id)
        for client_id in expired_white:
            self.white_black_list['white'].discard(client_id)
            del self.white_list_ttl[client_id]

        expired_black = []
        for client_id in self.black_list_ttl:
            self.black_list_ttl[client_id] -= 1
            if self.black_list_ttl[client_id] <= 0:
                expired_black.append(client_id)
        for client_id in expired_black:
            self.white_black_list['black'].discard(client_id)
            del self.black_list_ttl[client_id]

        print(f"White List - {self.white_black_list['white']}")
        print(f"Black List - {self.white_black_list['black']}")

        # === 計算聚合貢獻權重 ===
        if np.sum([self.performance_history[client] for client in selected_clients]) == 0:
            self.contribution_weights = [1 / len(selected_clients)] * len(selected_clients)
        else:
            total = sum([self.performance_history[client] for client in selected_clients])
            raw_weights = [self.performance_history[client] / total for client in selected_clients]
            norm_factor = sum(raw_weights)
            self.contribution_weights = [w / norm_factor for w in raw_weights]
        
        #print(f"Contribution Weights - {self.contribution_weights}")

        return selected_clients

    def update(self, selected_clients, rewards):
        """
        更新績效記錄、Thompson 參數與名單 TTL
        """
        decay = 0.9  # 衰減因子避免長期偏好特定客戶端

        for client, reward in zip(selected_clients, rewards):
            self.performance_history[client] = self.performance_history[client] * decay + reward
            self.posterior_alpha[client] = self.posterior_alpha[client] * decay + reward
            self.posterior_beta[client] = self.posterior_beta[client] * decay + (1 - reward)

            if reward > 0.9:
                self.white_black_list['white'].add(client)
                self.white_list_ttl[client] = self.max_ttl

        # 若客戶端平均績效改善則從黑名單移除
        for client_id in list(self.white_black_list['black']):
            avg = self.performance_history[client_id] / max(1, self.selection_counts[client_id])
            if avg > 0.8:
                self.white_black_list['black'].discard(client_id)
                if client_id in self.black_list_ttl:
                    del self.black_list_ttl[client_id]