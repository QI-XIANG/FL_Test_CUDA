import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.ensemble import IsolationForest
import torch
import math
import scipy.special
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from flcore.trainmodel.models import FedAvgCNN_V2
from torchvision.transforms.functional import normalize
import os

class EnhancedRSVDUCBThompson:
    def __init__(self, num_clients, num_join_clients, server_data_path, min_valid_clients=10, c=1, prior_alpha=1, prior_beta=1):
        # 基本參數
        self.num_clients = num_clients
        self.num_join_clients = num_join_clients
        self.min_valid_clients = max(min_valid_clients, num_join_clients)

        # Bandit 狀態參數
        self.selection_counts = np.zeros(num_clients)
        self.performance_history = np.zeros(num_clients)
        self.posterior_alpha = np.ones(num_clients) * prior_alpha
        self.posterior_beta = np.ones(num_clients) * prior_beta

        # 異常分數與惡意客戶偵測
        self.anomaly_score_history = np.zeros(num_clients)
        self.poisoned_penalty = np.zeros(num_clients)          
        self.poisoned_threshold = 0.5
        self.decay_factor = 0.9

        # 白名單與黑名單管理
        self.white_black_list = {"white": set(), "black": set()}
        self.white_list_ttl = {}
        self.black_list_ttl = {}
        self.max_ttl = 5

        # 多樣性與類似度追蹤
        self.client_similarity_history = np.zeros(num_clients)
        self.client_reward_history = np.zeros(num_clients)
        self.last_embeddings = None
        self.last_client_ids = None

        # 全域效能歷史（用於中毒影響評估）
        self.global_val_acc_history = []                       

        # 探索強度 (UCB)
        self.c = c

        # contribution_weights: 每輪各客戶對總模型的影響力
        self.contribution_weights = [1.0 / num_join_clients] * num_join_clients

        # Server 本地模型相關
        self.server_data = None
        self.server_model = FedAvgCNN_V2(in_features=3, num_classes=100).to("cuda")
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # 載入本地伺服器資料集
        train_data = np.load(server_data_path, allow_pickle=True)['data'].tolist()
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        self.server_data = (X_train, y_train)

        print(f"[DEBUG] raw image shape: {X_train.shape}")

        # 設定是否進行數據增強
        #self.data_augmentation = False

        # 初始化 baseline_gradients
        self.baseline_gradients = None
        # 初始化 server_optimizer
        self.optimizer = torch.optim.SGD(self.server_model.parameters(), lr=0.005)
        # 初始化 server_scheduler
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=0.99
        )

        if os.path.exists("server_model_latest.pth"):
            os.remove("server_model_latest.pth")

        # 如果有檔案，載入過去模型參數
        if os.path.exists("server_model_latest.pth"):
            self.load_server_model()
            print("[INFO] Loaded previous server model.")
        else:
            print("[INFO] No previous server model found.")
        
    
    def load_server_data(self):
        if self.server_data is None:
            raise ValueError("Server dataset not loaded.")
        train_loader = DataLoader(self.server_data, batch_size=10, drop_last=True, shuffle=True)
        
        return train_loader

    def reset_batchnorm_running_stats(self):
        for module in self.server_model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.reset_running_stats()

    def train_server_model(self, epochs=10, device='cuda'):
        if self.server_data is None:
            raise ValueError("Server dataset not loaded.")

        self.server_model.to(device)
        self.server_model.train()

        loader = self.load_server_data()

        # 只初始化一次 optimizer 和 scheduler（若未載入過）
        if self.optimizer is None:
            self.optimizer = self.optimizer = torch.optim.SGD(self.server_model.parameters(), lr=0.005)
        if self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=0.99
        )

        loss_fn = torch.nn.CrossEntropyLoss()
        # 只有模型首次初始化才 reset BN stats
        if not os.path.exists("server_model_latest.pth"):
            self.reset_batchnorm_running_stats()

        self.server_model.train()

        for epoch in range(epochs):
            #total_loss = 0
            #total_correct = 0
            #total_samples = 0

            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                output = self.server_model(xb)
                loss = loss_fn(output, yb)
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()

                #total_loss += loss.item() * xb.size(0)
                #total_correct += (output.argmax(dim=1) == yb).sum().item()
                #total_samples += xb.size(0)

            #avg_loss = total_loss / total_samples
            #acc = total_correct / total_samples
            #print(f"[Server Training] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {acc:.4f}")
            self.scheduler.step()

        self.server_model.eval()
        self.server_model.to('cpu')

        #self.print_model_checksum()
        self.save_server_model("server_model_latest.pth")

    
    def compute_server_gradients(self, model, loss_fn):
        """
        Compute the gradient vector of the given model on the server-side dataset (set via load_server_dataset).
        Returns a 1D NumPy array of gradients (flattened).
        """
        if self.server_data is None:
            return None

        x, y = self.server_data
        device = x.device
        model.to(device)
        model.zero_grad()

        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()

        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.detach().cpu().flatten())
        grad_vector = torch.cat(grads).numpy()
        return grad_vector

    def detect_poisoned_clients(self, gradients):
        # === 載入上一輪訓練好的 server model ===
        if os.path.exists("server_model_latest.pth"):
            self.load_server_model()  # 會同時設定 optimizer/scheduler
            print("[DEBUG] Loaded last round's server model before training.")
        else:
            print("[DEBUG] No previous model found, training from scratch.")
            self.reset_batchnorm_running_stats()

        # === 執行 server model 本地訓練以更新 baseline_gradient ===
        print("[DEBUG] Server training to refresh baseline gradients...")
        self.train_server_model()  # 將在目前 model 上進行 10 epoch 的 local training

        # 重新計算 baseline gradients（每一輪都會更新）
        self.baseline_gradients = self.compute_server_gradients(self.server_model, self.loss_fn)
        if self.baseline_gradients is None:
            raise ValueError("Server gradient computation failed.")

        print("[DEBUG] Baseline gradient re-computed.")

        client_ids = list(gradients.keys())
        grad_list = []
        for cid in client_ids:
            grad = gradients[cid].copy()
            norm = np.linalg.norm(grad)
            if norm > 10:
                grad = grad * (10.0 / norm)
            grad_list.append(grad)
        gradients_np = np.vstack(grad_list)

        global_mean = np.mean(gradients_np, axis=0)
        global_median = np.median(gradients_np, axis=0)

        l2_dists_mean = np.linalg.norm(gradients_np - global_mean, axis=1)
        l2_dists_median = np.linalg.norm(gradients_np - global_median, axis=1)

        # RSVD 特徵抽取
        max_components = min(gradients_np.shape[1], gradients_np.shape[0] // 2)
        n_components = min(max_components, max(2, int(np.log2(len(client_ids))) + 1))
        U, S, Vt = randomized_svd(gradients_np, n_components=n_components, random_state=42)
        reconstructed = np.dot(U, np.dot(np.diag(S), Vt))
        reconstruction_errors = np.linalg.norm(gradients_np - reconstructed, axis=1)
        self.last_embeddings = U * S
        self.last_client_ids = client_ids

        # Isolation Forest score
        iso_forest = IsolationForest(n_estimators=50, max_samples='auto', contamination=0.4, random_state=42)
        isolation_scores = -iso_forest.fit(gradients_np).decision_function(gradients_np)

        # Cosine distance to server-side baseline gradient
        baseline_norm = np.linalg.norm(self.baseline_gradients) + 1e-8
        baseline_cos_dist = np.array([
            1.0 - (np.dot(g, self.baseline_gradients) / ((np.linalg.norm(g) + 1e-8) * baseline_norm))
            for g in gradients_np
        ])

        # Combine scores
        combined_scores = (
            0.25 * reconstruction_errors +
            0.15 * isolation_scores +
            0.20 * l2_dists_mean +
            0.30 * l2_dists_median +
            0.20 * baseline_cos_dist
        )

        # Update EMA history
        for idx, cid in enumerate(client_ids):
            self.anomaly_score_history[cid] = (
                self.decay_factor * self.anomaly_score_history[cid] +
                (1 - self.decay_factor) * combined_scores[idx]
            )

        return self.anomaly_score_history
    
    def adjust_poisoned_threshold(self, anomaly_scores):
        base_percentile = 80
        if len(self.global_val_acc_history) >= 2 and (self.global_val_acc_history[-1] < self.global_val_acc_history[-2] - 0.01):
            base_percentile = 70
        threshold = np.percentile(anomaly_scores, base_percentile)

        if np.sum(anomaly_scores <= threshold) < 10:
            threshold = np.percentile(anomaly_scores, max(base_percentile + 10, 90))
        if np.sum(anomaly_scores <= threshold) < 10:
            perc = 100 * (self.num_join_clients / self.num_clients)
            threshold = np.percentile(anomaly_scores, perc)

        self.poisoned_threshold = threshold

    def select_clients(self, epoch, gradients):
        # 僅首次載入模型，若已載入過則不重複
        if self.optimizer is None or self.scheduler is None:
            if os.path.exists('server_model_latest.pth'):
                self.load_server_model()
                print("[DEBUG] Server model restored before client selection.")
            else:
                print("[DEBUG] No server model found before client selection.")
        
        anomaly_scores = self.detect_poisoned_clients(gradients)
        self.adjust_poisoned_threshold(anomaly_scores)

        valid = np.zeros(self.num_clients, dtype=bool)
        for cid in range(self.num_clients):
            if anomaly_scores[cid] + self.poisoned_penalty[cid] <= self.poisoned_threshold:
                valid[cid] = True

        for cid in self.white_black_list['black']:
            valid[cid] = False
        for cid in self.white_black_list['white']:
            valid[cid] = True

        if np.sum(valid) < self.num_join_clients:
            deficit = self.num_join_clients - np.sum(valid)
            black_candidates = list(self.white_black_list['black'])
            np.random.shuffle(black_candidates)
            for cid in black_candidates:
                if deficit <= 0:
                    break
                valid[cid] = True
                deficit -= 1

        mean_anomaly = np.mean(anomaly_scores)
        weight_ucb = np.clip(0.5 + 0.1 * (1 - mean_anomaly), 0.1, 0.9)
        weight_ts = 1.0 - weight_ucb

        diversity = np.zeros(self.num_clients)
        if self.last_embeddings is not None:
            client_index_map = {cid: idx for idx, cid in enumerate(self.last_client_ids)}
            for cid in range(self.num_clients):
                if not valid[cid] or cid not in client_index_map:
                    continue
                idx_i = client_index_map[cid]
                emb_i = self.last_embeddings[idx_i]
                distances = []
                for other_cid in self.last_client_ids:
                    if other_cid == cid or not valid[other_cid]:
                        continue
                    idx_j = client_index_map.get(other_cid)
                    if idx_j is None:
                        continue
                    emb_j = self.last_embeddings[idx_j]
                    cos_sim = np.dot(emb_i, emb_j) / ((np.linalg.norm(emb_i) + 1e-8) * (np.linalg.norm(emb_j) + 1e-8))
                    distances.append(1 - cos_sim)
                diversity[cid] = np.mean(distances) if distances else 0.0

        combined_scores = np.full(self.num_clients, -1e9)
        for cid in range(self.num_clients):
            if not valid[cid]:
                continue
            if self.selection_counts[cid] > 0:
                avg_reward = self.performance_history[cid] / self.selection_counts[cid]
                exploration_bonus = math.sqrt(2 * math.log(epoch + 1) / self.selection_counts[cid])
                ucb_score = avg_reward + self.c * exploration_bonus
            else:
                ucb_score = 1e6
            ts_score = np.random.beta(self.posterior_alpha[cid], self.posterior_beta[cid])
            combined_scores[cid] = weight_ucb * ucb_score + weight_ts * ts_score + 0.2 * diversity[cid]

        selected_indices = np.argsort(combined_scores)[-self.num_join_clients:]
        selected_clients = list(selected_indices)
        for cid in selected_clients:
            self.selection_counts[cid] += 1

        for cid in range(self.num_clients):
            if anomaly_scores[cid] > self.poisoned_threshold:
                self.poisoned_penalty[cid] = min(self.poisoned_penalty[cid] + 0.1, 1.0)
                if self.poisoned_penalty[cid] > 0.8:
                    self.white_black_list['black'].add(cid)
                    self.black_list_ttl[cid] = self.max_ttl
            else:
                self.poisoned_penalty[cid] = max(self.poisoned_penalty[cid] - 0.05, 0.0)

        for cid, ttl in list(self.black_list_ttl.items()):
            self.black_list_ttl[cid] -= 1
            if self.black_list_ttl[cid] <= 0:
                self.white_black_list['black'].discard(cid)
                del self.black_list_ttl[cid]
        for cid, ttl in list(self.white_list_ttl.items()):
            self.white_list_ttl[cid] -= 1
            if self.white_list_ttl[cid] <= 0:
                self.white_black_list['white'].discard(cid)
                del self.white_list_ttl[cid]

        return selected_clients

    def update(self, selected_clients, rewards, global_val_acc=None):
        decay = 0.9
        last_acc = self.global_val_acc_history[-1] if self.global_val_acc_history else 0.0
        if global_val_acc is not None:
            self.global_val_acc_history.append(global_val_acc)

        for cid in selected_clients:
            reward = rewards[cid]
            baseline = self.client_reward_history[cid]
            delta_reward = max(0.0, reward - baseline)
            self.client_reward_history[cid] = decay * baseline + (1 - decay) * reward
            self.performance_history[cid] = decay * self.performance_history[cid] + reward
            self.posterior_alpha[cid] = self.posterior_alpha[cid] * decay + delta_reward
            self.posterior_beta[cid] = self.posterior_beta[cid] * decay + (1 - delta_reward)

            if global_val_acc is not None and (global_val_acc - last_acc) > 0.01:
                self.performance_history[cid] += 0.05
            if reward > 0.9:
                self.white_black_list['white'].add(cid)
                self.white_list_ttl[cid] = self.max_ttl
                if cid in self.white_black_list['black']:
                    self.white_black_list['black'].discard(cid)
                    self.black_list_ttl.pop(cid, None)

        for cid in list(self.white_black_list['black']):
            if self.selection_counts[cid] > 0:
                avg_perf = self.performance_history[cid] / self.selection_counts[cid]
                if avg_perf > 0.8:
                    self.white_black_list['black'].discard(cid)
                    self.black_list_ttl.pop(cid, None)

    def compute_composite_rewards(self, acc_list, f1_list, auc_pr_list, avg_loss_list, selected_clients):
        def normalize(x):
            arr = np.array(x, dtype=float)
            return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)

        acc_norm = normalize(acc_list)
        f1_norm = normalize(f1_list)
        auc_pr_norm = normalize(auc_pr_list)
        loss_inv = 1.0 - normalize(avg_loss_list)

        norm_anomaly = normalize(self.anomaly_score_history)
        anomaly_inv = np.array([1.0 - norm_anomaly[cid] for cid in selected_clients])

        variances = np.array([
            np.var(acc_norm),
            np.var(f1_norm),
            np.var(loss_inv),
            np.var(anomaly_inv),
            np.var(auc_pr_norm)
        ])
        weights = scipy.special.softmax(variances * 10)
        min_w = 0.1 / len(weights)
        weights = np.clip(weights, min_w, None)
        weights /= np.sum(weights)

        raw_rewards = (weights[0] * acc_norm +
                       weights[1] * f1_norm +
                       weights[2] * loss_inv +
                       weights[3] * anomaly_inv +
                       weights[4] * auc_pr_norm)

        total_reward = np.sum(raw_rewards)
        if total_reward == 0:
            contribution_weights = [1.0 / len(selected_clients)] * len(selected_clients)
        else:
            contribution_weights = (raw_rewards / total_reward).tolist()
        self.contribution_weights = contribution_weights

        delta_rewards = [0.0] * self.num_clients
        ema = 0.9
        for idx, cid in enumerate(selected_clients):
            baseline = self.client_reward_history[cid]
            delta = raw_rewards[idx] - baseline
            delta_rewards[cid] = max(0.0, delta)
            self.client_reward_history[cid] = ema * baseline + (1 - ema) * raw_rewards[idx]

        return delta_rewards
    
    def print_model_checksum(self):
        total = 0.0
        for param in self.server_model.parameters():
            total += param.data.abs().sum().item()
        print(f"[DEBUG] Server model checksum: {total:.6f}")
    
    def save_server_model(self, path="server_model_latest.pth"):
        torch.save(self.server_model.state_dict(), path)

    def load_server_model(self, path="server_model_latest.pth"):
        state_dict = torch.load(path, map_location="cpu")
        self.server_model.load_state_dict(state_dict)

        self.server_model.to("cuda")  # 確保 optimizer 初始化在正確 device
        #self.server_optimizer = torch.optim.SGD(self.server_model.parameters(), lr=0.005, weight_decay=1e-4)
        #self.server_scheduler = torch.optim.lr_scheduler.StepLR(self.server_optimizer, step_size=3, gamma=0.8)
        #print(f"[INFO] Server model loaded from {path} and optimizer/scheduler reinitialized.")