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
import copy

# RSVDUCBTE PRO V11 胡搞版本 (定版)

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.utils.extmath import randomized_svd
from sklearn.ensemble import IsolationForest
import math
import os
import copy
from flcore.trainmodel.models import FedAvgCNN_V2, FedAvgCNN
from scipy.stats import rankdata
import scipy.special

class EnhancedRSVDUCBThompson:
    def __init__(self, num_clients, num_join_clients, min_valid_clients=10, c=1, prior_alpha=1, prior_beta=1):
        self.num_clients = num_clients
        self.num_join_clients = num_join_clients
        self.min_valid_clients = max(min_valid_clients, num_join_clients)

        # Bandit state and performance tracking
        self.selection_counts = np.zeros(num_clients)
        self.performance_history = np.zeros(num_clients)
        self.posterior_alpha = np.ones(num_clients) * prior_alpha
        self.posterior_beta = np.ones(num_clients) * prior_beta
        self.client_reward_history = np.zeros(num_clients)

        # Anomaly scoring
        self.anomaly_score_history = np.zeros(num_clients)
        self.poisoned_penalty = np.zeros(num_clients)
        self.poisoned_threshold = 0.5
        self.decay_factor = 0.8

        # Whitelist and blacklist
        self.white_black_list = {"white": set(), "black": set()}
        self.white_list_ttl = {}
        self.black_list_ttl = {}
        self.max_ttl = 5

        # 多樣性與類似度追蹤 (Diversity and similarity tracking)
        self.client_similarity_history = np.zeros(num_clients)
        self.client_reward_history = np.zeros(num_clients)
        self.last_embeddings = None
        self.last_client_ids = None

        # 全域效能歷史（用於中毒影響評估） (Global performance history for poisoning impact)
        self.global_val_acc_history = []

        # 探索強度 (UCB exploration factor)
        self.c = c

        # contribution_weights: 每輪各客戶對總模型的影響力 (Contribution weights of each selected client per round)
        self.contribution_weights = [1.0 / num_join_clients] * num_join_clients
        # [Improvement] baseline weight for server model contribution in aggregation (if used)
        self.baseline_aggregation_weight = 0.0

        # Server model and baseline
        self.server_model = FedAvgCNN_V2(in_features=3, num_classes=10).to("cuda")
        self.optimizer = torch.optim.SGD(self.server_model.parameters(), lr=0.005)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.99)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.baseline_gradients = None
        self.server_data_loader = None  # will be cached once loaded
        self.server_data_cache = None
        
        # 移除舊的 server 模型檔案 (Ensure no stale model file)
        if os.path.exists("server_model_latest.pth"):
            os.remove("server_model_latest.pth")

        if os.path.exists("server_model_latest.pth"):
            self.load_server_model()
            print("[INFO] Loaded previous server model.")
        else:
            print("[INFO] No previous server model found.")
    
    def load_server_data_new(self):
        server_data_path = "/home/dslab/qixiang/FL_Test_Env_CUDA/dataset/CINIC10_100_alpha01_server2/server_data.npz"
        train_data = np.load(server_data_path, allow_pickle=True)['data'].tolist()
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)
        self.server_data = [(x, y) for x, y in zip(X_train, y_train)]
        train_loader = DataLoader(self.server_data, batch_size=10, drop_last=True, shuffle=True)
        return train_loader

    def load_server_data_skewed(self, alpha=0.1, n_server_samples=1000, retained_classes=5, batch_size=10):
        if self.server_data_loader is not None:
            return self.server_data_loader

        server_data_path = "/home/dslab/qixiang/FL_Test_Env_CUDA/dataset/CINIC10_100_alpha01_server2/server_data.npz"
        train_data = np.load(server_data_path, allow_pickle=True)['data'].tolist()
        X_all = torch.Tensor(train_data['x']).type(torch.float32)
        y_all = torch.Tensor(train_data['y']).type(torch.int64)

        full_class_set = np.unique(y_all.numpy())
        selected_classes = np.random.choice(full_class_set, size=retained_classes, replace=False)
        selected_classes = sorted(selected_classes.tolist())

        class_indices = [torch.where(y_all == c)[0] for c in selected_classes]

        proportions = np.random.dirichlet([alpha] * retained_classes)
        proportions = (proportions / proportions.sum()) * n_server_samples
        proportions = proportions.astype(int)

        sampled_indices = []
        for i, idx_pool in enumerate(class_indices):
            if len(idx_pool) == 0: continue
            n = min(len(idx_pool), proportions[i])
            selected = idx_pool[torch.randperm(len(idx_pool))[:n]]
            sampled_indices.extend(selected.tolist())

        X_selected = X_all[sampled_indices]
        y_selected = y_all[sampled_indices]
        self.server_data_cache = [(x, y) for x, y in zip(X_selected, y_selected)]

        loader = DataLoader(self.server_data_cache, batch_size=batch_size, shuffle=True, drop_last=True)
        self.server_data_loader = loader
        return loader

    def train_server_model(self, epochs=3):
        self.server_model.to("cuda")
        self.server_model.train()

        loader = self.load_server_data_skewed()
        if self.check_current_server_model_performance() > self.global_val_acc_history[-1]:
            print("[INFO] Server model is overhead the global model...")
            pass
        else:
            for epoch in range(epochs):
                total_loss, correct, total = 0.0, 0, 0
                for xb, yb in loader:
                    xb, yb = xb.cuda(), yb.cuda()
                    out = self.server_model(xb)
                    loss = self.loss_fn(out, yb)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item() * xb.size(0)
                    correct += (out.argmax(1) == yb).sum().item()
                    total += xb.size(0)

                acc = correct / total
                avg_loss = total_loss / total
                print(f"[Server] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {acc:.4f}")
                self.scheduler.step()
                if acc > 0.98 or avg_loss < 0.01:
                    break

        self.save_server_model()
        self.server_model.eval()
    
    def check_current_server_model_performance(self):
        loader = self.load_server_data_skewed()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.cuda(), yb.cuda()
            out = self.server_model(xb)
            loss = self.loss_fn(out, yb)
            total_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.size(0)

        acc = correct / total
        return acc

    def compute_server_gradients(self, model, loss_fn):
        temp = copy.deepcopy(model)
        temp.to("cuda")
        temp.zero_grad()
        loader = self.load_server_data_skewed()

        for xb, yb in loader:
            xb, yb = xb.cuda(), yb.cuda()
            out = temp(xb)
            loss = loss_fn(out, yb)
            loss.backward()

        grads = [p.grad.cpu().flatten() for p in temp.parameters() if p.grad is not None]
        return torch.cat(grads).numpy()

    def robust_trimmed_mean(self, grads, beta=0.1):
        grads = np.stack(grads)
        sorted_grads = np.sort(grads, axis=0)
        n = grads.shape[0]
        b = int(n * beta)
        trimmed = sorted_grads[b:n - b, :]
        return np.mean(trimmed, axis=0)

    def detect_poisoned_clients(self, gradients, epoch):
        if epoch % 6 == 0:
            print("[INFO] Training server model... Epoch:", epoch)
            self.train_server_model()
        self.baseline_gradients = self.compute_server_gradients(self.server_model, self.loss_fn)

        grads_np = np.vstack([g for g in gradients.values()])
        diffs = grads_np - self.baseline_gradients

        recon_error = np.linalg.norm(diffs, axis=1)
        iso = IsolationForest(n_estimators=50).fit(grads_np)
        iso_scores = -iso.decision_function(grads_np)

        rank_matrix = np.array([rankdata(g) for g in grads_np])
        rank_std = np.std(rank_matrix, axis=1)

        combined_scores = 0.4 * recon_error + 0.3 * iso_scores + 0.3 * rank_std

        for i, cid in enumerate(gradients.keys()):
            self.anomaly_score_history[cid] = self.decay_factor * self.anomaly_score_history[cid] + (1 - self.decay_factor) * combined_scores[i]

        return self.anomaly_score_history
    
    def adjust_poisoned_threshold(self, anomaly_scores):
        # 動態調整懷疑客戶的異常閾值 (Adjust threshold for flagging clients as malicious)
        base_percentile = 80
        # If global validation accuracy dropped significantly, be more strict (lower threshold)
        if len(self.global_val_acc_history) >= 2 and (self.global_val_acc_history[-1] < self.global_val_acc_history[-2] - 0.01):
            base_percentile = 70
        threshold = np.percentile(anomaly_scores, base_percentile)
        # 保證至少有一定數量的客戶被視為正常 (Ensure at least some clients are considered valid)
        if np.sum(anomaly_scores <= threshold) < 10:
            threshold = np.percentile(anomaly_scores, max(base_percentile + 10, 90))
        if np.sum(anomaly_scores <= threshold) < 10:
            perc = 100 * (self.num_join_clients / self.num_clients)
            threshold = np.percentile(anomaly_scores, perc)
        self.poisoned_threshold = threshold

    def select_clients(self, epoch, gradients):
        # 如果 optimizer 或 scheduler 尚未載入，嘗試載入伺服器模型 (Load server model if not already loaded)
        if self.optimizer is None or self.scheduler is None:
            if os.path.exists('server_model_latest.pth'):
                self.load_server_model()
                print("[DEBUG] Server model restored before client selection.")
            else:
                print("[DEBUG] No server model found before client selection.")
        
        anomaly_scores = self.detect_poisoned_clients(gradients, epoch)
        self.adjust_poisoned_threshold(anomaly_scores)

        # 判斷每個客戶是否為有效客戶 (Determine valid clients based on anomaly scores and penalties)
        valid = np.zeros(self.num_clients, dtype=bool)
        for cid in range(self.num_clients):
            if anomaly_scores[cid] + self.poisoned_penalty[cid] <= self.poisoned_threshold:
                valid[cid] = True

        # 強制應用白名單/黑名單 (Apply whitelist/blacklist overrides)
        for cid in self.white_black_list['black']:
            valid[cid] = False
        for cid in self.white_black_list['white']:
            valid[cid] = True

        # 如果有效客戶不足，從黑名單中挑選一些以滿足 num_join_clients (Ensure enough clients selected by relaxing blacklist if needed)
        if np.sum(valid) < self.num_join_clients:
            deficit = self.num_join_clients - np.sum(valid)
            black_candidates = list(self.white_black_list['black'])
            np.random.shuffle(black_candidates)
            for cid in black_candidates:
                if deficit <= 0:
                    break
                valid[cid] = True
                deficit -= 1

        # 動態調整 UCB vs Thompson Sampling 權重 (Adjust exploration/exploitation balance based on anomaly level)
        mean_anomaly = np.mean(anomaly_scores)
        weight_ucb = np.clip(0.5 + 0.1 * (1 - mean_anomaly), 0.1, 0.9)
        weight_ts = 1.0 - weight_ucb

        # 計算每個有效客戶的多樣性分數 (Compute diversity score for each valid client)
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
                    # Cosine similarity -> diversity distance
                    cos_sim = np.dot(emb_i, emb_j) / ((np.linalg.norm(emb_i) + 1e-8) * (np.linalg.norm(emb_j) + 1e-8))
                    distances.append(1 - cos_sim)
                diversity[cid] = np.mean(distances) if distances else 0.0

        # 計算每個客戶的分數來進行選擇 (Calculate combined score for each client for selection)
        combined_scores = np.full(self.num_clients, -1e9)
        for cid in range(self.num_clients):
            if not valid[cid]:
                continue
            if self.selection_counts[cid] > 0:
                avg_reward = self.performance_history[cid] / self.selection_counts[cid]
                exploration_bonus = math.sqrt(2 * math.log(epoch + 1) / self.selection_counts[cid])
                ucb_score = avg_reward + self.c * exploration_bonus
            else:
                ucb_score = 1e6  # A very high score for never-selected clients to ensure they get explored at least once
            ts_score = np.random.beta(self.posterior_alpha[cid], self.posterior_beta[cid])
            # [Improvement] Include a small penalty for higher anomaly score to deprioritize borderline suspicious clients
            combined_scores[cid] = weight_ucb * ucb_score + weight_ts * ts_score + 0.2 * diversity[cid] - 0.1 * anomaly_scores[cid]

        # 選擇分數最高的 num_join_clients 個客戶 (Select the top-scoring clients)
        selected_indices = np.argsort(combined_scores)[-self.num_join_clients:]
        selected_clients = list(selected_indices)
        for cid in selected_clients:
            self.selection_counts[cid] += 1

        # 更新惡意懲罰和黑名單 (Update penalties and blacklist for highly suspicious clients)
        for cid in range(self.num_clients):
            if anomaly_scores[cid] > self.poisoned_threshold:
                # Increase penalty for suspected malicious clients
                self.poisoned_penalty[cid] = min(self.poisoned_penalty[cid] + 0.1, 1.0)
                if self.poisoned_penalty[cid] > 0.8:
                    self.white_black_list['black'].add(cid)
                    self.black_list_ttl[cid] = self.max_ttl
            else:
                # Gradually reduce penalty for normal clients
                self.poisoned_penalty[cid] = max(self.poisoned_penalty[cid] - 0.05, 0.0)

        # 更新黑/白名單 TTL (Update TTL for blacklist/whitelist entries)
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

    def update(self, selected_clients, rewards, global_val_acc=None, global_model=None):
        # 融合後模型的性能更新 (Update internal performance statistics after aggregation round)
        decay = 0.9
        last_acc = self.global_val_acc_history[-1] if self.global_val_acc_history else 0.0
        if global_val_acc is not None:
            self.global_val_acc_history.append(global_val_acc)

        for cid in selected_clients:
            reward = rewards[cid]
            baseline = self.client_reward_history[cid]
            delta_reward = max(0.0, reward - baseline)
            # Update exponential moving averages for reward and performance
            self.client_reward_history[cid] = decay * baseline + (1 - decay) * reward
            self.performance_history[cid] = decay * self.performance_history[cid] + reward
            self.posterior_alpha[cid] = self.posterior_alpha[cid] * decay + delta_reward
            self.posterior_beta[cid] = self.posterior_beta[cid] * decay + (1 - delta_reward)

            # If global validation accuracy improved significantly, give a small bonus to this client's performance history
            if global_val_acc is not None and (global_val_acc - last_acc) > 0.01:
                self.performance_history[cid] += 0.05
            if reward > 0.9:
                # [Improvement] Only whitelist if client has low anomaly score
                if self.anomaly_score_history[cid] <= self.poisoned_threshold * 0.5:
                    self.white_black_list['white'].add(cid)
                    self.white_list_ttl[cid] = self.max_ttl
                    if cid in self.white_black_list['black']:
                        self.white_black_list['black'].discard(cid)
                        self.black_list_ttl.pop(cid, None)

        # 從黑名單中移除表現良好的客戶 (Remove clients from blacklist if their average performance becomes high)
        for cid in list(self.white_black_list['black']):
            if self.selection_counts[cid] > 0:
                avg_perf = self.performance_history[cid] / self.selection_counts[cid]
                if avg_perf > 0.8:
                    self.white_black_list['black'].discard(cid)
                    self.black_list_ttl.pop(cid, None)
        
        if global_model is not None:
            self.fine_tune_global_model(global_model)
            pass

    def compute_composite_rewards(self, acc_list, f1_list, auc_pr_list, avg_loss_list, selected_clients):
        # 將多種指標正規化 (Normalize various metrics for combination)
        def normalize(x):
            arr = np.array(x, dtype=float)
            return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)

        acc_norm = normalize(acc_list)
        f1_norm = normalize(f1_list)
        auc_pr_norm = normalize(auc_pr_list)
        loss_inv = 1.0 - normalize(avg_loss_list)
        norm_anomaly = normalize(self.anomaly_score_history)
        anomaly_inv = np.array([1.0 - norm_anomaly[cid] for cid in selected_clients])

        # 動態計算每種指標的權重 (Dynamically compute weights for each metric via variance)
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

        # 計算每個客戶的綜合 raw reward (weighted sum of metrics)
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

        #print(f"[DEBUG] Contribution weights: {self.contribution_weights}")

        # 估計每個客戶相對於自身歷史表現的改善 (Compute delta rewards for bandit update)
        delta_rewards = [0.0] * self.num_clients
        ema = 0.85
        for idx, cid in enumerate(selected_clients):
            baseline = self.client_reward_history[cid]
            delta = raw_rewards[idx] - baseline
            delta_rewards[cid] = max(0.0, delta)
            self.client_reward_history[cid] = ema * baseline + (1 - ema) * raw_rewards[idx]

        return delta_rewards
    
    def print_model_checksum(self):
        # Debug function to print sum of absolute model parameters (to verify model state changes if needed)
        total = 0.0
        for param in self.server_model.parameters():
            total += param.data.abs().sum().item()
        print(f"[DEBUG] Server model checksum: {total:.6f}")
    
    def save_server_model(self, path="server_model_latest.pth"):
        torch.save(self.server_model.state_dict(), path)

    def load_server_model(self, path="server_model_latest.pth"):
        state_dict = torch.load(path, map_location="cpu")
        self.server_model.load_state_dict(state_dict)
        self.server_model.to("cuda")
        # Note: We could reinitialize optimizer/scheduler here if needed (not strictly necessary since we retrain server model each round)

    # fine tune global model on server's small dataset
    def fine_tune_global_model(self, global_model, epochs=3, device="cuda"):
        loader = self.load_server_data_new()
        global_model.to(device)
        opt = torch.optim.SGD(global_model.parameters(), lr=0.005)
        ce = torch.nn.CrossEntropyLoss()
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                out = global_model(xb)
                loss = ce(out, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        global_model.to("cuda")
        print(f"[INFO] Finished fine-tuning global model")