import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import torch
import torch.distributions as tdist
import math
import torch.nn.functional as F
import scipy.special

class EnhancedRSVDUCBThompson():
    def __init__(self, num_clients, num_join_clients, min_valid_clients=10, c=1, prior_alpha=1, prior_beta=1):
        self.num_clients = num_clients
        self.num_join_clients = num_join_clients
        self.min_valid_clients = min_valid_clients
        self.selection_counts = np.zeros(num_clients)
        self.performance_history = np.zeros(num_clients)
        self.anomaly_score_history = np.zeros(num_clients)
        self.poisoned_penalty = np.zeros(num_clients)
        self.poisoned_threshold = 0.5
        self.c = c
        self.posterior_alpha = torch.ones(num_clients) * prior_alpha
        self.posterior_beta = torch.ones(num_clients) * prior_beta
        self.decay_factor = 0.9
        self.white_black_list = {"white": set(), "black": set()}
        self.contribution_weights = None
        self.global_gradient_history = []
        self.client_similarity_history = [0.0 for _ in range(num_clients)]
        self.white_list_ttl = dict()
        self.black_list_ttl = dict()
        self.max_ttl = 5
        self.client_reward_history = np.zeros(num_clients)
        self.global_val_acc_history = []

    def detect_poisoned_clients(self, gradients):
        client_ids = list(gradients.keys())
        gradient_list = [gradients[k] for k in client_ids]

        clipped_gradients = []
        for grad in gradient_list:
            norm = np.linalg.norm(grad)
            if norm > 10:
                grad = grad * (10.0 / norm)
            clipped_gradients.append(grad)

        gradients_np = np.vstack(clipped_gradients)

        global_mean = np.mean(gradients_np, axis=0)
        cosine_similarities = F.cosine_similarity(
            torch.tensor(gradients_np), torch.tensor(global_mean).repeat(len(gradients_np), 1), dim=1
        ).numpy()

        max_components = min(gradients_np.shape[1], gradients_np.shape[0] // 2)
        n_components = max(2, int(np.log2(gradients_np.shape[0]) + 1))
        n_components = min(n_components, max_components)

        u, s, vt = randomized_svd(gradients_np, n_components=n_components, random_state=42)
        self.last_client_embeddings = u @ np.diag(s)

        reconstructed = np.dot(u, np.dot(np.diag(s), vt))
        reconstruction_errors = np.linalg.norm(gradients_np - reconstructed, axis=1)

        iso_forest = IsolationForest(n_estimators=50, max_samples='auto', contamination='auto', random_state=42)
        isolation_scores = -iso_forest.fit(gradients_np).decision_function(gradients_np)

        similarity_drifts = []
        for idx, sim in zip(client_ids, cosine_similarities):
            drift = abs(sim - self.client_similarity_history[idx])
            self.client_similarity_history[idx] = sim
            similarity_drifts.append(drift)

        self.global_gradient_history.append(global_mean)
        if len(self.global_gradient_history) > 5:
            self.global_gradient_history.pop(0)

        combined_scores = (
            0.4 * reconstruction_errors +
            0.3 * isolation_scores +
            0.3 * np.array(similarity_drifts)
        )

        for i, client_id in enumerate(client_ids):
            self.anomaly_score_history[client_id] = (
                self.decay_factor * self.anomaly_score_history[client_id] +
                (1 - self.decay_factor) * combined_scores[i]
            )

        return self.anomaly_score_history

    def adjust_poisoned_threshold(self, anomaly_scores):
        threshold = np.percentile(anomaly_scores, 80)
        valid_clients = (anomaly_scores <= threshold).astype(int)
        if np.sum(valid_clients) < 10:
            threshold = np.percentile(anomaly_scores, 90)
        self.poisoned_threshold = threshold

    def select_clients(self, epoch, gradients):
        anomaly_scores = self.detect_poisoned_clients(gradients)
        self.adjust_poisoned_threshold(anomaly_scores)

        valid_clients = (anomaly_scores + self.poisoned_penalty <= self.poisoned_threshold).astype(int)

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

        diversity_scores = [-1e400] * self.num_clients
        client_id_to_idx = {cid: idx for idx, cid in enumerate(gradients.keys())}

        for i in range(self.num_clients):
            if valid_clients[i] and i in client_id_to_idx:
                idx_i = client_id_to_idx[i]
                emb_i = self.last_client_embeddings[idx_i]
                diversities = []
                for j in range(self.num_clients):
                    if j != i and valid_clients[j] and j in client_id_to_idx:
                        idx_j = client_id_to_idx[j]
                        emb_j = self.last_client_embeddings[idx_j]
                        sim = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j) + 1e-8)
                        diversities.append(1 - sim)
                diversity_scores[i] = np.mean(diversities) if diversities else 0.0

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
                combined = weight_ucb * ucb_score + weight_ts * thompson_score + 0.1 * diversity_scores[i]
                combined_scores.append(combined)
            else:
                combined_scores.append(-1e400)

        selected_clients = np.argsort(combined_scores)[-self.num_join_clients:]
        for client in selected_clients:
            self.selection_counts[client] += 1

        for i in range(self.num_clients):
            if anomaly_scores[i] > self.poisoned_threshold:
                self.poisoned_penalty[i] = min(self.poisoned_penalty[i] + 0.1, 1.0)
                if self.poisoned_penalty[i] > 0.8:
                    self.white_black_list['black'].add(i)
                    self.black_list_ttl[i] = self.max_ttl
            else:
                self.poisoned_penalty[i] = max(self.poisoned_penalty[i] - 0.05, 0)

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

        contribution_weights_list = []
        if np.sum([self.performance_history[client] for client in selected_clients]) == 0:
            self.contribution_weights = [1 / len(selected_clients)] * len(selected_clients)
        else:
            total_performance_score = sum([self.performance_history[client] for client in selected_clients])
            for client in selected_clients:
                weight = self.performance_history[client] / total_performance_score
                contribution_weights_list.append(weight)
            sum_weights = sum(contribution_weights_list)
            self.contribution_weights = [w / sum_weights for w in contribution_weights_list]

        return selected_clients

    def update(self, selected_clients, rewards, global_val_acc=None):
        decay = 0.9
        last_global_acc = self.global_val_acc_history[-1] if self.global_val_acc_history else 0.0
        if global_val_acc is not None:
            self.global_val_acc_history.append(global_val_acc)

        for client, reward in zip(selected_clients, rewards):
            baseline = self.client_reward_history[client]
            delta_reward = max(0.0, reward - baseline)
            self.client_reward_history[client] = decay * baseline + (1 - decay) * reward

            self.performance_history[client] = self.performance_history[client] * decay + reward
            self.posterior_alpha[client] = self.posterior_alpha[client] * decay + delta_reward
            self.posterior_beta[client] = self.posterior_beta[client] * decay + (1 - delta_reward)

            if global_val_acc is not None and global_val_acc - last_global_acc > 0.01:
                self.performance_history[client] += 0.05

            if reward > 0.9:
                self.white_black_list['white'].add(client)
                self.white_list_ttl[client] = self.max_ttl

        for client_id in list(self.white_black_list['black']):
            avg = self.performance_history[client_id] / max(1, self.selection_counts[client_id])
            if avg > 0.8:
                self.white_black_list['black'].discard(client_id)
                if client_id in self.black_list_ttl:
                    del self.black_list_ttl[client_id]

    def compute_composite_rewards(self, acc_list, f1_list, auc_pr_list, avg_loss_list, selected_clients):
        def normalize(x):
            x = np.array(x)
            return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)

        acc_norm = normalize(acc_list)
        f1_norm = normalize(f1_list)
        auc_pr_norm = normalize(auc_pr_list)
        loss_inverted = 1.0 - normalize(avg_loss_list)
        anomaly_norm = 1.0 - normalize([self.anomaly_score_history[i] for i in selected_clients])

        variances = np.array([
            np.var(acc_norm),
            np.var(f1_norm),
            np.var(loss_inverted),
            np.var(anomaly_norm),
            np.var(auc_pr_norm)
        ])

        scaled_var = variances * 10
        weights = scipy.special.softmax(scaled_var)

        min_weight = 0.1 / len(weights)
        weights = np.clip(weights, min_weight, 1.0)
        weights /= np.sum(weights)

        raw_rewards = (
            weights[0] * acc_norm +
            weights[1] * f1_norm +
            weights[2] * loss_inverted +
            weights[3] * anomaly_norm +
            weights[4] * auc_pr_norm
        )

        delta_rewards = [0.0] * self.num_clients
        ema_decay = 0.9
        for idx, client_id in enumerate(selected_clients):
            baseline = self.client_reward_history[client_id]
            delta = raw_rewards[idx] - baseline
            delta_rewards[client_id] = max(0.0, delta)
            self.client_reward_history[client_id] = ema_decay * baseline + (1 - ema_decay) * raw_rewards[idx]

        return delta_rewards