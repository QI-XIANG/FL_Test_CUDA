import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.ensemble import IsolationForest
import torch
import torch.distributions as tdist
import math
import torch.nn.functional as F
import scipy.special

class EnhancedRSVDUCBThompson:
    """
    Robust federated client selector combining anomaly detection (SVD, IsolationForest),
    multi-armed bandit selection (UCB + Thompson), and diversity boosting.
    Optimized for faster convergence under up to 40% malicious clients (e.g., rare-label attacks).
    """

    def __init__(self, num_clients, num_join_clients, c=1, prior_alpha=1, prior_beta=1):
        self.num_clients = num_clients
        self.num_join_clients = num_join_clients
        # Bandit parameters and histories
        self.selection_counts = np.zeros(num_clients)   # Times selected per client
        self.performance_history = np.zeros(num_clients)  # Decayed cumulative reward per client
        self.posterior_alpha = torch.ones(num_clients) * prior_alpha  # Thompson alpha parameters
        self.posterior_beta = torch.ones(num_clients) * prior_beta    # Thompson beta parameters
        self.c = c  # UCB exploration parameter
        # Anomaly detection histories
        self.anomaly_score_history = np.zeros(num_clients)    # Exponentially-decayed anomaly score
        self.decay_factor = 0.9  # Decay factor for anomaly updates
        self.poisoned_penalty = np.zeros(num_clients)        # Accumulated penalty for suspicious clients
        # Blacklist of clients to de-prioritize based on anomaly
        self.black_list = set()
        # Store last SVD components for diversity embedding
        self.last_svd_u = None
        self.last_svd_s = None
        # Tracking similarity (for anomaly drift) and rewards (for baseline)
        self.client_similarity_history = np.zeros(num_clients)
        self.client_reward_history = np.zeros(num_clients)
        self.global_val_acc_history = []  # History of global validation accuracy

    def detect_poisoned_clients(self, gradients):
        """
        Compute an anomaly score for each client’s gradient update using:
        - SVD reconstruction error
        - Isolation Forest outlier score
        - Cosine-similarity drift from previous round
        - L2 distance from global mean
        """
        client_ids = list(gradients.keys())
        # Stack gradient vectors for computation
        grad_list = []
        for k in client_ids:
            grad = gradients[k].copy()
            norm = np.linalg.norm(grad)
            # Clip extremely large updates
            if norm > 10:
                grad = grad * (10.0 / norm)
            grad_list.append(grad)
        gradients_np = np.vstack(grad_list)

        # Global mean gradient
        global_mean = np.mean(gradients_np, axis=0)
        # Cosine similarity of each client to the global mean
        cos_sims = F.cosine_similarity(
            torch.tensor(gradients_np),
            torch.tensor(global_mean).repeat(len(gradients_np), 1),
            dim=1
        ).numpy()
        # L2 distance to global mean
        l2_dists = np.linalg.norm(gradients_np - global_mean, axis=1)

        # SVD for reconstruction error
        max_components = min(gradients_np.shape[1], gradients_np.shape[0] // 2)
        n_components = max(2, min(max_components, int(np.log2(gradients_np.shape[0]) + 1)))
        u, s, vt = randomized_svd(gradients_np, n_components=n_components, random_state=42)
        # Save SVD components for diversity embedding later
        self.last_svd_u = u
        self.last_svd_s = s
        reconstructed = np.dot(u, np.dot(np.diag(s), vt))
        reconstruction_errors = np.linalg.norm(gradients_np - reconstructed, axis=1)

        # Isolation Forest outlier score (higher means more outlier)
        iso_forest = IsolationForest(n_estimators=50, contamination='auto', random_state=42)
        iso_scores = -iso_forest.fit(gradients_np).decision_function(gradients_np)

        # Cosine-similarity drift from previous round for each client
        similarity_drifts = []
        for i, cid in enumerate(client_ids):
            sim = cos_sims[i]
            drift = abs(sim - self.client_similarity_history[cid])
            self.client_similarity_history[cid] = sim
            similarity_drifts.append(drift)

        # Combine signals (equal weighting)
        combined = (
            reconstruction_errors +
            iso_scores +
            np.array(similarity_drifts) +
            l2_dists
        )
        # Update exponentially-weighted anomaly history
        for i, cid in enumerate(client_ids):
            self.anomaly_score_history[cid] = (
                self.decay_factor * self.anomaly_score_history[cid] +
                (1 - self.decay_factor) * combined[i]
            )
        return self.anomaly_score_history

    def adjust_poisoned_threshold(self, anomaly_scores):
        """
        Dynamically set threshold so that at least num_join_clients are considered valid.
        """
        sorted_scores = np.sort(anomaly_scores)
        # Threshold is the anomaly score of the num_join-th client (ensures enough clients)
        if self.num_join_clients <= len(sorted_scores):
            threshold = sorted_scores[self.num_join_clients - 1]
        else:
            threshold = sorted_scores[-1]
        self.poisoned_threshold = threshold

    def select_clients(self, epoch, gradients):
        """
        Select a batch of clients for this round:
        1. Filter out high-anomaly updates.
        2. Rank remaining clients by a weighted sum of:
           - UCB score (average reward + confidence bonus)
           - Thompson sampling score
           - Diversity score (embedding distance in SVD space).
        """
        # 1. Compute anomaly scores and threshold
        anomaly_scores = self.detect_poisoned_clients(gradients)
        self.adjust_poisoned_threshold(anomaly_scores)

        # Flag valid clients (anomaly + penalty below threshold)
        valid = (anomaly_scores + self.poisoned_penalty <= self.poisoned_threshold).astype(int)
        # Exclude blacklisted clients
        for cid in self.black_list:
            if cid < self.num_clients:
                valid[cid] = 0
        # If too few valid clients, allow next-best (lowest anomaly) to fill
        valid_count = np.sum(valid)
        if valid_count < self.num_join_clients:
            needed = self.num_join_clients - valid_count
            # Pick clients with lowest anomaly that were excluded
            candidate_ids = np.argsort(anomaly_scores)
            for cid in candidate_ids:
                if valid[cid] == 0 and needed > 0:
                    valid[cid] = 1
                    needed -= 1
                    if needed <= 0:
                        break

        # 2. Compute scores
        weight_ucb = 0.7
        weight_ts = 0.3
        # Diversity boosting via SVD embeddings
        diversity = np.zeros(self.num_clients)
        if self.last_svd_u is not None:
            # Map client ID -> index in gradients list
            client_to_idx = {cid: i for i, cid in enumerate(gradients.keys())}
            for i in range(self.num_clients):
                if valid[i] == 1 and i in client_to_idx:
                    idx_i = client_to_idx[i]
                    emb_i = self.last_svd_u[idx_i] * self.last_svd_s  # weighted embedding
                    divers = []
                    for j in range(self.num_clients):
                        if j != i and valid[j] == 1 and j in client_to_idx:
                            idx_j = client_to_idx[j]
                            emb_j = self.last_svd_u[idx_j] * self.last_svd_s
                            sim = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i)*np.linalg.norm(emb_j) + 1e-8)
                            divers.append(1 - sim)
                    diversity[i] = np.mean(divers) if divers else 0.0

        # Combined score array
        combined_scores = np.full(self.num_clients, -np.inf)
        for i in range(self.num_clients):
            if valid[i]:
                # UCB component
                if self.selection_counts[i] > 0:
                    avg_reward = self.performance_history[i] / self.selection_counts[i]
                    bonus = math.sqrt(2 * math.log(epoch + 1) / self.selection_counts[i])
                    ucb_score = avg_reward + self.c * bonus
                else:
                    ucb_score = 1e6  # encourage exploration of new clients
                # Thompson sampling component
                ts_score = tdist.Beta(self.posterior_alpha[i], self.posterior_beta[i]).sample().item()
                # Final combined score: weighted sum + diversity
                combined_scores[i] = (
                    weight_ucb * ucb_score +
                    weight_ts * ts_score +
                    0.3 * diversity[i]
                )
        # Select top-k clients by combined score
        selected = np.argsort(combined_scores)[-self.num_join_clients:]
        # Update selection counts
        for cid in selected:
            self.selection_counts[cid] += 1

        # 3. Update anomaly penalties and blacklist
        for i in range(self.num_clients):
            if anomaly_scores[i] > self.poisoned_threshold:
                self.poisoned_penalty[i] = min(self.poisoned_penalty[i] + 0.1, 1.0)
                if self.poisoned_penalty[i] > 0.8:
                    self.black_list.add(i)
            else:
                self.poisoned_penalty[i] = max(self.poisoned_penalty[i] - 0.05, 0.0)

        return selected

    def update(self, selected_clients, rewards, global_val_acc=None):
        """
        After a training round, update bandit statistics using observed rewards.
        Rewards should be in [0,1] (e.g., normalized accuracy).
        Optionally incorporate global validation accuracy to boost good rounds.
        """
        last_acc = self.global_val_acc_history[-1] if self.global_val_acc_history else 0.0
        if global_val_acc is not None:
            self.global_val_acc_history.append(global_val_acc)
        decay = 0.9  # decay for histories

        for cid, reward in zip(selected_clients, rewards):
            # Update client-specific reward baseline (EMA)
            baseline = self.client_reward_history[cid]
            new_baseline = decay * baseline + (1 - decay) * reward
            delta_reward = max(0.0, reward - baseline)
            self.client_reward_history[cid] = new_baseline

            # Update bandit performance history (with decay) and Beta posteriors
            self.performance_history[cid] = decay * self.performance_history[cid] + reward
            self.posterior_alpha[cid] = self.posterior_alpha[cid] * decay + delta_reward
            self.posterior_beta[cid] = self.posterior_beta[cid] * decay + (1 - delta_reward)

            # Bonus for any significant global improvement this round
            if global_val_acc is not None and (global_val_acc - last_acc) > 0.01:
                self.performance_history[cid] += 0.05

        # Remove clients from blacklist if they recover performance
        for cid in list(self.black_list):
            avg_perf = self.performance_history[cid] / max(1, self.selection_counts[cid])
            if avg_perf > 0.8:
                self.black_list.discard(cid)

    def compute_composite_rewards(self, acc_list, f1_list, auc_pr_list, avg_loss_list, selected_clients):
        """
        (Optional) Compute composite rewards from multiple metrics (accuracy, F1, AUC-PR, etc.)
        for selected clients. Returns delta-rewards for bandit updates.
        """
        def normalize(x):
            arr = np.array(x, dtype=np.float32)
            return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

        acc_norm = normalize(acc_list)
        f1_norm = normalize(f1_list)
        auc_pr_norm = normalize(auc_pr_list)
        loss_inv = 1.0 - normalize(avg_loss_list)
        anomaly_inv = 1.0 - normalize([self.anomaly_score_history[cid] for cid in selected_clients])

        # Weight metrics by variance to emphasize the most variable metric
        variances = np.array([
            np.var(acc_norm), np.var(f1_norm), np.var(loss_inv),
            np.var(anomaly_inv), np.var(auc_pr_norm)
        ])
        weights = scipy.special.softmax(variances * 10)
        min_w = 0.1 / len(weights)
        weights = np.clip(weights, min_w, None)
        weights /= weights.sum()

        raw_rewards = (
            weights[0] * acc_norm +
            weights[1] * f1_norm +
            weights[2] * loss_inv +
            weights[3] * anomaly_inv +
            weights[4] * auc_pr_norm
        )
        total = raw_rewards.sum()
        if total == 0:
            self.contribution_weights = [1.0 / len(selected_clients)] * len(selected_clients)
        else:
            self.contribution_weights = (raw_rewards / total).tolist()

        # Compute delta rewards for Thompson updates
        delta_rewards = [0.0] * self.num_clients
        ema = 0.9
        for idx, cid in enumerate(selected_clients):
            baseline = self.client_reward_history[cid]
            delta = raw_rewards[idx] - baseline
            delta_rewards[cid] = max(0.0, delta)
            self.client_reward_history[cid] = ema * baseline + (1 - ema) * raw_rewards[idx]
        return delta_rewards