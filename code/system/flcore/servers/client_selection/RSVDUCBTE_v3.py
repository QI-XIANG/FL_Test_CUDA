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
    Robust federated client selector combining anomaly detection (SVD, IsolationForest, etc.),
    multi-armed bandit selection (UCB + Thompson Sampling), and diversity boosting.
    """

    def __init__(self, num_clients, num_join_clients, min_valid_clients=10, c=1, prior_alpha=1, prior_beta=1):
        self.num_clients = num_clients
        self.num_join_clients = num_join_clients
        self.min_valid_clients = min_valid_clients
        # Track selection counts and performance
        self.selection_counts = np.zeros(num_clients)        # Times selected per client
        self.performance_history = np.zeros(num_clients)     # Cumulative reward
        self.anomaly_score_history = np.zeros(num_clients)   # Exponentially-decayed anomaly score
        self.poisoned_penalty = np.zeros(num_clients)        # Penalty for suspected malicious clients
        self.poisoned_threshold = 0.5                       # Dynamic anomaly threshold
        self.c = c  # UCB exploration parameter
        # Beta priors for Thompson Sampling (initialized to Beta(1,1))
        self.posterior_alpha = torch.ones(num_clients) * prior_alpha
        self.posterior_beta = torch.ones(num_clients) * prior_beta
        self.decay_factor = 0.9  # Decay for anomaly score updates

        # White/black lists with TTL (time-to-live)
        self.white_black_list = {"white": set(), "black": set()}
        self.white_list_ttl = dict()
        self.black_list_ttl = dict()
        self.max_ttl = 5

        self.client_similarity_history = np.zeros(num_clients)  # For tracking cosine similarity drift
        self.client_reward_history = np.zeros(num_clients)      # For baseline reward calculations
        self.global_val_acc_history = []
        self.contribution_weights = None

    def detect_poisoned_clients(self, gradients):
        """
        Compute an anomaly score for each client’s gradient update.
        Uses gradient clipping, global mean/median differences, SVD reconstruction error,
        Isolation Forest, and cosine-similarity drift as signals.
        """
        client_ids = list(gradients.keys())
        # Stack gradient vectors for processing
        grad_list = []
        for k in client_ids:
            grad = gradients[k].copy()
            norm = np.linalg.norm(grad)
            # Clip extreme gradients
            if norm > 10:
                grad = grad * (10.0 / norm)
            grad_list.append(grad)
        gradients_np = np.vstack(grad_list)

        # Global statistics
        global_mean = np.mean(gradients_np, axis=0)
        #median_grad = np.median(gradients_np, axis=0)

        # Cosine similarity to mean (per client)
        cos_sims = F.cosine_similarity(
            torch.tensor(gradients_np), torch.tensor(global_mean).repeat(len(gradients_np), 1), dim=1
        ).numpy()
        # L2 distance to mean (helps detect outliers if updates deviate)
        l2_dists = np.linalg.norm(gradients_np - global_mean, axis=1)

        # SVD decomposition for reconstruction error
        max_components = min(gradients_np.shape[1], gradients_np.shape[0]//2)
        n_components = max(2, int(np.log2(gradients_np.shape[0]) + 1))
        n_components = min(n_components, max_components)
        u, s, vt = randomized_svd(gradients_np, n_components=n_components, random_state=42)
        reconstructed = np.dot(u, np.dot(np.diag(s), vt))
        reconstruction_errors = np.linalg.norm(gradients_np - reconstructed, axis=1)

        # Isolation Forest for outlier scores
        iso_forest = IsolationForest(n_estimators=50, max_samples='auto',
                                     contamination='auto', random_state=42)
        isolation_scores = -iso_forest.fit(gradients_np).decision_function(gradients_np)

        # Cosine-similarity drift from previous round
        similarity_drifts = []
        for i, cid in enumerate(client_ids):
            sim = cos_sims[i]
            drift = abs(sim - self.client_similarity_history[cid])
            self.client_similarity_history[cid] = sim
            similarity_drifts.append(drift)

        # Combine signals into one anomaly score
        combined = (
            0.3 * reconstruction_errors +    # SVD reconstruction error
            0.2 * isolation_scores +         # Isolation Forest
            0.2 * np.array(similarity_drifts) +  # Similarity drift
            0.3 * l2_dists                  # Distance from mean
        )
        # Update the exponentially-weighted anomaly history
        for i, cid in enumerate(client_ids):
            self.anomaly_score_history[cid] = (
                self.decay_factor * self.anomaly_score_history[cid] +
                (1 - self.decay_factor) * combined[i]
            )
        return self.anomaly_score_history

    def adjust_poisoned_threshold(self, anomaly_scores):
        """
        Set a threshold to admit enough clients as 'valid'.
        """
        threshold = np.percentile(anomaly_scores, 80)
        # Ensure at least min_valid_clients fall below the threshold
        if np.sum(anomaly_scores <= threshold) < 10:
            threshold = np.percentile(anomaly_scores, 90)
        self.poisoned_threshold = threshold

    def select_clients(self, epoch, gradients):
        """
        Select a batch of clients for this round.
        Applies anomaly filtering, then uses a weighted combination of UCB score,
        Thompson sample, and diversity to rank clients.
        """
        anomaly_scores = self.detect_poisoned_clients(gradients)
        self.adjust_poisoned_threshold(anomaly_scores)

        # Flag valid clients (anomaly + penalty below threshold)
        valid = (anomaly_scores + self.poisoned_penalty <= self.poisoned_threshold).astype(int)
        # Exclude permanently blacklisted clients
        for cid in self.white_black_list['black']:
            valid[cid] = 0
        # If too few valid clients, allow some blacklisted to fill up
        if np.sum(valid) < self.num_join_clients:
            deficit = self.num_join_clients - np.sum(valid)
            blist = list(self.white_black_list['black'])
            np.random.shuffle(blist)
            for cid in blist:
                if deficit <= 0: break
                valid[cid] = 1
                deficit -= 1

        # Determine UCB vs Thompson weights (more exploration if anomalies high)
        weight_ucb = 0.5 + 0.1 * (1 - np.mean(anomaly_scores))
        weight_ts = 1 - weight_ucb

        # Attempt to reuse last SVD embeddings for diversity (if available)
        embeddings = None
        try:
            embeddings = u * s  # from last SVD computation in detect_poisoned_clients
        except NameError:
            embeddings = None

        # Compute diversity score for each client (mean cosine distance to others)
        diversity = np.zeros(self.num_clients)
        if embeddings is not None:
            client_to_idx = {cid: i for i, cid in enumerate(gradients.keys())}
            for i in range(self.num_clients):
                if valid[i] and i in client_to_idx:
                    idx_i = client_to_idx[i]
                    emb_i = embeddings[idx_i]
                    divers = []
                    for j in range(self.num_clients):
                        if j != i and valid[j] and j in client_to_idx:
                            idx_j = client_to_idx[j]
                            emb_j = embeddings[idx_j]
                            sim = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i)*np.linalg.norm(emb_j) + 1e-8)
                            divers.append(1 - sim)
                    diversity[i] = np.mean(divers) if divers else 0.0

        # Calculate combined score for each client
        combined_scores = np.full(self.num_clients, -1e9)
        for i in range(self.num_clients):
            if valid[i]:
                # UCB part
                if self.selection_counts[i] > 0:
                    avg_reward = self.performance_history[i] / self.selection_counts[i]
                    delta = math.sqrt(2 * math.log(epoch+1) / self.selection_counts[i])
                    ucb_score = avg_reward + self.c * delta
                else:
                    ucb_score = 1e6  # Force exploration of new clients
                # Thompson sampling part
                ts_score = tdist.Beta(self.posterior_alpha[i], self.posterior_beta[i]).sample().item()
                combined_scores[i] = (weight_ucb * ucb_score + 
                                      weight_ts * ts_score + 
                                      0.2 * diversity[i])

        # Select top-k clients by combined score
        selected = np.argsort(combined_scores)[-self.num_join_clients:]
        for cid in selected:
            self.selection_counts[cid] += 1

        # Update penalties and black/white lists based on anomaly scores
        for i in range(self.num_clients):
            if anomaly_scores[i] > self.poisoned_threshold:
                self.poisoned_penalty[i] = min(self.poisoned_penalty[i] + 0.1, 1.0)
                if self.poisoned_penalty[i] > 0.8:
                    self.white_black_list['black'].add(i)
                    self.black_list_ttl[i] = self.max_ttl
            else:
                self.poisoned_penalty[i] = max(self.poisoned_penalty[i] - 0.05, 0.0)
        # Decrement TTLs for blacklisted clients
        expired = []
        for cid in list(self.black_list_ttl):
            self.black_list_ttl[cid] -= 1
            if self.black_list_ttl[cid] <= 0:
                expired.append(cid)
        for cid in expired:
            self.white_black_list['black'].discard(cid)
            del self.black_list_ttl[cid]

        return selected

    def update(self, selected_clients, rewards, global_val_acc=None):
        """
        After a round of training, update the bandit statistics using observed rewards.
        Rewards should be normalized (e.g., validation accuracy) in [0,1].
        """
        decay = 0.9
        last_acc = self.global_val_acc_history[-1] if self.global_val_acc_history else 0.0
        if global_val_acc is not None:
            self.global_val_acc_history.append(global_val_acc)

        for cid, reward in zip(selected_clients, rewards):
            baseline = self.client_reward_history[cid]
            new_baseline = decay * baseline + (1 - decay) * reward
            delta_reward = max(0.0, reward - baseline)
            self.client_reward_history[cid] = new_baseline

            # Update performance history and Beta parameters for Thompson Sampling
            self.performance_history[cid] = decay * self.performance_history[cid] + reward
            self.posterior_alpha[cid] = self.posterior_alpha[cid] * decay + delta_reward
            self.posterior_beta[cid] = self.posterior_beta[cid] * decay + (1 - delta_reward)

            # Reward bonus if global validation accuracy improved significantly
            if global_val_acc is not None and (global_val_acc - last_acc) > 0.01:
                self.performance_history[cid] += 0.05

            # If reward is very high, add to white-list temporarily
            if reward > 0.9:
                self.white_black_list['white'].add(cid)
                self.white_list_ttl[cid] = self.max_ttl

        # Optionally remove clients from blacklist if their average performance recovers
        for cid in list(self.white_black_list['black']):
            avg_perf = self.performance_history[cid] / max(1, self.selection_counts[cid])
            if avg_perf > 0.8:
                self.white_black_list['black'].discard(cid)
                if cid in self.black_list_ttl:
                    del self.black_list_ttl[cid]

    def compute_composite_rewards(self, acc_list, f1_list, auc_pr_list, avg_loss_list, selected_clients):
        """
        Compute a composite reward combining accuracy, F1, AUC-PR, (inverted) loss, and anomaly score.
        Weights are set via softmax on metric variances to focus on the most variable metrics.
        """
        def normalize(x):
            arr = np.array(x)
            return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)

        acc_norm = normalize(acc_list)
        f1_norm = normalize(f1_list)
        auc_pr_norm = normalize(auc_pr_list)
        loss_inv = 1.0 - normalize(avg_loss_list)
        anomaly_inv = 1.0 - normalize([self.anomaly_score_history[cid] for cid in selected_clients])

        # Compute variances and use softmax to get weights
        variances = np.array([
            np.var(acc_norm),
            np.var(f1_norm),
            np.var(loss_inv),
            np.var(anomaly_inv),
            np.var(auc_pr_norm)
        ])
        weights = scipy.special.softmax(variances * 10)
        min_w = 0.1 / len(weights)
        weights = np.clip(weights, min_w, 1.0)
        weights /= np.sum(weights)

        raw_rewards = (
            weights[0] * acc_norm +
            weights[1] * f1_norm +
            weights[2] * loss_inv +
            weights[3] * anomaly_inv +
            weights[4] * auc_pr_norm
        )

        # Normalize composite rewards to use as contribution weights
        total_reward = sum(raw_rewards)
        if total_reward == 0:
            contribution_weights = [1.0 / len(selected_clients)] * len(selected_clients)
            self.contribution_weights = contribution_weights
        else:
            contribution_weights = [r / total_reward for r in raw_rewards]
            self.contribution_weights = contribution_weights

        delta_rewards = [0.0] * self.num_clients
        ema = 0.9
        for idx, cid in enumerate(selected_clients):
            baseline = self.client_reward_history[cid]
            delta = raw_rewards[idx] - baseline
            delta_rewards[cid] = max(0.0, delta)
            self.client_reward_history[cid] = ema * baseline + (1 - ema) * raw_rewards[idx]
        return delta_rewards
