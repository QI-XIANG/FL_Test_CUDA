import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.ensemble import IsolationForest
import torch
import torch.distributions as tdist
import math
import torch.nn.functional as F
import scipy.special
#RSVDUCBTE PRO V5
class EnhancedRSVDUCBThompson():
    """
    Robust federated client selector combining anomaly detection (SVD, IsolationForest, etc.),
    multi-armed bandit selection (UCB + Thompson Sampling), and diversity boosting.
    Enhanced for faster convergence and tolerance to up to 40% stealthy poisoned clients.
    """

    def __init__(self, num_clients, num_join_clients, min_valid_clients=10, c=1, prior_alpha=1, prior_beta=1):
        self.num_clients = num_clients
        self.num_join_clients = num_join_clients
        # Ensure we require at least num_join_clients as valid to select each round
        self.min_valid_clients = max(min_valid_clients, num_join_clients)
        # Bandit tracking
        self.selection_counts = np.zeros(num_clients)        # Times selected per client
        self.performance_history = np.zeros(num_clients)     # Exponentially decayed cumulative reward
        # Exponentially-decayed anomaly score per client
        self.anomaly_score_history = np.zeros(num_clients)
        # Penalty accumulators for suspected malicious clients
        self.poisoned_penalty = np.zeros(num_clients)
        self.poisoned_threshold = 0.5   # Dynamic anomaly threshold (updated each round)
        self.c = c  # UCB exploration parameter
        # Beta posterior parameters for Thompson Sampling (initialized to Beta(1,1) for each client)
        self.posterior_alpha = torch.ones(num_clients) * prior_alpha
        self.posterior_beta = torch.ones(num_clients) * prior_beta
        self.decay_factor = 0.9  # Decay factor for anomaly score updates

        # Whitelist/Blacklist with time-to-live for entries
        self.white_black_list = {"white": set(), "black": set()}
        self.white_list_ttl = dict()
        self.black_list_ttl = dict()
        self.max_ttl = 5  # rounds to keep a client on white/black list without renewal

        # For tracking cosine similarity drift and baseline rewards
        self.client_similarity_history = np.zeros(num_clients)
        self.client_reward_history = np.zeros(num_clients)
        self.global_val_acc_history = []
        self.contribution_weights = None

        self.baseline_gradients = None  # Server gradient vector from clean reference data
        self.server_data = None         # Server-side (x, y) tensor tuple

    def detect_poisoned_clients(self, gradients):
        """
        Compute an anomaly score for each client’s gradient update using multiple signals:
        - Gradient norm clipping (limits extreme updates),
        - Distance from global mean and median update,
        - SVD reconstruction error (low-rank approximation error),
        - Isolation Forest outlier score,
        - Cosine similarity drift from previous round.
        Returns updated anomaly scores for all clients (stored in self.anomaly_score_history).
        """
        client_ids = list(gradients.keys())
        # Stack gradients into a matrix for analysis
        grad_list = []
        for cid in client_ids:
            grad = gradients[cid].copy()
            # Clip excessively large gradients to reduce undue influence (robust against magnitude attacks)
            norm = np.linalg.norm(grad)
            if norm > 10:
                grad = grad * (10.0 / norm)
            grad_list.append(grad)
        gradients_np = np.vstack(grad_list)

        # Compute global mean and median of updates (robust centers)
        global_mean = np.mean(gradients_np, axis=0)
        global_median = np.median(gradients_np, axis=0)

        # Cosine similarity of each update to the global mean (for drift calculation)
        cos_sims = F.cosine_similarity(
            torch.tensor(gradients_np), 
            torch.tensor(global_mean).repeat(len(gradients_np), 1), 
            dim=1
        ).numpy()
        # Euclidean distance of each update from global mean and median
        l2_dists_mean = np.linalg.norm(gradients_np - global_mean, axis=1)
        l2_dists_median = np.linalg.norm(gradients_np - global_median, axis=1)

        # SVD for low-rank approximation of updates matrix
        # Use a number of components ~ log2(n) to capture major trends while leaving out outliers
        max_components = min(gradients_np.shape[1], gradients_np.shape[0] // 2)
        n_components = min(max_components, max(2, int(np.log2(len(client_ids))) + 1))
        # randomized_svd for efficiency on potentially large dimension
        u, s, vt = randomized_svd(gradients_np, n_components=n_components, random_state=42)
        reconstructed = np.dot(u, np.dot(np.diag(s), vt))
        # Reconstruction error: how well each update is represented by the main components (larger error = more anomalous)
        reconstruction_errors = np.linalg.norm(gradients_np - reconstructed, axis=1)

        # Isolation Forest outlier detection on gradients
        # Set contamination to ~30% to account for a possible large fraction of outliers (up to 40%)
        iso_forest = IsolationForest(n_estimators=50, max_samples='auto',
                                     contamination=0.3, random_state=42)
        isolation_scores = -iso_forest.fit(gradients_np).decision_function(gradients_np)
        # (We use negative decision_function so that higher values mean more anomalous)

        # Cosine similarity drift: change in similarity to global mean compared to previous round
        similarity_drifts = []
        for idx, cid in enumerate(client_ids):
            sim = cos_sims[idx]
            drift = abs(sim - self.client_similarity_history[cid])
            self.client_similarity_history[cid] = sim  # update history for next round
            similarity_drifts.append(drift)
        
        # If server-side baseline exists, include cosine distance from it
        if self.baseline_gradients is not None:
            baseline_cos_dist = []
            for g in gradients_np:
                sim = np.dot(g, self.baseline_gradients) / (
                    np.linalg.norm(g) * np.linalg.norm(self.baseline_gradients) + 1e-8
                )
                baseline_cos_dist.append(1.0 - sim)
            baseline_cos_dist = np.array(baseline_cos_dist)
        else:
            baseline_cos_dist = np.zeros(len(gradients_np))

        # Combine signals into a composite anomaly score.
        # Weights are chosen to emphasize robust measures (median distance, SVD error) while still considering others.
        combined_scores = (
            0.25 * reconstruction_errors +        # large if update has components outside main subspace
            0.15 * isolation_scores +             # outlier score from Isolation Forest
            0.10 * np.array(similarity_drifts) +  # unusual change in behavior from last round
            0.20 * l2_dists_mean +                # distance from mean (outlier in overall distribution)
            0.30 * l2_dists_median+                # distance from median (robust outlier measure)
            0.15 * baseline_cos_dist
        )

        # Update exponentially-weighted anomaly history for each client
        for idx, cid in enumerate(client_ids):
            # Decay previous anomaly score and add current combined score
            self.anomaly_score_history[cid] = (
                self.decay_factor * self.anomaly_score_history[cid] +
                (1 - self.decay_factor) * combined_scores[idx]
            )
        return self.anomaly_score_history

    def adjust_poisoned_threshold(self, anomaly_scores):
        """
        Determine the anomaly score threshold for considering a client as 'poisoned'.
        We set the threshold so that a sufficient number of clients are under it (at least num_join_clients),
        ensuring we have enough valid clients to select. This threshold is typically a high percentile (e.g., 80th),
        but will be raised if needed to include the minimum required clients.
        """
        # Start with a high percentile (80th) as baseline threshold
        threshold = np.percentile(anomaly_scores, 80)
        # Ensure at least `min_valid_clients` (usually = num_join_clients) fall below the threshold
        if np.sum(anomaly_scores <= threshold) < self.min_valid_clients:
            # Increase threshold to 90th percentile if not enough valid clients
            threshold = np.percentile(anomaly_scores, 90)
        # As an extra safeguard, if still fewer than required, include top 50% by using 100*(join/total)% percentile
        if np.sum(anomaly_scores <= threshold) < self.min_valid_clients:
            perc = 100 * (self.num_join_clients / self.num_clients)
            threshold = np.percentile(anomaly_scores, perc)
        # Update the dynamic threshold
        self.poisoned_threshold = threshold

    def select_clients(self, epoch, gradients):
        """
        Select a set of clients for this training round.
        Steps:
        1. Detect anomalous (potentially poisoned) client updates.
        2. Adjust anomaly threshold to filter out likely poisoned clients while keeping enough clients.
        3. Compute a bandit score (UCB + Thompson Sampling + diversity) for each valid client.
        4. Select the top-k (num_join_clients) clients by this score.
        5. Update selection counts, and update white/black lists and penalties.
        """
        # 1. Anomaly detection for all clients present in `gradients`
        anomaly_scores = self.detect_poisoned_clients(gradients)
        # 2. Determine threshold to label clients as valid vs. suspicious
        self.adjust_poisoned_threshold(anomaly_scores)

        # Mark clients as valid (0/1) if their anomaly score plus any penalty is below threshold
        valid = np.zeros(self.num_clients, dtype=int)
        for cid in range(self.num_clients):
            if anomaly_scores[cid] + self.poisoned_penalty[cid] <= self.poisoned_threshold:
                valid[cid] = 1
        # Exclude any permanently blacklisted clients from valid set
        for cid in self.white_black_list['black']:
            valid[cid] = 0
        # Include whitelisted clients as valid regardless of anomaly score (they earned trust via high performance)
        for cid in self.white_black_list['white']:
            valid[cid] = 1

        # If we still have fewer valid clients than needed, relax filtering to include some blacklisted (worst-case fallback)
        if np.sum(valid) < self.num_join_clients:
            deficit = self.num_join_clients - np.sum(valid)
            black_candidates = list(self.white_black_list['black'])
            np.random.shuffle(black_candidates)
            for cid in black_candidates:
                if deficit <= 0:
                    break
                # Include blacklisted clients to meet the required number (this is not ideal but prevents stalling)
                valid[cid] = 1
                deficit -= 1

        # Determine weights for UCB vs Thompson sampling.
        # If overall anomaly levels are high, increase exploration (Thompson) to possibly find overlooked good clients.
        mean_anomaly = np.mean(anomaly_scores)
        weight_ucb = 0.5 + 0.1 * (1 - mean_anomaly)   # slightly more weight on UCB if anomalies are low
        weight_ucb = min(max(weight_ucb, 0.1), 0.9)   # keep in [0.1, 0.9] for safety
        weight_ts = 1.0 - weight_ucb

        # Compute a diversity score for each client using embeddings from SVD (if available).
        diversity = np.zeros(self.num_clients)
        # We use the low-rank embedding (u * s) as a feature for diversity measurement.
        try:
            embeddings = u * s  # from the last SVD computation in detect_poisoned_clients
        except NameError:
            embeddings = None
        if embeddings is not None:
            # Map client IDs to row indices in the embeddings matrix
            client_index_map = {cid: idx for idx, cid in enumerate(gradients.keys())}
            for cid in range(self.num_clients):
                if valid[cid] and cid in client_index_map:
                    idx_i = client_index_map[cid]
                    emb_i = embeddings[idx_i]
                    # Compute mean cosine distance from this client's embedding to other valid clients
                    distances = []
                    for other_cid in gradients.keys():
                        if other_cid == cid or not valid[other_cid]:
                            continue
                        idx_j = client_index_map[other_cid]
                        emb_j = embeddings[idx_j]
                        # Cosine similarity and convert to distance
                        cos_sim = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j) + 1e-8)
                        distances.append(1 - cos_sim)
                    diversity[cid] = np.mean(distances) if distances else 0.0

        # 3. Compute combined bandit score (UCB + Thompson + diversity) for each valid client
        combined_scores = np.full(self.num_clients, -1e9)  # initialize with large negative for invalid clients
        for cid in range(self.num_clients):
            if not valid[cid]:
                continue
            # UCB component: exploit known good clients, with exploration bonus
            if self.selection_counts[cid] > 0:
                avg_reward = self.performance_history[cid] / self.selection_counts[cid]
                # UCB exploration term (c * sqrt(log(epoch+1)/selection_count))
                delta = math.sqrt(2 * math.log(epoch + 1) / self.selection_counts[cid])
                ucb_score = avg_reward + self.c * delta
            else:
                # If never selected, give a high initial score to ensure exploration
                ucb_score = 1e6
            # Thompson Sampling component: sample from Beta posterior (represents optimism for this client's success probability)
            ts_score = tdist.Beta(self.posterior_alpha[cid], self.posterior_beta[cid]).sample().item()
            # Combined score: weighted sum of UCB and Thompson, plus a small diversity bonus
            combined_scores[cid] = weight_ucb * ucb_score + weight_ts * ts_score + 0.2 * diversity[cid]

        # 4. Select top-K clients by combined score
        selected_indices = np.argsort(combined_scores)[-self.num_join_clients:]
        selected_clients = list(selected_indices)  # numpy array to list
        for cid in selected_clients:
            self.selection_counts[cid] += 1  # update count for each selected client

        # 5. Update penalties and white/black lists based on anomaly outcomes for all clients
        for cid in range(self.num_clients):
            if anomaly_scores[cid] > self.poisoned_threshold:
                # High anomaly: increase penalty (up to 1.0). Penalty makes future selection harder.
                self.poisoned_penalty[cid] = min(self.poisoned_penalty[cid] + 0.1, 1.0)
                # If penalty exceeds 0.8 (persistent anomaly), blacklist this client
                if self.poisoned_penalty[cid] > 0.8:
                    self.white_black_list['black'].add(cid)
                    self.black_list_ttl[cid] = self.max_ttl
            else:
                # Low anomaly: gradually forgive by reducing penalty
                self.poisoned_penalty[cid] = max(self.poisoned_penalty[cid] - 0.05, 0.0)

        # Decrement TTL for blacklisted clients and remove if expired (allow rejoining after some rounds if they reform)
        expired_black = []
        for cid, ttl in list(self.black_list_ttl.items()):
            self.black_list_ttl[cid] -= 1
            if self.black_list_ttl[cid] <= 0:
                expired_black.append(cid)
        for cid in expired_black:
            self.white_black_list['black'].discard(cid)
            del self.black_list_ttl[cid]

        # Decrement TTL for whitelisted clients and remove if expired (so they don't stay trusted forever without merit)
        expired_white = []
        for cid, ttl in list(self.white_list_ttl.items()):
            self.white_list_ttl[cid] -= 1
            if self.white_list_ttl[cid] <= 0:
                expired_white.append(cid)
        for cid in expired_white:
            self.white_black_list['white'].discard(cid)
            del self.white_list_ttl[cid]

        return selected_clients

    def update(self, selected_clients, rewards, global_val_acc=None):
        """
        Update bandit statistics after a training round.
        - selected_clients: list of client IDs that were selected this round.
        - rewards: list of observed rewards (e.g., validation accuracy or composite metric) for each selected client.
        - global_val_acc: new global validation accuracy after this round (optional, for adaptive bonus).
        """
        decay = 0.9  # decay factor for integrating new rewards
        last_acc = self.global_val_acc_history[-1] if self.global_val_acc_history else 0.0
        if global_val_acc is not None:
            self.global_val_acc_history.append(global_val_acc)

        for cid, reward in zip(selected_clients, rewards):
            # Exponentially decayed baseline reward for client (to track its average contribution)
            baseline = self.client_reward_history[cid]
            new_baseline = decay * baseline + (1 - decay) * reward
            # Calculate positive reward difference from baseline (how much better this round was)
            delta_reward = max(0.0, reward - baseline)
            self.client_reward_history[cid] = new_baseline

            # Update cumulative performance history (decayed) for UCB
            self.performance_history[cid] = decay * self.performance_history[cid] + reward
            # Update Thompson sampling posteriors: treat delta_reward as success probability
            self.posterior_alpha[cid] = self.posterior_alpha[cid] * decay + delta_reward
            self.posterior_beta[cid] = self.posterior_beta[cid] * decay + (1 - delta_reward)

            # If global validation accuracy improved significantly this round, give a small bonus to these clients
            if global_val_acc is not None and (global_val_acc - last_acc) > 0.01:
                self.performance_history[cid] += 0.05  # bonus reward for contributing to global improvement

            # Whitelist clients with very high reward (trusted good contributors)
            if reward > 0.9:
                self.white_black_list['white'].add(cid)
                self.white_list_ttl[cid] = self.max_ttl
                # If it was blacklisted before, remove from blacklist since it proved its worth
                if cid in self.white_black_list['black']:
                    self.white_black_list['black'].discard(cid)
                    if cid in self.black_list_ttl:
                        del self.black_list_ttl[cid]

        # Optionally remove clients from blacklist if their average performance significantly recovers
        for cid in list(self.white_black_list['black']):
            if self.selection_counts[cid] > 0:
                avg_perf = self.performance_history[cid] / self.selection_counts[cid]
            else:
                avg_perf = 0.0
            # If a blacklisted client shows good average performance (e.g. >0.8), assume it may be rehabilitated
            if avg_perf > 0.8:
                self.white_black_list['black'].discard(cid)
                if cid in self.black_list_ttl:
                    del self.black_list_ttl[cid]

    def compute_composite_rewards(self, acc_list, f1_list, auc_pr_list, avg_loss_list, selected_clients):
        """
        Compute a composite reward for selected clients combining multiple metrics:
        accuracy, F1, AUC-PR, (inverted) loss, and (inverted) anomaly score.
        Weights for each component are determined by the variability of that metric across clients (higher variance -> higher weight),
        using a softmax on variances to focus on the most informative metrics.
        This reward is used to assess each client's true contribution.
        """
        # Normalization helper
        def normalize(x):
            arr = np.array(x, dtype=float)
            return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)

        # Normalize each metric list to [0,1]
        acc_norm = normalize(acc_list)
        f1_norm = normalize(f1_list)
        auc_pr_norm = normalize(auc_pr_list)
        loss_inv = 1.0 - normalize(avg_loss_list)  # invert loss: lower loss -> higher reward
        anomaly_inv = []  # 1 - anomaly score for each selected client (higher is better if anomaly is low)
        for cid in selected_clients:
            anomaly_inv.append(1.0 - normalize(self.anomaly_score_history)[cid])
        anomaly_inv = np.array(anomaly_inv)

        # Compute variance of each metric across the selected clients
        variances = np.array([
            np.var(acc_norm),
            np.var(f1_norm),
            np.var(loss_inv),
            np.var(anomaly_inv),
            np.var(auc_pr_norm)
        ])
        # Use softmax on variances (scaled) to determine weights (metrics with larger variance get higher weight)
        weights = scipy.special.softmax(variances * 10)
        # Ensure a minimum weight for each metric to avoid any being completely ignored
        min_w = 0.1 / len(weights)
        weights = np.clip(weights, min_w, None)
        weights /= np.sum(weights)  # re-normalize weights

        # Compute raw composite rewards as weighted sum of metrics for each client
        raw_rewards = (weights[0] * acc_norm +
                       weights[1] * f1_norm +
                       weights[2] * loss_inv +
                       weights[3] * anomaly_inv +
                       weights[4] * auc_pr_norm)

        # Normalize the composite rewards to derive contribution weights (sums to 1 across selected clients)
        total_reward = np.sum(raw_rewards)
        if total_reward == 0:
            contribution_weights = [1.0 / len(selected_clients)] * len(selected_clients)
        else:
            contribution_weights = (raw_rewards / total_reward).tolist()
        self.contribution_weights = contribution_weights  # store for potential use

        # Calculate delta rewards (improvement over each client's historical baseline)
        delta_rewards = [0.0] * self.num_clients
        ema = 0.9  # smoothing factor for updating baseline reward
        for idx, cid in enumerate(selected_clients):
            baseline = self.client_reward_history[cid]
            delta = raw_rewards[idx] - baseline
            delta_rewards[cid] = max(0.0, delta)
            # Update the client's baseline reward (EMA)
            self.client_reward_history[cid] = ema * baseline + (1 - ema) * raw_rewards[idx]
        return delta_rewards
    
    def load_server_dataset(self, path, global_model, loss_fn=torch.nn.CrossEntropyLoss(), device='cpu'):
        """
        Loads server_data.pt and computes gradient baseline using the current global model.
        Stores gradient vector to self.baseline_gradients for later anomaly comparison.
        """
        data = torch.load(path)
        images, labels = data['images'].to(device), data['labels'].to(device)
        self.server_data = (images, labels)

        global_model.to(device)
        global_model.eval()
        self.baseline_gradients = self.compute_server_gradients(global_model, loss_fn)
        global_model.cpu()
        print("Server baseline gradient computed and stored.")
    
    def compute_server_gradients(self, model, loss_fn):
        """
        Compute gradients of the global model on the small server-side dataset.
        Returns a flattened gradient vector.
        """
        model.zero_grad()
        x, y = self.server_data
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()

        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.detach().cpu().flatten())
        grad_vector = torch.cat(grads).numpy()
        return grad_vector
