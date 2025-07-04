import torch
import torch.distributions as tdist
import math
import numpy as np # Used only for final output type conversion

class ReputationGuardFL:
    """
    A high-performance, adaptive, and robust client selection and aggregation
    mechanism for Federated Learning, built on a unified PyTorch backend.
    """
    def __init__(self, num_clients, num_join_clients, global_accuracy_history=None,
                 min_valid_clients=10, c=1.0, prior_alpha=1.0, prior_beta=1.0,
                 decay_factor=0.9, gradient_subsample_dim=512,
                 attack_tolerance_threshold=0.5,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initializes the ReputationGuardFL parameters.

        Args:
            device (str): The device to run computations on ('cuda' or 'cpu').
            ... other parameters
        """
        self.num_clients = num_clients
        self.num_join_clients = num_join_clients
        self.min_valid_clients = min_valid_clients
        self.gradient_subsample_dim = gradient_subsample_dim
        self.attack_tolerance_threshold = attack_tolerance_threshold
        self.device = torch.device(device)
        
        self.global_accuracy_history = global_accuracy_history if global_accuracy_history is not None else []

        # --- Caches for single-pass analysis results ---
        self._cached_anomaly_scores = None
        self._cached_similarities = None

        # --- Client State Tracking (as PyTorch Tensors on the target device) ---
        self.selection_counts = torch.zeros(num_clients, device=self.device)
        self.performance_history = torch.zeros(num_clients, device=self.device)
        self.anomaly_score_history = torch.zeros(num_clients, device=self.device)
        self.poisoned_penalty = torch.zeros(num_clients, device=self.device)

        # --- Algorithm Parameters ---
        self.poisoned_threshold = 0.5
        self.c = c
        self.decay_factor = decay_factor
        self.posterior_alpha = torch.ones(num_clients, device=self.device) * prior_alpha
        self.posterior_beta = torch.ones(num_clients, device=self.device) * prior_beta
        
        print(f"ReputationGuardFL initialized on device: {self.device} with GPU-native Randomized SVD.")

    def _flatten_gradients(self, gradients_dict):
        """
        Efficiently flattens and stacks client gradients into a single tensor.
        """
        client_grads = [
            torch.cat([torch.as_tensor(p, device=self.device).flatten() for p in gradients_dict[cid]])
            for cid in gradients_dict.keys()
        ]
        return torch.stack(client_grads)

    @torch.compile
    def _analyze_gradients(self, gradients_dict):
        """
        Performs the main SVD analysis once per round using GPU-accelerated
        randomized SVD and caches the results.
        """
        client_ids = list(gradients_dict.keys())
        if len(client_ids) < 2:
            self._cached_anomaly_scores = {}
            self._cached_similarities = {}
            return

        gradient_matrix = self._flatten_gradients(gradients_dict)
        
        if self.gradient_subsample_dim and gradient_matrix.shape[1] > self.gradient_subsample_dim:
            perm = torch.randperm(gradient_matrix.shape[1], device=self.device)
            indices = perm[:self.gradient_subsample_dim]
            subsampled_gradients = gradient_matrix[:, indices]
        else:
            subsampled_gradients = gradient_matrix
        
        # --- GPU-native Randomized SVD ---
        # torch.svd_lowrank is the high-performance equivalent of sklearn's randomized_svd
        n_components = min(subsampled_gradients.shape[0] - 1, 5)
        if n_components < 1: return

        U, S, V = torch.svd_lowrank(subsampled_gradients, q=n_components)
        
        # Reconstruct the gradients using the low-rank approximation.
        # Note: torch.svd_lowrank returns V, so we use its transpose V.T
        reconstructed = U @ torch.diag(S) @ V.T

        # --- All subsequent calculations use PyTorch ---
        recon_errors = torch.linalg.norm(subsampled_gradients - reconstructed, dim=1)
        cosine_sim = torch.nn.functional.cosine_similarity(subsampled_gradients, reconstructed, dim=1)
        cosine_dist = 1 - cosine_sim
        
        norm_errors = (recon_errors - torch.min(recon_errors)) / (torch.max(recon_errors) - torch.min(recon_errors) + 1e-9)
        norm_dist = (cosine_dist - torch.min(cosine_dist)) / (torch.max(cosine_dist) - torch.min(cosine_dist) + 1e-9)
        combined_scores = 0.6 * norm_errors + 0.4 * norm_dist
        
        # Cache results
        self._cached_anomaly_scores = {cid: score.item() for cid, score in zip(client_ids, combined_scores)}
        self._cached_similarities = {cid: sim.item() for cid, sim in zip(client_ids, cosine_sim)}

    def _adjust_poisoned_threshold(self):
        """Dynamically adjusts threshold using PyTorch operations."""
        scores_tensor = torch.tensor(list(self._cached_anomaly_scores.values()), device=self.device)
        
        base_percentile = 0.5 # Median
        if len(self.global_accuracy_history) >= 2 and self.global_accuracy_history[-1] < self.global_accuracy_history[-2]:
            base_percentile = 0.4 # Stricter
            print("Note: Global accuracy dropped. Using stricter anomaly filtering.")
        
        if scores_tensor.numel() > 0:
            threshold = torch.quantile(scores_tensor, base_percentile)
            num_passing = torch.sum(scores_tensor <= threshold)
            if num_passing < self.min_valid_clients:
                new_percentile = 1.0 - (self.min_valid_clients / self.num_clients) if self.num_clients > 0 else 0.5
                threshold = torch.quantile(scores_tensor, new_percentile)
            self.poisoned_threshold = threshold.item()

    def select_clients(self, epoch, gradients_dict):
        """Selects clients using the unified analysis results."""
        available_client_ids = list(gradients_dict.keys())
        if not available_client_ids:
            return np.array([], dtype=int)

        # --- Perform SVD analysis only ONCE ---
        self._analyze_gradients(gradients_dict)

        # Update historical scores from the cached results
        for cid, score in self._cached_anomaly_scores.items():
            self.anomaly_score_history[cid] = (self.decay_factor * self.anomaly_score_history[cid] + (1 - self.decay_factor) * score)
        
        self._adjust_poisoned_threshold()

        # --- Stage 1 & 2 Filtering ---
        potential_client_ids = [cid for cid in available_client_ids if (self.anomaly_score_history[cid] + self.poisoned_penalty[cid]) <= self.poisoned_threshold]
        
        final_valid_client_ids = []
        if len(potential_client_ids) > 0:
            valid_grads_matrix = self._flatten_gradients({cid: gradients_dict[cid] for cid in potential_client_ids})
            robust_mean_gradient = torch.mean(valid_grads_matrix, dim=0)
            similarities_to_mean = torch.nn.functional.cosine_similarity(valid_grads_matrix, robust_mean_gradient.unsqueeze(0), dim=1)
            
            for i, cid in enumerate(potential_client_ids):
                if similarities_to_mean[i] >= self.attack_tolerance_threshold:
                    final_valid_client_ids.append(cid)
        
        # --- Scoring and Selection ---
        hybrid_scores = torch.full((self.num_clients,), -torch.inf, device=self.device)
        stability_factor = 1.0
        if len(self.global_accuracy_history) >= 5:
            if self.global_accuracy_history[-1] < np.mean(self.global_accuracy_history[-5:-1]):
                stability_factor = 0.5
        
        current_anomaly_scores = torch.tensor([self.anomaly_score_history[cid] for cid in available_client_ids], device=self.device)
        weight_ucb = stability_factor * (0.5 + 0.1 * (1.0 - torch.mean(current_anomaly_scores)))
        weight_ts = 1.0 - weight_ucb

        for i in final_valid_client_ids:
            if self.selection_counts[i] > 0:
                avg_reward = self.performance_history[i] / self.selection_counts[i]
                exploration_term = torch.sqrt(2 * torch.log(torch.tensor(epoch + 1, device=self.device)) / self.selection_counts[i])
                ucb_score = avg_reward + self.c * exploration_term
            else:
                ucb_score = torch.inf
            
            thompson_score = tdist.Beta(self.posterior_alpha[i], self.posterior_beta[i]).sample()
            hybrid_scores[i] = weight_ucb * ucb_score + weight_ts * thompson_score

        # Ensure we don't try to select more clients than are available and valid
        k_to_select = min(self.num_join_clients, len(final_valid_client_ids))
        if k_to_select > 0:
             selected_clients = torch.topk(hybrid_scores, k=k_to_select).indices
        else:
             selected_clients = torch.tensor([], dtype=torch.long, device=self.device)

        # Update counts and penalties
        self.selection_counts[selected_clients] += 1
        for cid in available_client_ids:
            if self.anomaly_score_history[cid] > self.poisoned_threshold:
                self.poisoned_penalty[cid] += 0.1
                
        return selected_clients.cpu().numpy()

    def update(self, selected_clients, rewards):
        """Updates client reputation based on rewards from the training round."""
        if len(selected_clients) == 0: return # Nothing to update
        selected_t = torch.tensor(selected_clients, device=self.device, dtype=torch.long)
        rewards_t = torch.tensor(rewards, device=self.device, dtype=torch.float)
        
        self.performance_history[selected_t] += rewards_t
        self.posterior_alpha[selected_t] += rewards_t
        self.posterior_beta[selected_t] += (1 - rewards_t)

    def get_aggregation_weights(self, selected_clients):
        """Calculates aggregation weights based on client reputation."""
        if len(selected_clients) == 0: return {}
        selected_t = torch.tensor(selected_clients, device=self.device, dtype=torch.long)
        scores = self.posterior_alpha[selected_t] / (self.posterior_alpha[selected_t] + self.posterior_beta[selected_t])
        
        if torch.sum(scores) == 0:
            scores = torch.ones_like(scores)
            
        normalized_scores = scores / torch.sum(scores)
        return {cid: score.item() for cid, score in zip(selected_clients, normalized_scores)}

    def calculate_robust_rewards(self, gradients_dict):
        """Calculates rewards by retrieving cached similarity scores."""
        rewards_dict = {}
        for cid in gradients_dict.keys():
            similarity = self._cached_similarities.get(cid, -1.0)
            reward = (similarity + 1.0) / 2.0
            rewards_dict[cid] = np.clip(reward, 0, 1)
        return rewards_dict