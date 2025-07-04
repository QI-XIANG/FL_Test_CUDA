import numpy as np
import torch
import torch.distributions as tdist
import math
from sklearn.utils.extmath import randomized_svd

class ReputationGuardFL:
    """
    An adaptive and robust client selection and aggregation mechanism for
    Federated Learning.
    """
    def __init__(self, num_clients, num_join_clients, global_accuracy_history=None,
                 min_valid_clients=10, c=1.0, prior_alpha=1.0, prior_beta=1.0,
                 decay_factor=0.9, gradient_subsample_dim=2048,
                 attack_tolerance_threshold=0.5):
        """
        Initializes the ReputationGuardFL parameters.

        Args:
            num_clients (int): The total number of clients in the pool.
            num_join_clients (int): The number of clients to select each round.
            global_accuracy_history (list, optional): A list tracking the history of
                                                     global model accuracy. Defaults to None.
            ... other parameters
        """
        self.num_clients = num_clients
        self.num_join_clients = num_join_clients
        self.min_valid_clients = min_valid_clients
        self.gradient_subsample_dim = gradient_subsample_dim
        self.attack_tolerance_threshold = attack_tolerance_threshold
        
        # --- NEW: Store the global accuracy history ---
        # Handle if the list is passed by reference or is None
        self.global_accuracy_history = global_accuracy_history if global_accuracy_history is not None else []


        # --- Client State Tracking (Reputation & Guarding) ---
        self.selection_counts = np.zeros(num_clients)
        self.performance_history = np.zeros(num_clients)
        self.anomaly_score_history = np.zeros(num_clients)
        self.poisoned_penalty = np.zeros(num_clients)

        # --- A variable to store results from the last SVD run ---
        self.last_run_similarities = {} 

        # --- Algorithm Parameters ---
        self.poisoned_threshold = 0.5
        self.c = c
        self.decay_factor = decay_factor
        self.posterior_alpha = torch.ones(num_clients) * prior_alpha
        self.posterior_beta = torch.ones(num_clients) * prior_beta

    def _flatten_and_subsample_gradients(self, gradients_dict):
        """Flattens and optionally subsamples gradients from a dictionary."""
        client_ids = list(gradients_dict.keys())
        flat_gradients = [torch.cat([torch.as_tensor(p, dtype=torch.float32).flatten() for p in gradients_dict[cid]]).cpu().numpy()
                  for cid in client_ids]
        gradient_matrix = np.vstack(flat_gradients)

        if self.gradient_subsample_dim and gradient_matrix.shape[1] > self.gradient_subsample_dim:
            rand_state = np.random.RandomState(42)
            indices = rand_state.choice(gradient_matrix.shape[1], self.gradient_subsample_dim, replace=False)
            return gradient_matrix[:, indices], client_ids, rand_state

        return gradient_matrix, client_ids, None

    @torch.compile
    def detect_anomalies(self, gradients_dict):
        # This method remains unchanged
        subsampled_gradients, client_ids, _ = self._flatten_and_subsample_gradients(gradients_dict)
        if len(client_ids) < 2: return self.anomaly_score_history
        n_components = min(subsampled_gradients.shape[0] - 1, 5)
        if n_components < 1: return self.anomaly_score_history
        u, s, vt = randomized_svd(subsampled_gradients, n_components=n_components, random_state=42)
        reconstructed = np.dot(u, np.dot(np.diag(s), vt))
        recon_errors = np.linalg.norm(subsampled_gradients - reconstructed, axis=1)
        norms_orig = np.linalg.norm(subsampled_gradients, axis=1)
        norms_recon = np.linalg.norm(reconstructed, axis=1)
        cosine_sim = np.einsum('ij,ij->i', subsampled_gradients, reconstructed) / (norms_orig * norms_recon + 1e-9)
        cosine_dist = 1 - cosine_sim
        # --- Store the raw similarity results for reuse ---
        self.last_run_similarities = {cid: sim for cid, sim in zip(client_ids, cosine_sim)}
        norm_errors = (recon_errors - np.min(recon_errors)) / (np.ptp(recon_errors) + 1e-8)
        norm_dist = (cosine_dist - np.min(cosine_dist)) / (np.ptp(cosine_dist) + 1e-8)
        combined_scores = 0.6 * norm_errors + 0.4 * norm_dist
        for i, client_id in enumerate(client_ids):
            self.anomaly_score_history[client_id] = (self.decay_factor * self.anomaly_score_history[client_id] + (1 - self.decay_factor) * combined_scores[i])
        return self.anomaly_score_history


    def _adjust_poisoned_threshold(self, anomaly_scores):
        """Dynamically adjusts the anomaly threshold based on score distribution
           and global model performance."""
        
        # --- NEW: Adaptive Defense Threshold ---
        # Default to the median (50th percentile)
        base_percentile = 50
        # If accuracy has dropped, tighten the defense
        if len(self.global_accuracy_history) >= 2 and self.global_accuracy_history[-1] < self.global_accuracy_history[-2]:
            base_percentile = 40 # Use a stricter percentile
            print("Note: Global accuracy dropped. Using stricter anomaly filtering.")
            
        threshold = np.percentile(anomaly_scores, base_percentile)
        
        num_passing = np.sum(anomaly_scores <= threshold)
        if num_passing < self.min_valid_clients:
            new_percentile = 100 * (1 - self.min_valid_clients / self.num_clients)
            threshold = np.percentile(anomaly_scores, new_percentile)
            print(f"Warning: Relaxed anomaly threshold to {threshold:.4f} to meet min client requirement.")
        self.poisoned_threshold = threshold

    def select_clients(self, epoch, gradients_dict):
        """Selects clients using a two-stage defense and a hybrid UCB-Thompson score."""
        
        # --- FIX: Get the list of clients that have gradients available this round ---
        available_client_ids = list(gradients_dict.keys())
        if not available_client_ids:
            print("Warning: No gradients received for selection. Returning empty set.")
            return np.array([], dtype=int)

        # --- The following logic now correctly operates only on available clients ---
        anomaly_scores = self.detect_anomalies(gradients_dict)
        self._adjust_poisoned_threshold(anomaly_scores)

        # --- Stage 1 filtering now iterates through available clients ONLY ---
        potential_client_ids = []
        for cid in available_client_ids:
            if (self.anomaly_score_history[cid] + self.poisoned_penalty[cid]) <= self.poisoned_threshold:
                potential_client_ids.append(cid)

        # --- Stage 2 defense now correctly uses the pre-filtered list ---
        final_valid_client_ids = []
        if len(potential_client_ids) > 0:
            # The line that previously caused the error is now safe,
            # as every cid in potential_client_ids is a key in gradients_dict.
            valid_gradients = [torch.cat([torch.as_tensor(p, dtype=torch.float32).flatten() for p in gradients_dict[cid]]).cpu().numpy()
                               for cid in potential_client_ids]
            robust_mean_gradient = np.mean(valid_gradients, axis=0)

            for client_id, client_grad_vec in zip(potential_client_ids, valid_gradients):
                similarity_to_mean = (np.dot(client_grad_vec, robust_mean_gradient) /
                                      (np.linalg.norm(client_grad_vec) * np.linalg.norm(robust_mean_gradient) + 1e-9))
                if similarity_to_mean >= self.attack_tolerance_threshold:
                    final_valid_client_ids.append(client_id)

        # The rest of the function proceeds as before, using the correctly filtered clients.
        hybrid_scores = np.full(self.num_clients, -np.inf)
        
        stability_factor = 1.0
        if len(self.global_accuracy_history) >= 5:
            recent_avg = np.mean(self.global_accuracy_history[-5:-1])
            if self.global_accuracy_history[-1] < recent_avg:
                stability_factor = 0.5
                print("Note: Accuracy stagnated. Promoting exploration.")

        weight_ucb = stability_factor * (0.5 + 0.1 * (1 - np.mean(anomaly_scores[available_client_ids])))
        weight_ts = 1 - weight_ucb

        for i in final_valid_client_ids:
            if self.selection_counts[i] > 0:
                avg_reward = self.performance_history[i] / self.selection_counts[i]
                exploration_term = math.sqrt(2 * math.log(epoch + 1) / self.selection_counts[i])
                ucb_score = avg_reward + self.c * exploration_term
            else:
                ucb_score = 1e10
            thompson_score = tdist.Beta(self.posterior_alpha[i], self.posterior_beta[i]).sample().item()
            hybrid_scores[i] = weight_ucb * ucb_score + weight_ts * thompson_score

        # Select from all clients, but only those in final_valid_client_ids have a non-infinite score
        p = np.random.permutation(len(hybrid_scores))
        selected_clients_indices = np.argsort(hybrid_scores[p])[::-1]
        selected_clients = p[selected_clients_indices][:self.num_join_clients]

        for client_id in selected_clients:
            self.selection_counts[client_id] += 1
            
        # Update penalty only for clients that were available this round
        for cid in available_client_ids:
            if self.anomaly_score_history[cid] > self.poisoned_threshold:
                self.poisoned_penalty[cid] += 0.1

        #print(f"Epoch {epoch}: Selected {len(selected_clients)} clients. Anomaly Threshold: {self.poisoned_threshold:.4f}")
        return selected_clients

    def update(self, selected_clients, rewards):
        """Updates client reputation based on rewards from the training round."""
        for client, reward in zip(selected_clients, rewards):
            reward = np.clip(reward, 0, 1)
            self.performance_history[client] += reward
            self.posterior_alpha[client] += reward
            self.posterior_beta[client] += (1 - reward)

    def get_aggregation_weights(self, selected_clients):
        """Calculates aggregation weights based on client reputation."""
        aggregation_weights = {}
        raw_scores = {}
        total_score = 0
        for cid in selected_clients:
            score = (self.posterior_alpha[cid] / (self.posterior_alpha[cid] + self.posterior_beta[cid])).item()
            raw_scores[cid] = score
            total_score += score
        if total_score == 0:
            num_selected = len(selected_clients)
            for cid in selected_clients:
                aggregation_weights[cid] = 1.0 / num_selected if num_selected > 0 else 0
        else:
            for cid in selected_clients:
                aggregation_weights[cid] = raw_scores[cid] / total_score
        return aggregation_weights

    def calculate_robust_rewards_old(self, gradients_dict):
        """Calculates a robust reward based on gradient conformity."""
        subsampled_gradients, client_ids, _ = self._flatten_and_subsample_gradients(gradients_dict)
        rewards_dict = {cid: 0 for cid in client_ids}
        if len(client_ids) < 2: return rewards_dict
        n_components = min(subsampled_gradients.shape[0] - 1, 10)
        if n_components < 1: return rewards_dict
        u, s, vt = randomized_svd(subsampled_gradients, n_components=n_components, random_state=42)
        reconstructed = np.dot(u, np.dot(np.diag(s), vt))
        norms_orig = np.linalg.norm(subsampled_gradients, axis=1)
        norms_recon = np.linalg.norm(reconstructed, axis=1)
        similarities = (np.einsum('ij,ij->i', subsampled_gradients, reconstructed) / (norms_orig * norms_recon + 1e-9))
        for i, client_id in enumerate(client_ids):
            reward = (similarities[i] + 1.0) / 2.0
            rewards_dict[client_id] = np.clip(reward, 0, 1)
        return rewards_dict
    
    def calculate_robust_rewards(self, gradients_dict):
        """
        Calculates robust rewards by retrieving the pre-computed similarity
        scores from the last anomaly detection phase, avoiding re-computation.
        """
        rewards_dict = {}
        client_ids = list(gradients_dict.keys())

        for cid in client_ids:
            # Retrieve the stored similarity score, defaulting to -1 (worst) if not found
            similarity = self.last_run_similarities.get(cid, -1.0)
            
            # Scale similarity from [-1, 1] to a reward in [0, 1]
            reward = (similarity + 1.0) / 2.0
            rewards_dict[cid] = np.clip(reward, 0, 1)
            
        return rewards_dict