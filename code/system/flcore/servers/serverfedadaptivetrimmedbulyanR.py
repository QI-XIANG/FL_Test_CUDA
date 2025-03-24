import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from sklearn.metrics.pairwise import cosine_similarity
from threading import Thread
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import mlflow
import torch
import numpy as np
import copy
from flcore.servers.client_selection.Random import Random
from flcore.servers.client_selection.Thompson import Thompson
from flcore.servers.client_selection.UCB import UCB
from flcore.servers.client_selection.RCS import RandomClusterSelection
from flcore.servers.client_selection.DECS import DiversityEnhancedClusterSelection
from flcore.servers.client_selection.ECS import EnhancedClusterSelection
from flcore.servers.client_selection.UCBECS import UCBEnhancedClusterSelection

class RobustFedBulyanR(Server):
    def __init__(self, args, times, agent=None, epsilon=0.1, decay_factor=0.99):
        super().__init__(args, times)
        self.args = args
        self.agent = agent
        self.epsilon = epsilon
        self.decay_factor = decay_factor
        self.set_slow_clients()
        self.set_clients(args, clientAVG)
        self.robustLR_threshold = 7
        self.server_lr = 1e-3

        # Initialize performance scores for clients
        self.performance_scores = np.ones(self.num_clients)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def get_vector_no_bn(self, model):
        bn_key = ['conv1.1.weight', 'conv1.1.bias', 'conv1.1.running_mean', 'conv1.1.running_var', 'conv1.1.num_batches_tracked']
        v = []
        for key in model.state_dict():
            if key in bn_key:
                continue 
            v.append(model.state_dict()[key].view(-1))
        return torch.cat(v)

    def pearson_correlation(self, x, y):
        return np.corrcoef(x, y)[0, 1]

    def robust_bulyan(self, weights, n_attackers, drop_percentage=0.1):
        num_clients = len(weights)
        dist_matrix = np.zeros((num_clients, num_clients))

        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                correlation = self.pearson_correlation(weights[i], weights[j])
                dist_matrix[i, j] = correlation
                dist_matrix[j, i] = correlation

        avg_correlations = np.mean(dist_matrix, axis=1)
        sorted_idx = np.argsort(avg_correlations)[::-1]
        num_clients_to_keep = int(num_clients * (1 - drop_percentage))
        selected_indices = sorted_idx[:num_clients_to_keep]

        print(f"Selected {len(selected_indices)} clients out of {num_clients} based on Pearson correlation.")
        return selected_indices

    def aggregate_models(self, selected_models, selected_indices):
        """
        Aggregates the selected client models using weighted averaging based on performance scores.
        Clients with better performance (higher accuracy, lower loss) contribute more.
        """
        model_params = torch.zeros_like(parameters_to_vector(selected_models[0].parameters()).clone())
        total_weight = 0

        # Normalize performance scores (so they sum to 1)
        performance_scores = np.array([self.performance_scores[idx] for idx in selected_indices])
        normalized_weights = performance_scores / np.sum(performance_scores)

        # Perform weighted aggregation
        for i, model in enumerate(selected_models):
            idx = selected_indices[i]
            weight = normalized_weights[i]  # Use normalized performance score as weight
            model_params += weight * parameters_to_vector(model.parameters()).clone()
            total_weight += weight

        # Normalize the aggregated parameters
        model_params /= total_weight
        aggregated_model = copy.deepcopy(selected_models[0])
        vector_to_parameters(model_params, aggregated_model.parameters())

        return aggregated_model


    def adjust_performance_scores(self, accuracies, participating_indices, decay_factor=0.95, std_threshold=1.5):
        """
        Update performance scores of the participating clients with a focus on penalizing poisoned clients.
        """
        # Step 1: Calculate the mean and standard deviation of performance across the participating clients
        performance_scores = [self.performance_scores[idx] for idx in participating_indices]
        mean_performance = np.mean(performance_scores)
        std_performance = np.std(performance_scores)

        # Step 2: Update performance scores based on relative performance (z-score style)
        for idx, client_acc in zip(participating_indices, accuracies):
            # Calculate how much the client's performance deviates from the group mean
            deviation = (client_acc - mean_performance) / (std_performance + 1e-5)  # Add small value to avoid divide-by-zero

            # If the client's performance deviates significantly from the mean, consider it poisoned
            if abs(deviation) > std_threshold:
                # Penalize severely if performance is outside the expected range
                self.performance_scores[idx] *= 0.8  # Reduce the score, penalizing the poisoned client
            else:
                # Otherwise, encourage normal clients slightly
                self.performance_scores[idx] *= 1.05  # Slightly boost the performance score

            # Ensure that the scores stay within a reasonable range
            self.performance_scores[idx] = max(0.1, min(self.performance_scores[idx], 2))

        # Step 3: Apply a gentle decay to all clients to avoid score convergence over time
        for idx in range(self.num_clients):
            if idx not in participating_indices:
                # Apply decay for clients that did not participate this round
                self.performance_scores[idx] *= decay_factor

        print("Updated performance scores for participating clients:", self.performance_scores[participating_indices])

    def select_clients_with_bias(self, select_agent, round_idx):
        """Select clients with performance-based bias and epsilon-greedy exploration."""
        selection_probs = self.performance_scores / np.sum(self.performance_scores)
        
        selected_ids = set()  # Use a set to track unique selections

        while len(selected_ids) < self.num_join_clients:
            if np.random.rand() < self.epsilon:
                # Randomly choose a client without bias
                selected_ids.add(np.random.choice(self.num_clients))
            else:
                # Select a client based on performance probabilities, ensuring uniqueness
                selected_ids.add(np.random.choice(self.num_clients, p=selection_probs))
            
        selected_ids = list(selected_ids)  # Convert set to list to match the format of selected clients
        print(f"Round {round_idx}: selected clients with performance-based bias and epsilon-greedy:", selected_ids)
        return selected_ids

    def train(self):
        self.send_models()
        testloaderfull = self.get_test_data()

        if self.args.select_clients_algorithm == "ECS":
            select_agent = EnhancedClusterSelection(self.num_clients, self.num_join_clients, self.random_join_ratio)
        elif self.args.select_clients_algorithm == "UCB":
            select_agent = UCB(self.num_clients, self.num_join_clients)
        elif self.args.select_clients_algorithm == "UCBECS":
            select_agent = UCBEnhancedClusterSelection(self.num_clients, self.num_join_clients, self.random_join_ratio)

        mlflow.set_experiment(self.select_clients_algorithm)
        with mlflow.start_run(run_name=f"noniid_wbn_{self.num_clients * self.poisoned_ratio}_FedBulyan"):
            mlflow.log_param("global_rounds", self.global_rounds)
            mlflow.log_param("dataset", self.dataset)
            mlflow.log_param("algorithm", self.algorithm)
            mlflow.log_param("num_clients", self.num_clients)

            for i in range(self.global_rounds + 1):
                s_t = time.time()

                selected_ids = self.select_clients_with_bias(select_agent, i)
                self.selected_clients = [self.clients[c] for c in selected_ids]

                print(f"\n-------------Round number: {i}-------------")
                print(f"history acc: {self.acc_his}")

                # Calculate individual client accuracies
                client_accuracies = []
                for client in self.selected_clients:
                    test_acc, test_num, auc = client.test_metrics()  # Or use your own method to get accuracy
                    client_accuracies.append(test_acc / test_num)

                # Send the list of accuracies to adjust_performance_scores
                self.adjust_performance_scores(client_accuracies, selected_ids)

                # Simulate clients training
                for client in self.selected_clients:
                    client.train()

                self.receive_models()
                clients_weight = [parameters_to_vector(i.parameters()).cpu().detach().numpy() for i in self.uploaded_models]

                bulyan_client_indices = self.robust_bulyan(clients_weight, int(self.num_join_clients * self.poisoned_ratio), drop_percentage=0.1)
                print("Selected clients after FedRFB:", bulyan_client_indices)

                # Identify poisoned clients among selected clients
                poisoned_selected = [idx for idx in bulyan_client_indices if self.clients[idx].poisoned]
                print(f"Poisoned clients among selected clients: {poisoned_selected}")

                aggregated_model = self.aggregate_models([self.uploaded_models[idx] for idx in bulyan_client_indices], bulyan_client_indices)

                self.global_model = aggregated_model
                self.send_models()

                # Evaluate and log metrics every `eval_gap` rounds
                if i % self.eval_gap == 0:
                    print("\nEvaluate global model")
                    acc, train_loss, auc = self.evaluate()  # You might need to adjust this depending on your `evaluate` method
                    self.acc_data.append(acc)
                    self.loss_data.append(train_loss)
                    self.auc_data.append(auc)

                    mlflow.log_metric("global_accuracy", acc, step=i)
                    mlflow.log_metric("train_loss", train_loss, step=i)
                    mlflow.log_metric("test_auc", auc, step=i)

                    # Adjust performance scores based on the evaluation results
                    self.adjust_performance_scores([acc], bulyan_client_indices)  # Adjust the scores of the clients involved in this round

                # Track and log round execution time
                self.Budget.append(time.time() - s_t)
                print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

                # Check if the training process should terminate early based on the stopping condition
                if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                    break

            # After training is complete, print the best accuracy and average time cost
            print("\nBest accuracy.")
            print(max(self.rs_test_acc))
            print("\nAverage time cost per round.")
            print(sum(self.Budget[1:]) / len(self.Budget[1:]))

            # Save the final results and the global model
            self.save_results()
            self.save_global_model()

