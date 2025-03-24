import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import mlflow
import torch
from sklearn.cluster import KMeans
import numpy as np
import copy
from flcore.servers.client_selection.Random import Random
from flcore.servers.client_selection.Thompson import Thompson
from flcore.servers.client_selection.UCB import UCB
from flcore.servers.client_selection.RCS import RandomClusterSelection
from flcore.servers.client_selection.DECS import DiversityEnhancedClusterSelection

class FedEpsilonDecayBulyan(Server):
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
        bn_key = ['conv1.1.weight', 'conv1.1.bias', 'conv1.1.running_mean', 'conv1.1.running_var', 'conv1.1.num_batches_tracked',
                  'conv2.1.weight', 'conv2.1.bias', 'conv2.1.running_mean', 'conv2.1.running_var', 'conv2.1.num_batches_tracked']
        v = []
        for key in model.state_dict():
            if key in bn_key:
                continue 
            v.append(model.state_dict()[key].view(-1))
        return torch.cat(v)

    def euclidean_distance(self, x, y):
        return np.linalg.norm(x - y)

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
        model_params = torch.zeros_like(parameters_to_vector(selected_models[0].parameters()).clone())

        total_weight = 0
        for i, model in enumerate(selected_models):
            idx = selected_indices[i]
            weight = self.performance_scores[idx]
            model_params += weight * parameters_to_vector(model.parameters()).clone()
            total_weight += weight

        model_params /= total_weight
        aggregated_model = copy.deepcopy(selected_models[0])
        vector_to_parameters(model_params, aggregated_model.parameters())

        return aggregated_model

    def adjust_performance_scores(self, acc, participating_indices):
        for idx in participating_indices:
            if acc > 0.5:
                self.performance_scores[idx] = min(self.performance_scores[idx] * 1.1, 2)
            else:
                self.performance_scores[idx] = max(self.performance_scores[idx] * 0.9, 0.1)
        
        # Decay all scores to encourage diversity over rounds
        self.performance_scores *= self.decay_factor
        print("Updated performance scores for participating clients:", self.performance_scores[participating_indices])

    def select_clients_with_bias(self, select_agent, round_idx):
        selection_probs = self.performance_scores / np.sum(self.performance_scores)

        selected_ids = []
        for _ in range(self.num_join_clients):
            if np.random.rand() < self.epsilon:
                selected_ids.append(np.random.choice(self.num_clients))
            else:
                selected_ids.append(np.random.choice(self.num_clients, p=selection_probs))

        print(f"Round {round_idx}: selected clients with performance-based bias and epsilon-greedy:", selected_ids)
        return np.unique(selected_ids)

    def train(self):
        self.send_models()
        testloaderfull = self.get_test_data()

        if self.args.select_clients_algorithm == "Random":
            select_agent = Random(self.num_clients, self.num_join_clients, self.random_join_ratio)
        elif self.args.select_clients_algorithm == "RCS":
            select_agent = RandomClusterSelection(self.num_clients, self.num_join_clients, self.random_join_ratio)
        elif self.args.select_clients_algorithm == "DECS":
            select_agent = DiversityEnhancedClusterSelection(self.num_clients, self.num_join_clients, self.random_join_ratio)
        elif self.args.select_clients_algorithm == "UCB":
            select_agent = UCB(self.num_clients, self.num_join_clients)
        elif self.args.select_clients_algorithm == "Thompson":
            select_agent = Thompson(num_clients=self.num_clients, num_selections=self.num_join_clients)

        mlflow.set_experiment(self.select_clients_algorithm)
        with mlflow.start_run(run_name=f"noniid_wbn_{self.num_clients * self.poisoned_ratio}_RobustBULYAN"):
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

                for client in self.selected_clients:
                    client.train()

                self.receive_models()
                clients_weight = [parameters_to_vector(i.parameters()).cpu().detach().numpy() for i in self.uploaded_models]

                bulyan_client_indices = self.robust_bulyan(clients_weight, int(self.num_join_clients * self.poisoned_ratio), drop_percentage=0.1)
                print("Selected clients after Bulyan:", bulyan_client_indices)

                # Identify poisoned clients among Bulyan-selected clients
                poisoned_selected = [idx for idx in bulyan_client_indices if self.clients[idx].poisoned]
                print(f"Poisoned clients among Bulyan-selected clients: {poisoned_selected}")

                aggregated_model = self.aggregate_models([self.uploaded_models[idx] for idx in bulyan_client_indices], bulyan_client_indices)

                if self.dlg_eval and i % self.dlg_gap == 0:
                    self.call_dlg(i)

                self.global_model = aggregated_model
                self.send_models()

                if i % self.eval_gap == 0:
                    print("\nEvaluate global model")
                    acc, train_loss, auc = self.evaluate()
                    self.acc_data.append(acc)
                    self.loss_data.append(train_loss)
                    self.auc_data.append(auc)
                    mlflow.log_metric("global accuracy", acc, step=i)
                    mlflow.log_metric("train_loss", train_loss, step=i)
                    mlflow.log_metric("test_auc", auc, step=i)

                    self.adjust_performance_scores(acc, bulyan_client_indices)

                self.Budget.append(time.time() - s_t)
                print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

                if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                    break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()
