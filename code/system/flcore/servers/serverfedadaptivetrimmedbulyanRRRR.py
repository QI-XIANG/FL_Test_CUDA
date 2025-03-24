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
from flcore.servers.client_selection.AUCB import AdaptiveUCB
from flcore.servers.client_selection.RSVD import RSVDClientDetection
from flcore.servers.client_selection.RSVDUCB_old import RSVDUCBClientSelection
from flcore.servers.client_selection.RSVDUCBT import RSVDUCBThompson

class RobustFedBulyanRRRR(Server):
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

        # Initialize client gradients (RSVD purpose)
        self.client_gradients = {}  # Store gradients for each client
        self.gradients_available = False  # Flag to track if gradients are available

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
        model_params = torch.zeros_like(parameters_to_vector(selected_models[0].parameters()).clone())
        total_weight = 0

        performance_scores = np.array([self.performance_scores[idx] for idx in selected_indices])
        normalized_weights = performance_scores / np.sum(performance_scores)

        for i, model in enumerate(selected_models):
            idx = selected_indices[i]
            weight = normalized_weights[i]
            model_params += weight * parameters_to_vector(model.parameters()).clone()
            total_weight += weight

        model_params /= total_weight
        aggregated_model = copy.deepcopy(selected_models[0])
        vector_to_parameters(model_params, aggregated_model.parameters())

        return aggregated_model

    def adjust_performance_scores(self, accuracies, losses, participating_indices, decay_factor=0.95, std_threshold=1.5, boost_factor=1.05, penalize_factor=0.95):
        performance_scores = [self.performance_scores[idx] for idx in participating_indices]
        mean_performance = np.mean(performance_scores)
        std_performance = np.std(performance_scores)

        for idx, (client_acc, client_loss) in zip(participating_indices, zip(accuracies, losses)):
            acc_deviation = (client_acc - mean_performance) / (std_performance + 1e-5)
            loss_deviation = (client_loss - np.mean(losses)) / (np.std(losses) + 1e-5)

            combined_deviation = 0.7 * acc_deviation - 0.3 * loss_deviation

            if abs(combined_deviation) > std_threshold:
                self.performance_scores[idx] *= penalize_factor
            else:
                self.performance_scores[idx] *= boost_factor

            self.performance_scores[idx] = max(0.1, min(self.performance_scores[idx], 2))

        for idx in range(self.num_clients):
            if idx not in participating_indices:
                self.performance_scores[idx] *= decay_factor

        self.performance_scores = np.clip(self.performance_scores, 0.2, 1.8)
        print("Updated performance scores for participating clients:", self.performance_scores[participating_indices])

    def select_clients_with_bias(self, select_agent, round_idx):
        selection_probs = self.performance_scores / np.sum(self.performance_scores)
        selected_ids = set()

        while len(selected_ids) < self.num_join_clients:
            if np.random.rand() < self.epsilon:
                selected_ids.add(np.random.choice(self.num_clients))
            else:
                selected_ids.add(np.random.choice(self.num_clients, p=selection_probs))

        selected_ids = list(selected_ids)
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
        elif self.args.select_clients_algorithm == "AUCB":
            select_agent = AdaptiveUCB(self.num_clients, self.num_join_clients)
        elif self.args.select_clients_algorithm == "Thompson":
            select_agent = Thompson(num_clients=self.num_clients, num_selections=self.num_join_clients)
        elif self.args.select_clients_algorithm == "RSVDUCBT":
            select_agent = RSVDUCBThompson(self.num_clients, self.num_join_clients)

        mlflow.set_experiment(self.select_clients_algorithm)
        with mlflow.start_run(run_name=f"noniid_wbn_{self.num_clients * self.poisoned_ratio}_FedBulyan"):
            mlflow.log_param("global_rounds", self.global_rounds)
            mlflow.log_param("dataset", self.dataset)
            mlflow.log_param("algorithm", self.algorithm)
            mlflow.log_param("num_clients", self.num_clients)

            for i in range(self.global_rounds + 1):
                s_t = time.time()

                # If RSVD is selected, we pass gradients for each round
                if self.select_clients_algorithm == "RSVD":
                    # For the first round, use random selection
                    if not self.gradients_available:
                        select_agent = Random(self.num_clients, self.num_join_clients, self.random_join_ratio)
                        selected_ids = select_agent.select_clients(i)
                    else:
                        # After the first round, pass actual gradients
                        select_agent = RSVDClientDetection(self.num_clients, self.num_join_clients)
                        selected_ids = select_agent.select_clients(i, self.client_gradients)
                elif self.select_clients_algorithm == "RSVDUCB":
                    # For the first round, use random selection
                    if not self.gradients_available:
                        select_agent = Random(self.num_clients, self.num_join_clients, self.random_join_ratio)
                        selected_ids = select_agent.select_clients(i)
                    else:
                        # After the first round, pass actual gradients
                        select_agent = RSVDUCBClientSelection(self.num_clients, self.num_join_clients)
                        selected_ids = select_agent.select_clients(i, self.client_gradients)
                elif self.select_clients_algorithm == "RSVDUCBT":
                    # For the first round, use random selection
                    if not self.gradients_available:
                        select_agent = Random(self.num_clients, self.num_join_clients, self.random_join_ratio)
                        selected_ids = select_agent.select_clients(i)
                    else:
                        # After the first round, pass actual gradients
                        select_agent = RSVDUCBThompson(self.num_clients, self.num_join_clients)
                        selected_ids = select_agent.select_clients(i, self.client_gradients)
                else:
                    # For other algorithms, select clients without gradients (or as needed)
                    selected_ids = self.select_clients_with_bias(select_agent, i)

                self.selected_clients = [self.clients[c] for c in selected_ids]

                print(f"\n-------------Round number: {i}-------------")
                print(f"history acc: {self.acc_his}")

                client_accuracies = []
                client_losses = []
                for client in self.selected_clients:
                    test_acc, test_num, auc = client.test_metrics()
                    client_loss = client.compute_loss()
                    client_accuracies.append(test_acc / test_num)
                    client_losses.append(client_loss)

                self.adjust_performance_scores(client_accuracies, client_losses, selected_ids)

                for client in self.selected_clients:
                    client.train()
                    if self.select_clients_algorithm in ["RSVD", "RSVDUCB", "RSVDUCBT"]:
                        gradients = client.get_training_gradients()  # Get gradients after training
                        self.client_gradients[client.id] = gradients  # Store gradients
                
                # After the first round, set the flag to True to start using gradients in future rounds
                if not self.gradients_available:
                    self.gradients_available = True

                self.receive_models()

                '''
                calculate each model's accuracy
                '''
                if self.select_clients_algorithm in ["RSVD", "RSVDUCB", "RSVDUCBT"] and self.gradients_available:
                    clients_acc = []
                    for client_model, client in zip(self.uploaded_models, self.selected_clients):
                        test_acc, test_num, auc= self.test_metrics_all(client_model, testloaderfull)
                        #print(test_acc/test_num)
                        clients_acc.append(test_acc/test_num)

                    #clients_acc_weight = list(map(lambda x: x/sum(clients_acc), clients_acc))

                    reward_decay = 1
                    for reward, client in zip(clients_acc, self.selected_clients):
                        self.sums_of_reward[client.id] =  self.sums_of_reward[client.id] * reward_decay + reward
                        self.numbers_of_selections[client.id] += 1
                    
                    rewards = clients_acc
                    select_agent.update(selected_ids, rewards)
                
                if self.select_clients_algorithm in ["UCB", "GAC"]:
                    clients_acc = []
                    for client_model, client in zip(self.uploaded_models, self.selected_clients):
                        test_acc, test_num, auc= self.test_metrics_all(client_model, testloaderfull)
                        #print(test_acc/test_num)
                        clients_acc.append(test_acc/test_num)

                    #clients_acc_weight = list(map(lambda x: x/sum(clients_acc), clients_acc))

                    reward_decay = 1
                    for reward, client in zip(clients_acc, self.selected_clients):
                        self.sums_of_reward[client.id] =  self.sums_of_reward[client.id] * reward_decay + reward
                        self.numbers_of_selections[client.id] += 1
                    
                    rewards = clients_acc
                    select_agent.update(selected_ids, rewards)
                
                clients_weight = [parameters_to_vector(i.parameters()).cpu().detach().numpy() for i in self.uploaded_models]

                bulyan_client_indices = self.robust_bulyan(clients_weight, int(self.num_join_clients * self.poisoned_ratio), drop_percentage=0.1)
                print("Selected clients after FedRFB:", bulyan_client_indices)

                poisoned_selected = [idx for idx in bulyan_client_indices if self.clients[idx].poisoned]
                print(f"Poisoned clients among selected clients: {poisoned_selected}")

                aggregated_model = self.aggregate_models([self.uploaded_models[idx] for idx in bulyan_client_indices], bulyan_client_indices)

                self.global_model = aggregated_model
                self.send_models()

                if i % self.eval_gap == 0:
                    print("\nEvaluate global model")

                    if self.select_clients_algorithm in ["RSVD", "RSVDUCB", "RSVDUCBT"] and self.gradients_available:
                        acc, train_loss, auc = self.evaluate_trust()
                    elif self.select_clients_algorithm in ["UCB", "GAC"]:
                        acc, train_loss, auc = self.evaluate_trust()
                    else:
                        acc, train_loss, auc = self.evaluate()

                    self.acc_data.append(acc)
                    self.loss_data.append(train_loss)
                    self.auc_data.append(auc)

                    mlflow.log_metric("global_accuracy", acc, step=i)
                    mlflow.log_metric("train_loss", train_loss, step=i)
                    mlflow.log_metric("test_auc", auc, step=i)

                    self.adjust_performance_scores([acc], [train_loss], bulyan_client_indices)

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
