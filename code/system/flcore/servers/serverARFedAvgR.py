import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import mlflow
import torch
import pandas as pd

from flcore.servers.client_selection.Random import Random
from flcore.servers.client_selection.Thompson import Thompson
from flcore.servers.client_selection.UCB import UCB
from flcore.servers.client_selection.AUCB import AdaptiveUCB
from flcore.servers.client_selection.RCS import RandomClusterSelection
from flcore.servers.client_selection.UCBECS import UCBEnhancedClusterSelection

class AdaptiveRobustFedAvgR(Server):
    def __init__(self, args, times, agent=None):
        super().__init__(args, times)
        
        self.agent = agent
        self.robustLR_threshold = 7
        self.server_lr = 1e-3
        
        # Use correct attribute names from args
        self.client_selection_algorithm = args.select_clients_algorithm  # Fixed name from args
        self.num_clients = args.num_clients
        self.num_join_clients = int(args.join_ratio * args.num_clients)  # Correct number of clients joining
        self.eval_gap = args.eval_gap
        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.auto_break = args.auto_break
        self.top_cnt = 10  # Use a fixed top count threshold (you may adjust this as needed)

        # Initialize clients and set slow clients
        self.set_slow_clients()
        self.set_clients(args, clientAVG)
        
        print(f"\nJoin ratio / total clients: {self.num_join_clients} / {self.num_clients}")
        print("Finished creating server and clients.")

    def get_vector_no_bn(self, model):
        bn_key = ['conv1.1.weight', 'conv1.1.bias', 'conv1.1.running_mean', 'conv1.1.running_var', 
                  'conv1.1.num_batches_tracked', 'conv2.1.weight', 'conv2.1.bias', 'conv2.1.running_mean',
                  'conv2.1.running_var', 'conv2.1.num_batches_tracked']
        v = [model.state_dict()[key].view(-1) for key in model.state_dict() if key not in bn_key]
        return torch.cat(v)

    def train(self):
        self.send_models()  # Initialize model
        testloaderfull = self.get_test_data()

        # Select client selection algorithm
        select_agent = self._get_select_agent()

        mlflow.set_experiment(self.client_selection_algorithm)
        with mlflow.start_run(run_name=f"noniid_wbn_{self.num_clients * self.poisoned_ratio}_contribution"):
            mlflow.log_param("global_rounds", self.global_rounds)
            mlflow.log_param("dataset", self.dataset)
            mlflow.log_param("algorithm", self.algorithm)
            mlflow.log_param("num_clients", self.num_clients)

            for i in range(self.global_rounds + 1):
                s_t = time.time()

                selected_ids = select_agent.select_clients(i)
                print("Selected clients:", selected_ids)
                self.selected_clients = [self.clients[c] for c in selected_ids]

                print(f"\n-------------Round number: {i}-------------")
                print(f"History acc: {self.acc_his}")

                # Parallelize client training using ThreadPoolExecutor
                threads = [Thread(target=client.train) for client in self.selected_clients]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

                self.receive_models()

                # Calculate each model's accuracy and update weights
                clients_acc, clients_acc_weight, trust_scores = self._evaluate_clients(testloaderfull)

                # Update rewards and client selection
                reward_decay = 1
                for reward, client in zip(clients_acc, self.selected_clients):
                    self.sums_of_reward[client.id] = self.sums_of_reward[client.id] * reward_decay + reward
                    self.numbers_of_selections[client.id] += 1

                select_agent.update(selected_ids, clients_acc)

                # Evaluate if necessary
                if self.dlg_eval and i % self.dlg_gap == 0:
                    self.call_dlg(i)

                # Aggregate parameters using adaptive weights
                self._aggregate_parameters(clients_acc_weight)

                self.send_models_bn()

                # Evaluate global model
                if i % self.eval_gap == 0:
                    print("\nEvaluate global model")
                    acc, train_loss, auc = self.evaluate_trust()
                    self.acc_data.append(acc)
                    self.loss_data.append(train_loss)
                    self.auc_data.append(auc)
                    mlflow.log_metric("global_accuracy", acc, step=i)
                    mlflow.log_metric("train_loss", train_loss, step=i)

                self.Budget.append(time.time() - s_t)
                print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

                if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                    break

        print("\nBest accuracy:", max(self.rs_test_acc))
        print("\nAverage time cost per round:", sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

    def _get_select_agent(self):
        """Helper function to instantiate the client selection agent based on the selected algorithm."""
        if self.client_selection_algorithm == "Random":
            return Random(self.num_clients, self.num_join_clients, self.random_join_ratio)
        elif self.client_selection_algorithm == "RCS":
            return RandomClusterSelection(self.num_clients, self.num_join_clients, self.random_join_ratio)
        elif self.client_selection_algorithm == "UCB":
            return UCB(self.num_clients, self.num_join_clients)
        elif self.client_selection_algorithm == "AUCB":
            return AdaptiveUCB(self.num_clients, self.num_join_clients)
        elif self.client_selection_algorithm == "UCBECS":
            return UCBEnhancedClusterSelection(self.num_clients, self.num_join_clients, self.random_join_ratio)
        elif self.client_selection_algorithm == "Thompson":
            return Thompson(num_clients=self.num_clients, num_selections=self.num_join_clients)
        else:
            raise ValueError(f"Unsupported client selection algorithm: {self.client_selection_algorithm}")

    def _evaluate_clients(self, testloader):
        """Calculate accuracy for each selected client and compute normalized weights and trust scores."""
        clients_acc = []
        trust_scores = []  # Trust scores initialized

        for client_model, client in zip(self.uploaded_models, self.selected_clients):
            test_acc, test_num, auc = self.test_metrics_all(client_model, testloader)
            print(f"Test accuracy: {test_acc / test_num:.4f}")
            clients_acc.append(test_acc / test_num)

            # Calculate a simple trust score based on accuracy and number of updates
            trust_score = self.calculate_trust_score(client, testloader)
            trust_scores.append(trust_score)

        # Normalize the accuracies to use as weights
        total_acc = sum(clients_acc)
        clients_acc_weight = [acc / total_acc for acc in clients_acc]

        return clients_acc, clients_acc_weight, trust_scores

    def calculate_trust_score(self, client, testloader):
        """Calculate the trust score based on client performance and updates."""
        
        # Make sure num_updates exists, otherwise set it to 0
        num_updates = getattr(client, 'num_updates', 0)
        
        # You can also use other attributes or methods of the client to calculate trust score
        accuracy = self.test_client_accuracy(client, testloader)  # Assuming you already have a method to calculate accuracy
        
        # Trust score calculation logic (you can customize this)
        trust_score = accuracy * (num_updates + 1)  # Example: Trust score is weighted by the number of updates
        
        return trust_score


    def test_client_accuracy(self, client, testloader):
        """Test accuracy of a client using their model on the provided testloader."""
        # Use the client's model to compute the accuracy on the test data
        model = client.model  # Assuming each client has a model attribute
        
        # Check if CUDA is available, then move the model and input to the correct device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)  # Move model to the correct device
        
        model.eval()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in testloader:
                # Move data and target to the same device as the model
                data, target = data.to(device), target.to(device)
                
                # Forward pass through the model
                output = model(data)
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total
        return accuracy


    def _aggregate_parameters(self, clients_acc_weight):
        """Aggregate parameters using the weights of the clients."""
        same_weight = [1 / self.num_join_clients] * self.num_join_clients
        weight = clients_acc_weight if self.weight_option != "same" else same_weight
        self.aggregate_parameters_bn(weight)

    def compute_robustLR(self, agent_updates):
        """Computes robust learning rate for model updates based on client updates."""
        agent_updates_sign = [torch.sign(update) for update in agent_updates]
        sm_of_signs = torch.abs(sum(agent_updates_sign))

        sm_of_signs[sm_of_signs < self.robustLR_threshold] = -self.server_lr
        sm_of_signs[sm_of_signs >= self.robustLR_threshold] = self.server_lr
        return sm_of_signs.to(self.device)
