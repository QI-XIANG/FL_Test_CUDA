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
from flcore.servers.client_selection.GAC import GAClientSelection
from flcore.servers.client_selection.RSVD import RSVDClientDetection
from flcore.servers.client_selection.RSVDUCB_old import RSVDUCBClientSelection
from flcore.servers.client_selection.RSVDUCBT import RSVDUCBThompson


class AdaptiveRobustFedAvg(Server):
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
        self.weight_option = "adaptive"  # Fixed weight option
        # Initialize clients and set slow clients
        self.set_slow_clients()
        self.set_clients(args, clientAVG)

        # Initialize client gradients (RSVD purpose)
        self.client_gradients = {}  # Store gradients for each client
        self.gradients_available = False  # Flag to track if gradients are available
        
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
                    selected_ids = select_agent.select_clients(i)

                print("Selected clients:", selected_ids)
                self.selected_clients = [self.clients[c] for c in selected_ids]

                print(f"\n-------------Round number: {i}-------------")
                print(f"History acc: {self.acc_his}")

                # Train each selected client and collect gradients
                for client in self.selected_clients:
                    client.train()
                    if self.select_clients_algorithm in ["RSVD", "RSVDUCB", "RSVDUCBT"]:
                        gradients = client.get_training_gradients()  # Get gradients after training
                        self.client_gradients[client.id] = gradients  # Store gradients

                # After the first round, set the flag to True to start using gradients in future rounds
                if not self.gradients_available:
                    self.gradients_available = True

                self.receive_models()

                # Calculate each model's accuracy and update weights
                clients_acc, clients_acc_weight = self._evaluate_clients(testloaderfull)

                # Update rewards and client selection
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
                
                if self.select_clients_algorithm in ["UCB"]:
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
        elif self.client_selection_algorithm == "UCB":
            return UCB(self.num_clients, self.num_join_clients)
        elif self.client_selection_algorithm == "AUCB":
            return AdaptiveUCB(self.num_clients, self.num_join_clients)
        elif self.client_selection_algorithm == "Thompson":
            return Thompson(num_clients=self.num_clients, num_selections=self.num_join_clients)
        elif self.select_clients_algorithm == "GAC":
            select_agent = GAClientSelection(self.num_clients, self.num_join_clients)
        elif self.select_clients_algorithm == "RSVD":
            select_agent = RSVDClientDetection(self.num_clients, self.num_join_clients)
        elif self.select_clients_algorithm == "RSVDUBC":
            select_agent = RSVDUCBClientSelection(self.num_clients, self.num_join_clients)
        elif self.select_clients_algorithm == "RSVDUCBT":
            select_agent = RSVDUCBThompson(self.num_clients, self.num_join_clients)
        else:
            raise ValueError(f"Unsupported client selection algorithm: {self.client_selection_algorithm}")

    def _evaluate_clients(self, testloader):
        """Calculate accuracy for each selected client and compute normalized weights."""
        clients_acc = []
        for client_model, client in zip(self.uploaded_models, self.selected_clients):
            test_acc, test_num, auc = self.test_metrics_all(client_model, testloader)
            #print(f"Test accuracy: {test_acc / test_num:.4f}")
            clients_acc.append(test_acc / test_num)

        # Normalize the accuracies to use as weights
        total_acc = sum(clients_acc)
        clients_acc_weight = [acc / total_acc for acc in clients_acc]

        return clients_acc, clients_acc_weight

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
