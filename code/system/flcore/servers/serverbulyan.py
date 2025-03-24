import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import mlflow
import torch
from sklearn.cluster import KMeans
import pandas as pd

import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

from flcore.servers.client_selection.Random import Random
from flcore.servers.client_selection.Thompson import Thompson
from flcore.servers.client_selection.UCB import UCB
from flcore.servers.client_selection.RCS import RandomClusterSelection
from flcore.servers.client_selection.DECS import DiversityEnhancedClusterSelection

class FedBulyan(Server):
    def __init__(self, args, times, agent=None):
        super().__init__(args, times)

        self.args = args
        self.agent = agent
        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientAVG)
        self.robustLR_threshold = 7
        self.server_lr = 1e-3

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
    
    def bulyan(self, weights, n_attackers):
        num_clients = len(weights)
        dist_matrix = np.zeros((num_clients, num_clients))
        
        # Calculate pairwise distances between clients
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                dist = self.euclidean_distance(weights[i], weights[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        # Krum-like selection step: Find the k closest clients for each client
        sorted_idx = np.sum(dist_matrix, axis=0).argsort()[:num_clients - n_attackers]
        krum_selected_indices = sorted_idx[:num_clients - n_attackers]
        
        # Apply clustering (KMeans) to further refine the selection
        client_updates = [weights[i] for i in krum_selected_indices]
        
        # Clustering the selected clients' updates
        kmeans = KMeans(n_clusters=1, random_state=0).fit(client_updates)
        # Get the centroid of the cluster
        centroid = kmeans.cluster_centers_[0]

        # Find the client whose update is closest to the centroid (central model)
        closest_client_idx = np.argmin([self.euclidean_distance(centroid, update) for update in client_updates])
        
        # Return the selected client index from the KMeans clustering result
        return krum_selected_indices[closest_client_idx]
    
    def train(self):
        self.send_models()  # Initialize model
        testloaderfull = self.get_test_data()

        # Select clients based on the selected algorithm
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
        with mlflow.start_run(run_name=f"noniid_wbn_{self.num_clients * self.poisoned_ratio}_BULYAN"):
            mlflow.log_param("global_rounds", self.global_rounds)
            mlflow.log_param("dataset", self.dataset)
            mlflow.log_param("algorithm", self.algorithm)
            mlflow.log_param("num_clients", self.num_clients)

            for i in range(self.global_rounds + 1):
                s_t = time.time()
                
                selected_ids = select_agent.select_clients(i)
                print("selected clients:", selected_ids)
                self.selected_clients = [self.clients[c] for c in selected_ids]

                print(f"\n-------------Round number: {i}-------------")
                print(f"history acc: {self.acc_his}")

                for client in self.selected_clients:
                    client.train()

                self.receive_models()
                clients_weight = [parameters_to_vector(i.parameters()).cpu().detach().numpy() for i in self.uploaded_models]
                bulyan_client_index = self.bulyan(clients_weight, int(self.num_join_clients * self.poisoned_ratio))
                print(bulyan_client_index)

                if self.dlg_eval and i % self.dlg_gap == 0:
                    self.call_dlg(i)

                # Update the global model with the selected Bulyan model
                self.global_model = copy.deepcopy(self.uploaded_models[bulyan_client_index])

                self.send_models()

                if i % self.eval_gap == 0:
                    print("\nEvaluate global model")
                    # Unpack three values from evaluate() function
                    acc, train_loss, auc = self.evaluate()  # Unpacked to handle 3 return values
                    self.acc_data.append(acc)
                    self.loss_data.append(train_loss)
                    self.auc_data.append(auc)
                    mlflow.log_metric("global accuracy", acc, step=i)
                    mlflow.log_metric("train_loss", train_loss, step=i)
                    mlflow.log_metric("test_auc", auc, step=i)  # Log the AUC value too

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

    def compute_robustLR(self, agent_updates):
        agent_updates_sign = [torch.sign(update) for update in agent_updates]  
        sm_of_signs = torch.abs(sum(agent_updates_sign))

        sm_of_signs[sm_of_signs < self.robustLR_threshold] = -self.server_lr
        sm_of_signs[sm_of_signs >= self.robustLR_threshold] = self.server_lr   
        return sm_of_signs.to(self.device)
