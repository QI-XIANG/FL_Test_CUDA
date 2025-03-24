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



class FedTrimmed(Server):
    def __init__(self, args, times, agent = None):
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

        # self.load_model()

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
    
    def Trimmed(self, weights, n_attackers):
        num_clients = len(weights)
        n = num_clients - 2 * n_attackers
        dist_matrix = np.zeros((num_clients, num_clients))
        # 计算权重之间的距离
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                dist = self.euclidean_distance(weights[i], weights[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        distance = np.sum(dist_matrix, axis=1)
        med = np.median(distance)
        chosen = abs(distance - med).argsort()
        chosen_index = chosen[: n]

    
        return chosen_index
    
    def train(self):
        self.send_models() #initialize model
        testloaderfull = self.get_test_data()

        if self.args.select_clients_algorithm == "Random":
            select_agent = Random(self.num_clients, self.num_join_clients, self.random_join_ratio)

        elif self.args.select_clients_algorithm == "UCB":
            select_agent = UCB(self.num_clients, self.num_join_clients)

        # elif self.args.selected_clients_algorithm == "DQN":
        #     state = self.get_state()
        #     action = self.agent.select_action(state)
        #     self.selected_clients = [self.clients[c] for c in action]
        
        elif self.args.select_clients_algorithm == "Thompson":
            select_agent = Thompson(num_clients=self.num_clients, num_selections=self.num_join_clients)

        
        mlflow.set_experiment(self.select_clients_algorithm)
        with mlflow.start_run(run_name = f"noniid_wbn_{self.num_clients*self.poisoned_ratio}_KRUM"):
            mlflow.log_param("global_rounds", self.global_rounds)
            mlflow.log_param("dataset", self.dataset)
            mlflow.log_param("algorithm", self.algorithm)
            mlflow.log_param("num_clients", self.num_clients)

            for i in range(self.global_rounds+1):
                s_t = time.time()
                
                selected_ids = select_agent.select_clients(i)
                print("selected clients:", selected_ids)
                self.selected_clients = [self.clients[c] for c in selected_ids]

                # self.selected_clients = self.select_clients()
                # s = [c.id for c in self.selected_clients]
                # print(s)

                # => mh code 
                
                '''
                select client by UCB
                '''
                # self.selected_clients = self.select_clients_UCB(i)
                # s = [c.id for c in self.selected_clients]
                # print(s)

                print(f"\n-------------Round number: {i}-------------")

                print(f"history acc: {self.acc_his}")
                # <= mh code 

                for client in self.selected_clients:
                    client.train()

                # threads = [Thread(target=client.train)
                #            for client in self.selected_clients]
                # [t.start() for t in threads]
                # [t.join() for t in threads]


                self.receive_models()
                clients_weight = [parameters_to_vector(i.parameters()).cpu().detach().numpy() for i in self.uploaded_models]
                trimmed_clients_index = self.Trimmed(clients_weight, int(self.num_join_clients*self.poisoned_ratio))
                print(trimmed_clients_index)
                self.uploaded_models = [self.uploaded_models[ti] for ti in trimmed_clients_index]

                if self.dlg_eval and i%self.dlg_gap == 0:
                    self.call_dlg(i)

                same_weight = [1/len(trimmed_clients_index)] * len(trimmed_clients_index)
                self.aggregate_parameters_bn(same_weight)


                self.send_models_bn()
                # self.send_models()

                if i%self.eval_gap == 0:
                    # print(f"\n-------------Round number: {i}-------------")
                    print("\nEvaluate global model")
                    acc, train_loss, auc = self.evaluate()
                    # acc, train_loss = self.evaluate_trust()
                    self.acc_data.append(acc)
                    self.loss_data.append(train_loss)
                    self.auc_data.append(auc)
                    mlflow.log_metric("global accuracy", acc, step = i)
                    mlflow.log_metric("train_loss", train_loss, step = i)

                # => mh code
                '''
                use selected clients to test accuracy
                '''
                # acc_p = 0
                # for client in self.selected_clients:
                #     ct, ns, auc = client.test_metrics()
                #     acc_p += ct/ns
                # acc_p = acc_p / len(self.selected_clients)
                # print(f"acc_p: {acc_p}")
                # <= mh code

                self.Budget.append(time.time() - s_t)
                print('-'*25, 'time cost', '-'*25, self.Budget[-1])

                if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                    break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()


    def compute_robustLR(self, agent_updates):
        agent_updates_sign = [torch.sign(update) for update in agent_updates]  
        sm_of_signs = torch.abs(sum(agent_updates_sign))

        sm_of_signs[sm_of_signs < self.robustLR_threshold] = -self.server_lr
        sm_of_signs[sm_of_signs >= self.robustLR_threshold] = self.server_lr   
        return sm_of_signs.to(self.device)
