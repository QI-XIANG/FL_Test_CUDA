import time
from flcore.clients.clientavg_lstm import clientAVG
from flcore.servers.serverbase_lstm import Server
from threading import Thread
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import mlflow
import torch
from sklearn.cluster import KMeans
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.data_utils import read_client_data  # 假設這是您的數據工具模組
from sklearn.preprocessing import label_binarize
from sklearn import metrics

from flcore.servers.client_selection.Random import Random
from flcore.servers.client_selection.Thompson import Thompson
from flcore.servers.client_selection.UCB import UCB
from flcore.servers.client_selection.RCS import RandomClusterSelection
from flcore.servers.client_selection.DECS import DiversityEnhancedClusterSelection
from flcore.servers.client_selection.GAC import GAClientSelection
from flcore.servers.client_selection.RSVD import RSVDClientDetection
from flcore.servers.client_selection.RSVDUCB_old import RSVDUCBClientSelection
from flcore.servers.client_selection.RSVDUCBT import RSVDUCBThompson
from flcore.servers.client_selection.RSVDUCBT_forTest import RSVDUCBThompsonEnhanced

class FedAvgLSTM(Server):
    def __init__(self, args, times, agent=None):
        super().__init__(args, times)
        self.agent = agent
        self.set_slow_clients()
        self.set_clients(args, clientAVG)
        self.robustLR_threshold = 7
        self.server_lr = 1e-3
        self.client_gradients = {}
        self.gradients_available = False
        self.global_perplexity_history = []
        self.model = args.model
        print(f"Join ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def train(self):
        """執行聯邦平均訓練"""
        self.send_models()
        testloaderfull = self.get_test_data()
        # 根據選擇演算法初始化客戶端選擇代理
        if self.select_clients_algorithm == "Random":
            select_agent = Random(self.num_clients, self.num_join_clients, self.random_join_ratio)
        elif self.select_clients_algorithm == "RCS":
            select_agent = RandomClusterSelection(self.num_clients, self.num_join_clients, self.random_join_ratio)
        elif self.select_clients_algorithm == "DECS":
            select_agent = DiversityEnhancedClusterSelection(self.num_clients, self.num_join_clients, self.random_join_ratio)
        elif self.select_clients_algorithm == "UCB":
            select_agent = UCB(self.num_clients, self.num_join_clients)
        elif self.select_clients_algorithm == "GAC":
            select_agent = GAClientSelection(self.num_clients, self.num_join_clients)
        elif self.select_clients_algorithm == "RSVD":
            select_agent = RSVDClientDetection(self.num_clients, self.num_join_clients)
        elif self.select_clients_algorithm == "RSVDUBC":
            select_agent = RSVDUCBClientSelection(self.num_clients, self.num_join_clients)
        elif self.select_clients_algorithm == "RSVDUCBT":
            select_agent = RSVDUCBThompson(self.num_clients, self.num_join_clients)
        elif self.select_clients_algorithm == "RSVDUCBTE":
            select_agent = RSVDUCBThompsonEnhanced(self.num_clients, self.num_join_clients, self.global_accuracy_history)
        elif self.select_clients_algorithm == "Thompson":
            select_agent = Thompson(num_clients=self.num_clients, num_selections=self.num_join_clients)
        else:
            raise ValueError(f"未知的客戶端選擇演算法: {self.select_clients_algorithm}")

        mlflow.set_experiment(self.select_clients_algorithm)
        with mlflow.start_run(run_name=f"noniid_{self.num_clients*self.poisoned_ratio}_same"):
            mlflow.log_param("global_rounds", self.global_rounds)
            mlflow.log_param("dataset", self.dataset)
            mlflow.log_param("algorithm", self.algorithm)
            mlflow.log_param("num_clients", self.num_clients)

            counter_for_RSVD = 0
            for i in range(self.global_rounds + 1):
                s_t = time.time()
                selected_ids = select_agent.select_clients(i)
                if self.select_clients_algorithm in ["RSVD", "RSVDUCB", "RSVDUCBT", "RSVDUCBTE"]:
                    counter_for_RSVD += 1
                    if counter_for_RSVD == 1 and self.gradients_available:
                        select_agent = type(select_agent)(self.num_clients, self.num_join_clients, self.global_perplexity_history if self.select_clients_algorithm == "RSVDUCBTE" else None)

                print(f"Selected clients: {selected_ids}")
                self.selected_clients = [self.clients[c] for c in selected_ids]
                poisoned_selected = [idx for idx in selected_ids if self.clients[idx].poisoned]
                print(f"Poisoned clients: {poisoned_selected}")
                print(f"\n-------------Round number: {i}-------------")
                print(f"History perplexity: {self.perplexity_his}")

                for client in self.selected_clients:
                    client.train()
                    if self.select_clients_algorithm in ["RSVD", "RSVDUCB", "RSVDUCBT", "RSVDUCBTE"]:
                        self.client_gradients[client.id] = client.get_training_gradients()

                if not self.gradients_available:
                    self.gradients_available = True

                self.receive_models()
                if self.select_clients_algorithm in ["RSVD", "RSVDUCB", "RSVDUCBT", "RSVDUCBTE", "UCB", "GAC"]:
                    clients_perplexity = []
                    for client_model, client in zip(self.uploaded_models, self.selected_clients):
                        perplexity, test_num = self.test_metrics_all(client_model, testloaderfull)
                        clients_perplexity.append(perplexity)
                    reward_decay = 1
                    rewards = [-p for p in clients_perplexity]  # Lower perplexity is better
                    for reward, client in zip(rewards, self.selected_clients):
                        self.sums_of_reward[client.id] = self.sums_of_reward[client.id] * reward_decay + reward
                        self.numbers_of_selections[client.id] += 1
                    select_agent.update(selected_ids, rewards)

                same_weight = [1/self.num_join_clients] * self.num_join_clients
                self.aggregate_parameters(same_weight)
                self.send_models()

                if i % self.eval_gap == 0:
                    print("\nEvaluate global model")
                    perplexity, train_loss = self.evaluate() if self.select_clients_algorithm not in ["UCB", "GAC"] else self.evaluate_trust()
                    self.global_perplexity_history.append(perplexity)
                    mlflow.log_metric("global perplexity", perplexity, step=i)
                    mlflow.log_metric("train_loss", train_loss, step=i)

                if self.dlg_eval and i % self.dlg_gap == 0:
                    self.call_dlg(i)

                self.Budget.append(time.time() - s_t)
                print('-'*25, 'Time Cost', '-'*25, self.Budget[-1])

                if self.auto_break and self.check_done(acc_lss=[self.rs_test_perplexity], top_cnt=self.top_cnt):
                    break

        print("\nBest Perplexity:")
        print(min(self.rs_test_perplexity))
        print("\nAverage Time Cost Per Round:")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))
        self.save_results()
        self.save_global_model()

    def compute_robustLR(self, agent_updates):
        """計算 RobustLR 更新"""
        agent_updates_sign = [torch.sign(update) for update in agent_updates]
        sm_of_signs = torch.abs(sum(agent_updates_sign))
        sm_of_signs[sm_of_signs < self.robustLR_threshold] = -self.server_lr
        sm_of_signs[sm_of_signs >= self.robustLR_threshold] = self.server_lr
        return sm_of_signs.to(self.device)