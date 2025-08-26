import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import mlflow
import torch
from sklearn.cluster import KMeans
import pandas as pd
import threading
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

from flcore.servers.client_selection.Random import Random
from flcore.servers.client_selection.Thompson import Thompson
from flcore.servers.client_selection.UCB import UCB
from flcore.servers.client_selection.RCS import RandomClusterSelection
from flcore.servers.client_selection.DECS import DiversityEnhancedClusterSelection

class FedKrum(Server):
    def __init__(self, args, times, agent=None):
        """初始化 FedAvg 伺服器物件"""
        super().__init__(args, times)

        self.agent = agent  # 選擇代理（若有）
        self.set_slow_clients()  # 設置緩慢客戶端
        self.set_clients(args, clientAVG)  # 設置客戶端，使用 clientAVG 類
        self.robustLR_threshold = 7  # RobustLR 閾值
        self.server_lr = 1e-3  # 伺服器學習率
        
        # 初始化客戶端梯度（用於 RSVD）
        self.client_gradients = {}  # 儲存每個客戶端的梯度
        self.gradients_available = False  # 標誌是否已有梯度可用
        self.global_accuracy_history = []  # 全局準確率歷史記錄

        self.model = args.model  # 全局模型（依資料集動態指定）
        
        # 判斷是否為多標籤資料集（僅 CelebA 為多標籤）
        self.is_multilabel = (args.dataset.lower() == 'celeba')

        self.dynamic = args.dynamic_training

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def get_vector_no_bn(self, model):
        """獲取模型參數向量，排除批次正規化層"""
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
    
    def krum(self, weights, n_attackers):
        num_clients = len(weights)
        dist_matrix = np.zeros((num_clients, num_clients))
        # 计算权重之间的距离
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                dist = self.euclidean_distance(weights[i], weights[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        # 计算每个参与者的距离和，并选择距离和最小的模型
        # min_sum_dist = float('inf')
        # selected_index = -1
        # for i in range(num_clients):
        #     sorted_indices = np.argsort(dist_matrix[i])
        #     sum_dist = np.sum(dist_matrix[i, sorted_indices[1:(num_clients - n_attackers)]])
        #     if sum_dist < min_sum_dist:
        #         min_sum_dist = sum_dist
        #         selected_index = i
        sorted_idx = np.sum(dist_matrix, axis=0).argsort()[:num_clients-n_attackers]

        chosen_index = int(sorted_idx[0])
    
        return chosen_index

    def train(self):
        self.send_models() #initialize model
        #testloaderfull = self.get_test_data()

        if self.select_clients_algorithm == "Random":
            select_agent = Random(self.num_clients, self.num_join_clients, self.random_join_ratio)
        elif self.select_clients_algorithm == "RCS":
            select_agent = RandomClusterSelection(self.num_clients, self.num_join_clients, self.random_join_ratio)
        elif self.select_clients_algorithm == "DECS":
            select_agent = DiversityEnhancedClusterSelection(self.num_clients, self.num_join_clients, self.random_join_ratio)
        elif self.select_clients_algorithm == "UCB":
            select_agent = UCB(self.num_clients, self.num_join_clients)

        # elif self.args.selected_clients_algorithm == "DQN":
        #     state = self.get_state()
        #     action = self.agent.select_action(state)
        #     self.selected_clients = [self.clients[c] for c in action]
        
        elif self.select_clients_algorithm == "Thompson":
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

                # Identify poisoned clients among Bulyan-selected clients
                poisoned_selected = [idx for idx in selected_ids if self.clients[idx].poisoned]
                print(f"Poisoned clients in selected clients: {poisoned_selected}")

                # 紀錄有毒的客戶端數量
                self.poisoned_clients_selected.append(len(poisoned_selected))
                # <= mh code
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

                def split_and_map(input_number, total_range, parts=4):
                    # Compute thresholds using split_number logic
                    part_size = total_range / parts
                    thresholds = [round(part_size * i) for i in range(1, parts + 1)]

                    # Map input_number to a value based on thresholds
                    for i, threshold in enumerate(thresholds):
                        if input_number <= threshold:
                            return i
                    return parts - 1  # Return the last index if number exceeds all thresholds

                if self.dynamic > 0:
                    current_index = split_and_map(i, self.global_rounds)
                    #print(f"current index: {current_index}")
                else:
                    current_index = 0

                streams = [torch.cuda.Stream(device=self.device) for _ in self.selected_clients]

                def client_train_with_stream(client, stream, current_index):
                    with torch.cuda.stream(stream):
                        client.set_current_index(current_index)
                        client.train()
                    stream.synchronize()  # 確保這個 client 訓練完成

                threads = []
                for client, stream in zip(self.selected_clients, streams):
                    t = threading.Thread(target=client_train_with_stream, args=(client, stream, current_index))
                    t.start()
                    threads.append(t)

                for t in threads:
                    t.join()

                # threads = [Thread(target=client.train)
                #            for client in self.selected_clients]
                # [t.start() for t in threads]
                # [t.join() for t in threads]


                self.receive_models()
                clients_weight = [parameters_to_vector(i.parameters()).cpu().detach().numpy() for i in self.uploaded_models]
                print(len(self.uploaded_models))
                krum_clients_index = self.krum(clients_weight, int(self.num_join_clients*self.poisoned_ratio))
                print(krum_clients_index)
                # self.uploaded_models = [self.uploaded_models[krum_clients_index]]

                if self.dlg_eval and i%self.dlg_gap == 0:
                    self.call_dlg(i)
                # self.aggregate_parameters([1])
                self.global_model = copy.deepcopy(self.uploaded_models[krum_clients_index])
                self.aggregate_parameters_bn([1])


                self.send_models_bn()
                #self.send_models()

                if i%self.eval_gap == 0:
                    # print(f"\n-------------Round number: {i}-------------")
                    # Unpack three values from evaluate() function
                    acc, train_loss, auc, f1, auc_pr_score = self.evaluate()  # Unpacked to handle 5 return values
                    #self.acc_data.append(acc)
                    #self.loss_data.append(train_loss)
                    #self.auc_data.append(auc)
                    mlflow.log_metric("global accuracy", acc, step=i)
                    mlflow.log_metric("train_loss", train_loss, step=i)
                    mlflow.log_metric("test_auc", auc, step=i)  # Log the AUC value too

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

                self.time_cost_list.append(self.Budget[-1])

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
