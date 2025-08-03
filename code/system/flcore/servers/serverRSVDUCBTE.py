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
from utils.data_utils import read_client_data  # 假設這是您的數據工具模組
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import random

from flcore.servers.client_selection.Random import Random
from flcore.servers.client_selection.Thompson import Thompson
from flcore.servers.client_selection.UCB import UCB
from flcore.servers.client_selection.RCS import RandomClusterSelection
from flcore.servers.client_selection.DECS import DiversityEnhancedClusterSelection
from flcore.servers.client_selection.GAC import GAClientSelection
from flcore.servers.client_selection.RSVD import RSVDClientDetection
from flcore.servers.client_selection.RSVDUCB_old import RSVDUCBClientSelection
from flcore.servers.client_selection.RSVDUCBT import RSVDUCBThompson
from flcore.servers.client_selection.RSVDUCBTE import EnhancedRSVDUCBThompson
import threading

class FedRSVDUCBTE(Server):
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

    def train(self):
        """執行聯邦平均訓練過程"""
        self.send_models()  # 初始化模型分發
        testloaderfull = self.testdataloader  # 取得測試資料

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
            #select_agent = EnhancedRSVDUCBThompson(self.num_clients, self.num_join_clients, self.global_accuracy_history)
            #server_data_path = "/home/dslab/qixiang/FL_Test_Env_CUDA/dataset/Cifar100_100_alpha01_server_2/server_data.npz"
            #select_agent = EnhancedRSVDUCBThompson(self.num_clients, self.num_join_clients, server_data_path)
            select_agent = EnhancedRSVDUCBThompson(self.num_clients, self.num_join_clients)
            # random select one client
            #random_client = random.randint(0, self.num_clients - 1)
            #select_agent.random_load_data_from_client(random_client, self.clients[random_client].trainDataLoader_clean)
        elif self.select_clients_algorithm == "Thompson":
            select_agent = Thompson(num_clients=self.num_clients, num_selections=self.num_join_clients)
        else:
            raise ValueError(f"未知的客戶端選擇演算法: {self.select_clients_algorithm}")

        # 設置 MLflow 實驗
        mlflow.set_experiment(self.select_clients_algorithm)
        with mlflow.start_run(run_name=f"noniid_wbn_{self.num_clients*self.poisoned_ratio}_same"):
            mlflow.log_param("global_rounds", self.global_rounds)
            mlflow.log_param("dataset", self.dataset)
            mlflow.log_param("algorithm", self.algorithm)
            mlflow.log_param("num_clients", self.num_clients)

            counter_for_RSVD = 0  # RSVD 計數器

            for i in range(self.global_rounds + 1):
                s_t = time.time()  # 記錄開始時間
                
                # 客戶端選擇邏輯
                if self.select_clients_algorithm == "RSVDUCBTE":
                    if not self.gradients_available:
                        select_agent = Random(self.num_clients, self.num_join_clients, self.random_join_ratio)
                        selected_ids = select_agent.select_clients(i)
                        counter_for_RSVD += 1
                    else:
                        if counter_for_RSVD == 1:
                            #select_agent = EnhancedRSVDUCBThompson(self.num_clients, self.num_join_clients, self.global_accuracy_history)
                            #server_data_path = "/home/dslab/qixiang/FL_Test_Env_CUDA/dataset/Cifar100_100_alpha01_server_2/server_data.npz"
                            #select_agent = EnhancedRSVDUCBThompson(self.num_clients, self.num_join_clients, server_data_path)
                            select_agent = EnhancedRSVDUCBThompson(self.num_clients, self.num_join_clients)
                            #random_client = random.randint(0, self.num_clients - 1)
                            #select_agent.random_load_data_from_client(random_client, self.clients[random_client].trainDataLoader_clean)
                        #elif counter_for_RSVD > 1:
                            # random select one client
                            #random_client = random.randint(0, self.num_clients - 1)
                            #select_agent.random_load_data_from_client(random_client, self.clients[random_client].trainDataLoader_clean)
                        selected_ids = select_agent.select_clients(i, self.client_gradients)
                        counter_for_RSVD += 1
                else:
                    selected_ids = select_agent.select_clients(i)
                                
                print("Selected clients:", selected_ids)
                self.selected_clients = [self.clients[c] for c in selected_ids]

                poisoned_selected = [idx for idx in selected_ids if self.clients[idx].poisoned]
                print(f"Poisoned clients among FedRSVDUCBTE clients: {poisoned_selected}")

                print(f"\n-------------Round number: {i}-------------")

                print(f"history acc: {self.acc_his}")

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

                def client_train_with_stream(client, stream, current_index, collect_gradients, gradient_store):
                    """
                    每個 client 的訓練任務，配合 CUDA stream 並收集梯度（如果啟用）
                    """
                    with torch.cuda.stream(stream):
                        client.set_current_index(current_index)
                        client.train()
                        if collect_gradients:
                            gradients = client.get_training_gradients()
                            gradient_store[client.id] = gradients
                    stream.synchronize()  # 等待這個 stream 的 client 完成

                # --- 準備多個 CUDA stream 和 client gradient 緩存 ---
                streams = [torch.cuda.Stream() for _ in self.selected_clients]
                collect_gradients = self.select_clients_algorithm in ["RSVDUCBTE"]
                gradient_store = {}  # client.id -> gradients

                # --- 啟動多執行緒 client 訓練 ---
                threads = []
                for client, stream in zip(self.selected_clients, streams):
                    t = threading.Thread(
                        target=client_train_with_stream,
                        args=(client, stream, current_index, collect_gradients, gradient_store)
                    )
                    t.start()
                    threads.append(t)

                # --- 等待所有執行緒完成 ---
                for t in threads:
                    t.join()

                # --- 如果是 RSVDUCBTE 模式，記錄所有 client 的梯度 ---
                if collect_gradients:
                    self.client_gradients = gradient_store

                '''for client in self.selected_clients:
                    client.set_current_index(current_index)
                    #print(f"client: {client.id}, current_index: {current_index}")
                    client.train()
                    if self.select_clients_algorithm in ["RSVDUCBTE"]:
                        gradients = client.get_training_gradients()
                        self.client_gradients[client.id] = gradients'''

                if not self.gradients_available:
                    self.gradients_available = True

                self.receive_models()

                if self.select_clients_algorithm in ["RSVDUCBTE"] and self.gradients_available:
                    clients_acc = []
                    clients_f1 = []
                    clients_auc_pr = []
                    clients_avg_loss = []
                    clients_composite_reward = []
                    clients_global_test_acc = None

                    # === Step 1: Evaluate selected clients on global test set ===
                    for client_model, client in zip(self.uploaded_models, self.selected_clients):
                        test_acc, test_num, auc, f1, auc_pr, avg_loss = self.test_metrics_all(client_model, testloaderfull)
                        
                        clients_global_test_acc = test_acc

                        # Normalize and filter each metric to avoid negative/NaN values
                        acc_score = test_acc / test_num if test_num > 0 else 0
                        f1_score = f1 if f1 > 0 else 0
                        auc_pr_score = auc_pr if auc_pr > 0 else 0
                        
                        clients_acc.append(acc_score)
                        clients_f1.append(f1_score)
                        clients_auc_pr.append(auc_pr_score)
                        clients_avg_loss.append(avg_loss)

                    # === Step 2: Define composite reward ===
                    # You can tune the weights below depending on your dataset imbalance or priority
                    '''weight_acc = 0.4
                    weight_f1 = 0.3
                    weight_auc_pr = 0.3

                    for acc, f1, auc_pr in zip(clients_acc, clients_f1, clients_auc_pr):
                        # Composite reward combines all three metrics
                        composite_reward = weight_acc * acc + weight_f1 * f1 + weight_auc_pr * auc_pr
                        clients_composite_reward.append(composite_reward)'''

                    # === Step 3: Apply reward decay and update reward history ===
                    '''reward_decay = 0.9
                    for reward, client in zip(clients_composite_reward, self.selected_clients):
                        # Exponentially decay past rewards and add current
                        self.sums_of_reward[client.id] = self.sums_of_reward[client.id] * reward_decay + reward
                        self.numbers_of_selections[client.id] += 1'''

                    # === Step 4: Provide updated reward list to selection algorithm ===
                    if counter_for_RSVD > 1:
                        rewards = select_agent.compute_composite_rewards(clients_acc, clients_f1, clients_auc_pr, clients_avg_loss, selected_ids)
                        select_agent.update(selected_ids, rewards, clients_global_test_acc, self.global_model)
                    '''rewards = clients_composite_reward
                    select_agent.update(selected_ids, rewards)'''
                
                same_weight = None # initialize the variable

                # first round use equal weight
                if counter_for_RSVD <= 1:
                    same_weight = [1/self.num_join_clients] * self.num_join_clients
                
                # second round use adaptive weight
                if self.select_clients_algorithm in ["RSVDUCBTE"] and counter_for_RSVD > 1 :
                    print("Use Adaptive Weights")
                    same_weight = select_agent.contribution_weights
                    
                self.aggregate_parameters_bn(same_weight)

                self.send_models_bn()

                if i % self.eval_gap == 0:
                    print("\nEvaluate global model")
                    if self.select_clients_algorithm in ["RSVD", "RSVDUCB", "RSVDUCBT", "RSVDUCBTE"] and self.gradients_available:
                        acc, train_loss, auc, f1, auc_pr_score = self.evaluate()
                    elif self.select_clients_algorithm in ["UCB", "GAC"]:
                        acc, train_loss, auc, f1, auc_pr_score = self.evaluate_trust()
                    else:
                        acc, train_loss, auc, f1, auc_pr_score = self.evaluate()
                    
                    self.global_accuracy_history.append(acc)
                    #self.acc_data.append(acc)
                    #self.loss_data.append(train_loss)
                    #self.auc_data.append(auc)
                    mlflow.log_metric("global accuracy", acc, step=i)
                    mlflow.log_metric("train_loss", train_loss, step=i)

                if self.dlg_eval and i % self.dlg_gap == 0:
                    self.call_dlg(i)

                self.Budget.append(time.time() - s_t)
                print('-'*25, 'Time Cost', '-'*25, self.Budget[-1])

                self.time_cost_list.append(self.Budget[-1])

                if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                    break

        print("\nBest Accuracy:")
        print(max(self.rs_test_acc))
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