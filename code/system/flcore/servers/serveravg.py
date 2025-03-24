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

class FedAvg(Server):
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
        testloaderfull = self.get_test_data()  # 獲取測試資料

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
                if self.select_clients_algorithm == "RSVD":
                    if not self.gradients_available:
                        select_agent = Random(self.num_clients, self.num_join_clients, self.random_join_ratio)
                        selected_ids = select_agent.select_clients(i)
                        counter_for_RSVD += 1
                    else:
                        if counter_for_RSVD == 1:
                            select_agent = RSVDClientDetection(self.num_clients, self.num_join_clients)
                        selected_ids = select_agent.select_clients(i, self.client_gradients)
                        counter_for_RSVD += 1
                elif self.select_clients_algorithm == "RSVDUCB":
                    if not self.gradients_available:
                        select_agent = Random(self.num_clients, self.num_join_clients, self.random_join_ratio)
                        selected_ids = select_agent.select_clients(i)
                    else:
                        select_agent = RSVDUCBClientSelection(self.num_clients, self.num_join_clients)
                        selected_ids = select_agent.select_clients(i, self.client_gradients)
                elif self.select_clients_algorithm == "RSVDUCBT":
                    if not self.gradients_available:
                        select_agent = Random(self.num_clients, self.num_join_clients, self.random_join_ratio)
                        selected_ids = select_agent.select_clients(i)
                        counter_for_RSVD += 1
                    else:
                        if counter_for_RSVD == 1:
                            select_agent = RSVDUCBThompson(self.num_clients, self.num_join_clients)
                        selected_ids = select_agent.select_clients(i, self.client_gradients)
                        counter_for_RSVD += 1
                elif self.select_clients_algorithm == "RSVDUCBTE":
                    if not self.gradients_available:
                        select_agent = Random(self.num_clients, self.num_join_clients, self.random_join_ratio)
                        selected_ids = select_agent.select_clients(i)
                        counter_for_RSVD += 1
                    else:
                        if counter_for_RSVD == 1:
                            select_agent = RSVDUCBThompsonEnhanced(self.num_clients, self.num_join_clients, self.global_accuracy_history)
                        selected_ids = select_agent.select_clients(i, self.client_gradients)
                        counter_for_RSVD += 1
                else:
                    selected_ids = select_agent.select_clients(i)
                
                print("Selected clients:", selected_ids)
                self.selected_clients = [self.clients[c] for c in selected_ids]

                poisoned_selected = [idx for idx in selected_ids if self.clients[idx].poisoned]
                print(f"Poisoned clients among FedAvg clients: {poisoned_selected}")

                print(f"\n-------------Round number: {i}-------------")

                print(f"history acc: {self.acc_his}")

                for client in self.selected_clients:
                    client.train()
                    if self.select_clients_algorithm in ["RSVD", "RSVDUCB", "RSVDUCBT", "RSVDUCBTE"]:
                        gradients = client.get_training_gradients()
                        self.client_gradients[client.id] = gradients

                if not self.gradients_available:
                    self.gradients_available = True

                self.receive_models()

                if self.select_clients_algorithm in ["RSVD", "RSVDUCB", "RSVDUCBT", "RSVDUCBTE"] and self.gradients_available:
                    clients_acc = []
                    for client_model, client in zip(self.uploaded_models, self.selected_clients):
                        test_acc, test_num, auc = self.test_metrics_all(client_model, testloaderfull)
                        clients_acc.append(test_acc / test_num if test_num > 0 else 0)

                    reward_decay = 1
                    for reward, client in zip(clients_acc, self.selected_clients):
                        self.sums_of_reward[client.id] = self.sums_of_reward[client.id] * reward_decay + reward
                        self.numbers_of_selections[client.id] += 1
                    
                    rewards = clients_acc
                    select_agent.update(selected_ids, rewards)
                
                elif self.select_clients_algorithm in ["UCB", "GAC"]:
                    clients_acc = []
                    for client_model, client in zip(self.uploaded_models, self.selected_clients):
                        test_acc, test_num, auc = self.test_metrics_all(client_model, testloaderfull)
                        clients_acc.append(test_acc / test_num if test_num > 0 else 0)

                    reward_decay = 1
                    for reward, client in zip(clients_acc, self.selected_clients):
                        self.sums_of_reward[client.id] = self.sums_of_reward[client.id] * reward_decay + reward
                        self.numbers_of_selections[client.id] += 1
                    
                    rewards = clients_acc
                    select_agent.update(selected_ids, rewards)

                same_weight = [1/self.num_join_clients] * self.num_join_clients
                self.aggregate_parameters_bn(same_weight)

                self.send_models_bn()

                if i % self.eval_gap == 0:
                    print("\nEvaluate global model")
                    if self.select_clients_algorithm in ["RSVD", "RSVDUCB", "RSVDUCBT", "RSVDUCBTE"] and self.gradients_available:
                        acc, train_loss, auc = self.evaluate_trust()
                    elif self.select_clients_algorithm in ["UCB", "GAC"]:
                        acc, train_loss, auc = self.evaluate_trust()
                    else:
                        acc, train_loss, auc = self.evaluate()
                    
                    self.global_accuracy_history.append(acc)
                    self.acc_data.append(acc)
                    self.loss_data.append(train_loss)
                    self.auc_data.append(auc)
                    mlflow.log_metric("global accuracy", acc, step=i)
                    mlflow.log_metric("train_loss", train_loss, step=i)

                if self.dlg_eval and i % self.dlg_gap == 0:
                    self.call_dlg(i)

                self.Budget.append(time.time() - s_t)
                print('-'*25, 'Time Cost', '-'*25, self.Budget[-1])

                if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                    break

        print("\nBest Accuracy:")
        print(max(self.rs_test_acc))
        print("\nAverage Time Cost Per Round:")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))
        
        self.save_results()
        self.save_global_model()

    def test_metrics_all(self, client_model, testloaderfull):
        """測試模型效能，動態適配單標籤與多標籤資料集"""
        client_model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                x = x.to(self.device)
                if self.is_multilabel:
                    y = y.to(self.device).float()  # 多標籤資料集（如 CelebA）使用浮點數
                else:
                    y = y.to(self.device).long()  # 單標籤資料集使用整數

                output = client_model(x)

                if self.is_multilabel:
                    pred = (torch.sigmoid(output) > 0.5).float()
                    test_acc += torch.sum(pred == y).item() / (y.shape[0] * self.num_classes)
                else:
                    test_acc += torch.sum(torch.argmax(output, dim=1) == y).item()

                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                if self.is_multilabel:
                    y_true.append(y.detach().cpu().numpy())
                else:
                    nc = self.num_classes
                    if self.num_classes == 2:
                        nc += 1
                    lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                    if self.num_classes == 2:
                        lb = lb[:, :2]
                    y_true.append(lb)

        if test_num == 0:
            return 0, 0, 0.0

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro', multi_class='ovr') if y_true.size > 0 else 0.0
        
        return test_acc, test_num, auc

    def compute_robustLR(self, agent_updates):
        """計算 RobustLR 更新"""
        agent_updates_sign = [torch.sign(update) for update in agent_updates]  
        sm_of_signs = torch.abs(sum(agent_updates_sign))

        sm_of_signs[sm_of_signs < self.robustLR_threshold] = -self.server_lr
        sm_of_signs[sm_of_signs >= self.robustLR_threshold] = self.server_lr   
        return sm_of_signs.to(self.device)