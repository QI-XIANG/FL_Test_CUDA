import torch
import os
import numpy as np
import pandas as pd
import h5py
import copy
import time
import random
import matplotlib.pyplot as plt
from utils.data_utils import read_client_data
from utils.dlg import DLG
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import math
import heapq
from sklearn.cluster import KMeans

# 定義伺服器類別，用於聯邦學習中的全局協調
class Server(object):
    def __init__(self, args, times):
        """初始化伺服器物件，設定主要屬性"""
        self.device = args.device  # 設備（CPU 或 GPU）
        self.dataset = args.dataset  # 資料集名稱（例如 'CelebA', 'MNIST', 'GTSRB', 'CIFAR10'）
        self.num_classes = args.num_classes  # 類別數（CelebA: 40, MNIST: 10, GTSRB: 43, CIFAR10: 10）
        self.global_rounds = args.global_rounds  # 全局訓練輪數
        self.local_epochs = args.local_epochs  # 客戶端本地訓練輪數
        self.batch_size = args.batch_size  # 批次大小
        self.learning_rate = args.local_learning_rate  # 學習率
        self.global_model = copy.deepcopy(args.model)  # 全局模型（依資料集動態指定）
        self.num_clients = args.num_clients  # 客戶端數量
        self.join_ratio = args.join_ratio  # 參與比例
        self.random_join_ratio = args.random_join_ratio  # 是否隨機參與
        self.num_join_clients = int(self.num_clients * self.join_ratio)  # 參與客戶端數
        self.algorithm = args.algorithm  # 聯邦學習演算法
        self.time_select = args.time_select  # 時間選擇參數
        self.goal = args.goal  # 訓練目標
        self.time_threthold = args.time_threthold  # 時間閾值
        self.save_folder_name = args.save_folder_name  # 儲存資料夾名稱
        self.top_cnt = 20  # 最高計數（用於收斂檢查）
        self.auto_break = args.auto_break  # 是否自動終止

        self.clients = []  # 客戶端列表
        self.selected_clients = []  # 選中的客戶端
        self.train_slow_clients = []  # 訓練緩慢的客戶端
        self.send_slow_clients = []  # 傳送緩慢的客戶端

        self.uploaded_weights = []  # 上傳的權重
        self.uploaded_ids = []  # 上傳的客戶端 ID
        self.uploaded_models = []  # 上傳的模型

        self.rs_test_acc = []  # 測試準確率記錄
        self.rs_test_auc = []  # 測試 AUC 記錄
        self.rs_train_loss = []  # 訓練損失記錄

        self.times = times  # 執行次數
        self.eval_gap = args.eval_gap  # 評估間隔
        self.client_drop_rate = args.client_drop_rate  # 客戶端退出率
        self.train_slow_rate = args.train_slow_rate  # 訓練緩慢率
        self.send_slow_rate = args.send_slow_rate  # 傳送緩慢率

        self.dlg_eval = args.dlg_eval  # 是否啟用 DLG 攻擊評估
        self.dlg_gap = args.dlg_gap  # DLG 評估間隔
        self.batch_num_per_client = args.batch_num_per_client  # 每個客戶端的批次數

        # MH 程式碼：中毒相關參數
        self.poisoned_ratio = args.poisoned_ratio  # 中毒客戶端比例
        self.random_seed = args.random_seed  # 隨機種子
        self.poisoned_clients = self.select_poisoned_client()  # 選擇中毒客戶端
        print(f"poisoned_clients: {self.poisoned_clients}")

        self.interact = [[] for i in range(self.num_clients)]  # 客戶端互動記錄
        self.acc_his = []  # 準確率歷史
        self.clients_acc_his = [[] for i in range(self.num_clients)]  # 客戶端準確率歷史

        self.numbers_of_selections = [0] * self.num_clients  # 客戶端選擇次數
        self.sums_of_reward = [0] * self.num_clients  # 客戶端獎勵總和
        self.clients_loss = [0] * self.num_clients  # 客戶端損失

        self.acc_data = []  # 準確率數據
        self.loss_data = []  # 損失數據
        self.auc_data = []  # AUC 數據
        self.select_clients_algorithm = args.select_clients_algorithm  # 客戶端選擇演算法
        self.server = args.algorithm  # 伺服器演算法
        self.Budget = []  # 預算記錄
        self.weight_option = args.weight_option  # 權重選項

        # 判斷是否為多標籤資料集
        self.is_multilabel = (args.dataset.lower() == 'celeba')

    def select_poisoned_client(self):
        """選擇中毒客戶端，動態適配不同資料集"""
        np.random.seed(self.random_seed)
        label_one_clients = []
        
        for i in range(self.num_clients):
            temp = read_client_data(self.dataset, i, is_train=False)  # 讀取測試資料
            if len(temp) == 0:  # 確保資料不為空
                continue
            for image in temp:
                if self.dataset.lower() == 'celeba':
                    if 1 in image[1]:  # CelebA 多標籤，檢查是否有屬性為 1
                        label_one_clients.append(i)
                        break
                else:
                    if image[1] == 1:  # 單標籤資料集（MNIST, GTSRB, CIFAR10），檢查標籤是否為 1
                        label_one_clients.append(i)
                        break
        
        # 確保中毒客戶端數不超過可用數量
        num_poisoned_clients = min(int(self.num_clients * self.poisoned_ratio), len(label_one_clients))
        if num_poisoned_clients == 0:  # 若無符合條件的客戶端，隨機選擇
            num_poisoned_clients = min(int(self.num_clients * self.poisoned_ratio), self.num_clients)
            poisoned_clients = list(np.random.choice(range(self.num_clients), num_poisoned_clients, replace=False))
        else:
            poisoned_clients = list(np.random.choice(label_one_clients, num_poisoned_clients, replace=False))
        return poisoned_clients

    def get_test_data(self):
        """獲取測試資料，適用於所有資料集"""
        batch_size = self.batch_size
        test_data = read_client_data(self.dataset, 0, is_train=False)  # 從客戶端 0 開始

        for i in range(1, self.num_clients):
            test_data += read_client_data(self.dataset, i, is_train=False)  # 合併所有客戶端測試資料
        
        if len(test_data) == 0:  # 若無測試資料，返回空 DataLoader
            return DataLoader([], batch_size, drop_last=False, shuffle=False)
        
        random.shuffle(test_data)  # 隨機打亂資料
        sampling_data = test_data[:min(3000, len(test_data))]  # 取樣最多 3000 筆資料

        return DataLoader(sampling_data, batch_size, drop_last=False, shuffle=False)

    def select_clients_by_trust(self):
        """根據信任度選擇客戶端"""
        if self.random_join_ratio:
            num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            num_join_clients = self.num_join_clients

        clients_trust = [0.5] * self.num_clients  # 初始信任度
        for i, (record, cah) in enumerate(zip(self.interact, self.clients_acc_his)):
            if len(record) == 0: continue
            clients_trust[i] = sum(record)/len(record)  # 計算平均信任度

        total_trust = sum(clients_trust)
        if total_trust == 0:  # 避免除以 0
            clients_trust = [1.0 / self.num_clients] * self.num_clients
        else:
            clients_trust = [x / total_trust for x in clients_trust]  # 正規化

        selected_clients_id = np.random.choice(np.arange(self.num_clients), size=num_join_clients, replace=False, p=clients_trust)
        selected_clients = [self.clients[id] for id in selected_clients_id]
        print(f"客戶端信任度: {clients_trust}")
        print(f"選中的客戶端 ID: {selected_clients_id}")
        return selected_clients

    def test_metrics_all(self, client_model, testloaderfull):
        """測試模型效能，返回總正確屬性數和有效屬性數"""
        client_model.eval()

        test_acc = 0  # 總正確屬性數
        test_num = 0
        y_prob = []
        y_true = []
        
        if self.is_multilabel:
            correct_per_label = torch.zeros(self.num_classes).to(self.device)
            total_per_label = torch.zeros(self.num_classes).to(self.device)
            valid_attributes = torch.zeros(self.num_classes).to(self.device)

        with torch.no_grad():
            for x, y in testloaderfull:
                x = x.to(self.device)
                if self.is_multilabel:
                    y = y.to(self.device).float()
                    pred = (torch.sigmoid(output) > 0.5).float()
                    test_acc += torch.sum(pred == y).item()
                    correct_per_label += torch.sum(pred == y, dim=0)
                    total_per_label += y.shape[0]
                    valid_attributes += (torch.sum(y, dim=0) > 0).float() * (torch.sum(1 - y, dim=0) > 0).float()
                else:
                    y = y.to(self.device).long()
                    output = client_model(x)
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
            if self.is_multilabel:
                return 0, 0, 0.0, torch.zeros(self.num_classes)
            return 0, 0, 0.0

        if self.is_multilabel:
            valid_mask = (valid_attributes > 0).float()
            effective_num_attributes = torch.sum(valid_mask).item() or 1
            test_acc = test_acc / (test_num * effective_num_attributes) if test_num > 0 else 0
            label_acc = (correct_per_label / total_per_label).cpu().numpy() * valid_mask.cpu().numpy()

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        auc = metrics.roc_auc_score(y_true, y_prob, average='micro', multi_class='ovr') if y_true.size > 0 else 0.0
        
        if self.is_multilabel:
            return test_acc, test_num, auc, label_acc
        return test_acc / test_num if test_num > 0 else 0, test_num, auc

    def params_to_vector(self, model):
        """將模型參數轉為向量"""
        params = []
        for param in model.parameters():
            params.append(param.view(-1))
        return torch.cat(params)

    def set_clients(self, args, clientObj):
        """設置客戶端，適配所有資料集"""
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            poisoned = 1 if i in self.poisoned_clients else 0  # 是否為中毒客戶端
            train_data = read_client_data(self.dataset, i, is_train=True)  # 讀取訓練資料
            test_data = read_client_data(self.dataset, i, is_train=False)  # 讀取測試資料
            client = clientObj(args, 
                               id=i, 
                               train_samples=len(train_data), 
                               test_samples=len(test_data), 
                               train_slow=train_slow, 
                               send_slow=send_slow,
                               poisoned=poisoned)
            self.clients.append(client)

    def select_slow_clients(self, slow_rate):
        """隨機選擇緩慢客戶端"""
        slow_clients = [False for i in range(self.num_clients)]
        idx = list(range(self.num_clients))
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients), replace=False)
        for i in idx_:
            slow_clients[i] = True
        return slow_clients

    def set_slow_clients(self):
        """設置緩慢客戶端"""
        self.train_slow_clients = self.select_slow_clients(self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(self.send_slow_rate)

    def select_clients(self):
        """隨機選擇客戶端"""
        if self.random_join_ratio:
            num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, num_join_clients, replace=False))
        return selected_clients

    def select_clients_UCB(self, epoch):
        """使用 UCB 演算法選擇客戶端"""
        clients_upper_bound = []
        for i in range(self.num_clients):
            if self.numbers_of_selections[i] > 0:
                average_reward = self.sums_of_reward[i] / self.numbers_of_selections[i]
                delta_i = math.sqrt(2 * math.log(epoch+1) / self.numbers_of_selections[i])
                upper_bound = average_reward + delta_i
            else:
                upper_bound = 1e400
            clients_upper_bound.append(upper_bound)

        t = copy.deepcopy(clients_upper_bound)
        max_number = []
        max_index = []
        for _ in range(self.num_join_clients):
            number = max(t)
            index = t.index(number)
            t[index] = 0
            max_number.append(number)
            max_index.append(index)
        t = []

        selected_clients_id = max_index
        selected_clients = []
        for id in selected_clients_id:
            self.numbers_of_selections[id] += 1
            selected_clients.append(self.clients[id])
        return selected_clients

    def send_models_bn(self):
        """傳送模型參數（含批次正規化層）"""
        assert (len(self.clients) > 0)
        for client in self.clients:
            start_time = time.time()
            client.set_parameters_bn(self.global_model)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def send_models(self):
        """傳送模型參數"""
        assert (len(self.clients) > 0)
        for client in self.clients:
            start_time = time.time()
            client.set_parameters(self.global_model)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        """接收客戶端模型"""
        assert (len(self.selected_clients) > 0)
        active_clients = self.selected_clients
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                                   client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        if tot_samples > 0:  # 避免除以 0
            for i, w in enumerate(self.uploaded_weights):
                self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters_bn_lr(self, clients_weight):
        """聚合參數（含批次正規化層和學習率調整）"""
        bn_key = ['conv1.1.weight', 'conv1.1.bias', 'conv1.1.running_mean', 'conv1.1.running_var', 'conv1.1.num_batches_tracked',
                  'conv2.1.weight', 'conv2.1.bias', 'conv2.1.running_mean', 'conv2.1.running_var', 'conv2.1.num_batches_tracked']
        for key in self.global_model.state_dict().keys():
            if key not in bn_key:
                temp = torch.zeros_like(self.global_model.state_dict()[key], dtype=torch.float32)
                for weight, model in zip(clients_weight, self.uploaded_models):
                    temp += weight * model.state_dict()[key]
                self.global_model.state_dict()[key].data.copy_(temp)

    def aggregate_parameters_bn(self, clients_weight):
        """聚合參數（含批次正規化層）"""
        bn_key = ['conv1.1.weight', 'conv1.1.bias', 'conv1.1.running_mean', 'conv1.1.running_var', 'conv1.1.num_batches_tracked',
                  'conv2.1.weight', 'conv2.1.bias', 'conv2.1.running_mean', 'conv2.1.running_var', 'conv2.1.num_batches_tracked']
        for key in self.global_model.state_dict().keys():
            if key not in bn_key:
                temp = torch.zeros_like(self.global_model.state_dict()[key], dtype=torch.float32)
                for weight, model in zip(clients_weight, self.uploaded_models):
                    if key in model.state_dict():
                        temp += weight * model.state_dict()[key]
                self.global_model.state_dict()[key].data.copy_(temp)

    def aggregate_parameters(self, clients_weight):
        """聚合參數"""
        assert (len(self.uploaded_models) > 0)
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
        for w, client_model in zip(clients_weight, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        """添加客戶端參數至全局模型"""
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        """儲存全局模型"""
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        """載入全局模型"""
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        """檢查模型是否存在"""
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)

    def save_results(self):
        """儲存訓練結果"""
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if len(self.rs_test_acc):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = os.path.join(result_path, "{}.h5".format(algo))
            print("File path: " + file_path)
            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

        acc_df = pd.DataFrame(self.acc_data)
        loss_df = pd.DataFrame(self.loss_data)
        auc_df = pd.DataFrame(self.auc_data)

        name = f"{self.algorithm}_{self.select_clients_algorithm}_{self.poisoned_ratio*self.num_clients}_{self.random_seed}"
        acc_df.columns = [name]
        loss_df.columns = [name]
        auc_df.columns = [name]

        auc_dir = f"../results/{self.num_clients}/auc"
        os.makedirs(auc_dir, exist_ok=True)
        auc_df.to_csv(os.path.join(auc_dir, f"{name}.csv"), index=False)

        acc_dir = f"../results/{self.num_clients}/accuracy"
        loss_dir = f"../results/{self.num_clients}/loss"
        os.makedirs(acc_dir, exist_ok=True)
        os.makedirs(loss_dir, exist_ok=True)
        acc_df.to_csv(os.path.join(acc_dir, f"{name}.csv"), index=False)
        loss_df.to_csv(os.path.join(loss_dir, f"{name}.csv"), index=False)

        # 繪製並儲存 AUC 圖表
        plt.figure()
        plt.plot(auc_df, label='AUC', color='blue')
        plt.title('AUC 隨時間變化')
        plt.xlabel('輪次')
        plt.ylabel('AUC')
        plt.legend()
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(auc_dir, f"{name}_auc.png"))
        plt.close()

        # 繪製並儲存準確率圖表
        plt.figure()
        plt.plot(acc_df, label='準確率', color='green')
        plt.title('準確率隨時間變化')
        plt.xlabel('輪次')
        plt.ylabel('準確率')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.savefig(os.path.join(acc_dir, f"{name}_accuracy.png"))
        plt.close()

        # 繪製並儲存損失圖表
        plt.figure()
        plt.plot(loss_df, label='損失', color='red')
        plt.title('損失隨時間變化')
        plt.xlabel('輪次')
        plt.ylabel('損失')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.savefig(os.path.join(loss_dir, f"{name}_loss.png"))
        plt.close()

    def save_item(self, item, item_name):
        """儲存特定項目"""
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        """載入特定項目"""
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_selected_clients_metrics(self):
        """測試選中客戶端的效能"""
        clients_accuracy = []
        for c in self.selected_clients:
            ct, ns, auc = c.test_metrics_all()
            clients_accuracy.append(ct/ns if ns > 0 else 0)

    def test_metrics(self):
        """測試所有客戶端的效能，返回適配多標籤的統計數據"""
        num_samples = []
        tot_correct = []
        tot_auc = []
        label_acc_per_client = [] if self.is_multilabel else None

        for c in self.clients:
            if self.is_multilabel:
                ct, ns, auc, label_acc = c.test_metrics()  # 多標籤返回 4 個值
                label_acc_per_client.append(label_acc)
            else:
                ct, ns, auc = c.test_metrics()  # 單標籤返回 3 個值
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]
        if self.is_multilabel:
            return ids, num_samples, tot_correct, tot_auc, label_acc_per_client
        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        """計算訓練效能"""
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)
        ids = [c.id for c in self.clients]
        return ids, num_samples, losses

    def test_metrics_trust(self, min_trust_index):
        """測試信任客戶端的效能"""
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            if c.id in min_trust_index:
                continue
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        return ids, num_samples, tot_correct, tot_auc

    def train_metrics_trust(self, min_trust_index):
        """計算信任客戶端的訓練效能"""
        num_samples = []
        losses = []
        for c in self.clients:
            if c.id in min_trust_index:
                continue
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)
        ids = [c.id for c in self.clients]
        return ids, num_samples, losses

    def evaluate(self, acc=None, loss=None):
        """評估全局模型，按客戶端加權平均計算準確率"""
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        total_samples = sum(stats[1])
        if total_samples == 0:
            test_acc = 0.0
            test_auc = 0.0
            if self.is_multilabel:
                label_acc = np.zeros(self.num_classes)
        else:
            if self.is_multilabel:
                client_acc = [ct for ct in stats[2]]  # 每個客戶端的有效屬性準確率
                weights = [ns / total_samples for ns in stats[1]]
                test_acc = sum(a * w for a, w in zip(client_acc, weights))
                test_auc = sum(stats[3]) * 1.0 / total_samples
                label_acc_per_client = [stats[4][i] for i in range(len(stats[1]))]
                label_acc = np.average(label_acc_per_client, weights=weights, axis=0)
            else:
                test_acc = sum(stats[2]) * 1.0 / total_samples
                test_auc = sum(stats[3]) * 1.0 / total_samples

        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1]) if sum(stats_train[1]) > 0 else 0.0
        accs = [a / n if n > 0 else 0 for a, n in zip(stats[2], stats[1])]
        aucs = [a / n if n > 0 else 0 for a, n in zip(stats[3], stats[1])]

        if acc is None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss is None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

        if self.is_multilabel:
            print("\nEach Attribute Prediction Accurancy（CelebA）：")
            attribute_names = [
                "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
                "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
                "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
                "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache",
                "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
                "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
                "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"
            ]
            for i, (attr_name, attr_acc) in enumerate(zip(attribute_names, label_acc)):
                print(f"Attribute {i+1}: {attr_name:<20} - Prediction Accuracy: {attr_acc:.4f}")

        if len(self.acc_his) >= 3:
            self.acc_his.pop(0)
        self.acc_his.append(test_acc)

        return test_acc, train_loss, test_auc

    def get_n_min(self, number, target):
        """獲取最小的 n 個值及其索引"""
        t = copy.deepcopy(target)
        min_number = []
        min_index = []
        for _ in range(min(number, len(t))):
            number = min(t)
            index = t.index(number)
            t[index] = float('inf')
            min_number.append(number)
            min_index.append(index)
        print(min_number)
        print(min_index)
        return min_index

    def get_not_evaluate_index(self):
        """獲取不參與評估的客戶端索引"""
        threshold = np.percentile(self.numbers_of_selections, 40)
        pass_ = [1 if i >= threshold else 0 for i in self.numbers_of_selections]
        return list(np.where(np.array(pass_) == 0)[0])

    def evaluate_trust(self, acc=None, loss=None):
        """評估信任客戶端的效能"""
        not_join = self.get_not_evaluate_index()
        print("不參與評估的客戶端: ", not_join)
        stats = self.test_metrics_trust(not_join)
        stats_train = self.train_metrics_trust(not_join)

        total_samples = sum(stats[1])
        if total_samples == 0:  # 避免除以 0
            test_acc = 0.0
            test_auc = 0.0
        else:
            test_acc = sum(stats[2])*1.0 / total_samples
            test_auc = sum(stats[3])*1.0 / total_samples

        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1]) if sum(stats_train[1]) > 0 else 0.0
        accs = [a / n if n > 0 else 0 for a, n in zip(stats[2], stats[1])]
        aucs = [a / n if n > 0 else 0 for a, n in zip(stats[3], stats[1])]

        if acc is None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss is None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("信任評估")
        print("平均訓練損失: {:.4f}".format(train_loss))
        print("平均測試準確率: {:.4f}".format(test_acc))
        print("平均測試 AUC: {:.4f}".format(test_auc))
        print("標準差測試準確率: {:.4f}".format(np.std(accs)))
        print("標準差測試 AUC: {:.4f}".format(np.std(aucs)))

        if len(self.acc_his) >= 3:
            self.acc_his.pop(0)
        self.acc_his.append(test_acc)

        return test_acc, train_loss, test_auc

    def print_(self, test_acc, test_auc, train_loss):
        """列印效能指標"""
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("Averaged Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        """檢查是否完成訓練"""
        for acc_ls in acc_lss:
            if top_cnt is not None and div_value is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if not (find_top and find_div):
                    return False
            elif top_cnt is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if not find_top:
                    return False
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if not find_div:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        """執行 DLG 攻擊評估"""
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break
                    x = x.to(self.device)
                    if self.is_multilabel:
                        y = y.to(self.device).float()  # 多標籤資料集使用浮點數
                    else:
                        y = y.to(self.device).long()  # 單標籤資料集使用整數
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1

        if cnt > 0:
            print('PSNR 值為 {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR 錯誤')