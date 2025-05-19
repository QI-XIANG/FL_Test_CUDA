import torch
import os
import numpy as np
import pandas as pd
import h5py
import copy
import time
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader, Dataset
import math
from flcore.clients.clientavg import clientAVG
from threading import Thread
import mlflow

# 定義 Shakespeare 資料集類，用於載入 .npz 文件
class ShakespeareDataset(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path)
        self.x = data['x']  # [num_sequences, seq_length]
        self.y = data['y']  # [num_sequences, seq_length]
        self.num_samples = len(self.x)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)

# 伺服器基類，支援 Shakespeare 語言建模
class Server(object):
    def __init__(self, args, times):
        self.device = args.device
        self.dataset = args.dataset.lower()
        self.vocab_size = args.vocab_size if hasattr(args, 'vocab_size') else 8000
        self.seq_length = args.seq_length if hasattr(args, 'seq_length') else 80
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 20
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_perplexity = []  # 測試困惑度
        self.rs_train_loss = []  # 訓練損失
        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.poisoned_ratio = args.poisoned_ratio
        self.random_seed = args.random_seed
        self.poisoned_clients = self.select_poisoned_client()
        print(f"Poisoned clients: {self.poisoned_clients}")

        self.interact = [[] for _ in range(self.num_clients)]
        self.perplexity_his = []
        self.clients_perplexity_his = [[] for _ in range(self.num_clients)]
        self.numbers_of_selections = [0] * self.num_clients
        self.sums_of_reward = [0] * self.num_clients
        self.clients_loss = [0] * self.num_clients

        self.perplexity_data = []
        self.loss_data = []
        self.select_clients_algorithm = args.select_clients_algorithm
        self.server = args.algorithm
        self.Budget = []
        self.weight_option = args.weight_option
        self.data_dir = f"Shakespeare_{self.num_clients}"

    def select_poisoned_client(self):
        """隨機選擇中毒客戶端（語言建模無明確標籤）"""
        np.random.seed(self.random_seed)
        num_poisoned_clients = int(self.num_clients * self.poisoned_ratio)
        return list(np.random.choice(range(self.num_clients), num_poisoned_clients, replace=False))

    def get_test_data(self):
        """獲取所有客戶端的測試資料"""
        batch_size = self.batch_size
        test_data = []
        for i in range(self.num_clients):
            data_path = os.path.join(self.data_dir, 'test', f'{i}.npz')
            if os.path.exists(data_path):
                dataset = ShakespeareDataset(data_path)
                test_data.extend([(x, y) for x, y in dataset])
        if not test_data:
            return DataLoader([], batch_size, drop_last=False, shuffle=False)
        random.shuffle(test_data)
        sampling_data = test_data[:min(3000, len(test_data))]
        return DataLoader(sampling_data, batch_size, drop_last=False, shuffle=False)

    def select_clients_by_trust(self):
        """根據信任度選擇客戶端（使用困惑度）"""
        if self.random_join_ratio:
            num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            num_join_clients = self.num_join_clients
        clients_trust = [0.5] * self.num_clients
        for i, (record, cph) in enumerate(zip(self.interact, self.clients_perplexity_his)):
            if len(record) == 0:
                continue
            clients_trust[i] = sum(record) / len(record)
        total_trust = sum(clients_trust)
        if total_trust == 0:
            clients_trust = [1.0 / self.num_clients] * self.num_clients
        else:
            clients_trust = [x / total_trust for x in clients_trust]
        selected_clients_id = np.random.choice(np.arange(self.num_clients), size=num_join_clients, replace=False, p=clients_trust)
        selected_clients = [self.clients[id] for id in selected_clients_id]
        print(f"Client trust: {clients_trust}")
        print(f"Selected client IDs: {selected_clients_id}")
        return selected_clients

    def test_metrics_all(self, client_model, testloaderfull):
        """測試模型困惑度"""
        client_model.eval()
        total_loss = 0.0
        test_num = 0
        with torch.no_grad():
            for x, y in testloaderfull:
                x, y = x.to(self.device), y.to(self.device)
                output = client_model(x)  # [batch_size, seq_length, vocab_size]
                loss = nn.CrossEntropyLoss()(output.view(-1, self.vocab_size), y.view(-1))
                total_loss += loss.item() * x.size(0) * self.seq_length
                test_num += x.size(0) * self.seq_length
        if test_num == 0:
            return float('inf'), 0
        avg_loss = total_loss / test_num
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return perplexity, test_num

    def params_to_vector(self, model):
        """將模型參數轉為向量"""
        params = []
        for param in model.parameters():
            params.append(param.view(-1))
        return torch.cat(params)

    def set_clients(self, args, clientObj):
        """設置客戶端"""
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            poisoned = i in self.poisoned_clients
            data_path_train = os.path.join(self.data_dir, 'train', f'{i}.npz')
            data_path_test = os.path.join(self.data_dir, 'test', f'{i}.npz')
            train_samples = len(ShakespeareDataset(data_path_train)) if os.path.exists(data_path_train) else 0
            test_samples = len(ShakespeareDataset(data_path_test)) if os.path.exists(data_path_test) else 0
            client = clientObj(args, id=i, train_samples=train_samples, test_samples=test_samples,
                               train_slow=train_slow, send_slow=send_slow, poisoned=poisoned)
            self.clients.append(client)

    def select_slow_clients(self, slow_rate):
        """隨機選擇緩慢客戶端"""
        slow_clients = [False] * self.num_clients
        idx = np.random.choice(range(self.num_clients), int(slow_rate * self.num_clients), replace=False)
        for i in idx:
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
        return list(np.random.choice(self.clients, num_join_clients, replace=False))

    def select_clients_UCB(self, epoch):
        """使用 UCB 演算法選擇客戶端（基於困惑度）"""
        clients_upper_bound = []
        for i in range(self.num_clients):
            if self.numbers_of_selections[i] > 0:
                average_reward = self.sums_of_reward[i] / self.numbers_of_selections[i]
                delta_i = math.sqrt(2 * math.log(epoch+1) / self.numbers_of_selections[i])
                upper_bound = average_reward + delta_i
            else:
                upper_bound = 1e400
            clients_upper_bound.append(upper_bound)
        selected_clients_id = heapq.nlargest(self.num_join_clients, range(len(clients_upper_bound)), key=clients_upper_bound.__getitem__)
        for id in selected_clients_id:
            self.numbers_of_selections[id] += 1
        return [self.clients[id] for id in selected_clients_id]

    def send_models(self):
        """傳送模型參數"""
        for client in self.clients:
            start_time = time.time()
            client.set_parameters(self.global_model)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        """接收客戶端模型"""
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in self.selected_clients:
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
        if tot_samples > 0:
            self.uploaded_weights = [w / tot_samples for w in self.uploaded_weights]

    def aggregate_parameters(self, clients_weight):
        """聚合參數"""
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
        for w, client_model in zip(clients_weight, self.uploaded_models):
            for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
                server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        """儲存全局模型"""
        model_path = os.path.join("models", self.dataset)
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.global_model, os.path.join(model_path, f"{self.algorithm}_server.pt"))

    def load_model(self):
        """載入全局模型"""
        model_path = os.path.join("models", self.dataset, f"{self.algorithm}_server.pt")
        if os.path.exists(model_path):
            self.global_model = torch.load(model_path)

    def model_exists(self):
        """檢查模型是否存在"""
        return os.path.exists(os.path.join("models", self.dataset, f"{self.algorithm}_server.pt"))

    def save_results(self):
        """儲存訓練結果"""
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)
            
        algo = f"{self.dataset}_{self.algorithm}_{self.goal}_{self.times}"
        file_path = os.path.join(result_path, f"{algo}.h5")
        print("File path: " + file_path)
        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('rs_test_perplexity', data=self.rs_test_perplexity)
            hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

        perplexity_df = pd.DataFrame(self.perplexity_data, columns=[f"{self.algorithm}_{self.select_clients_algorithm}_{self.poisoned_ratio*self.num_clients}_{self.random_seed}"])
        loss_df = pd.DataFrame(self.loss_data, columns=[f"{self.algorithm}_{self.select_clients_algorithm}_{self.poisoned_ratio*self.num_clients}_{self.random_seed}"])
        perplexity_df = perplexity_df.iloc[::2]
        loss_df = loss_df.iloc[::2]

        perplexity_dir = os.path.join(result_path, f"{self.num_clients}/perplexity")
        loss_dir = os.path.join(result_path, f"{self.num_clients}/loss")
        os.makedirs(perplexity_dir, exist_ok=True)
        os.makedirs(loss_dir, exist_ok=True)
        perplexity_df.to_csv(os.path.join(perplexity_dir, f"{algo}.csv"), index=False)
        loss_df.to_csv(os.path.join(loss_dir, f"{algo}.csv"), index=False)

        plt.figure()
        plt.plot(perplexity_df, label='Perplexity', color='blue')
        plt.title('Perplexity Over Time')
        plt.xlabel('Epochs')
        plt.ylabel('Perplexity')
        plt.legend()
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(perplexity_dir, f"{algo}_perplexity.png"))
        plt.close()

        plt.figure()
        plt.plot(loss_df, label='Loss', color='red')
        plt.title('Loss Over Time')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(loss_dir, f"{algo}_loss.png"))
        plt.close()

    def save_item(self, item, item_name):
        """儲存特定項目"""
        os.makedirs(self.save_folder_name, exist_ok=True)
        torch.save(item, os.path.join(self.save_folder_name, f"server_{item_name}.pt"))

    def load_item(self, item_name):
        """載入特定項目"""
        return torch.load(os.path.join(self.save_folder_name, f"server_{item_name}.pt"))

    def test_metrics(self):
        """測試所有客戶端的困惑度"""
        num_samples = []
        perplexities = []
        for c in self.clients:
            perplexity, ns = c.test_metrics()
            perplexities.append(perplexity * ns)
            num_samples.append(ns)
        return [c.id for c in self.clients], num_samples, perplexities

    def train_metrics(self):
        """計算訓練損失"""
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl)
        return [c.id for c in self.clients], num_samples, losses

    def test_metrics_trust(self, min_trust_index):
        """測試信任客戶端的困惑度"""
        num_samples = []
        perplexities = []
        for c in self.clients:
            if c.id in min_trust_index:
                continue
            perplexity, ns = c.test_metrics()
            perplexities.append(perplexity * ns)
            num_samples.append(ns)
        return [c.id for c in self.clients], num_samples, perplexities

    def train_metrics_trust(self, min_trust_index):
        """計算信任客戶端的訓練損失"""
        num_samples = []
        losses = []
        for c in self.clients:
            if c.id in min_trust_index:
                continue
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl)
        return [c.id for c in self.clients], num_samples, losses

    def evaluate(self, perplexity=None, loss=None):
        """評估全局模型困惑度"""
        stats = self.test_metrics()
        stats_train = self.train_metrics()
        total_samples = sum(stats[1])
        if total_samples == 0:
            test_perplexity = float('inf')
            train_loss = float('inf')
        else:
            test_perplexity = sum(stats[2]) / total_samples
            train_loss = sum(stats_train[2]) / sum(stats_train[1])
        perplexities = [p / n if n > 0 else float('inf') for p, n in zip(stats[2], stats[1])]
        if perplexity is None:
            self.rs_test_perplexity.append(test_perplexity)
            self.perplexity_data.append(test_perplexity)
        else:
            perplexity.append(test_perplexity)
        if loss is None:
            self.rs_train_loss.append(train_loss)
            self.loss_data.append(train_loss)
        else:
            loss.append(train_loss)
        print(f"Average train loss: {train_loss:.4f}")
        print(f"Average test perplexity: {test_perplexity:.4f}")
        print(f"Perplexity std: {np.std(perplexities):.4f}")
        if len(self.perplexity_his) >= 3:
            self.perplexity_his.pop(0)
        self.perplexity_his.append(test_perplexity)
        return test_perplexity, train_loss

    def evaluate_trust(self, perplexity=None, loss=None):
        """評估信任客戶端的困惑度"""
        not_join = self.get_not_evaluate_index()
        print(f"Clients not evaluated: {not_join}")
        stats = self.test_metrics_trust(not_join)
        stats_train = self.train_metrics_trust(not_join)
        total_samples = sum(stats[1])
        if total_samples == 0:
            test_perplexity = float('inf')
            train_loss = float('inf')
        else:
            test_perplexity = sum(stats[2]) / total_samples
            train_loss = sum(stats_train[2]) / sum(stats_train[1])
        perplexities = [p / n if n > 0 else float('inf') for p, n in zip(stats[2], stats[1])]
        if perplexity is None:
            self.rs_test_perplexity.append(test_perplexity)
            self.perplexity_data.append(test_perplexity)
        else:
            perplexity.append(test_perplexity)
        if loss is None:
            self.rs_train_loss.append(train_loss)
            self.loss_data.append(train_loss)
        else:
            loss.append(train_loss)
        print("Trust evaluation")
        print(f"Average train loss: {train_loss:.4f}")
        print(f"Average test perplexity: {test_perplexity:.4f}")
        print(f"Perplexity std: {np.std(perplexities):.4f}")
        if len(self.perplexity_his) >= 3:
            self.perplexity_his.pop(0)
        self.perplexity_his.append(test_perplexity)
        return test_perplexity, train_loss

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        """檢查是否完成訓練（基於困惑度）"""
        for acc_ls in acc_lss:
            if top_cnt is not None and div_value is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1, largest=False).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if not (find_top and find_div):
                    return False
            elif top_cnt is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1, largest=False).indices[0] > top_cnt
                if not find_top:
                    return False
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if not find_div:
                    return False
        return True

    def call_dlg(self, R):
        """執行 DLG 攻擊評估（暫時禁用）"""
        print("DLG attack not implemented for text data")
        return