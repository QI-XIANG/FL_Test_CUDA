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


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
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

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        # => mh code
        self.poisoned_ratio = args.poisoned_ratio
        self.random_seed = args.random_seed

        self.poisoned_clients = self.select_poisoned_client()
        # self.poisoned_clients = []
        print(f"poisoned clients: {self.poisoned_clients}")

        self.interact = [[] for i in range(self.num_clients)]
        self.acc_his = []

        self.clients_acc_his = [[] for i in range(self.num_clients)]

        self.numbers_of_selections = [0] * self.num_clients
        self.sums_of_reward = [0] * self.num_clients
        self.clients_loss = [0] * self.num_clients

        self.acc_data = []
        self.loss_data = []
        self.auc_data = []
        self.select_clients_algorithm = args.select_clients_algorithm
        self.server = args.algorithm
        self.Budget = []
        self.weight_option = args.weight_option
        
        
        # <= mh code

    # => mh code
    '''def select_poisoned_client(self):
        np.random.seed(self.random_seed)
        label_one_clients = []
        for i in range(self.num_clients):
            temp = read_client_data(self.dataset, i, is_train=False)
            
            for image in temp:
                if image[1] == 1:
                    label_one_clients.append(i)
                    break
        # label_one_clients = [0,4,5,8] #[0,2,4,5,6,8,9]
        poisoned_clients = list(np.random.choice(label_one_clients, int(self.num_clients*self.poisoned_ratio), replace=False))
        # poisoned_clients = list(np.random.choice(np.arange(self.num_clients), 4, replace=False))
        return poisoned_clients'''
    
    def select_poisoned_client(self):
        np.random.seed(self.random_seed)
        label_one_clients = []
        
        for i in range(self.num_clients):
            temp = read_client_data(self.dataset, i, is_train=False)
            
            for image in temp:
                if image[1] == 1:
                    label_one_clients.append(i)
                    break
        
        # 確保可以抽樣的客戶端數量不超過可用客戶端的數量
        num_poisoned_clients = int(self.num_clients * self.poisoned_ratio)
        if num_poisoned_clients > len(label_one_clients):
            num_poisoned_clients = len(label_one_clients)

        poisoned_clients = list(np.random.choice(label_one_clients, num_poisoned_clients, replace=False))
        return poisoned_clients

    
    def get_test_data(self):
        batch_size = self.batch_size
        test_data = read_client_data(self.dataset, 0, is_train=False)

        for i in range(1, self.num_clients):
            test_data += read_client_data(self.dataset, i, is_train=False)
        from random import shuffle
        shuffle(test_data)
        sampling_data = test_data[0:3000]

        return DataLoader(sampling_data, batch_size, drop_last=False, shuffle=False)
 

    def select_clients_by_trust(self):
        if self.random_join_ratio:
            num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            num_join_clients = self.num_join_clients
        # selected_clients = list(np.random.choice(self.clients, num_join_clients, replace=False))

        # => mh code
        clients_trust = [0.5] * self.num_clients
        for i, (record, cah) in enumerate(zip(self.interact, self.clients_acc_his)):
            if len(record) == 0: continue
            clients_trust[i] = sum(record)/len(record) #+ np.mean(cah)

        clients_trust = list(map(lambda x: x/sum(clients_trust), clients_trust))
        selected_clients_id = np.random.choice(np.arange(self.num_clients),size= num_join_clients,replace=False, p=clients_trust)
        selected_clients = [self.clients[id] for id in selected_clients_id]
        print(f"clients_trust: {clients_trust}")
        print(f"selected_clients_id: {selected_clients_id}")
        # <= mh code

        return selected_clients
    
    def test_metrics_all(self, client_model, testloaderfull):
        
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        client_model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = client_model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc
    
    def params_to_vector(self, model):
        params = []
        for param in model.parameters():
            params.append(param.view(-1))
        params = torch.cat(params)
        return params
    
    # <= mh code
    


    def set_clients(self, args, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            # => mh code
            if i in self.poisoned_clients: poisoned = 1
            else: poisoned = 0
            # <= mh code

            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow,
                            poisoned = poisoned)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, num_join_clients, replace=False))

        return selected_clients
    
    def select_clients_UCB(self, epoch):
        
        clients_upper_bound = []

        for i in range(self.num_clients):
            if (self.numbers_of_selections[i] > 0):
                average_reward = self.sums_of_reward[i] / self.numbers_of_selections[i]
                delta_i = math.sqrt(2 * math.log(epoch+1) / self.numbers_of_selections[i])
                upper_bound = average_reward + delta_i
            else:
                
                upper_bound = 1e400
            
            clients_upper_bound.append(upper_bound)
            # if upper_bound > max_upper_bound:
            #     max_upper_bound = upper_bound
            #     ad = i
        
        t = copy.deepcopy(clients_upper_bound)
        # 求m个最大的数值及其索引
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
        assert (len(self.clients) > 0)
        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters_bn(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def send_models(self):
        assert (len(self.clients) > 0)
        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        # active_clients = random.sample(
        #     self.selected_clients, int((1-self.client_drop_rate) * self.num_join_clients))
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
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters_bn_lr(self, clients_weight):
        bn_key = ['conv1.1.weight', 'conv1.1.bias', 'conv1.1.running_mean', 'conv1.1.running_var', 'conv1.1.num_batches_tracked',
                  'conv2.1.weight', 'conv2.1.bias', 'conv2.1.running_mean', 'conv2.1.running_var', 'conv2.1.num_batches_tracked']
        
        
        for key in self.global_model.state_dict().keys():
            if key not in bn_key:
                temp = torch.zeros_like(self.global_model.state_dict()[key], dtype=torch.float32)
                for weight, model in zip(clients_weight, self.uploaded_models):
                    temp += weight * model.state_dict()[key]
                self.global_model.state_dict()[key].data.copy_(temp)

    def aggregate_parameters_bn(self, clients_weight):
        bn_key = ['conv1.1.weight', 'conv1.1.bias', 'conv1.1.running_mean', 'conv1.1.running_var', 'conv1.1.num_batches_tracked',
                  'conv2.1.weight', 'conv2.1.bias', 'conv2.1.running_mean', 'conv2.1.running_var', 'conv2.1.num_batches_tracked']
        
        for key in self.global_model.state_dict().keys():
            if key not in bn_key:
                temp = torch.zeros_like(self.global_model.state_dict()[key], dtype=torch.float32)
                for weight, model in zip(clients_weight, self.uploaded_models):
                    if key in model.state_dict():
                        temp += weight * model.state_dict()[key]
                    else:
                        pass
                self.global_model.state_dict()[key].data.copy_(temp)

    def aggregate_parameters(self, clients_weight):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        
        for param in self.global_model.parameters():
            param.data.zero_()
            
        # for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
        #     self.add_parameters(w, client_model)
        
        for w, client_model in zip(clients_weight, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        
        # Create the base results directory if it doesn't exist
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # Check if there are test accuracies to save
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

        # Construct the name for the results files
        name = f"{self.algorithm}_{self.select_clients_algorithm}_{self.poisoned_ratio*self.num_clients}_{self.random_seed}"
        acc_df.columns = [name]
        loss_df.columns = [name]
        auc_df.columns = [name]

        # Create necessary directories for saving CSV files
        auc_dir = f"../results/{self.num_clients}/auc"
        os.makedirs(auc_dir, exist_ok=True)  # Create the auc directory if it doesn't exist

        # Save the DataFrames to CSV files
        auc_df.to_csv(os.path.join(auc_dir, f"{name}.csv"), index=False)
        # Uncomment and modify the following lines if needed:
        acc_dir = f"../results/{self.num_clients}/accuracy"
        loss_dir = f"../results/{self.num_clients}/loss"
        os.makedirs(acc_dir, exist_ok=True)
        os.makedirs(loss_dir, exist_ok=True)
        acc_df.to_csv(os.path.join(acc_dir, f"{name}.csv"), index=False)
        loss_df.to_csv(os.path.join(loss_dir, f"{name}.csv"), index=False)

        # Plot and save AUC graph
        plt.figure()
        plt.plot(auc_df, label='AUC', color='blue')
        plt.title('AUC Over Time')
        plt.xlabel('Epochs')
        plt.ylabel('AUC')
        plt.legend()
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Only show integer ticks on x-axis
        plt.savefig(os.path.join(auc_dir, f"{name}_auc.png"))
        plt.close()

        # Plot and save Accuracy graph
        plt.figure()
        plt.plot(acc_df, label='Accuracy', color='green')
        plt.title('Accuracy Over Time')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Only show integer ticks on x-axis
        plt.legend()
        plt.savefig(os.path.join(acc_dir, f"{name}_accuracy.png"))
        plt.close()

        # Plot and save Loss graph
        plt.figure()
        plt.plot(loss_df, label='Loss', color='red')
        plt.title('Loss Over Time')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Only show integer ticks on x-axis
        plt.legend()
        plt.savefig(os.path.join(loss_dir, f"{name}_loss.png"))
        plt.close()

        
        # time_df = pd.DataFrame(self.Budget, columns= [name])
        # time_df.to_csv(f"../results/multiple_round/{self.num_clients}/time/{name}.csv", index=False)


        # poisoned_df = pd.DataFrame(self.poisoned_clients, columns=[name])
        # poisoned_df.to_csv(f"../results/{self.num_clients}/poisoned/{name}.csv", index=False)



        # if self.algorithm == "FedUCBN":
        #     client_select_times = pd.DataFrame(self.numbers_of_selections)
        #     client_select_times.columns = [self.select_clients_algorithm]

        #     client_select_times.to_csv(f"../results/multiple_round/{self.num_clients}/select_times/{name}.csv", index=False)

 


    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_selected_clients_metrics(self):
        clients_accuracy = []
        for c in self.selected_clients:
            ct, ns, auc = c.test_metrics_all()
            clients_accuracy.append(ct/ns)
        # breakpoint()
            
        
    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses
    
    def test_metrics_trust(self, min_trust_index):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            # if len(self.interact[c.id]) != 0 and np.mean(self.interact[c.id]) == 0:
            #     print(f"not join evaluate: {c.id}")
            #     continue
            if c.id in min_trust_index:
                # print(f"not join evaluate: {c.id}")
                continue
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]
        
        return ids, num_samples, tot_correct, tot_auc

    def train_metrics_trust(self, min_trust_index):
        num_samples = []
        losses = []
        for c in self.clients:
            # if len(self.interact[c.id]) != 0 and np.mean(self.interact[c.id]) == 0:
            #     print(f"not join evaluate: {c.id}")
            #     continue
            if c.id in min_trust_index:
                # print(f"not join evaluate: {c.id}")
                continue
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

        # => mh code
        if len(self.acc_his) >= 3:
            self.acc_his.pop(0)
        self.acc_his.append(test_acc)

        return test_acc, train_loss, test_auc
    
    def get_n_min(self, number, target):
        t = copy.deepcopy(target)
        min_number = []
        min_index = []
        for _ in range(number):
            number = min(t)
            index = t.index(number)
            t[index] = float('inf')
            min_number.append(number)
            min_index.append(index)
        t = []
        print(min_number)
        print(min_index)

        return min_index
    
    def get_not_evaluate_index(self):
        # from sklearn import preprocessing
        # from sklearn_extra.cluster import KMedoids
        # zscore = preprocessing.StandardScaler()
        # sum_reward_zs = zscore.fit_transform(list(np.array(self.sums_of_reward).reshape(-1,1)))
        # sum_select_zs = zscore.fit_transform(list(np.array(self.numbers_of_selections).reshape(-1,1)))
        # sum_reward_zs_df = pd.DataFrame(sum_reward_zs, columns=["sum_reward_zs"])
        # sum_select_zs_df = pd.DataFrame(sum_select_zs, columns=["sum_select_zs"])
        # df = pd.concat([sum_reward_zs_df, sum_select_zs_df], axis=1)

        # kmeans = KMeans(n_clusters=2).fit(data_zs)
        # print("kmeans: ", kmeans.labels_)
        
        # pred_KMedoids = KMedoids(n_clusters=2).fit(df)
        # pred_labels = pred_KMedoids.labels_
        # print("KMedoids: ", pred_labels)
        
        # benchmark = (epoch + 1) * self.num_join_clients / self.num_clients
        # threshold = benchmark - np.std(self.numbers_of_selections)
        # std = np.std(self.sums_of_reward)
        # threshold = np.mean(self.sums_of_reward) - 0.25*std
        threshold = np.percentile(self.numbers_of_selections, 40)
        pass_ = [1 if i >= threshold else 0 for i in self.numbers_of_selections]

        return list(np.where(np.array(pass_) == 0)[0])
        
    def evaluate_trust(self, acc=None, loss=None):
        # min_trust_index = self.get_n_min(4, self.sums_of_reward)
        not_join = self.get_not_evaluate_index()
        print("not join: ", not_join)
        stats = self.test_metrics_trust(not_join)
        stats_train = self.train_metrics_trust(not_join)

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Trust")
        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

        # => mh code
        if len(self.acc_his) >= 3:
            self.acc_his.pop(0)
        self.acc_his.append(test_acc)

        return test_acc, train_loss, test_auc

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
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

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')
