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

from flcore.servers.client_selection.Random import Random
from flcore.servers.client_selection.Thompson import Thompson
from flcore.servers.client_selection.UCB import UCB
from flcore.servers.client_selection.GAC import GAClientSelection
from flcore.servers.client_selection.DMSS import DynamicMultiStrategySelection

import threading


class FedUCBN(Server):
    def __init__(self, args, times, agent = None):
        super().__init__(args, times)

        # self.args = args
        self.agent = agent
        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientAVG)
        self.robustLR_threshold = 7
        self.server_lr = 1e-3

        self.dynamic = args.dynamic_training

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
    
    def train(self):
        self.send_models() #initialize model
        testloaderfull = self.testdataloader

        if self.select_clients_algorithm == "Random":
            select_agent = Random(self.num_clients, self.num_join_clients, self.random_join_ratio)

        elif self.select_clients_algorithm == "UCB":
            select_agent = UCB(self.num_clients, self.num_join_clients)
        
        elif self.select_clients_algorithm == "GAC":
            select_agent = GAClientSelection(self.num_clients, self.num_join_clients)

        elif self.select_clients_algorithm == "DMSS":
            select_agent = DynamicMultiStrategySelection(self.num_clients, self.num_join_clients)
        
        elif self.select_clients_algorithm == "Thompson":
            select_agent = Thompson(num_clients=self.num_clients, num_selections=self.num_join_clients)

        mlflow.set_experiment(self.select_clients_algorithm)
        with mlflow.start_run(run_name = f"noniid_wbn_{self.num_clients*self.poisoned_ratio}_contribution"):
            mlflow.log_param("global_rounds", self.global_rounds)
            mlflow.log_param("dataset", self.dataset)
            mlflow.log_param("algorithm", self.algorithm)
            mlflow.log_param("num_clients", self.num_clients)

            for i in range(self.global_rounds+1):
                s_t = time.time()

                selected_ids = select_agent.select_clients(i)
                print("selected clients:", selected_ids)
                self.selected_clients = [self.clients[c] for c in selected_ids]
                # 紀錄有毒的客戶端數量
                poisoned_selected = [idx for idx in selected_ids if self.clients[idx].poisoned]
                self.poisoned_clients_selected.append(len(poisoned_selected))
                # => mh code 
                

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


                self.receive_models()

                # => mh code 
                '''
                calculate each model's accuracy
                '''
                clients_acc = []
                for client_model, client in zip(self.uploaded_models, self.selected_clients):
                    test_acc, test_num, auc, f1, auc_pr, avg_loss = self.test_metrics_all(client_model, testloaderfull)
                    #print(test_acc/test_num)
                    clients_acc.append(test_acc/test_num)

                clients_acc_weight = list(map(lambda x: x/sum(clients_acc), clients_acc))

                reward_decay = 1
                for reward, client in zip(clients_acc, self.selected_clients):
                    self.sums_of_reward[client.id] =  self.sums_of_reward[client.id] * reward_decay + reward
                    self.numbers_of_selections[client.id] += 1
                
                rewards = clients_acc
                select_agent.update(selected_ids, rewards)

                # <= mh code 
                if self.dlg_eval and i%self.dlg_gap == 0:
                    self.call_dlg(i)
                
                same_weight = [1/self.num_join_clients] * self.num_join_clients
                weight = clients_acc_weight
                if self.weight_option == "same":
                    weight = same_weight
                
                self.aggregate_parameters_bn(weight)


                self.send_models_bn()
                # self.send_models()

                if i%self.eval_gap == 0:
                    # print(f"\n-------------Round number: {i}-------------")
                    print("\nEvaluate global model")
                    # acc, train_loss = self.evaluate()
                    acc, train_loss, auc, test_f1, test_auc_pr = self.evaluate_trust()
                    #self.acc_data.append(acc)
                    #self.loss_data.append(train_loss)
                    #self.auc_data.append(auc)
                    mlflow.log_metric("global accuracy", acc, step = i)
                    mlflow.log_metric("train_loss", train_loss, step = i)


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
