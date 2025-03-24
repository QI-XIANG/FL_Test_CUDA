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



class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        
        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientAVG)
        self.robustLR_threshold = 7
        self.server_lr = 1e-3

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

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
        testloaderfull = self.get_test_data()

        mlflow.set_experiment("UCB")
        with mlflow.start_run(run_name = "noniid_wbn_4_contribution"):
            mlflow.log_param("global_rounds", self.global_rounds)
            mlflow.log_param("dataset", self.dataset)
            mlflow.log_param("algorithm", self.algorithm)
            mlflow.log_param("num_clients", self.num_clients)

            for i in range(self.global_rounds+1):
                s_t = time.time()
                

                # self.send_models()

                # if i%self.eval_gap == 0:
                #     print(f"\n-------------Round number: {i}-------------")
                #     print("\nEvaluate global model")
                #     self.evaluate()

            
                
                # self.selected_clients = self.select_clients()
                # s = [c.id for c in self.selected_clients]
                # print(s)

                # => mh code 
                '''
                select client by DQN
                '''

                
                '''
                select client by UCB
                '''
                self.selected_clients = self.select_clients_UCB(i)
                s = [c.id for c in self.selected_clients]
                print(s)

                '''
                select client by trust
                '''

                print(f"\n-------------Round number: {i}-------------")

                print(f"history acc: {self.acc_his}")
                # if len(self.acc_his) == 3 and (max(self.acc_his) - min(self.acc_his)) < 0.015:
                #     print("acc convergence!!!")
                #     break
                # if len(self.acc_his) >= 1 and self.acc_his[-1] >= 0.75:
                #     print("acc to the goal!!")
                #     break

                # self.selected_clients = self.select_clients_by_trust()
                # <= mh code 

                for client in self.selected_clients:
                    client.train()

                # threads = [Thread(target=client.train)
                #            for client in self.selected_clients]
                # [t.start() for t in threads]
                # [t.join() for t in threads]


                self.receive_models()
                
                # global_model_vector = parameters_to_vector(self.global_model.parameters())
                # print(global_model_vector)
                # update = [parameters_to_vector(model.parameters()) - global_model_vector for model in self.uploaded_models]
                # lr_vector = torch.Tensor([self.server_lr]*len(global_model_vector)).to(self.device)
                # self.compute_robustLR(update)

                # aggregated_updates = torch.zeros_like(update[0])
                # for u in update:
                #     aggregated_updates += u
                # aggregated_updates = aggregated_updates / self.num_join_clients

                # cur_global_params = parameters_to_vector(self.global_model.parameters())
                # new_global_params =  (cur_global_params + lr_vector*aggregated_updates).float() 
                # vector_to_parameters(new_global_params, self.global_model.parameters())

                # => mh code 
                '''
                consine similarity
                ''' 
                # import numpy as np
                # global_model_vector = parameters_to_vector(self.global_model.parameters())
                # update = [parameters_to_vector(model.parameters()) - global_model_vector for model in self.uploaded_models]
                # # global_model_vector = self.get_vector_no_bn(self.global_model)
                # # update = []
                # # for model in self.uploaded_models:
                # #     client_model_vector = self.get_vector_no_bn(model)
                # #     update.append(client_model_vector - global_model_vector)

                # detect_anomaly_model = torch.load('vae_noniid_pat.pt').to("cuda")
                # detect_anomaly_model.eval()
                # error = []
                # for u in update:
                #     # predict = detect_anomaly_model(u)
                #     predict, mean, logvar = detect_anomaly_model(u)
                #     # reconstruction_loss = torch.nn.BCELoss(reduction='none')(predict, u)
                #     # kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
                #     # total_loss = reconstruction_loss + kl_divergence
                #     # error.append(torch.mean(total_loss).item())
                #     r_error = torch.mean(torch.pow(predict - u, 2))
                #     error.append(r_error.item())
                
                # threshold = np.quantile(error, 0.3)
                
                # attacker = error > threshold
                # if i > 50: breakpoint()
                
                '''
                calculate each model's accuracy
                '''
                clients_acc = []
                for client_model, client in zip(self.uploaded_models, self.selected_clients):
                    test_acc, test_num, auc= self.test_metrics_all(client_model, testloaderfull)
                    print(test_acc/test_num)
                    if client.poisoned:
                        clients_acc.append(test_acc/test_num)
                    else:
                        clients_acc.append(test_acc/test_num)
                clients_acc_weight = list(map(lambda x: x/sum(clients_acc), clients_acc))

                reward_decay = 0.9
                for reward, client in zip(clients_acc, self.selected_clients):
                    self.sums_of_reward[client.id] =  self.sums_of_reward[client.id] * reward_decay + reward
                # breakpoint()
                
                # print(clients_acc)
                
                # kmeans = KMeans(n_clusters= 2)
                # label = kmeans.fit_predict(pd.DataFrame(clients_acc))
                # print(label)


                '''
                check whether it is melicious node and record
                '''
                # trust_value = 1
                # for client in self.selected_clients:
                #     if client.poisoned:
                #         self.interact[client.id].append(1-trust_value)
                #         # self.clients_acc_his[index].append(0.1*acc)
                #     else:
                #         self.interact[client.id].append(trust_value)
                #         # self.clients_acc_his[index].append(acc)
                # acc_decay = 1
                # for index, (client, cacc) in enumerate(zip(self.selected_clients, clients_acc_weight)):
                #     if client.poisoned:
                #         clients_acc_weight[index] = cacc*acc_decay

                # mlflow.log_param("trust_value", trust_value)
                # mlflow.log_param("acc_deacy", acc_decay)

                same_weight = [1/self.num_join_clients] * self.num_join_clients
                # <= mh code 

                if self.dlg_eval and i%self.dlg_gap == 0:
                    self.call_dlg(i)
                # self.aggregate_parameters(same_weight)
                # self.aggregate_parameters_bn(same_weight)
                self.aggregate_parameters_bn(clients_acc_weight)


                self.send_models_bn()
                # self.send_models()

                if i%self.eval_gap == 0:
                    # print(f"\n-------------Round number: {i}-------------")
                    print("\nEvaluate global model")
                    # acc, train_loss = self.evaluate()
                    acc, train_loss = self.evaluate_trust()

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
