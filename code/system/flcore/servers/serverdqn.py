import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import mlflow
import torch

import torch.nn as nn
import torch.optim as optim
import numpy as np

class FedDQN(Server):
    def __init__(self, args, times, agent):
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

        self.agent = agent

    def get_vector_no_bn(self, model):
        bn_key = ['conv1.1.weight', 'conv1.1.bias', 'conv1.1.running_mean', 'conv1.1.running_var', 'conv1.1.num_batches_tracked',
                  'conv2.1.weight', 'conv2.1.bias', 'conv2.1.running_mean', 'conv2.1.running_var', 'conv2.1.num_batches_tracked']
        v = []
        for key in model.state_dict():
            if key in bn_key:
                continue 
            v.append(model.state_dict()[key].view(-1))
        return torch.cat(v)
    
    def get_state(self):
        from sklearn.decomposition import PCA
        state = [parameters_to_vector(self.global_model.parameters()).cpu().detach().numpy()]
        for c in self.clients:
            client_vector = parameters_to_vector(c.model.parameters()).cpu().detach().numpy()
            state.append(parameters_to_vector(c.model.parameters()).cpu().detach().numpy())

        pca = PCA(n_components=2)
        pca.fit(state)
        state_pca = pca.transform(state)
        return torch.tensor(state_pca.reshape(1,-1)).to(self.device)


    
    def train(self):
        self.send_models() #initialize model
        testloaderfull = self.get_test_data()
        total_reward = 0
        done = False
        pre_acc = 0
        mlflow.set_experiment("DQN")
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
                state = self.get_state()
                action = self.agent.select_action(state)
                self.selected_clients = [self.clients[c] for c in action]

                # action = agent.select_action(state)
                # local_data = [torch.tensor([self.sums_of_reward[id]], dtype = torch.float) for id in range(num_clients)]
                # states = torch.stack(local_data)
                # breakpoint()
                # actions = [agent.select_action(state) for state in states]
                # breakpoint()



                print(f"\n-------------Round number: {i}-------------")

                # print(f"history acc: {self.acc_his}")

                for client in self.selected_clients:
                    client.train()

                # threads = [Thread(target=client.train)
                #            for client in self.selected_clients]
                # [t.start() for t in threads]
                # [t.join() for t in threads]


                self.receive_models()

                
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


                same_weight = [1/self.num_join_clients] * self.num_join_clients
                # <= mh code 

                if self.dlg_eval and i%self.dlg_gap == 0:
                    self.call_dlg(i)
                # self.aggregate_parameters(same_weight)
                # self.aggregate_parameters_bn(same_weight)
                self.aggregate_parameters_bn(clients_acc_weight)

                
                self.send_models_bn()
                # self.send_models()
                next_state = self.get_state()

                if i%self.eval_gap == 0:
                    # print(f"\n-------------Round number: {i}-------------")
                    print("\nEvaluate global model")
                    # acc, train_loss = self.evaluate()
                    acc, train_loss = self.evaluate_trust()

                    mlflow.log_metric("global accuracy", acc, step = i)
                    mlflow.log_metric("train_loss", train_loss, step = i)
                
                reward = acc - pre_acc
                total_reward += reward
                pre_acc = acc

                if acc >= 98: 
                    print("~~~~")
                    done = True
                self.agent.append_to_replay_buffer(state, list(action)[0], reward, next_state, done)
                self.agent.train(i)


                self.Budget.append(time.time() - s_t)
                print('-'*25, 'time cost', '-'*25, self.Budget[-1])

                if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                    break
                if done: break

                if i%10 == 0:
                    print("agent start to update epsilon")
                    self.agent.update_epsilon()

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

