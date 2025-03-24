import torch
import torch.distributions as tdist

import math
import copy

class UCB:
    def __init__(self, num_clients, num_join_clients):
        self.num_clients = num_clients
        self.num_join_clients = num_join_clients
        self.numbers_of_selections = [0] * self.num_clients
        self.sums_of_reward = [0] * self.num_clients

    def get_n_max(self, n, target):
        t = copy.deepcopy(target)
        # 求m个最大的数值及其索引
        max_number = []
        max_index = []
        for _ in range(n):
            number = max(t)
            index = t.index(number)
            t[index] = 0
            max_number.append(number)
            max_index.append(index)
        t = []

        return max_index
    
    def select_clients(self, epoch):
        clients_upper_bound = []
        c = 1
        for i in range(self.num_clients):
            if (self.numbers_of_selections[i] > 0):
                average_reward = self.sums_of_reward[i] / self.numbers_of_selections[i]
                delta_i = math.sqrt(2 * math.log(epoch+1) / self.numbers_of_selections[i])
                # delta_i = math.sqrt(2 * math.log((epoch+1)*self.num_join_clients) / self.numbers_of_selections[i])

                # delta_i = math.sqrt(math.log((epoch+1)*self.num_join_clients) / self.numbers_of_selections[i])
                upper_bound = average_reward + c * delta_i
            else:
                upper_bound = 1e400
            
            clients_upper_bound.append(upper_bound)
            # if upper_bound > max_upper_bound:
            #     max_upper_bound = upper_bound
            #     ad = i
        
        selected_clients = self.get_n_max(self.num_join_clients, clients_upper_bound)
        
        
        # selected_clients = []
        # for id in selected_clients_id:
        #     self.numbers_of_selections[id] += 1
        #     selected_clients.append(self.clients[id])    

        return selected_clients

    def update(self, clients, rewards):
        reward_decay = 1
        for client, reward in zip(clients, rewards):
            self.sums_of_reward[client] = self.sums_of_reward[client] * reward_decay + reward
            self.numbers_of_selections[client] += 1
        print("sums of reward: ", self.sums_of_reward)
        print("number of selection: ", self.numbers_of_selections)
        