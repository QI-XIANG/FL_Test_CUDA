import torch
import math
import copy

class AdaptiveUCB:
    def __init__(self, num_clients, num_join_clients, trust_decay=0.9, exploration_factor=1):
        self.num_clients = num_clients
        self.num_join_clients = num_join_clients
        self.numbers_of_selections = [0] * self.num_clients
        self.sums_of_reward = [0] * self.num_clients
        self.trust_scores = [1] * self.num_clients  # Trust scores for clients
        self.exploration_factor = exploration_factor
        self.trust_decay = trust_decay

    def get_n_max(self, n, target):
        t = copy.deepcopy(target)
        max_number = []
        max_index = []
        for _ in range(n):
            number = max(t)
            index = t.index(number)
            t[index] = 0
            max_number.append(number)
            max_index.append(index)
        return max_index

    def select_clients(self, epoch):
        clients_upper_bound = []
        c = self.exploration_factor
        for i in range(self.num_clients):
            if self.numbers_of_selections[i] > 0:
                average_reward = self.sums_of_reward[i] / self.numbers_of_selections[i]
                delta_i = math.sqrt(2 * math.log(epoch + 1) / self.numbers_of_selections[i])
                upper_bound = average_reward + c * delta_i * self.trust_scores[i]
            else:
                upper_bound = 1e400  # Select untested clients with maximum priority
            clients_upper_bound.append(upper_bound)

        #print(f"Epoch {epoch}, client upper bounds: {clients_upper_bound}")  # Debugging line
        selected_clients = self.get_n_max(self.num_join_clients, clients_upper_bound)
        #print(f"Selected clients at epoch {epoch}: {selected_clients}")  # Debugging line
        return selected_clients

    def update(self, clients, rewards, anomalies=None):
        reward_decay = 0.9
        for client, reward in zip(clients, rewards):
            self.sums_of_reward[client] = self.sums_of_reward[client] * reward_decay + reward
            self.numbers_of_selections[client] += 1

            # Update trust score based on anomaly detection (if available)
            if anomalies is not None and anomalies[client] > 0.5:  # Threshold for anomaly detection
                self.trust_scores[client] *= 0.5  # Penalize client if it's anomalous
            else:
                self.trust_scores[client] *= self.trust_decay  # Decay trust score over time

        print("Updated trust scores:", self.trust_scores)
        print("sums of reward: ", self.sums_of_reward)
        print("number of selections: ", self.numbers_of_selections)  # Debugging line
