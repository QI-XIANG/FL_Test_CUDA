import torch
import torch.distributions as tdist

class Thompson:
    def __init__(self, num_clients, num_selections, prior_alpha=1, prior_beta=1):
        self.num_clients = num_clients
        self.num_selections = num_selections
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.posterior_alpha = torch.ones(num_clients) * prior_alpha
        self.posterior_beta = torch.ones(num_clients) * prior_beta
        self.numbers_of_selections = [0] * self.num_clients


    def select_clients(self, epoch):
        samples = torch.zeros(self.num_clients)
        for client in range(self.num_clients):
            samples[client] = tdist.Beta(self.posterior_alpha[client], self.posterior_beta[client]).sample()
        _, selected_clients = torch.topk(samples, self.num_selections)
        return selected_clients.tolist()
    
    def update(self, clients, rewards):
        for client, reward in zip(clients, rewards):
            self.posterior_alpha[client] += reward
            self.posterior_beta[client] += (1 - reward)
            self.numbers_of_selections[client] += 1
        print("number of selection: ", self.numbers_of_selections)
