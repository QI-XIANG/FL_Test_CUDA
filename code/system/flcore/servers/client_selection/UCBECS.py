import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

class UCBEnhancedClusterSelection:
    def __init__(self, num_clients, num_join_clients, random_join_ratio, feature_dim=10, max_poisoned_ratio=0.4, epsilon=0.1, c=2):
        self.num_clients = num_clients
        self.num_join_clients = num_join_clients
        self.random_join_ratio = random_join_ratio
        self.feature_dim = feature_dim  # Dimensionality of client features
        self.max_poisoned_ratio = max_poisoned_ratio  # Maximum acceptable poisoned ratio
        self.epsilon = epsilon  # Exploration-exploitation trade-off (epsilon-greedy)
        self.c = c  # UCB parameter (controls the confidence bound)
        
        # Randomly initialize clients' feature vectors (or use real data like model updates)
        self.client_features = np.random.randn(self.num_clients, self.feature_dim)

        # Initialize UCB tracking
        self.performance_estimates = np.zeros(self.num_clients)  # Mean performance (accuracy, etc.)
        self.confidence_bounds = np.ones(self.num_clients)  # Confidence bounds for each client
        self.selected_clients_history = []

    def calculate_similarity(self, features):
        """ Calculate cosine similarity between the clients """
        return cosine_similarity(features)

    def cluster_clients(self):
        """ Cluster clients into groups using k-means to identify outliers better """
        kmeans = KMeans(n_clusters=min(self.num_clients, 5), random_state=42)
        clusters = kmeans.fit_predict(self.client_features)
        return clusters, kmeans.cluster_centers_

    def estimate_poisoned_clients(self, cluster_clients, cluster_center):
        """ Estimate poisoned clients based on distance from cluster center (outliers) """
        distances = cdist(self.client_features[cluster_clients], [cluster_center])
        threshold = np.percentile(distances, 90)  # Poisoned clients are those far from the center
        poisoned_estimates = cluster_clients[distances.flatten() > threshold]
        return poisoned_estimates

    def update_confidence_bounds(self, client_idx, performance, round_num):
        """ Update the UCB performance estimate and confidence bound for a given client """
        self.performance_estimates[client_idx] = (self.performance_estimates[client_idx] * (round_num - 1) + performance) / round_num
        self.confidence_bounds[client_idx] = np.sqrt(self.c * np.log(round_num) / round_num)  # UCB formula

    def calculate_ucb(self, round_num):
        """ Calculate UCB values for all clients """
        ucb_values = self.performance_estimates + self.confidence_bounds
        return ucb_values

    def select_clients_with_ucb(self, round_num):
        """ Select clients based on UCB strategy (exploitation + exploration) """
        ucb_values = self.calculate_ucb(round_num)
        selected_clients = np.argsort(ucb_values)[-self.num_join_clients:]  # Select clients with highest UCB
        return selected_clients

    def select_clients(self, epoch):
        """ Select clients based on enhanced clustering and UCB strategy """
        if self.random_join_ratio:
            num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients + 1), 1, replace=False)[0]
        else:
            num_join_clients = self.num_join_clients
        
        # Cluster clients based on their features
        clusters, cluster_centers = self.cluster_clients()
        
        # Sort clusters by performance and spread
        cluster_similarities = self.calculate_similarity(cluster_centers)
        cluster_spreads = []
        for i in range(len(cluster_centers)):
            cluster_clients = np.where(clusters == i)[0]
            distances = cdist(self.client_features[cluster_clients], [cluster_centers[i]])
            spread = np.var(distances)
            cluster_spreads.append(spread)
        cluster_spreads = np.array(cluster_spreads)
        
        avg_similarities = np.mean(cluster_similarities, axis=1)
        sorted_clusters_by_similarity = np.argsort(avg_similarities - cluster_spreads)[::-1]  # descending order
        
        selected_clients = []
        selected_poisoned_clients = 0
        for cluster_idx in sorted_clusters_by_similarity:
            cluster_clients = np.where(clusters == cluster_idx)[0]
            poisoned_estimates = self.estimate_poisoned_clients(cluster_clients, cluster_centers[cluster_idx])
            num_poisoned_in_cluster = len(poisoned_estimates)

            # Select clients while controlling the proportion of poisoned clients
            num_clients_to_select = min(len(cluster_clients), num_join_clients - len(selected_clients))
            max_poisoned_in_cluster = int(num_clients_to_select * self.max_poisoned_ratio)
            num_poisoned_to_select = min(num_poisoned_in_cluster, num_poisoned_in_cluster)
            num_normal_to_select = num_clients_to_select - num_poisoned_to_select
            
            normal_clients = [client for client in cluster_clients if client not in poisoned_estimates]
            selected_from_cluster = np.random.choice(normal_clients, size=num_normal_to_select, replace=False)
            poisoned_from_cluster = np.random.choice(poisoned_estimates, size=num_poisoned_to_select, replace=False)
            
            selected_clients.extend(selected_from_cluster)
            selected_clients.extend(poisoned_from_cluster)
            selected_poisoned_clients += num_poisoned_to_select
            
            if len(selected_clients) >= num_join_clients:
                break

        # Enhance selection by UCB
        selected_clients_with_ucb = self.select_clients_with_ucb(epoch)
        print(f"Clients selected with UCB at epoch {epoch}: {selected_clients_with_ucb}")

        return selected_clients_with_ucb[:num_join_clients]
    
    def update(self, selected_ids, clients_acc):
        """
        Update the performance estimates and confidence bounds for selected clients.
        
        Args:
            selected_ids (list): Indices of the selected clients.
            clients_acc (list): Corresponding accuracies for the selected clients.
        """
        for client_idx, performance in zip(selected_ids, clients_acc):
            round_num = len(self.selected_clients_history) + 1
            self.update_confidence_bounds(client_idx, performance, round_num)
        
        # Track selected clients for this round
        self.selected_clients_history.append(selected_ids)
