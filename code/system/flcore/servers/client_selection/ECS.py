import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

class EnhancedClusterSelection():
    def __init__(self, num_clients, num_join_clients, random_join_ratio, feature_dim=10, max_poisoned_ratio=0.4):
        self.num_clients = num_clients
        self.num_join_clients = num_join_clients
        self.random_join_ratio = random_join_ratio
        self.feature_dim = feature_dim  # Dimensionality of client features
        self.max_poisoned_ratio = max_poisoned_ratio  # Maximum acceptable poisoned ratio
        
        # Randomly initialize clients' feature vectors (or use real data like model updates)
        self.client_features = np.random.randn(self.num_clients, self.feature_dim)

    def calculate_similarity(self, features):
        """ Calculate cosine similarity between the clients """
        return cosine_similarity(features)

    def cluster_clients(self):
        """ Cluster clients into groups using k-medoids algorithm to identify outliers better """
        kmeans = KMeans(n_clusters=min(self.num_clients, 5), random_state=42)
        clusters = kmeans.fit_predict(self.client_features)
        return clusters, kmeans.cluster_centers_

    def estimate_poisoned_clients(self, cluster_clients, cluster_center):
        """ Estimate poisoned clients based on distance from cluster center (outliers) """
        distances = cdist(self.client_features[cluster_clients], [cluster_center])
        threshold = np.percentile(distances, 90)  # Poisoned clients are those far from the center
        poisoned_estimates = cluster_clients[distances.flatten() > threshold]
        return poisoned_estimates

    def select_clients(self, epoch):
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
        
        return selected_clients[:num_join_clients]
