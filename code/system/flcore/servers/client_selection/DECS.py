import numpy as np
import torch
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from sklearn.neighbors import LocalOutlierFactor

class DiversityEnhancedClusterSelection:
    def __init__(self, num_clients, num_join_clients, random_join_ratio, feature_dim=10, max_poisoned_ratio=0.4):
        self.num_clients = num_clients
        self.num_join_clients = num_join_clients
        self.random_join_ratio = random_join_ratio
        self.feature_dim = feature_dim
        self.max_poisoned_ratio = max_poisoned_ratio
        self.client_features = np.random.randn(self.num_clients, self.feature_dim)  # Example client features

    def calculate_similarity(self, features):
        """Calculate cosine similarity between client feature vectors."""
        return cosine_similarity(features)

    def robust_cluster_clients(self):
        """Cluster clients using DBSCAN with fallback to KMeans if clustering is insufficient."""
        try:
            dbscan = DBSCAN(eps=0.5, min_samples=3)
            clusters = dbscan.fit_predict(self.client_features)
            
            # Check if DBSCAN resulted in too many single-point clusters
            cluster_indices = [i for i in range(max(clusters) + 1) if np.sum(clusters == i) > 1]
            
            if len(cluster_indices) < 2:
                raise ValueError("DBSCAN produced insufficient clusters.")
            
            return clusters, cluster_indices

        except Exception as e:
            print("DBSCAN failed, falling back to KMeans clustering.")
            # Fallback to KMeans with a minimum number of clusters
            kmeans = KMeans(n_clusters=min(5, self.num_clients // 2), random_state=0)
            clusters = kmeans.fit_predict(self.client_features)
            cluster_indices = np.unique(clusters)
            return clusters, cluster_indices

    def detect_outliers_within_cluster(self, cluster_clients):
        """Detect potential poisoned clients within a cluster using LOF or distance-based method."""
        if len(cluster_clients) < 2:
            # Skip outlier detection for very small clusters
            print("Skipping LOF as cluster has 1 or fewer clients.")
            return []

        if len(cluster_clients) <= 5:
            # Use distance-based outlier detection for small clusters
            distances = cdist(self.client_features[cluster_clients], [self.client_features[cluster_clients].mean(axis=0)])
            threshold = np.percentile(distances, 90)  # Define outliers as points farthest from mean
            outliers = cluster_clients[distances.flatten() > threshold]
            return outliers
        else:
            # Use LOF for larger clusters
            n_neighbors = min(5, len(cluster_clients) - 1)
            lof = LocalOutlierFactor(n_neighbors=n_neighbors)
            labels = lof.fit_predict(self.client_features[cluster_clients])
            outliers = cluster_clients[labels == -1]
            return outliers

    def select_clients(self, epoch):
        """Select clients for federated learning with diversity and outlier rejection."""
        if self.random_join_ratio:
            num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients + 1), 1, replace=False)[0]
        else:
            num_join_clients = self.num_join_clients
        
        # Cluster clients robustly
        clusters, valid_cluster_indices = self.robust_cluster_clients()

        selected_clients = []
        for cluster_idx in valid_cluster_indices:
            cluster_clients = np.where(clusters == cluster_idx)[0]
            outliers = self.detect_outliers_within_cluster(cluster_clients)
            non_outliers = [client for client in cluster_clients if client not in outliers]
            
            # Limit the number of selected clients from each cluster
            max_clients_per_cluster = int(num_join_clients / len(valid_cluster_indices))
            num_clients_to_select = min(len(non_outliers), max_clients_per_cluster)
            
            if num_clients_to_select > 0:
                selected_from_cluster = np.random.choice(non_outliers, size=num_clients_to_select, replace=False)
                selected_clients.extend(selected_from_cluster)
            
            if len(selected_clients) >= num_join_clients:
                break
        
        # Fallback in case no clients were selected from clusters
        if len(selected_clients) < num_join_clients:
            print("Fallback: No clients selected from clusters. Selecting random clients.")
            remaining_clients = [client for client in range(self.num_clients) if client not in selected_clients]
            additional_clients = np.random.choice(remaining_clients, size=num_join_clients - len(selected_clients), replace=False)
            selected_clients.extend(additional_clients)
        
        return selected_clients[:num_join_clients]  # Ensure exactly `num_join_clients` are returned
    
    def update(self, clients, rewards):
        """Update strategy based on feedback (optional)."""
        pass
