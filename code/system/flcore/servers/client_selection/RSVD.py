import numpy as np
from sklearn.utils.extmath import randomized_svd

class RSVDClientDetection:
    def __init__(self, num_clients, num_join_clients, min_valid_clients=10):
        """
        初始化參數
        :param num_clients: 客戶端總數
        :param num_join_clients: 每輪選擇的客戶端數量
        :param min_valid_clients: 每輪至少保留的有效客戶端數量
        """
        self.num_clients = num_clients
        self.num_join_clients = num_join_clients
        self.min_valid_clients = min_valid_clients
        self.stability_scores = np.ones(num_clients)  # 穩定性分數
        self.trust_scores = np.ones(num_clients)  # 信任分數
        self.selection_counts = np.zeros(num_clients)  # 選擇次數記錄
        self.performance_history = np.zeros(num_clients)  # 歷史績效記錄
        self.anomaly_score_history = np.zeros(num_clients)  # 異常分數歷史
        self.poisoned_threshold = 0.5  # 初始異常閾值
        
    def detect_poisoned_clients(self, gradients):
        """
        使用隨機奇異值分解檢測中毒客戶端
        :param gradients: 各客戶端的本地梯度更新 (形狀為 [num_clients, gradient_dim])
        :return: 各客戶端的異常分數
        """
        client_ids = list(gradients.keys())
        gradients = np.vstack(list(gradients.values())) 
        u, s, vt = randomized_svd(gradients, n_components=2, random_state=42)
        reconstructed = np.dot(np.dot(u, np.diag(s)), vt)
        errors = np.linalg.norm(gradients - reconstructed, axis=1)

        # 更新異常分數歷史（平滑處理）
        for i, client_id in enumerate(client_ids):
            self.anomaly_score_history[client_id] = 0.8 * self.anomaly_score_history[client_id] + 0.2 * errors[i]
        
        return self.anomaly_score_history

    def adjust_poisoned_threshold(self, anomaly_scores):
        """
        動態調整 poisoned_threshold
        :param anomaly_scores: 異常分數列表
        :return: 調整後的 poisoned_threshold
        """
        # 計算異常分數的 80% 分位數作為新的閾值
        threshold = np.percentile(anomaly_scores, 80)
        
        # 如果有效客戶端少於最低數量，放寬閾值
        valid_clients = np.where(anomaly_scores <= threshold, 1, 0)
        if np.sum(valid_clients) < self.min_valid_clients:
            # 放寬閾值至 90% 分位數
            threshold = np.percentile(anomaly_scores, 90)
            print(f"Relaxed anomaly threshold to {threshold}")
        
        # 可以根據具體情況進一步設計更精細的調整方式
        self.poisoned_threshold = threshold
        
    def select_clients(self, epoch, gradients):
        """
        客戶端選擇邏輯
        :param epoch: 當前訓練輪次
        :param gradients: 各客戶端的本地梯度更新
        :return: 選定的客戶端列表
        """
        anomaly_scores = self.detect_poisoned_clients(gradients)

        # 動態調整 poisoned_threshold
        self.adjust_poisoned_threshold(anomaly_scores)

        # 根據 adjusted threshold 選擇有效客戶端
        valid_clients = np.where(anomaly_scores <= self.poisoned_threshold, 1, 0)
        
        # 計算多樣性懲罰與分數
        diversity_penalty = 1 / (self.selection_counts + 1)
        scores = (
            (self.stability_scores * self.trust_scores) 
            + 0.5 * diversity_penalty
        ) - 0.5 * anomaly_scores

        # 應用有效性掩碼
        scores = scores * valid_clients

        # 隨機性噪聲
        noise_level = max(0.05, np.std(scores) * 0.1)
        noise = np.random.uniform(0, noise_level, self.num_clients)
        adjusted_scores = scores + noise

        # 選擇分數最高的 num_join_clients 客戶端
        selected_clients = np.argsort(adjusted_scores)[-self.num_join_clients:]

        # 更新選擇次數
        for client in selected_clients:
            self.selection_counts[client] += 1

        print(f"Epoch {epoch}: Anomaly Scores - {anomaly_scores}")
        return selected_clients

    def update(self, selected_clients, rewards):
        """
        更新信任分數與穩定性分數
        :param selected_clients: 被選中的客戶端列表
        :param rewards: 客戶端的績效分數
        """
        # 計算當前選定客戶端的平均績效作為基準
        avg_rewards = np.mean(rewards)

        for client, reward in zip(selected_clients, rewards):
            # 使用移動平均更新歷史績效
            self.performance_history[client] = (
                self.performance_history[client] * 0.8 + reward * 0.2
            )

            # 動態調整更新幅度（根據與基準的差距）
            delta = abs(reward - avg_rewards) / (avg_rewards + 1e-8)  # 防止除零
            if reward < 0:  # 表現不佳，降低分數
                self.trust_scores[client] *= max(1 - delta, 0.8)
                self.stability_scores[client] *= max(1 - delta, 0.8)
            else:  # 表現良好，提高分數
                self.trust_scores[client] = min(self.trust_scores[client] + delta * 0.1, 1.0)
                self.stability_scores[client] = min(self.stability_scores[client] + delta * 0.1, 1.0)

        #print("Updated Trust Scores:", self.trust_scores)
        #print("Updated Stability Scores:", self.stability_scores)



'''
函數說明
    1.detect_poisoned_clients 函數：
        使用 rSVD 對梯度矩陣進行降維。
        透過重建誤差，計算異常分數以識別中毒客戶端。
    2.select_clients 函數：
        使用穩定性和信任分數作為選擇指標。
        中毒的客戶端會降低穩定性分數，使其未來被選中的可能性降低。
    3.update 函數：
        根據當前輪次的績效分數，動態更新信任分數與穩定性分數。
        表現不佳的客戶端將受到懲罰，而表現良好的客戶端獲得分數提升。
整合邏輯
    1.在訓練過程中，根據梯度矩陣調用 select_clients，確定參與訓練的客戶端。
    2.訓練結束後，根據績效數據調用 update，更新信任與穩定性分數。
優點
    1.動態適應性：結合異常檢測與動態分數調整，對中毒客戶端的抵抗能力更強。
    2.簡化特徵提取：rSVD 提供高效的異常分數計算。
    3.穩定性提升：綜合使用穩定性與信任度，確保模型的穩定訓練。
改進策略
    1. 隨機性引入:在選擇時結合隨機性，避免選擇過於固定的客戶端。這可以通過在分數計算中引入隨機噪聲實現。
    2. 多樣性機制:記錄每個客戶端的參與次數，優先選擇參與次數較少的客戶端。
    3. 歷史績效結合:根據歷史績效分數（例如過去幾輪的 reward 平均值）調整穩定性和信任分數。
    4. 調整懲罰與回復力度:對中毒客戶端加重懲罰，對正常客戶端加速穩定性恢復，從而縮短訓練收斂時間。
    5.異常分數平滑：使用指數平滑方法對異常分數進行歷史更新，避免因一次波動對穩定 clients 過度懲罰。
    6.動態閾值調整：確保在有效 clients 不足時，適當放寬異常分數的篩選閾值，避免訓練中斷。
    7.穩定性恢復機制：被懲罰的 clients 可在後續輪次逐步恢復穩定性，為可能的誠實 clients 提供重新參與的機會。
    8.最低有效保障：確保每輪訓練中有效 clients 的數量，防止過濾過多導致訓練失敗。
'''