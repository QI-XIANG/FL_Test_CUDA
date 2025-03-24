import numpy as np
import math

class DynamicMultiStrategySelection:
    def __init__(self, num_clients, num_join_clients, poisoned_ratio=0.4, exploration_rate=0.2):
        """
        初始化參數
        :param num_clients: 客戶端總數
        :param num_join_clients: 每輪選擇的客戶端數量
        :param poisoned_ratio: 可接受的中毒客戶端比例
        :param exploration_rate: 隨機探索的比例，用於平衡探索與利用
        """
        self.num_clients = num_clients
        self.num_join_clients = num_join_clients
        self.poisoned_ratio = poisoned_ratio
        self.exploration_rate = exploration_rate
        
        # 初始化績效記錄與選擇次數
        self.performance_history = np.zeros(num_clients)  # 績效分數
        self.numbers_of_selections = np.zeros(num_clients)  # 客戶端被選擇的次數
        
        # 假設初始每個客戶端的穩定性與信任分數均為高
        self.stability_scores = np.ones(num_clients)  # 穩定性分數
        self.trust_scores = np.ones(num_clients)  # 信任分數

    def calculate_scores(self):
        """
        計算每個客戶端的動態分數，結合績效、穩定性與信任度。
        """
        # 避免除以零的情況
        with np.errstate(divide='ignore', invalid='ignore'):
            average_performance = self.performance_history / (self.numbers_of_selections + 1e-10)
            scores = average_performance * self.stability_scores * self.trust_scores
            scores = np.nan_to_num(scores)  # 將 NaN 轉為 0
            
        return scores

    def select_clients(self, epoch):
        """
        客戶端選擇邏輯
        :param epoch: 當前訓練輪次
        :return: 選定的客戶端列表
        """
        # 計算動態分數
        scores = self.calculate_scores()
        
        # 隨機探索 (以一定比例選擇分數較低的客戶端)
        exploration_size = int(self.num_join_clients * self.exploration_rate)
        exploitation_size = self.num_join_clients - exploration_size
        
        # 基於分數選擇高績效的客戶端
        top_clients = np.argsort(scores)[-exploitation_size:]  # 分數最高者
        # 基於隨機性選擇部分客戶端
        random_clients = np.random.choice(np.arange(self.num_clients), exploration_size, replace=False)
        
        # 合併選擇的客戶端
        selected_clients = np.unique(np.concatenate([top_clients, random_clients]))
        
        # 確保選擇的客戶端數量符合要求
        if len(selected_clients) > self.num_join_clients:
            selected_clients = selected_clients[:self.num_join_clients]
        
        # Debug: 印出選擇過程
        print(f"Epoch {epoch}: Selected Clients - {selected_clients}")
        print(f"Scores - {scores}")
        
        return selected_clients

    def update(self, selected_clients, rewards, stability_updates):
        """
        更新績效、穩定性與信任度
        :param selected_clients: 被選中的客戶端列表
        :param rewards: 客戶端的績效分數
        :param stability_updates: 穩定性更新，對中毒客戶端進行懲罰
        """
        for client, reward, stability in zip(selected_clients, rewards, stability_updates):
            # 更新績效分數
            self.performance_history[client] += reward
            self.numbers_of_selections[client] += 1
            
            # 更新穩定性分數
            self.stability_scores[client] = max(0, self.stability_scores[client] + stability)
            
            # 懲罰信任度對於懷疑中毒的客戶端
            if stability < 0:
                self.trust_scores[client] *= 0.9  # 每次降低 10%
        
        # Debug: 印出更新後的統計數據
        print("Updated Performance History:", self.performance_history)
        print("Updated Stability Scores:", self.stability_scores)
        print("Updated Trust Scores:", self.trust_scores)

'''
核心設計理念
    1.多策略結合：結合 UCB 和隨機探索策略，在穩定性和探索性之間找到平衡。
    2.動態評分：對每個客戶端根據其歷史績效和穩定性進行打分。
    3.懲罰與獎勵機制：對於疑似中毒的客戶端施加懲罰，降低其被選擇的機會。
    4.歷史記憶：保留多輪的歷史績效數據，避免過於倚賴當前結果。
核心改進點
    1.分數計算結合多維度：
        績效：基於歷史表現。
        穩定性：根據模型訓練過程中的異常行為（如梯度差異）進行更新。
        信任度：對於懷疑中毒的客戶端進行逐步懲罰。
    2.動態探索與利用：
        將部分資源分配給隨機探索，避免陷入局部最優解。
        主要資源仍然用於高績效客戶端，保證穩定學習。
    3.穩定性更新與懲罰機制：
        當客戶端表現出不穩定（如中毒行為），即降低其穩定性和信任度。
'''