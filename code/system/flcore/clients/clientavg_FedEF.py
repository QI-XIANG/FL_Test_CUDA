import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *

###############################################################################
# clientAVG 類別：基於 FedAvg 的客戶端，整合 FedEF 訓練機制並加入前次模型載入與保存步驟
###############################################################################
class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.criterion = nn.CrossEntropyLoss()  # 分類任務損失函數
        # 初始化 AMP 的梯度縮放器（僅在 cuda 裝置上使用）
        self.scaler = torch.cuda.amp.GradScaler() if self.device == "cuda" else None
        # 設定溫度參數 (τ)，預設 0.5
        self.tau = 0.5
        # 全域訓練輪次 T，用於動態調整權重係數
        self.total_rounds = args.global_rounds

    def train(self):
        """
        客戶端本地訓練函式，執行以下步驟：
         1. 嘗試載入前次保存的本地模型（prev_model）；若無則跳過（第一次訓練）
         2. 更新全域模型參考（由 server 傳入最新全域模型）
         3. 使用 AMP、餘弦相似度優化及梯度裁剪進行本地訓練
         4. 訓練結束後保存目前本地模型，供下輪作為 prev_model 使用
        """
        trainloader = self.load_train_data()
        self.model.train()

        # 若啟用差分隱私 (DP)，初始化相關模組（保持原有邏輯）
        if self.privacy:
            self.model, self.optimizer, trainloader, privacy_engine = initialize_dp(
                self.model, self.optimizer, trainloader, self.dp_sigma)

        # 嘗試載入前次保存的本地模型（prev_model）
        try:
            self.load_local_model()
            #print(f"Client {self.id} 載入前次本地模型成功。")
        except Exception as e:
            print(f"Client {self.id} 尚無 prev_model，跳過載入。")

        # 更新全域模型參考（實際上由 server 傳入，此處以 self.model 為示範）
        self.set_global_model(self.model)

        # 使用 CUDA Event 計算訓練時間
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()

        max_local_steps = self.local_epochs
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        # 根據訓練輪次動態調整權重係數 λ，至少保證 20% 用於對比損失
        current_round = self.train_time_cost['num_rounds'] + 1
        lam = max(0.2, 1 - current_round / self.total_rounds)

        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                y = y.to(self.device)

                with torch.cuda.amp.autocast(enabled=(self.scaler is not None)):
                    # -------------------- 計算對比損失 ℓE --------------------
                    H_current = self.model.feature_extractor(x)
                    H_global = self.global_model.feature_extractor(x)
                    H_prev = self.prev_model.feature_extractor(x)
                    norm_current = F.normalize(H_current, p=2, dim=1)
                    norm_global = F.normalize(H_global, p=2, dim=1)
                    norm_prev = F.normalize(H_prev, p=2, dim=1)
                    sim_positive = (norm_current * norm_global).sum(dim=1)
                    sim_negative = (norm_current * norm_prev).sum(dim=1)
                    exp_positive = torch.exp(sim_positive / self.tau)
                    exp_negative = torch.exp(sim_negative / self.tau)
                    loss_E = -torch.log(exp_positive / (exp_positive + exp_negative) + 1e-8)
                    loss_E = loss_E.mean() * 0.1

                    # -------------------- 計算修改後的分類損失 ℓF --------------------
                    output = self.model(x)
                    probs = F.softmax(output, dim=1)
                    probs_correct = probs.gather(1, y.view(-1, 1)).squeeze(1)
                    penalty = (1 - probs_correct) ** 2
                    unique_classes, counts = torch.unique(y, return_counts=True)
                    weight_dict = {}
                    total_weight = 0.0
                    for cls, count in zip(unique_classes, counts):
                        weight = 1.0 / count.item()
                        weight_dict[int(cls.item())] = weight
                        total_weight += weight
                    for cls in weight_dict:
                        weight_dict[cls] /= total_weight
                    weights = torch.tensor([weight_dict[int(label.item())] for label in y], device=self.device)
                    log_probs = torch.log(probs_correct + 1e-8)
                    loss_F = - (weights * penalty * log_probs).mean()

                    # -------------------- 組合最終損失 --------------------
                    loss = lam * loss_E + (1 - lam) * loss_F

                self.optimizer.zero_grad()
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
                    self.optimizer.step()

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        end_time.record()
        torch.cuda.synchronize()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += start_time.elapsed_time(end_time) / 1000.0

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}: epsilon = {eps:.2f}, delta = {DELTA}")

        # 保存本次訓練後的本地模型，供下輪作為 prev_model 使用
        self.save_local_model()

    def compute_loss(self):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        testloader = self.load_test_data()
        with torch.no_grad():
            for x, y in testloader:
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)
        return total_loss / total_samples

    def get_training_gradients(self):
        gradient_matrix = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradient_matrix.append(param.grad.view(-1))
        gradient_matrix = torch.cat(gradient_matrix)
        return gradient_matrix.cpu().numpy()