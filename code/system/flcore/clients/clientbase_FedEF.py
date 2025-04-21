import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data

###############################################################################
# Client 類別：聯邦學習中每個客戶端的基本類別
###############################################################################
class Client(object):
    """
    聯邦學習中客戶端的基本類別
    此類別初始化時會建立三個模型：
      - self.model：當前本地模型
      - self.prev_model：保存前一輪的本地模型（用於對比學習）
      - self.global_model：保存最新收到的全域模型（用於知識整合）
    """
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        # 初始化模型（此處假設 args.model 為已建立好的模型物件）
        self.model = copy.deepcopy(args.model)            # 當前本地模型
        self.prev_model = copy.deepcopy(args.model)         # 儲存前一輪本地模型
        self.global_model = copy.deepcopy(args.model)       # 儲存最新收到的全域模型

        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        # 檢查模型是否含有 BatchNorm 層
        self.has_BatchNorm = any(isinstance(layer, nn.BatchNorm2d) for layer in self.model.children())

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma

        self.loss = nn.CrossEntropyLoss()
        # 使用 Adam 優化器以加快收斂速度
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay
        self.poisoned = kwargs['poisoned']

    def load_train_data(self, batch_size=None):
        """
        載入訓練資料，若裝置為 cuda 則啟用 pin_memory 加速資料搬移
        """
        if batch_size is None:
            batch_size = self.batch_size
        pin_mem = True if self.device == "cuda" else False
        train_data = read_client_data(self.dataset, self.id, is_train=True)

        # 資料攻擊處理邏輯（略，同前）
        train_data_poison = []
        if self.dataset == 'CelebA':
            for data in train_data:
                data = list(data)
                if data[1] == 1:
                    data[1] = torch.tensor(1)
                train_data_poison.append(tuple(data))
            train_data = train_data_poison
        else:
            if self.poisoned:
                for data in train_data:
                    data = list(data)
                    if data[1] == 1:
                        data[1] = torch.tensor(9)
                    if data[1] == 2:
                        data[1] = torch.tensor(7)
                    if data[1] == 9:
                        data[1] = torch.tensor(1)
                    train_data_poison.append(tuple(data))
                train_data = train_data_poison

        return DataLoader(train_data, batch_size, drop_last=True, shuffle=False, pin_memory=pin_mem)

    def load_test_data(self, batch_size=None):
        """
        載入測試資料，若裝置為 cuda 則啟用 pin_memory
        """
        if batch_size is None:
            batch_size = self.batch_size
        pin_mem = True if self.device == "cuda" else False
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False, pin_memory=pin_mem)

    def set_parameters_bn(self, model):
        """
        僅更新模型中非 BatchNorm 層的參數
        """
        bn_key = [k for k in model.state_dict().keys() if 'running' in k or 'num_batches_tracked' in k]
        for key in self.model.state_dict().keys():
            if key not in bn_key and key in model.state_dict():
                self.model.state_dict()[key].data.copy_(model.state_dict()[key])

    def set_parameters(self, model):
        """
        用全域模型參數更新本地模型參數
        """
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        """
        將 model 的參數複製到 target 模型中
        """
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()

    def update_parameters(self, model, new_params):
        """
        更新模型參數
        """
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def set_global_model(self, model):
        """
        更新全域模型參考，供本地訓練使用
        """
        self.clone_model(model, self.global_model)

    def save_local_model(self):
        """
        將當前本地模型儲存，供下輪作為 prev_model 使用
        """
        self.save_item(self.model.state_dict(), "prev_model")

    def load_local_model(self):
        """
        載入前次儲存的本地模型，更新 self.prev_model
        """
        state_dict = self.load_item("prev_model")
        self.prev_model.load_state_dict(state_dict)

    def test_metrics(self):
        """
        計算測試資料的準確率、樣本數以及 AUC
        """
        testloader = self.load_test_data()
        self.model.eval()
        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloader:
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
                y_prob.append(output.cpu().numpy())
                nc = self.num_classes if self.num_classes != 2 else 3
                lb = label_binarize(y.cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        return test_acc, test_num, auc

    def train_metrics(self):
        """
        計算訓練資料上的損失與樣本數
        """
        trainloader = self.load_train_data()
        self.model.eval()
        total_loss = 0
        total_num = 0
        with torch.no_grad():
            for x, y in trainloader:
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                total_loss += loss.item() * y.shape[0]
                total_num += y.shape[0]
        return total_loss, total_num

    def save_item(self, item, item_name, item_path=None):
        """
        儲存 item 至指定資料夾，檔名格式：client_{id}_{item_name}.pt
        """
        if item_path is None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, f"client_{self.id}_{item_name}.pt"))

    def load_item(self, item_name, item_path=None):
        """
        載入指定檔名的 item
        """
        if item_path is None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, f"client_{self.id}_{item_name}.pt"))