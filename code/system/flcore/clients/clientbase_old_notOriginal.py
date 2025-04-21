import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn import metrics
from utils.data_utils import read_client_data
from sklearn.preprocessing import label_binarize

# 聯邦學習客戶端類，支援 CelebA 多標籤分類
class Client(object):
    """
    聯邦學習客戶端基類，支援多標籤分類（例如 CelebA）。
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        """
        初始化客戶端。
        
        參數：
            args: 包含模型、資料集、學習率等參數的物件
            id (int): 客戶端編號
            train_samples (int): 訓練樣本數
            test_samples (int): 測試樣本數
            **kwargs: 額外參數（如毒化標記）
        """
        self.model = copy.deepcopy(args.model) 
        
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # 客戶端編號
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes if self.dataset.lower() != 'celeba' else 40  # CelebA 有 40 個屬性
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.is_multilabel = (self.dataset.lower() == 'celeba')  # 添加多標籤標誌

        # 檢查是否有 BatchNorm 層（這裡使用 GroupNorm，無需調整）
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma

        # 根據資料集選擇損失函數
        if self.is_multilabel:
            self.loss = nn.BCEWithLogitsLoss()  # 多標籤分類使用二元交叉熵
        else:
            self.loss = nn.CrossEntropyLoss()  # 單標籤分類使用交叉熵

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate) 
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay
        self.poisoned = kwargs['poisoned']

    def load_train_data(self, batch_size=None):
        """
        載入客戶端的訓練資料。
        
        參數：
            batch_size (int, optional): 批量大小，預設使用 self.batch_size
        
        返回：
            DataLoader: 訓練資料載入器
        """
        if batch_size is None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        train_data_poison = []

        if self.is_multilabel:
            # CelebA 的毒化邏輯：隨機翻轉部分屬性標籤
            if self.poisoned:
                for data in train_data:
                    data = list(data)
                    labels = data[1].clone()  # 假設標籤為 (40,) 的張量
                    # 隨機選擇 10% 的屬性進行翻轉
                    flip_indices = np.random.choice(40, int(40 * 0.1), replace=False)
                    for idx in flip_indices:
                        labels[idx] = 1 - labels[idx]  # 0->1 或 1->0
                    data[1] = labels
                    train_data_poison.append(tuple(data))
                train_data = train_data_poison
        else:
            # 其他資料集的毒化邏輯（單標籤翻轉）
            if self.poisoned:
                for data in train_data:
                    data = list(data)
                    if data[1] == 1:
                        data[1] = torch.tensor(9)
                    elif data[1] == 2:
                        data[1] = torch.tensor(7)
                    elif data[1] == 9:
                        data[1] = torch.tensor(1)
                    train_data_poison.append(tuple(data))
                train_data = train_data_poison

        return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)

    def load_test_data(self, batch_size=None):
        """
        載入客戶端的測試資料。
        
        參數：
            batch_size (int, optional): 批量大小，預設使用 self.batch_size
        
        返回：
            DataLoader: 測試資料載入器
        """
        if batch_size is None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)
    
    '''def set_parameters_bn(self, model):
        
        #設定模型參數（保留 GroupNorm 或 BatchNorm 層）。
        
        bn_key = ['conv1.1.weight', 'conv1.1.bias', 'conv1.1.running_mean', 'conv1.1.running_var', 'conv1.1.num_batches_tracked',
                  'conv2.1.weight', 'conv2.1.bias', 'conv2.1.running_mean', 'conv2.1.running_var', 'conv2.1.num_batches_tracked']
        for key in self.model.state_dict().keys():
            if key not in bn_key:
                if key in model.state_dict():
                    self.model.state_dict()[key].data.copy_(model.state_dict()[key])'''
    
    def set_parameters_bn(self, model):
        bn_keys = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                # BN parameters: weight, bias, running_mean, running_var, num_batches_tracked
                bn_keys.extend([
                    f"{name}.weight",
                    f"{name}.bias",
                    f"{name}.running_mean",
                    f"{name}.running_var",
                    f"{name}.num_batches_tracked"
                ])

        for key in self.model.state_dict().keys():
            if key not in bn_keys:
                if key in model.state_dict():
                    self.model.state_dict()[key].data.copy_(model.state_dict()[key])

    def set_parameters(self, model):
        """
        設定模型參數。
        """
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        """
        複製模型參數到目標模型。
        """
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()

    def update_parameters(self, model, new_params):
        """
        更新模型參數。
        """
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        """
        測試模型性能，適配多標籤和單標籤資料集。
        
        返回：
            tuple: 
                - 多標籤 (CelebA): (test_acc, test_num, auc, label_acc)
                - 單標籤: (test_acc, test_num, auc)
        """
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        if self.is_multilabel:
            # 多標籤（CelebA）評估
            correct_per_label = torch.zeros(self.num_classes).to(self.device)  # 每個屬性的正確預測數
            valid_attributes = torch.zeros(self.num_classes).to(self.device)  # 記錄有效屬性

            with torch.no_grad():
                for x, y in testloaderfull:
                    if isinstance(x, list):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device).float()  # 轉為浮點數，因為標籤是 0/1
                    output = self.model(x)
                    preds = (torch.sigmoid(output) > 0.5).float()  # 預測值（閾值 0.5）
                    
                    # 計算總正確屬性數和每個屬性的正確數
                    test_acc += torch.sum(preds == y).item()
                    correct_per_label += torch.sum(preds == y, dim=0)
                    test_num += y.shape[0]
                    
                    # 檢查每個屬性是否有正負樣本
                    valid_attributes += (torch.sum(y, dim=0) > 0).float() * (torch.sum(1 - y, dim=0) > 0).float()
                    
                    y_prob.append(output.detach().cpu().numpy())  # logits 用於 AUC
                    y_true.append(y.detach().cpu().numpy())

            if test_num == 0:
                return 0, 0, 0.0, np.zeros(self.num_classes)

            # 計算有效屬性數並調整準確率
            valid_mask = (valid_attributes > 0).float()
            effective_num_attributes = torch.sum(valid_mask).item() or 1  # 避免除以零
            test_acc = test_acc / (test_num * effective_num_attributes)  # 平均準確率（基於有效屬性）
            label_acc = (correct_per_label / test_num).cpu().numpy() * valid_mask.cpu().numpy()  # 每個屬性的準確率

            y_prob = np.concatenate(y_prob, axis=0)
            y_true = np.concatenate(y_true, axis=0)
            auc = metrics.roc_auc_score(y_true, y_prob, average='macro') if y_true.size > 0 else 0.0

            return test_acc, test_num, auc, label_acc

        else:
            # 單標籤評估
            with torch.no_grad():
                for x, y in testloaderfull:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = self.model(x)

                    test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                    test_num += y.shape[0]

                    y_prob.append(output.detach().cpu().numpy())
                    nc = self.num_classes
                    if self.num_classes == 2:
                        nc += 1
                    lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                    if self.num_classes == 2:
                        lb = lb[:, :2]
                    y_true.append(lb)

            # self.model.cpu()
            # self.save_model(self.model, 'model')

            y_prob = np.concatenate(y_prob, axis=0)
            y_true = np.concatenate(y_true, axis=0)

            auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
            
            return test_acc, test_num, auc

    def train_metrics(self):
        """
        計算訓練損失。
        
        返回：
            tuple: (總損失, 訓練樣本數)
        """
        trainloader = self.load_train_data()
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                if self.is_multilabel:
                    y = y.float()  # 多標籤需要浮點數標籤
                loss = self.loss(output, y)
                train_num += y.shape[0] if not self.is_multilabel else y.numel()
                losses += loss.item() * (y.shape[0] if not self.is_multilabel else y.numel())

        return losses, train_num

    def save_item(self, item, item_name, item_path=None):
        """
        儲存物件（例如模型參數）。
        """
        if item_path is None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        """
        載入物件。
        """
        if item_path is None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))