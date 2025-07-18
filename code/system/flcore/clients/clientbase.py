import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn import metrics
from utils.data_utils import read_client_data
from sklearn.preprocessing import label_binarize
from torchvision import transforms
from sklearn.metrics import f1_score, precision_recall_curve, auc as sklearn_auc
import random
import torch.nn.functional as F

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

        self.dynamic = args.dynamic_training
        
        self.current_index = 0

        self.label_datasize = [0] * args.num_classes
        self.finishCount = True

        self.attack_type = args.attack_type
        self.static_target = {}
        self.static_flag = 1

        # CelebA 有 40 個屬性，其他資料集使用 args.num_classes
        self.num_classes = args.num_classes if self.dataset.lower() != 'celeba' else 40
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.is_multilabel = (self.dataset.lower() == 'celeba')  # 添加多標籤標誌
        self.data_augmentation = True # 是否使用數據增強

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

        self.global_rounds = args.global_rounds  # Server聚合執行次數

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

        self.testDataLoader = self.load_test_data()
        self.trainDataLoader = self.load_train_data()

    def _apply_data_augmentation_cifar100(self, images):
        """
        對 CIFAR-100 的輸入圖像應用本地資料增強。
        應用多種隨機變換組合，而不是單一變換。

        參數：
            images (torch.Tensor): 一批圖像資料 (N, C, H, W)。

        返回：
            torch.Tensor: 增強後的圖像資料。
        """
        augmented_images_batch = []
        for i in range(images.shape[0]):
            image = images[i] # 處理單張圖像
            
            random_number = random.randint(1, 5) # 對每張圖像隨機選擇一個增強

            if random_number == 1:  # flip
                if random.random() > 0.5:
                    image = torch.flip(image, dims=[-1])  # 水平翻轉
            elif random_number == 2:  # rotate
                angle = random.uniform(-15, 15)
                # rotate 函數需要 (H, W) 或 (C, H, W)，如果輸入是 (C, H, W) 則沒問題
                image = transforms.functional.rotate(image, angle)
            elif random_number == 3:  # crop
                # 對於批量圖像，需要確保 crop 參數對應每張圖像
                # 這裡假設圖像形狀為 (C, H, W)，因此我們對每個圖像進行裁剪
                # CIFAR-100 是 32x32，裁剪到 28x28，然後調整回 32x32
                i_crop, j_crop, h_crop, w_crop = transforms.RandomCrop.get_params(
                    image, output_size=(28, 28)
                )
                image = transforms.functional.crop(image, i_crop, j_crop, h_crop, w_crop)
                image = F.interpolate(image.unsqueeze(0), size=(32, 32), mode='bilinear', align_corners=False).squeeze(0)
            elif random_number == 4:  # random affine
                degrees = 15
                translate = (0.1, 0.1)
                scale = (0.9, 1.1)
                shear = (-10, 10)
                image = transforms.functional.affine(image, 
                                                    angle=random.uniform(-degrees, degrees),
                                                    translate=(int(random.uniform(-translate[0] * 32, translate[0] * 32)),
                                                                int(random.uniform(-translate[1] * 32, translate[1] * 32))),
                                                    scale=random.uniform(scale[0], scale[1]),
                                                    shear=(random.uniform(-shear[0], shear[0]), random.uniform(-shear[1], shear[1])))
            elif random_number == 5:
                image = image  # do not change the image
            augmented_images_batch.append(image)
        
        return torch.stack(augmented_images_batch) # 將單張圖像重新堆疊成批次

    def load_train_data(self, batch_size=None):
        """
        載入客戶端的訓練資料，並可選擇性地應用資料增強（僅針對 CIFAR-100）。

        參數：
            batch_size (int, optional): 批量大小，預設使用 self.batch_size

        返回：
            DataLoader: 訓練資料載入器
        """
        if batch_size is None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)

        #count label in train data
        if self.finishCount:
            for img, label in train_data:
                self.label_datasize[label] += 1
            self.finishCount = False
            #print("Client ", self.id, " finish count label")
        
        if self.finishCount == False:
            #print("Client ", self.id, " already finish count label")
            pass
        
        # 毒化邏輯
        if self.poisoned and self.attack_type == 'label_flipping':
            poisoned_data = []
            for img, label in train_data:
                if self.is_multilabel:
                    # CelebA 的毒化邏輯：隨機翻轉部分屬性標籤
                    labels = label.clone()  # 假設標籤為 (40,) 的張量
                    # 隨機選擇 10% 的屬性進行翻轉
                    flip_indices = np.random.choice(self.num_classes, int(self.num_classes * 0.1), replace=False)
                    for idx in flip_indices:
                        labels[idx] = 1 - labels[idx]  # 0->1 或 1->0
                    poisoned_data.append((img, labels))
                else:
                    # 其他資料集的毒化邏輯（單標籤翻轉）
                    # 這裡的毒化規則是硬編碼的，您可以根據需求修改
                    if label == 1:
                        label = torch.tensor(9)
                    elif label == 2:
                        label = torch.tensor(7)
                    elif label == 9:
                        label = torch.tensor(1)
                    poisoned_data.append((img, label))
            train_data = poisoned_data

        # 毒化邏輯
        if self.poisoned and self.attack_type == 'adaptive_label_flipping':
            poisoned_data = []
            for img, label in train_data:
                if self.is_multilabel:
                    # CelebA 的毒化邏輯：隨機翻轉部分屬性標籤
                    labels = label.clone()  # 假設標籤為 (40,) 的張量
                    # 隨機選擇 10% 的屬性進行翻轉
                    flip_indices = np.random.choice(self.num_classes, int(self.num_classes * 0.1), replace=False)
                    for idx in flip_indices:
                        labels[idx] = 1 - labels[idx]  # 0->1 或 1->0
                    poisoned_data.append((img, labels))
                else:
                    # find the less represented class with 10% label
                    attack_label_count = max(1, int(len(self.label_datasize) * 0.1))
                    # get attack label index base on attack_label_count from self.label_datasize
                    label_counts_with_indices = [(i, self.label_datasize[i]) for i in range(self.num_classes)]
                    # Sort by count (x[1]) in ascending order
                    sorted_labels_by_count = sorted(label_counts_with_indices, key=lambda x: x[1])
                    # Extract only the label indices of the least represented classes
                    attack_label_index = [item[0] for item in sorted_labels_by_count[:attack_label_count]]

                    #print("Client ", self.id, " ,attack_label_index: ", attack_label_index)
                    
                    for label in attack_label_index:
                        if self.static_flag <= 0:
                            # assign random label other than attack_label
                            random_label = random.randint(0, self.num_classes - 1)
                            #convert label to int
                            label_t = int(label)
                            while random_label == label_t:
                                random_label = random.randint(0, self.num_classes - 1)
                            try:
                                label = torch.tensor(random_label)
                            except:
                                pass
                        elif self.static_flag >= 1:
                            if len(self.static_target) < attack_label_count:
                                # assign random label other than attack_label
                                random_label = random.randint(0, self.num_classes - 1)
                                #convert label to int
                                label_t = int(label)
                                while random_label == label_t:
                                    random_label = random.randint(0, self.num_classes - 1)
                                self.static_target[label_t] = random_label
                            try:
                                label = torch.tensor(self.static_target[label_t])
                                #print("Client ", self.id, "original label:", label_t,  " ,attack label: ", self.static_target[label_t])
                            except:
                                pass
                    else:
                        #print("Client ", self.id, "original")
                        label = torch.tensor(label)  # keep the original label      
                    
                    poisoned_data.append((img, label))
            
            train_data = poisoned_data

        augmented_train_data = []
        # 僅對 CIFAR-100 應用本地資料增強
        if self.data_augmentation and self.dataset.lower() == 'cifar100_100_alpha01':
            #print("Local Data Enhancement...")
            for images, labels in train_data:
                augmented_images = self._apply_data_augmentation_cifar100(images.unsqueeze(0)).squeeze(0) # 傳入單張圖像，並處理維度
                augmented_train_data.append((augmented_images, labels))
                augmented_train_data.append((images, labels))  # original data
            train_loader = DataLoader(augmented_train_data, batch_size, drop_last=True, shuffle=True)  # 增強後需要 shuffle
        else:
            train_loader = DataLoader(train_data, batch_size, drop_last=True, shuffle=True) # 訓練資料通常需要 shuffle

        return train_loader

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
        測試模型性能，適配多標籤和單標籤資料集，計算準確率、AUC、F1 分數和 AUC-PR。

        返回：
            tuple:
                - 多標籤 (CelebA): (test_acc, test_num, auc, label_acc, f1, auc_pr)
                - 單標籤: (test_acc, test_num, auc, f1, auc_pr)
        """
        #testloaderfull = self.load_test_data()
        testloaderfull = self.testDataLoader
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        y_pred = []

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

                    y_prob.append(output.detach().cpu().numpy())  # logits 用於 AUC 和 AUC-PR
                    y_true.append(y.detach().cpu().numpy())
                    y_pred.append(preds.detach().cpu().numpy())  # 預測標籤用於 F1 分數

            if test_num == 0:
                return 0, 0, 0.0, np.zeros(self.num_classes), 0.0, 0.0

            # 計算有效屬性數並調整準確率
            valid_mask = (valid_attributes > 0).float()
            effective_num_attributes = torch.sum(valid_mask).item() or 1  # 避免除以零
            test_acc = test_acc / (test_num * effective_num_attributes)  # 平均準確率（基於有效屬性）
            label_acc = (correct_per_label / test_num).cpu().numpy() * valid_mask.cpu().numpy()  # 每個屬性的準確率

            y_prob = np.concatenate(y_prob, axis=0)
            y_true = np.concatenate(y_true, axis=0)
            y_pred = np.concatenate(y_pred, axis=0)

            # 計算 AUC（宏平均）
            auc = metrics.roc_auc_score(y_true, y_prob, average='macro') if y_true.size > 0 else 0.0

            # 計算 F1 分數（微平均）
            f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)

            # 計算 AUC-PR（微平均）
            precision, recall, _ = precision_recall_curve(y_true.ravel(), y_prob.ravel())
            auc_pr = sklearn_auc(recall, precision) if precision.size > 0 else 0.0

            return test_acc, test_num, auc, label_acc, f1, auc_pr

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
                    preds = torch.argmax(output, dim=1)  # 預測類別

                    test_acc += (torch.sum(preds == y)).item()
                    test_num += y.shape[0]

                    y_prob.append(output.detach().cpu().numpy())
                    y_pred.append(preds.detach().cpu().numpy())
                    nc = self.num_classes
                    if self.num_classes == 2:
                        nc += 1
                    lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                    if self.num_classes == 2:
                        lb = lb[:, :2]
                    y_true.append(lb)

            if test_num == 0:
                return 0, 0, 0.0, 0.0, 0.0

            y_prob = np.concatenate(y_prob, axis=0)
            y_true = np.concatenate(y_true, axis=0)
            y_pred = np.concatenate(y_pred, axis=0)

            # 計算 AUC（微平均）
            auc = metrics.roc_auc_score(y_true, y_prob, average='micro') if y_true.size > 0 else 0.0

            # 計算 F1 分數（加權平均）
            f1 = f1_score(y_true.argmax(axis=1), y_pred, average='weighted', zero_division=0)

            # 計算 AUC-PR（各類別平均）
            auc_pr_score = 0.0
            for i in range(self.num_classes):
                precision, recall, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
                auc_pr_score += sklearn_auc(recall, precision) if precision.size > 0 else 0.0
            auc_pr_score /= self.num_classes if self.num_classes > 0 else 1

            return test_acc, test_num, auc, f1, auc_pr_score

    def train_metrics(self):
        """
        計算訓練損失。

        返回：
            tuple: (總損失, 訓練樣本數)
        """
        #trainloader = self.load_train_data()
        trainloader = self.trainDataLoader
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
    
    def set_current_index(self, index):
        self.current_index = index