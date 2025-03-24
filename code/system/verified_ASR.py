import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import Sequential, Conv2d, GroupNorm, ReLU, MaxPool2d, Linear
from flcore.trainmodel.models import FedAvgCNN

# 加入必要的全域變數到安全全域名單
torch.serialization.add_safe_globals([Sequential, Conv2d, GroupNorm, ReLU, MaxPool2d, Linear, FedAvgCNN, set])

# **自定義數據集**
class VerifiedDataset(Dataset):
    def __init__(self, verified_folder, transform=None):
        self.verified_folder = verified_folder
        self.files = [f for f in os.listdir(verified_folder) if f.endswith('.npy')]
        self.transform = transform  # 可選的轉換 (resize, normalization)

        self.data = []
        self.labels = []

        for file in self.files:
            file_path = os.path.join(verified_folder, file)
            loaded_data = np.load(file_path, allow_pickle=True).item()
            
            images = loaded_data['x'].astype(np.uint8)  # **確保是 uint8 格式的 numpy.ndarray**
            labels = loaded_data['y']

            self.data.extend(images)
            self.labels.extend(labels)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]  # **numpy.ndarray (28, 28)** 確保沒有額外維度
        label = self.labels[idx]

        # 確保圖像形狀為 (28, 28)
        if image.shape != (28, 28):
            image = np.squeeze(image)  # 去除多餘的維度

        # 將圖像形狀調整為 (1, 28, 28)
        image = image[np.newaxis, :, :]  # 新增一個維度，變為 (1, 28, 28)

        # 檢查圖像形狀，應該是 (1, 28, 28)
        assert image.shape == (1, 28, 28), f"錯誤: image.shape = {image.shape}, 預期為 (1, 28, 28)"

        if self.transform:
            image = self.transform(image)  # **ToTensor() 會自動轉換為 (1, 28, 28)**

        return image, torch.tensor(label, dtype=torch.long)

# **定義圖像轉換**
transform = transforms.Compose([
    transforms.ToTensor(),  # **確保格式為 (1, 28, 28)**
    transforms.Normalize([0.5], [0.5])  # 與訓練數據一致的歸一化
])

# **Collate Function 確保 batch 維度正確**
def collate_fn(batch):
    images, labels = zip(*batch)

    # **確保所有圖片都是 [batch_size, 1, 28, 28]**
    images = torch.stack(images, dim=0)  # (batch_size, 1, 28, 28)
    labels = torch.tensor(labels, dtype=torch.long)

    return images, labels

# **載入模型**
model_path = './models/fmnist40_Tadaptive/FedAvg_server.pt'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型權重檔案 {model_path} 不存在，請檢查路徑。")

model = torch.load(model_path, weights_only=True)
model.eval()

# **準備數據加載器**
verified_folder = '../../dataset/fmnist40_Tadaptive/verified'
if not os.path.exists(verified_folder):
    raise FileNotFoundError(f"驗證資料夾 {verified_folder} 不存在，請確認已產生攻擊數據。")

verified_dataset = VerifiedDataset(verified_folder, transform=transform)
verified_loader = DataLoader(verified_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# **開始攻擊成功率計算**
attack_success_count = 0
total_count = 0

with torch.no_grad():
    for images, labels in verified_loader:
        print(f"Model input shape: {images.shape}")  # 檢查輸入形狀
        output = model(images)  # **正確輸入模型**
        _, predicted_classes = torch.max(output, 1)
        
        # 計算攻擊成功次數
        target_label = 1  # 假設攻擊者希望模型將毒化數據分類為 "1"
        attack_success_count += (predicted_classes == target_label).sum().item()
        total_count += len(labels)

# **計算攻擊成功率**
if total_count == 0:
    print("未找到任何驗證數據，請檢查 verified 資料夾是否包含 .npy 檔案。")
else:
    attack_success_rate = 100 * attack_success_count / total_count
    print(f"Total samples: {total_count}")
    print(f"Successfully attacked samples: {attack_success_count}")
    print(f"Attack success rate: {attack_success_rate:.2f}%")