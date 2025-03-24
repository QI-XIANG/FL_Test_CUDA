import numpy as np
import os
import torch


def read_data(dataset, idx, is_train=True):
    """
    從指定路徑讀取客戶端的資料（訓練或測試），適配包含 'x' 和 'y' 鍵的 .npz 文件。
    
    參數：
        dataset (str): 資料集名稱
        idx (int): 客戶端編號
        is_train (bool): 是否為訓練資料，預設為 True
    
    返回：
        dict: 包含 'x'（特徵）和 'y'（標籤）的資料字典
    """
    # 根據訓練或測試設置文件路徑
    if is_train:
        data_dir = os.path.join('../../dataset', dataset, 'train/')
        file_path = os.path.join(data_dir, f"{idx}.npz")
    else:
        data_dir = os.path.join('../../dataset', dataset, 'test/')
        file_path = os.path.join(data_dir, f"{idx}.npz")

    # 檢查文件是否存在
    if not os.path.exists(file_path):
        print(f"警告: 文件 {file_path} 不存在，返回空字典")
        return {'x': [], 'y': []}

    # 讀取 .npz 文件
    try:
        with open(file_path, 'rb') as f:
            npz_data = np.load(f, allow_pickle=True)
            # 檢查並提取 'x' 和 'y' 鍵
            if 'x' in npz_data and 'y' in npz_data:
                data = {
                    'x': npz_data['x'],  # 圖像或特徵數據
                    'y': npz_data['y']   # 標籤數據
                }
            else:
                raise KeyError(f"文件中缺少 'x' 或 'y' 鍵，實際鍵名: {npz_data.files}")
            return data
    except Exception as e:
        print(f"錯誤: 讀取 {file_path} 失敗 ({e})，返回空字典")
        return {'x': [], 'y': []}


def read_client_data(dataset, idx, is_train=True):
    """
    讀取客戶端資料並轉換為 PyTorch 張量格式，支援 CelebA 多標籤分類及其他單標籤資料集。
    
    參數：
        dataset (str): 資料集名稱
        idx (int): 客戶端編號
        is_train (bool): 是否為訓練資料，預設為 True
    
    返回：
        list: 包含 (特徵, 標籤) 元組的資料列表
    """
    # 特殊資料集（文本）處理
    if dataset[:2] == "ag" or dataset[:2] == "SS":
        return read_client_data_text(dataset, idx, is_train)
    elif dataset[:2] == "sh":
        return read_client_data_shakespeare(dataset, idx)

    # 讀取原始資料
    train_data = read_data(dataset, idx, is_train)

    # 檢查資料是否為空
    if not train_data['x'].size or not train_data['y'].size:
        print(f"警告: 客戶端 {idx} 的資料為空，返回空列表")
        return []

    # 根據資料集類型處理資料
    if dataset.lower() == "celeba":
        # CelebA 多標籤處理
        X_data = torch.tensor(train_data['x'], dtype=torch.float32)  # 圖像資料轉為浮點數張量
        y_data = torch.tensor(train_data['y'], dtype=torch.float32)  # 多標籤屬性轉為浮點數張量 (0 或 1)
        data = [(x, y) for x, y in zip(X_data, y_data)]
    else:
        # 其他單標籤資料集處理
        X_data = torch.tensor(train_data['x'], dtype=torch.float32)  # 圖像資料轉為浮點數張量
        y_data = torch.tensor(train_data['y'], dtype=torch.int64)    # 單標籤轉為整數張量
        data = [(x, y) for x, y in zip(X_data, y_data)]

    return data


def read_client_data_text(dataset, idx, is_train=True):
    """
    讀取文本資料集（例如 ag、SS）的客戶端資料。
    
    參數：
        dataset (str): 資料集名稱
        idx (int): 客戶端編號
        is_train (bool): 是否為訓練資料，預設為 True
    
    返回：
        list: 包含 ((特徵, 長度), 標籤) 元組的資料列表
    """
    train_data = read_data(dataset, idx, is_train)

    if not train_data['x'].size or not train_data['y'].size:
        print(f"警告: 客戶端 {idx} 的資料為空，返回空列表")
        return []

    if is_train:
        X_train, X_train_lens = list(zip(*train_data['x']))
        y_train = train_data['y']

        X_train = torch.tensor(X_train, dtype=torch.int64)
        X_train_lens = torch.tensor(X_train_lens, dtype=torch.int64)
        y_train = torch.tensor(y_train, dtype=torch.int64)

        train_data = [((x, lens), y) for x, lens, y in zip(X_train, X_train_lens, y_train)]
        return train_data
    else:
        X_test, X_test_lens = list(zip(*train_data['x']))
        y_test = train_data['y']

        X_test = torch.tensor(X_test, dtype=torch.int64)
        X_test_lens = torch.tensor(X_test_lens, dtype=torch.int64)
        y_test = torch.tensor(y_test, dtype=torch.int64)

        test_data = [((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)]
        return test_data


def read_client_data_shakespeare(dataset, idx, is_train=True):
    """
    讀取 Shakespeare 資料集的客戶端資料。
    
    參數：
        dataset (str): 資料集名稱
        idx (int): 客戶端編號
        is_train (bool): 是否為訓練資料，預設為 True
    
    返回：
        list: 包含 (特徵, 標籤) 元組的資料列表
    """
    train_data = read_data(dataset, idx, is_train)

    if not train_data['x'].size or not train_data['y'].size:
        print(f"警告: 客戶端 {idx} 的資料為空，返回空列表")
        return []

    X_data = torch.tensor(train_data['x'], dtype=torch.int64)
    y_data = torch.tensor(train_data['y'], dtype=torch.int64)

    data = [(x, y) for x, y in zip(X_data, y_data)]
    return data


# 測試程式碼（可選）
if __name__ == "__main__":
    # 測試 CelebA 資料集
    data = read_client_data('CelebA', 0, is_train=False)
    print(f"數據長度: {len(data)}")
    if data:
        image, label = data[0]
        print(f"圖像形狀: {image.shape}, 標籤: {label}")