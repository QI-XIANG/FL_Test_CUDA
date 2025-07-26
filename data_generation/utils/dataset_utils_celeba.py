import os
import ujson
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

batch_size = 10
train_size = 0.75
least_samples = batch_size / (1 - train_size)
alpha = 2.0

def check(config_path, train_path, test_path, num_clients, num_classes, niid=False, 
          balance=True, partition=None):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
           config['num_classes'] == num_classes and \
           config['non_iid'] == niid and \
           config['balance'] == balance and \
           config['partition'] == partition and \
           config['alpha'] == alpha and \
           config['batch_size'] == batch_size:
            print("\nDataset already generated.\n")
            return True

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False

def separate_data(data, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=2):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data
    N = len(dataset_label)
    idxs = np.arange(N)

    dataidx_map = {}

    if not niid:
        num_per_client = N // num_clients
        for client in range(num_clients):
            start_idx = client * num_per_client
            end_idx = (client + 1) * num_per_client if client < num_clients - 1 else N
            dataidx_map[client] = idxs[start_idx:end_idx]
    else:
        if partition == "dir":
            min_size = 0
            while min_size < least_samples:
                idx_batch = [[] for _ in range(num_clients)]
                for k in tqdm(range(num_classes), desc="Distributing Attributes (Dirichlet)"):
                    idx_k = np.where(dataset_label[:, k] == 1)[0]
                    if len(idx_k) == 0:
                        continue
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                    proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            for j in range(num_clients):
                dataidx_map[j] = idx_batch[j]
        else:
            num_per_client = N // num_clients
            for client in range(num_clients):
                start_idx = client * num_per_client
                end_idx = (client + 1) * num_per_client if client < num_clients - 1 else N
                dataidx_map[client] = idxs[start_idx:end_idx]

    for client in tqdm(range(num_clients), desc="Assigning Data"):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]
        for i in range(num_classes):
            count = int(np.sum(y[client][:, i]))
            if count > 0:
                statistic[client].append((i, count))

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Attributes: ", [i[0] for i in statistic[client]])
        print(f"\t\t Samples per attribute: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic

def split_data(X, y, train_size=0.8):
    train_data, test_data = [], []
    num_samples = {'train': [], 'test': []}

    for i in tqdm(range(len(y)), desc="Splitting Data"):
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_size, shuffle=True)
        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()
    del X, y
    return train_data, test_data

def save_file(config_path, train_path, test_path, train_data, test_data, num_clients, 
              num_classes, statistic, niid=False, balance=True, partition=None):
    config = {
        'num_clients': num_clients, 
        'num_classes': num_classes, 
        'non_iid': niid, 
        'balance': balance, 
        'partition': partition, 
        'Size of samples for labels in clients': statistic, 
        'alpha': alpha, 
        'batch_size': batch_size, 
    }

    print("Saving to disk.\n")

    for idx, train_dict in tqdm(enumerate(train_data), total=len(train_data), desc="Saving Train Data"):
        with open(f"{train_path}{idx}.npz", 'wb') as f:
            np.savez_compressed(f, **train_dict, pickle_kwargs={'protocol': 4})  # 使用協議 4

    for idx, test_dict in tqdm(enumerate(test_data), total=len(test_data), desc="Saving Test Data"):
        with open(f"{test_path}{idx}.npz", 'wb') as f:
            np.savez_compressed(f, **test_dict, pickle_kwargs={'protocol': 4})  # 使用協議 4

    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")