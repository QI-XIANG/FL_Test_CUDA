import numpy as np
import os
import sys
import random
import torch
from torch.utils.data import Dataset
from collections import Counter
from sklearn.model_selection import train_test_split
import requests
import ujson
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = f"Shakespeare_{num_clients}/"
train_size = 0.8
alpha = 0.1
seq_length = 80

def download_shakespeare(dir_path):
    data_url = 'https://raw.githubusercontent.com/teropa/nlp/master/resources/corpora/gutenberg/shakespeare-hamlet.txt'
    raw_text_file = os.path.join(dir_path, 'shakespeare.txt')

    if not os.path.exists(raw_text_file):
        os.makedirs(dir_path, exist_ok=True)
        print(f"Downloading Shakespeare dataset from {data_url} to {raw_text_file}...")
        response = requests.get(data_url, stream=True)
        if response.status_code != 200:
            raise Exception(f"Failed to download data, status code: {response.status_code}")
        total_size = int(response.headers.get('content-length', 0))
        t = tqdm(total=total_size, unit='i', unit_scale=True)
        with open(raw_text_file, 'wb') as f:
            for data in response.iter_content(chunk_size=1024):
                t.update(len(data))
                f.write(data)
        t.close()
        print("Download complete.")
    return raw_text_file

def create_vocabulary(text, vocab_size=8000):
    print("Creating vocabulary...")
    words = text.lower().replace('\n', ' ').split()
    word_counts = Counter(words)
    common_words = [word[0] for word in word_counts.most_common(vocab_size - 1)]
    vocabulary = ['<unk>'] + common_words
    word_to_index = {word: index for index, word in enumerate(tqdm(vocabulary, desc="Building word to index"))}
    index_to_word = {index: word for index, word in enumerate(tqdm(vocabulary, desc="Building index to word"))}
    print("Vocabulary created.")
    return vocabulary, word_to_index, index_to_word

def text_to_indices(text, word_to_index):
    print("Converting text to indices...")
    words = text.lower().replace('\n', ' ').split()
    indices = [word_to_index.get(word, word_to_index['<unk>']) for word in tqdm(words, desc="Mapping words to indices")]
    print("Text to indices complete.")
    return indices

def separate_data(indices, num_clients, niid=False, balance=False, partition='uniform', alpha=0.1):
    print(f"Separating data among {num_clients} clients (Non-IID: {niid}, Partition: {partition})...")
    client_data = [[] for _ in range(num_clients)]
    num_total_samples = len(indices)

    if partition == 'uniform':
        base_size = num_total_samples // num_clients
        remainder = num_total_samples % num_clients
        start = 0
        for i in tqdm(range(num_clients), desc="Distributing data"):
            size = base_size + (1 if i < remainder else 0)
            client_data[i] = indices[start:start + size]
            start += size
    elif partition == 'dirichlet':
        while True:
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = (np.array(proportions) * num_total_samples).astype(int)
            client_data = [[] for _ in range(num_clients)]
            current_index = 0
            possible = True
            for i in range(num_clients):
                size = proportions[i]
                client_data[i] = indices[current_index : current_index + size]
                current_index += size
                if size == 0 and num_total_samples > 0:
                    possible = False
                    break
            if possible or num_total_samples == 0:
                break
        remaining_indices = [idx for client_list in client_data for idx in client_list]
        client_data = [[] for _ in range(num_clients)]
        indices_per_client = len(remaining_indices) // num_clients
        remainder_final = len(remaining_indices) % num_clients
        start_final = 0
        for i in range(num_clients):
            end_final = start_final + indices_per_client + (1 if i < remainder_final else 0)
            client_data[i] = remaining_indices[start_final:end_final]
            start_final = end_final

    else:
        raise ValueError(f"Partition method '{partition}' not supported.")

    print("Data separation complete.")
    return client_data

class ShakespeareDatasetNPZ:
    def __init__(self, indices, seq_length):
        self.indices = np.array(indices, dtype=np.int64)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.indices) - self.seq_length

    def __getitem__(self, idx):
        input_seq = self.indices[idx:idx + self.seq_length]
        target_seq = self.indices[idx + 1:idx + self.seq_length + 1]
        return {'x': input_seq, 'y': target_seq}

def split_client_data_npz(client_data, seq_length, train_size=0.8):
    print("Splitting client data into train and test sets...")
    train_data = []
    test_data = []
    for i, data in enumerate(tqdm(client_data, desc="Splitting clients")):
        if len(data) > seq_length + 1:
            train_indices, test_indices = train_test_split(data, train_size=train_size, shuffle=True)
            train_data.append(ShakespeareDatasetNPZ(train_indices, seq_length))
            test_data.append(ShakespeareDatasetNPZ(test_indices, seq_length))
        else:
            train_data.append(None)
            test_data.append(None)
    print("Data splitting complete.")
    return train_data, test_data

def save_file_npz(dir_path, train_data, test_data, vocabulary, word_to_index, index_to_word, num_clients, seq_length):
    print("Saving processed data to .npz files...")
    config = {
        'num_clients': num_clients,
        'vocab_size': len(vocabulary),
        'seq_length': seq_length,
        'word_to_index': word_to_index,
        'index_to_word': index_to_word
    }
    os.makedirs(dir_path, exist_ok=True)
    train_path = os.path.join(dir_path, 'train')
    test_path = os.path.join(dir_path, 'test')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    with open(os.path.join(dir_path, 'config.json'), 'w') as f:
        ujson.dump(config, f)

    for i, data in enumerate(tqdm(train_data, desc="Saving train data")):
        if data is not None:
            all_data = [data[j] for j in range(len(data))]
            input_seqs = np.array([item['x'] for item in all_data])
            target_seqs = np.array([item['y'] for item in all_data])
            np.savez_compressed(os.path.join(train_path, f'{i}.npz'), x=input_seqs, y=target_seqs)
        else:
            np.savez_compressed(os.path.join(train_path, f'{i}.npz'), x=np.array([]), y=np.array([]))

    for i, data in enumerate(tqdm(test_data, desc="Saving test data")):
        if data is not None:
            all_data = [data[j] for j in range(len(data))]
            input_seqs = np.array([item['x'] for item in all_data])
            target_seqs = np.array([item['y'] for item in all_data])
            np.savez_compressed(os.path.join(test_path, f'{i}.npz'), x=input_seqs, y=target_seqs)
        else:
            np.savez_compressed(os.path.join(test_path, f'{i}.npz'), x=np.array([]), y=np.array([]))

    print("Data saving complete.")

if __name__ == "__main__":
    niid = True if len(sys.argv) > 1 and sys.argv[1] == "noniid" else False
    partition_method = 'dirichlet' if niid else 'uniform'

    raw_text_file = download_shakespeare(dir_path)
    with open(raw_text_file, 'r', encoding='utf-8') as f:
        text = f.read()

    vocabulary, word_to_index, index_to_word = create_vocabulary(text)
    indices = text_to_indices(text, word_to_index)

    client_indices = separate_data(indices, num_clients, niid=niid, partition=partition_method, alpha=alpha)

    train_data_npz, test_data_npz = split_client_data_npz(client_indices, seq_length, train_size=train_size)
    save_file_npz(dir_path, train_data_npz, test_data_npz, vocabulary, word_to_index, index_to_word, num_clients, seq_length)

    print(f"\nShakespeare dataset processed and saved as .npz to {dir_path}")
    print(f"Number of clients: {num_clients}")
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Sequence length: {seq_length}")
    print(f"Non-IID partition: {niid} (method: {partition_method})")