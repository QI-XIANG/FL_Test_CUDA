import h5py
import numpy as np
import os


def average_data(algorithm="", dataset="", goal="", times=10, num_clients=10, expID=""):
    test_acc = get_all_results_for_one_algo(algorithm, dataset, goal, times, num_clients, expID)

    max_accurancy = []
    for i in range(times):
        max_accurancy.append(test_acc[i].max())

    print("std for best accurancy:", np.std(max_accurancy))
    print("mean for best accurancy:", np.mean(max_accurancy))


def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10, num_clients=10, expID=""):
    test_acc = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + algorithms_list[i] + "_" + goal + "_" + str(i)
        test_acc.append(np.array(read_data_then_delete(file_name, delete=False, num_clients=num_clients, expID=expID)))

    return test_acc


def read_data_then_delete(file_name, delete=False, num_clients=10, expID=""):
    file_path = f"../results/{num_clients}_{expID}/h5/" + file_name + ".h5"
    if not os.path.exists(file_path):
        file_path = "../results/" + "shakespeare_20_FedAvgLSTM_test_0" + ".h5"
    #print("File path in read_data: " + file_path)
    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))

    if delete:
        os.remove(file_path)
    try:
        print("Length: ", len(rs_test_acc))
    except TypeError:
        print("Caught TypeError: Cannot get length of rs_test_acc. Skipping length check.")
        rs_test_acc = [0, 0]
        pass  # Simply pass if a TypeError occurs

    return rs_test_acc