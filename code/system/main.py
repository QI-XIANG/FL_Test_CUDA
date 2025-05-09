#!/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging

from flcore.servers.serveravg import FedAvg
# from flcore.servers.serverCollectData import CollectData
from flcore.servers.serverdqn import FedDQN
from flcore.servers.serverthompson import FedThompson
from flcore.servers.serverkrum import FedKrum
from flcore.servers.servertrimmed import FedTrimmed
from flcore.servers.myserver import FedUCBN
from flcore.servers.serverbulyan import FedBulyan
from flcore.servers.serverbulyan_advanced import FedBulyanAdvanced
from flcore.servers.serverbulyanrobust import FedRobustBulyan
from flcore.servers.serverbulyancosinerobust import FedCosineRobustBulyan
from flcore.servers.serverbulyanepsilondecay import FedEpsilonDecayBulyan
from flcore.servers.serverfedadaptivetrimmedbulyan import RobustFedBulyan
from flcore.servers.serverfedadaptivetrimmedbulyanR import RobustFedBulyanR
from flcore.servers.serverfedadaptivetrimmedbulyanRR import RobustFedBulyanRR
from flcore.servers.serverfedadaptivetrimmedbulyanRRR import RobustFedBulyanRRR
from flcore.servers.serverfedadaptivetrimmedbulyanRRRR import RobustFedBulyanRRRR
from flcore.servers.serverARFedAvg import AdaptiveRobustFedAvg
from flcore.servers.serverARFedAvgR import AdaptiveRobustFedAvgR

from flcore.trainmodel.models import *

from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.resnetCifar100 import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *

from utils.result_utils import average_data
from utils.mem_utils import MemReporter


logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)

# hyper-params for Text tasks
vocab_size = 98635
max_len=200
emb_dim=32

def run(args):

    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "mlr": # convex
            if "mnist" in args.dataset:
                args.model = Mclr_Logistic(1*28*28, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = Mclr_Logistic(3*32*32, num_classes=args.num_classes).to(args.device)
            else:
                args.model = Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)

        elif model_str == "cnn": # non-convex
            if "mnist20" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif "fmnist600" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif "Cifar10_20" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "Cifar100_100" in args.dataset:
                args.model = FedProtoCifar100().to(args.device)
            elif "omniglot" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
                # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
            elif "Digit5" in args.dataset:
                args.model = Digit5CNN().to(args.device)
            elif "CelebA" in args.dataset:
                args.model = CelebAMultiLabelCNN().to(args.device)
            elif "GTSRB20" in args.dataset:
                args.model = FedAvgCNN3(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "SVHN20" in args.dataset:
                args.model = FedAvgCNN3(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)
    

        else:
            raise NotImplementedError

        print(args.model)

        # select algorithm
        if args.algorithm == "FedAvg":
            server = FedAvg(args, i)

        # elif args.algorithm == "FedDqn":
        #     server = FedDQN(args, i, agent)
        elif args.algorithm == "FedThompson":
            server = FedThompson(args, i)

        elif args.algorithm == "FedKrum":
            server = FedKrum(args, i)

        elif args.algorithm == "FedTrimmed":
            server = FedTrimmed(args, i)
        
        elif args.algorithm == "FedUCBN":
            server = FedUCBN(args, i)

        elif args.algorithm == "FedBulyan":
            server = FedBulyan(args, i)
        
        elif args.algorithm == "FedBulyanRobust":
            server = FedRobustBulyan(args, i)

        elif args.algorithm == "FedBulyanCosineRobust":
            server = FedCosineRobustBulyan(args, i)
        
        elif args.algorithm == "FedEpsilonDecayBulyan":
            server = FedEpsilonDecayBulyan(args, i)
        
        elif args.algorithm == "FedBulyanAdvanced":
            server = FedBulyanAdvanced(args, i)
        
        elif args.algorithm == "FedRFB":
            server = RobustFedBulyan(args, i)
        
        elif args.algorithm == "FedRFBR":
           server = RobustFedBulyanR(args, i)
        
        elif args.algorithm == "FedRFBRR":
            server = RobustFedBulyanRR(args, i)
        
        elif args.algorithm == "FedRFBRRR":
            server = RobustFedBulyanRRR(args, i)

        elif args.algorithm == "FedRFBRRRR":
            server = RobustFedBulyanRRRR(args, i)

        elif args.algorithm == "ARFedAvg":
            server = AdaptiveRobustFedAvg(args, i)

        elif args.algorithm == "ARFedAvgR":
            server = AdaptiveRobustFedAvgR(args, i)

        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time()-start)
        
        # torch.save(agent, 'dqn_model_formal.pt')
        print("model saved")

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    

    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="mnist")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=50)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=0.5,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=5,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-dp', "--privacy", type=bool, default=False,
                        help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")


    #select client
    parser.add_argument('-sca', "--select_clients_algorithm", type=str, default = "Random")

    parser.add_argument('-pr', "--poisoned_ratio", type=float, default = 0.4)
    parser.add_argument('-wo', "--weight_option", type=str, default = "same")
    parser.add_argument('-rs', "--random_seed", type=int, default=309)


    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Clients randomly join: {}".format(args.random_join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Client select regarding time: {}".format(args.time_select))
    if args.time_select:
        print("Time threthold: {}".format(args.time_threthold))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model))
    print("Using device: {}".format(args.device))
    print("Using DP: {}".format(args.privacy))
    if args.privacy:
        print("Sigma for DP: {}".format(args.dp_sigma))
    print("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("DLG attack: {}".format(args.dlg_eval))
    if args.dlg_eval:
        print("DLG attack round gap: {}".format(args.dlg_gap))
    print("=" * 50)

    run(args)

    
    print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")
