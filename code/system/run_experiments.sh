#!/bin/bash

#echo "Starting experiment with poisoned_ratio 0 FedAvgCNN with dtat enhancement"
#python main.py -data Cifar100_100_alpha01 -nc 100 -nb 100 -algo FedRSVDUCBTE -sca RSVDUCBTE -pr 0.4 -ls 5 -gr 100 -dynamic -1 -de 1 -at adaptive_label_flipping -eid test520

#echo "Starting experiment with poisoned_ratio 0.4 FedAvgCNN with dtat enhancement"
#python main.py -data Cifar100_100_alpha01 -nc 100 -nb 100 -pr 0.4 -ls 5 -gr 100 -sca UCB -algo FedUCBN -dynamic -1 -de 1 -at adaptive_label_flipping -eid test521

echo "Starting experiment with poisoned_ratio 0.4 FedAvgCNN_V2 without dtat enhancement"
python main.py -data Cifar100_100_alpha01 -nc 100 -nb 100 -algo FedRSVDUCBTE -sca RSVDUCBTE -pr 0.4 -ls 5 -gr 150 -dynamic -1 -de -1 -at adaptive_label_flipping -eid test01_RSVD

echo "Starting experiment with poisoned_ratio 0.4 FedAvgCNN_V2 without dtat enhancement"
python main.py -data Cifar100_100_alpha01 -nc 100 -nb 100 -pr 0.4 -ls 5 -gr 100 -sca UCB -algo FedUCBN -dynamic -1 -de -1 -at adaptive_label_flipping -eid test02_UCB

echo "Starting experiment with poisoned_ratio 0.4 FedAvgCNN_V2 without dtat enhancement"
python main.py -data Cifar100_100_alpha01 -nc 100 -nb 100 -algo FedTrimmed -pr 0.4 -ls 5 -gr 100 -dynamic -1 -de -1 -at adaptive_label_flipping -eid test03_Trimmed

echo "Starting experiment with poisoned_ratio 0.4 FedAvgCNN_V2 without dtat enhancement"
python main.py -data Cifar100_100_alpha01 -nc 100 -nb 100 -algo FedKrum -pr 0.4 -ls 5 -gr 100 -dynamic -1 -de -1 -at adaptive_label_flipping -eid test04_Krum

echo "Starting experiment with poisoned_ratio 0.4 FedAvgCNN_V2 without dtat enhancement"
python main.py -data Cifar100_100_alpha01 -nc 100 -nb 100 -algo FedThompson -pr 0.4 -ls 5 -gr 100 -dynamic -1 -de -1 -at adaptive_label_flipping -eid test05_Thompson

echo "All experiments finished!"