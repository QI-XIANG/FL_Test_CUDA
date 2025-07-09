#!/bin/bash

#echo "Starting experiment with poisoned_ratio 0 mobileNetV2_new without dtat enhancement"
#python main.py -data Cifar100_100_alpha01 -nc 100 -nb 100 -pr 0 -ls 1 -gr 200 -dynamic 1 -at adaptive_label_flipping -eid test00

#echo "Starting experiment with poisoned_ratio 0.0 mobileNetV2_new without dtat enhancement"
#python main.py -data Cifar100_100_alpha01 -nc 100 -nb 100 -pr 0 -ls 1 -gr 200 -dynamic 1 -at adaptive_label_flipping -eid test01

#echo "Starting experiment with poisoned_ratio 0 normalCNN without dtat enhancement"
#python main.py -data CINIC10_300_alpha01 -nc 300 -nb 10 -pr 0 -ls 1 -gr 200 -dynamic 1 -at adaptive_label_flipping -eid test01

#echo "Starting experiment with poisoned_ratio 0.1 normalCNN without dtat enhancement"
#python main.py -data CINIC10_300_alpha01 -nc 300 -nb 10 -pr 0.1 -ls 1 -gr 200 -dynamic 1 -at adaptive_label_flipping -eid test02

#echo "Starting experiment with poisoned_ratio 0.2 normalCNN without dtat enhancement"
#python main.py -data CINIC10_300_alpha01 -nc 300 -nb 10 -pr 0.2 -ls 1 -gr 200 -dynamic 1 -at adaptive_label_flipping -eid test03

#echo "Starting experiment with poisoned_ratio 0.4 normalCNN without dtat enhancement"
#python main.py -data CINIC10_300_alpha01 -nc 300 -nb 10 -pr 0.4 -ls 1 -gr 200 -dynamic 1 -at adaptive_label_flipping -eid test04

echo "Starting experiment with poisoned_ratio 0 MobileNetV2 without dtat enhancement"

python main.py -data CINIC10_100_alpha01 -nc 100 -nb 10 -pr 0 -ls 1 -gr 50 -dynamic -1 -at adaptive_label_flipping -eid test01

python main.py -data CINIC10_100_alpha01 -nc 100 -nb 10 -pr 0 -ls 3 -gr 50 -dynamic -1 -at adaptive_label_flipping -eid test02

python main.py -data CINIC10_100_alpha01 -nc 100 -nb 10 -pr 0 -ls 5 -gr 50 -dynamic -1 -at adaptive_label_flipping -eid test03

python main.py -data CINIC10_100_alpha01 -nc 100 -nb 10 -pr 0 -ls 10 -gr 50 -dynamic -1 -at adaptive_label_flipping -eid test04

python main.py -data CINIC10_100_alpha01 -nc 100 -nb 10 -pr 0 -ls 5 -gr 50 -dynamic 1 -at adaptive_label_flipping -eid test05

echo "All experiments finished!"