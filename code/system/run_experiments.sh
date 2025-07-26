#!/bin/bash

#echo "Starting experiment with poisoned_ratio 0 mobileNetV2_new without dtat enhancement"
#python main.py -data Cifar100_100_alpha01 -nc 100 -nb 100 -pr 0 -ls 1 -gr 200 -dynamic 1 -at adaptive_label_flipping -eid test00

#echo "Starting experiment with poisoned_ratio 0.0 mobileNetV2_new without dtat enhancement"
#python main.py -data Cifar100_100_alpha01 -nc 100 -nb 100 -pr 0 -ls 1 -gr 200 -dynamic 1 -at adaptive_label_flipping -eid test01

# Clean function to reset memory (especially GPU) between runs
#clean_memory() {
    #echo "Cleaning CPU page cache..."
    #sync && sudo bash -c 'echo 3 > /proc/sys/vm/drop_caches'
    #sleep 2

    #echo "Cleaning GPU memory..."
    #if command -v nvidia-smi &> /dev/null; then
        # 強制釋放 zombie 程序占用的 GPU 記憶體（如有）
        #PIDS=$(nvidia-smi | grep python | awk '{print $5}')
        #for pid in $PIDS; do
            #echo "Killing GPU process $pid"
            #kill -9 $pid
        #done
        #sleep 3
    #fi

    #echo "Memory cleanup complete."
#}

echo "Starting experiment with poisoned_ratio 0 mobileNetV2_new with dtat enhancement"
python main.py -data Cifar100_100_alpha01 -nc 100 -nb 100 -pr 0.1 -ls 10 -gr 100 -dynamic -1 -de 1 -at adaptive_label_flipping -eid test01

echo "Starting experiment with poisoned_ratio 0 mobileNetV2_new with dtat enhancement"
python main.py -data Cifar100_100_alpha01 -nc 100 -nb 100 -pr 0.1 -ls 5 -gr 100 -dynamic -1 -de 1 -at adaptive_label_flipping -eid test02

echo "Starting experiment with poisoned_ratio 0 mobileNetV2_new with dtat enhancement"
python main.py -data Cifar100_100_alpha01 -nc 100 -nb 100 -pr 0.1 -ls 3 -gr 100 -dynamic -1 -de 1 -at adaptive_label_flipping -eid test03

echo "Starting experiment with poisoned_ratio 0 mobileNetV2_new with dtat enhancement"
python main.py -data Cifar100_100_alpha01 -nc 100 -nb 100 -pr 0.1 -ls 1 -gr 100 -dynamic -1 -de 1 -at adaptive_label_flipping -eid test04

echo "Starting experiment with poisoned_ratio 0 mobileNetV2_new with dtat enhancement"
python main.py -data Cifar100_100_alpha01 -nc 100 -nb 100 -pr 0.1 -ls 5 -gr 100 -dynamic 1 -de 1 -at adaptive_label_flipping -eid test05

echo "Starting experiment with poisoned_ratio 0 mobileNetV2_new with dtat enhancement"
python main.py -data Cifar100_100_alpha01 -nc 100 -nb 100 -pr 0.2 -ls 10 -gr 100 -dynamic -1 -de 1 -at adaptive_label_flipping -eid test06

echo "Starting experiment with poisoned_ratio 0 mobileNetV2_new with dtat enhancement"
python main.py -data Cifar100_100_alpha01 -nc 100 -nb 100 -pr 0.2 -ls 5 -gr 100 -dynamic -1 -de 1 -at adaptive_label_flipping -eid test07

echo "Starting experiment with poisoned_ratio 0 mobileNetV2_new with dtat enhancement"
python main.py -data Cifar100_100_alpha01 -nc 100 -nb 100 -pr 0.2 -ls 3 -gr 100 -dynamic -1 -de 1 -at adaptive_label_flipping -eid test08

echo "Starting experiment with poisoned_ratio 0 mobileNetV2_new with dtat enhancement"
python main.py -data Cifar100_100_alpha01 -nc 100 -nb 100 -pr 0.2 -ls 1 -gr 100 -dynamic -1 -de 1 -at adaptive_label_flipping -eid test09

echo "Starting experiment with poisoned_ratio 0 mobileNetV2_new with dtat enhancement"
python main.py -data Cifar100_100_alpha01 -nc 100 -nb 100 -pr 0.2 -ls 5 -gr 100 -dynamic 1 -de 1 -at adaptive_label_flipping -eid test10

echo "Starting experiment with poisoned_ratio 0 mobileNetV2_new with dtat enhancement"
python main.py -data Cifar100_100_alpha01 -nc 100 -nb 100 -pr 0.3 -ls 10 -gr 100 -dynamic -1 -de 1 -at adaptive_label_flipping -eid test11

echo "Starting experiment with poisoned_ratio 0 mobileNetV2_new with dtat enhancement"
python main.py -data Cifar100_100_alpha01 -nc 100 -nb 100 -pr 0.3 -ls 5 -gr 100 -dynamic -1 -de 1 -at adaptive_label_flipping -eid test12

echo "Starting experiment with poisoned_ratio 0 mobileNetV2_new with dtat enhancement"
python main.py -data Cifar100_100_alpha01 -nc 100 -nb 100 -pr 0.3 -ls 3 -gr 100 -dynamic -1 -de 1 -at adaptive_label_flipping -eid test13

echo "Starting experiment with poisoned_ratio 0 mobileNetV2_new with dtat enhancement"
python main.py -data Cifar100_100_alpha01 -nc 100 -nb 100 -pr 0.3 -ls 1 -gr 100 -dynamic -1 -de 1 -at adaptive_label_flipping -eid test14

echo "Starting experiment with poisoned_ratio 0 mobileNetV2_new with dtat enhancement"
python main.py -data Cifar100_100_alpha01 -nc 100 -nb 100 -pr 0.3 -ls 5 -gr 100 -dynamic 1 -de 1 -at adaptive_label_flipping -eid test15

echo "Starting experiment with poisoned_ratio 0 mobileNetV2_new with dtat enhancement"
python main.py -data Cifar100_100_alpha01 -nc 100 -nb 100 -pr 0.4 -ls 10 -gr 100 -dynamic -1 -de 1 -at adaptive_label_flipping -eid test16

echo "Starting experiment with poisoned_ratio 0 mobileNetV2_new with dtat enhancement"
python main.py -data Cifar100_100_alpha01 -nc 100 -nb 100 -pr 0.4 -ls 5 -gr 100 -dynamic -1 -de 1 -at adaptive_label_flipping -eid test17

echo "Starting experiment with poisoned_ratio 0 mobileNetV2_new with dtat enhancement"
python main.py -data Cifar100_100_alpha01 -nc 100 -nb 100 -pr 0.4 -ls 3 -gr 100 -dynamic -1 -de 1 -at adaptive_label_flipping -eid test18

echo "Starting experiment with poisoned_ratio 0 mobileNetV2_new with dtat enhancement"
python main.py -data Cifar100_100_alpha01 -nc 100 -nb 100 -pr 0.4 -ls 1 -gr 100 -dynamic -1 -de 1 -at adaptive_label_flipping -eid test19

echo "Starting experiment with poisoned_ratio 0 mobileNetV2_new with dtat enhancement"
python main.py -data Cifar100_100_alpha01 -nc 100 -nb 100 -pr 0.4 -ls 5 -gr 100 -dynamic 1 -de 1 -at adaptive_label_flipping -eid test20


echo "All experiments finished!"