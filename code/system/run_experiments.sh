#!/bin/bash

echo "Starting experiment with poisoned_ratio 0 mobileNetV2_new with dtat enhancement"
python main.py -data SVHN100_alpha01 -nc 100 -nb 10 -pr 0.4 -ls 5 -gr 100 -algo FedRSVDUCBTE -sca RSVDUCBTE -dynamic -1 -de -1 -at adaptive_label_flipping -eid test525_svhn
echo "Starting experiment with poisoned_ratio 0 mobileNetV2_new with dtat enhancement"
python main.py -data SVHN100_alpha01 -nc 100 -nb 10 -pr 0.3 -ls 5 -gr 100 -algo FedRSVDUCBTE -sca RSVDUCBTE -dynamic -1 -de -1 -at adaptive_label_flipping -eid test526_svhn
echo "Starting experiment with poisoned_ratio 0 mobileNetV2_new with dtat enhancement"
python main.py -data SVHN100_alpha01 -nc 100 -nb 10 -pr 0.2 -ls 5 -gr 100 -algo FedRSVDUCBTE -sca RSVDUCBTE -dynamic -1 -de -1 -at adaptive_label_flipping -eid test527_svhn
echo "Starting experiment with poisoned_ratio 0 mobileNetV2_new with dtat enhancement"
python main.py -data SVHN100_alpha01 -nc 100 -nb 10 -pr 0.1 -ls 5 -gr 100 -algo FedRSVDUCBTE -sca RSVDUCBTE -dynamic -1 -de -1 -at adaptive_label_flipping -eid test528_svhn
echo "Starting experiment with poisoned_ratio 0 mobileNetV2_new with dtat enhancement"
python main.py -data SVHN100_alpha01 -nc 100 -nb 10 -pr 0.0 -ls 5 -gr 100 -algo FedRSVDUCBTE -sca RSVDUCBTE -dynamic -1 -de -1 -at adaptive_label_flipping -eid test529_svhn

echo "All experiments finished!"