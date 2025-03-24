#2
# python main.py -data fmnist -m cnn -algo FedUCBN -gr 499 -did 0 -go cnn --num_clients 10 -lbs 32 -pr 0.4 -sca UCB
# python main.py -data fmnist -m cnn -algo FedKrum -gr 499 -did 0 -go cnn --num_clients 10 -lbs 32 -pr 0.4


#3
# python main.py -data fmnist -m cnn -algo FedUCBN -gr 499 -did 0 -go cnn --num_clients 10 -lbs 32 -pr 0.4
# python main.py -data fmnist -m cnn -algo FedUCBN -gr 499 -did 0 -go cnn --num_clients 10 -lbs 32 -pr 0.4 -sca UCB

# time

# python main.py -data fmnist -m cnn -algo FedAvg -gr 499 -did 0 -go cnn --num_clients 10 -lbs 32 -pr 0.4
# python main.py -data fmnist -m cnn -algo FedUCBN -gr 499 -did 0 -go cnn --num_clients 10 -lbs 32 -pr 0.4
# python main.py -data fmnist -m cnn -algo FedUCBN -gr 499 -did 0 -go cnn --num_clients 10 -lbs 32 -pr 0.4 -sca UCB
# python main.py -data fmnist -m cnn -algo FedKrum -gr 499 -did 0 -go cnn --num_clients 10 -lbs 32 -pr 0.4


# python main.py -data fmnist50 -m cnn -algo FedAvg -gr 499 -did 0 -go cnn --num_clients 50 -lbs 32 -pr 0.4
# python main.py -data fmnist50 -m cnn -algo FedUCBN -gr 499 -did 0 -go cnn --num_clients 50 -lbs 32 -pr 0.4
# python main.py -data fmnist50 -m cnn -algo FedUCBN -gr 499 -did 0 -go cnn --num_clients 50 -lbs 32 -pr 0.4 -sca UCB
# python main.py -data fmnist50 -m cnn -algo FedKrum -gr 499 -did 0 -go cnn --num_clients 50 -lbs 32 -pr 0.4


# python main.py -data fmnist100 -m cnn -algo FedAvg -gr 499 -did 0 -go cnn --num_clients 100 -lbs 32 -pr 0.4
# python main.py -data fmnist100 -m cnn -algo FedUCBN -gr 499 -did 0 -go cnn --num_clients 100 -lbs 32 -pr 0.4
# python main.py -data fmnist100 -m cnn -algo FedUCBN -gr 499 -did 0 -go cnn --num_clients 100 -lbs 32 -pr 0.4 -sca UCB
# python main.py -data fmnist100 -m cnn -algo FedKrum -gr 499 -did 0 -go cnn --num_clients 100 -lbs 32 -pr 0.4

# Trimmed
# python main.py -data fmnist -m cnn -algo FedTrimmed -gr 499 -did 0 -go cnn --num_clients 10 -lbs 32 -pr 0.4
# python main.py -data fmnist50 -m cnn -algo FedTrimmed -gr 499 -did 0 -go cnn --num_clients 50 -lbs 32 -pr 0.4
# python main.py -data fmnist100 -m cnn -algo FedTrimmed -gr 499 -did 0 -go cnn --num_clients 100 -lbs 32 -pr 0.4

#same
# python main.py -data fmnist -m cnn -algo FedUCBN -gr 499 -did 0 -go cnn --num_clients 10 -lbs 32 -pr 0.4 -sca UCB
# python main.py -data fmnist50 -m cnn -algo FedUCBN -gr 499 -did 0 -go cnn --num_clients 50 -lbs 32 -pr 0.4 -sca UCB
# python main.py -data fmnist100 -m cnn -algo FedUCBN -gr 499 -did 0 -go cnn --num_clients 100 -lbs 32 -pr 0.4 -sca UCB

#attack fedbn
# round=499
# for num_clients in 15
# do
#     python main.py -data fmnist"$num_clients" -m cnn -algo FedAvg -gr $round -did 0 -go cnn --num_clients $num_clients -lbs 32 -pr 0
#     python main.py -data fmnist"$num_clients" -m cnn -algo FedAvg -gr $round -did 0 -go cnn --num_clients $num_clients -lbs 32 -pr 0.4
#     python main.py -data fmnist"$num_clients" -m cnn -algo FedUCBN -gr $round -did 0 -go cnn --num_clients $num_clients -lbs 32 -pr 0.4 -wo contribution
#     python main.py -data fmnist"$num_clients" -m cnn -algo FedUCBN -gr $round -did 0 -go cnn --num_clients $num_clients -lbs 32 -pr 0.4 -sca UCB -wo contribution
#     python main.py -data fmnist"$num_clients" -m cnn -algo FedTrimmed -gr $round -did 0 -go cnn --num_clients $num_clients -lbs 32 -pr 0.4
# done


# round=499
# for num_clients in 10 20 30
# do
#     python main.py -data fmnist"$num_clients" -m cnn -algo FedUCBN -gr $round -did 0 -go cnn --num_clients $num_clients -lbs 32 -pr 0.4 -rs 309
#     python main.py -data fmnist"$num_clients" -m cnn -algo FedUCBN -gr $round -did 0 -go cnn --num_clients $num_clients -lbs 32 -pr 0.4 -sca UCB -rs 309
# done



# python main.py -data fmnist100 -m cnn -algo FedUCBN -gr 1 -did 0 -go cnn --num_clients 100 -lbs 32 -pr 0.4 -sca UCB -wo contribution
# for round in 1000 2000
# do
#     python main.py -data fmnist100 -m cnn -algo FedUCBN -gr $round -did 0 -go cnn --num_clients 100 -lbs 32 -pr 0.4 -wo contribution
#     python main.py -data fmnist100 -m cnn -algo FedUCBN -gr $round -did 0 -go cnn --num_clients 100 -lbs 32 -pr 0.4 -sca UCB -wo contribution
# done


round=499
for num_clients in 10 20 30
do
    for random_seed in 309 #1 2 3 4 5 6 7 8 9
    do
        # python main.py -data fmnist"$num_clients" -m cnn -algo FedAvg -gr $round -did 0 -go cnn --num_clients $num_clients -lbs 32 -pr 0 -rs $random_seed
        python main.py -data fmnist"$num_clients" -m cnn -algo FedAvg -gr $round -did 0 -go cnn --num_clients $num_clients -lbs 32 -pr 0.4 -rs $random_seed
        python main.py -data fmnist"$num_clients" -m cnn -algo FedUCBN -gr $round -did 0 -go cnn --num_clients $num_clients -lbs 32 -pr 0.4 -rs $random_seed
        python main.py -data fmnist"$num_clients" -m cnn -algo FedUCBN -gr $round -did 0 -go cnn --num_clients $num_clients -lbs 32 -pr 0.4 -sca UCB -rs $random_seed
        python main.py -data fmnist"$num_clients" -m cnn -algo FedTrimmed -gr $round -did 0 -go cnn --num_clients $num_clients -lbs 32 -pr 0.4 -rs $random_seed
    done
done


# for random_seed in 7 8 9
# do
#     python main.py -data fmnist30 -m cnn -algo FedUCBN -gr 499 -did 0 -go cnn --num_clients 30 -lbs 32 -pr 0.4 -sca UCB -rs $random_seed
# done
