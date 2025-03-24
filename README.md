### 1.注意事項

1. 強烈建議建一個python的虛擬環境來跑實驗 (怕套件跟自己電腦套件衝突引發大爆炸的話)!
2. 我目前已經修正並新增學長在requirement.txt中缺少列出的套件
3. 學長的code/utils/data_utils.py存在路徑上的錯誤，這部分已修正
4. 修正`python main.py`無法執行的問題，原因是沒有去執行dataset/generate_mnist.py (我這邊使用的sys args是noniid balance dir)

### 2.使用說明

1. 建好並啟動python虛擬環境 (強烈建議!!!) (本儲存庫已包含虛擬環境和必要套件的安裝，可以直接啟動虛擬環境開始跑實驗!)
2. 在命令提示字元執行`pip install -r requirements.txt`
3. 安裝完套件後切換路徑到code/system
4. 在命令提示字元執行`python main.py` (沒輸入參數的情況下會按照預設值跑，詳細資訊請參考main.py)
5. 正常來說，命令提示字元的視窗內會開始跑2000輪的訓練 (會跑很久...我家裡的電腦比較舊會占用很多CPU資源，沒獨顯只有內顯QQ)

(建議只是測試不要跑滿2000輪，可以到main.py裡面改預設值!)

### 3.已知錯誤 (在其他電腦上)

1. OSError: [WinError 126] (暫時無解，我只知道是缺少torch需要的.dll檔，但不知道是哪個) 錯誤已解決!
2. 無法儲存實驗跑完結果 (有解但未解，問題出在路徑不存在) 錯誤已解決!
![](https://i.imgur.com/y7oxlVR.png)

### 4.看完部分程式碼後的建議

1. 可以新增early stopping功能

### 5.執行過程範例影片

[影片連結](https://www.youtube.com/embed/GGPfxRIfAWY?si=wSSsqsiAZZHvhKOK)

---

### 欲增加實驗
1. 要增加不同的資料及進行實驗，那資料集的部分可以參考以下的Generate data 和 Attack。因為資料集不同，所以有可能攻擊的手法就要再設計過。 本來是使用FashionMnist，這個只有10類，我是把label 1 換成 9。那如果是cifar 100的話，這樣應該是有100類，如果只換一類可能影響不大!?(不太確定)
2. 另外有關別的防禦手法，這個可能要去看論文並且實做了，那這部分應該就是要做在server端中。檔案在code/system/flcore/servers/。這樣就需要去繼承 serverbase.py，然後再修改一些聚合或防禦的手段。(這只是我個人的看法，看你們要怎麼實做進行)

### Generate data
1. dataset dir
2. modify client number in generate_fmnist.py
3. python generate_fmnist.py noniid - dir
4. dir -> 迪立克雷分布

dataset dir中有 cifar 10 跟 cifar 100，如果想要調整client的數量，那就要修改這裡頭檔案的client number，迪立克雷分布是為了滿足資料集是非獨立同分布的。

### Attack
這實驗室做label flip attack，是把數字1換成9(這攻擊還可以做延伸或替換)

code/system/flcore/servers/serverbase.py
在function select_poisoned_client 中，選取哪個client是會被汙染的

code/system/flcore/clients/clientbase.py
在function load_train_data 中，將惡意client的資料的1 label換成9

### run exp command
python main.py -data fmnist30 -m cnn -algo FedUCBN -gr 499 -did 0 -num_clients 30 -lbs 32 -pr 0.4 -sca UCB

- data : which dataset do you want to train
- m : which model used to trainging
- algo : server aggregate method
- gr : round of experiments (start from zero)
- num_clients : clients number in environment
- lbs : batch size
- pr : the ratio of poisoned clients in the environment
- sca : client select algorithm

or use exp.sh to run multiple exp

---

#### CUDA版本查詢

* 查cuda版本
    * `nvidia-smi`
* 找對應的PyTorch安裝
    * [PyTorch](https://pytorch.org/get-started/previous-versions/)

requirements.txt裡面的檔案適用於python3.12.0，並且預設安裝的PyTorch版本是不能用GPU訓練的，要自己去找顯卡對應的PyTorch安裝

正常使用顯卡訓練會比較快，可以參考下面的圖片

![](https://i.imgur.com/6uGcJMq.png)

* CPU
    * 50 global round
    * 20 clients
    * Cifar100 (Dir 0.1)

![](https://i.imgur.com/PkKJo9i.png)

* CPU
    * 50 global round
    * 20 clients
    * Cifar100 (Dir 1.0)

![](https://i.imgur.com/PovlhV2.png)

* GPU
    * 100 global round
    * 20 clients
    * Cifar100 (Dir 0.1)

![](https://i.imgur.com/HwTjndu.png)

* GPU
    * 100 global round
    * 20 clients
    * Cifar100 (Dir 1.0)

![](https://i.imgur.com/xX4DgnO.png)

---

#### 特別感謝原始碼提供者 : 鍾明翰 學長 [學長筆記連結](https://hackmd.io/XyJWVGecSRWu4jn0haT8mg)
#### 2024.10.14 編輯 by 棨翔 [Federated Learning 學長交接討論重點摘要](https://hackmd.io/@qixiang1009/BkubNnkj6)

