import datetime
import os


import torch
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data.sampler import SubsetRandomSampler


import model
import dataset
import graph
from parameters import *
import graph




device = torch.device(DEVICE)
net = model.CNNs()
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)


train_loss_value=[]      #trainingのlossを保持するlist
train_acc_value=[]       #trainingのaccuracyを保持するlist
val_loss_value=[]      #valのlossを保持するlist
val_acc_value=[]       #valのaccuracyを保持するlist






def learning():
    # make result dir
    os.makedirs(RESULT_DIR_PATH, exist_ok=True)

    #load Dataset
    trainval_dataset = dataset.MyDataset(TRAIN_DATASET_PATH + "/train", (RESIZE[0], RESIZE[1]))    #画像のリサイズはいくらにするか？　これは学習とテストに影響を与える

    '''
    n_samples = len(trainval_dataset)
    train_size = int(n_samples * 0.8)   # ここの割合は要件等
    val_size = n_samples - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])
    '''



    ## cross val
    splits = KFold(n_splits=5, shuffle=True, random_state=26)   # random_stateの値は要検討
    for fold, (train_idx, val_idx) in enumerate(splits.split(trainval_dataset)):


        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)


        #load Dataloader
        train_dataloader = torch.utils.data.DataLoader(
            trainval_dataset, batch_size=TRAIN_BATCH_SIZE,
            sampler=train_sampler, num_workers=2, drop_last=True
        )

        val_dataloader = torch.utils.data.DataLoader(
            trainval_dataset, batch_size=TRAIN_BATCH_SIZE,
            sampler=val_sampler, num_workers=2, drop_last=True
        )





    #loop of epoch
        for epoch in range(1, EPOCH+1):
            dt_now = datetime.datetime.now()
            epoch_time = dt_now.strftime('%Y-%m-%d %H:%M:%S')
            path = RESULT_DIR_PATH + "/" + EXPT_NUMBER + '.log'
            with open(path, 'a') as f:
                print('Fold {}'.format(fold + 1), 'epoch', epoch, file=f)
                print(epoch_time, file=f)


            # training

            train_sum_loss = 0.0          #lossの合計
            train_sum_correct = 0         #正解率の合計
            train_sum_total = 0           #dataの数の合計

            for (inputs, labels) in train_dataloader:

                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_sum_loss += loss.item()
                _, predicted = outputs.max(1)
                train_sum_total += labels.size(0)
                train_sum_correct += (predicted == labels).sum().item()


            with open(path, 'a') as f:

                print("train mean loss={}, accuracy={}".format(
                    train_sum_loss*TRAIN_BATCH_SIZE/len(train_dataloader.dataset), float(train_sum_correct/train_sum_total)), file=f)  #lossとaccuracy出力 ここのグラフの出力を確認する！！！
                train_loss_value.append(train_sum_loss*TRAIN_BATCH_SIZE/len(train_dataloader.dataset))  #traindataのlossをグラフ描画のためにlistに保持
                train_acc_value.append(float(train_sum_correct/train_sum_total))   #traindataのaccuracyをグラフ描画のためにlistに保持




            val_sum_loss = 0.0          #lossの合計
            val_sum_correct = 0         #正解率の合計
            val_sum_total = 0           #dataの数の合計



            # validation

            for (inputs, labels) in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_sum_loss += loss.item()                            #lossを足していく
                _, predicted = outputs.max(1)                      #出力の最大値の添字(予想位置)を取得
                val_sum_total += labels.size(0)                        #labelの数を足していくことでデータの総和を取る
                val_sum_correct += (predicted == labels).sum().item()  #予想位置と実際の正解を比べ,正解している数だけ足す






            with open(path, 'a') as f:

                print("val mean loss={}, accuracy={}".format(
                    val_sum_loss*TRAIN_BATCH_SIZE/len(val_dataloader.dataset), float(val_sum_correct/val_sum_total)), file=f)  #lossとaccuracy出力
                val_loss_value.append(val_sum_loss*TRAIN_BATCH_SIZE/len(val_dataloader.dataset))  #traindataのlossをグラフ描画のためにlistに保持 open内に書かなくてよい
                val_acc_value.append(float(val_sum_correct/val_sum_total))   #traindataのaccuracyをグラフ描画のためにlistに保持


    model_path = RESULT_DIR_PATH + '/model.pth'
    torch.save(net.state_dict(), model_path)




if __name__ == "__main__":
    learning()
    graph.plot_loss_acc(train_loss_value, train_acc_value, val_loss_value, val_acc_value)
