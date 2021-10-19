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
from parameters import *
import main





device = torch.device(DEVICE)
net = model.CNNs()
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)

train_loss_value=[]      #trainingのlossを保持するlist
train_acc_value=[]       #trainingのaccuracyを保持するlist







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
            sampler=train_sampler, num_workers=0, drop_last=True
        )

        val_dataloader = torch.utils.data.DataLoader(
            trainval_dataset, batch_size=TRAIN_BATCH_SIZE,
            sampler=val_sampler, num_workers=0, drop_last=True
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
            for (inputs, labels) in train_dataloader:

                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()


            sum_loss = 0.0          #lossの合計
            sum_correct = 0         #正解率の合計
            sum_total = 0           #dataの数の合計



            # validation

            for (inputs, labels) in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                sum_loss += loss.item()                            #lossを足していく
                _, predicted = outputs.max(1)                      #出力の最大値の添字(予想位置)を取得
                sum_total += labels.size(0)                        #labelの数を足していくことでデータの総和を取る
                sum_correct += (predicted == labels).sum().item()  #予想位置と実際の正解を比べ,正解している数だけ足す






            with open(path, 'a') as f:

                print("val mean loss={}, accuracy={}".format(
                    sum_loss*TRAIN_BATCH_SIZE/len(train_dataloader.dataset), float(sum_correct/sum_total)), file=f)  #lossとaccuracy出力
                train_loss_value.append(sum_loss*TRAIN_BATCH_SIZE/len(train_dataloader.dataset))  #traindataのlossをグラフ描画のためにlistに保持
                train_acc_value.append(float(sum_correct/sum_total))   #traindataのaccuracyをグラフ描画のためにlistに保持


    model_path = 'model.pth'
    torch.save(net.state_dict(), model_path)






if __name__ == "__main__":
    learning()
    main.main()
    main.syn_image()
