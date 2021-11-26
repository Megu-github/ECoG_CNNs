import datetime
import os

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data.sampler import SubsetRandomSampler
from statistics import mean
from tqdm.notebook import tqdm


import model
import dataset
import graph
from parameters import *
import graph








def learning():

    # make result dir
    os.makedirs(RESULT_DIR_PATH, exist_ok=True)

    #load Dataset
    trainval_dataset = dataset.MyDataset(TRAIN_DATASET_PATH + "/train", (RESIZE[0], RESIZE[1]))    #画像のリサイズはいくらにするか？　これは学習とテストに影響を与える

    device = torch.device(DEVICE)

    criterion = nn.CrossEntropyLoss()



    ## cross val
    nets, accs, losses = [], [], []

    splits = KFold(n_splits=5, shuffle=True, random_state=26)   # random_stateの値は要検討
    for fold, (train_idx, val_idx) in enumerate(splits.split(trainval_dataset)):

        net = model.CNNs(p_dropout1=0.25, p_dropout2=0.5, use_Barch_Norm=False).to(device)
        optimizer = optim.SGD(net.parameters(),lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)   #Adams検討

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_dataloader = torch.utils.data.DataLoader(
            trainval_dataset, batch_size=TRAIN_BATCH_SIZE,
            sampler=train_sampler, num_workers=2, drop_last=True
        )
        val_dataloader = torch.utils.data.DataLoader(
            trainval_dataset, batch_size=TRAIN_BATCH_SIZE,
            sampler=val_sampler, num_workers=2, drop_last=True
        )

        net, loss, acc, history = fit(net, optimizer, criterion, EPOCH, train_dataloader, val_dataloader, device, fold)
        nets.append(net)
        losses.append(loss)
        accs.append(acc)

        #graph.plot_loss_acc(history[:,1], history[:,2], history[:,3], history[:,4], fold)
        graph.evaluate_history(history, fold)

        model_path = RESULT_DIR_PATH  + '/model' + str(fold+1) + '.pth'
        torch.save(net.state_dict(), model_path)

    file_path = RESULT_DIR_PATH + "/" + EXPT_NUMBER + '.log'
    with open(file_path, 'a') as f:
        print("oof loss: {:4f}".format(mean(losses)), file=f)
        print("oof acc: {:4f}".format(mean(accs)), file=f)



def fit(net, optimizer, criterion, EPOCH, train_dataloader, val_dataloader, device, fold):
    history = np.zeros((0,5))


    #loop of epoch
    for epoch in range(EPOCH):

        # training
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0
        train_sum_correct = 0
        train_sum_total = 0

        #訓練フェーズ
        net.train()
        used_datasize = 0

        for inputs, labels in tqdm(train_dataloader):
            used_datasize += len(labels)    # 要確認
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            predicted = torch.max(outputs, 1)[1]
            train_acc += (predicted == labels).sum()
            avg_train_loss = train_loss / used_datasize
            avg_train_acc = train_acc / used_datasize





        val_sum_loss = 0.0          #lossの合計
        val_sum_correct = 0         #正解率の合計
        val_sum_total = 0           #dataの数の合計



        # validation
        #予測フェーズ
        net.eval()
        used_datasize = 0

        for inputs, labels in val_dataloader:
            used_datasize += len(labels)
            inputs, labels = inputs.to(device), labels.to(device)

            # 予測計算
            outputs = net(inputs)

            # 損失計算
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # 予測値算出
            predicted = torch.max(outputs, 1)[1]

            # 正解件数算出
            val_acc += (predicted == labels).sum()

            # 損失と精度の計算
            avg_val_loss = val_loss / used_datasize
            avg_val_acc = val_acc / used_datasize


        dt_now = datetime.datetime.now()    #jstに設定しなおす。
        epoch_time = dt_now.strftime('%Y-%m-%d %H:%M:%S')
        file_path = RESULT_DIR_PATH + "/" + EXPT_NUMBER + '.log'

        with open(file_path, 'a') as f:
            '''
            print("train mean loss={}, accuracy={}".format(
                train_loss*TRAIN_BATCH_SIZE/len(train_dataloader.dataset), float(train_sum_correct/train_sum_total)), file=f)  #lossとaccuracy出力 ここのグラフの出力を確認する！！！
            train_loss_value.append(train_loss*TRAIN_BATCH_SIZE/len(train_dataloader.dataset))  #traindataのlossをグラフ描画のためにlistに保持
            train_acc_value.append(float(train_sum_correct/train_sum_total))   #traindataのaccuracyをグラフ描画のためにlistに保持

            print("val mean loss={}, accuracy={}".format(
                val_sum_loss*TRAIN_BATCH_SIZE/len(val_dataloader.dataset), float(val_sum_correct/val_sum_total)), file=f)  #lossとaccuracy出力
            val_loss_value.append(val_sum_loss*TRAIN_BATCH_SIZE/len(val_dataloader.dataset))  #traindataのlossをグラフ描画のためにlistに保持 open内に書かなくてよい
            val_acc_value.append(float(val_sum_correct/val_sum_total))   #traindataのaccuracyをグラフ描画のためにlistに保持

            '''


        with open(file_path, 'a') as f:
            print('Fold {}'.format(fold + 1), 'epoch', epoch, file=f)
            print(epoch_time, file=f)


            print (f'Epoch [{(epoch+1)}/{EPOCH}], loss: {avg_train_loss:.5f} acc: {avg_train_acc:.5f} val_loss: {avg_val_loss:.5f}, val_acc: {avg_val_acc:.5f}', file=f)
        item = np.array([epoch+1, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc])
        history = np.vstack((history, item))

    return net, avg_val_loss, avg_val_acc, history

if __name__ == "__main__":
    learning()
