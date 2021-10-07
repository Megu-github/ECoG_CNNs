import datetime
import os

import torch
import torch.optim as optim
import torch.utils.data
import torch.nn as nn

import model
import dataset
from parameters import *



# 結果を保存するpathを生成
dirname = os.path.dirname(os.path.abspath(__file__))
result_dir_path = dirname + '/Result/' + TRAIN_EXPT_NUMBER



device = torch.device(DEVICE)
net = model.CNNs()
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)

train_loss_value=[]      #trainingのlossを保持するlist
train_acc_value=[]       #trainingのaccuracyを保持するlist


def learning():
    # make result dir
    os.makedirs(result_dir_path, exist_ok=True)

    #load Dataset
    trainval_dataset = dataset.MyDataset(TRAIN_DATASET_PATH + "/train", (RESIZE[0], RESIZE[1]))    #画像のリサイズはいくらにするか？　これは学習とテストに影響を与える
    n_samples = len(trainval_dataset)
    train_size = int(n_samples * 0.8)   # ここの割合は要件等
    val_size = n_samples - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])

    #load Dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
        num_workers=0, drop_last=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
        num_workers=0, drop_last=True
    )




    #loop of epoch


    for epoch in range(1, EPOCH+1):
        dt_now = datetime.datetime.now()
        epoch_time = dt_now.strftime('%Y-%m-%d %H:%M:%S')
        path = result_dir_path + "/" + TRAIN_EXPT_NUMBER + '.log'
        with open(path, 'a') as f:
            print('epoch', epoch, file=f)
            print(epoch_time, file=f)


        #training
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





        for (inputs, labels) in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            #labels = labels.view(-1, 1)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            sum_loss += loss.item()                            #lossを足していく
            _, predicted = outputs.max(1)                      #出力の最大値の添字(予想位置)を取得
            sum_total += labels.size(0)                        #labelの数を足していくことでデータの総和を取る
            sum_correct += (predicted == labels).sum().item()  #予想位置と実際の正解を比べ,正解している数だけ足す


        with open(path, 'a') as f:

            print("train mean loss={}, accuracy={}".format(
                sum_loss*TRAIN_BATCH_SIZE/len(train_dataloader.dataset), float(sum_correct/sum_total)), file=f)  #lossとaccuracy出力
            train_loss_value.append(sum_loss*TRAIN_BATCH_SIZE/len(train_dataloader.dataset))  #traindataのlossをグラフ描画のためにlistに保持
            train_acc_value.append(float(sum_correct/sum_total))   #traindataのaccuracyをグラフ描画のためにlistに保持


        model_path = 'model.pth'
        torch.save(net.state_dict(), model_path)






if __name__ == "__main__":
    learning()
