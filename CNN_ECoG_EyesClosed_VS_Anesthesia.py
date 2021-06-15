# -*- coding: utf-8 -*-
#https://ohke.hateblo.jp/entry/2019/12/28/230000


from pathlib import Path
import datetime
import csv
import os
import datetime

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn


# グローバル変数
BATCH_SIZE = 16
WEIGHT_DECAY = 0.005
LEARNING_RATE = 0.0001
EPOCH = 20
RESIZE = [224, 224]
DEVICE = "cuda" # サーバー上なら"cuda"
DATASET_PATH = '/home/megu/CNN_Dataset/MK1_expt.1' # セーバーにDATASETをコピーして、そのpathを書く
EXPT_NUMBER = 'MK1_expt.1'

# 結果を保存するpathを生成
dirname = os.path.dirname(os.path.abspath(__file__))
result_dir_path = dirname + '/Result/' + EXPT_NUMBER



# PytorchでのCNNのモデル作り
# モデルは次のサイトを参考にした　https://qiita.com/mathlive/items/8e1f9a8467fff8dfd03c
class CNNs(nn.Module):
    def __init__(self):
        super(CNNs, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = nn.Conv2d(3,16,3)
        self.conv2 = nn.Conv2d(16,32,3)

        self.fc1 = nn.Linear(32 * int(RESIZE[0]/4 - 1.5) * int(RESIZE[1]/4 - 1.5), 32*min(int(RESIZE[0]/4 - 1.5), int(RESIZE[1]/4 - 1.5)))
        self.fc2 = nn.Linear(32*min(int(RESIZE[0]/4 - 1.5), int(RESIZE[1]/4 - 1.5)), 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x




class MyDataset(data.Dataset):
    def __init__(self, dir_path, input_size):
        super().__init__()

        self.dir_path = dir_path
        self.input_size = input_size

        self.image_paths = [str(p) for p in Path(self.dir_path).glob("**/*.png")]
        self.len = len(self.image_paths)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        p = self.image_paths[index]

        # 入力
        image = Image.open(p)
        image = image.resize(self.input_size)
        image = np.array(image)
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).float()

        # ラベル (0: Eyes_Closed, 1: Anesthesia)
        label = p.split("/")[6]     #ここはpath名が変わると変更することになるので、いつかうまい具合に書き換える
        label = 1 if label == "Anesthesia" else 0

        return image, label

'''
device = torch.device(DEVICE)
net = CNNs()
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)


def log_observe(dataloader, name):
    sum_loss = 0.0
    sum_correct = 0
    sum_total = 0

    loss_value=[]   # lossを保持するlist
    acc_value=[]    # accを保持するlist


    for (inputs, labels) in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        sum_loss += loss.item()
        _, predicted = outputs.max(1)
        sum_total += labels.size(0)
        sum_correct += (predicted == labels).sum().item()


    print(name + " mean loss={}, accuracy={}".format(
        sum_loss*BATCH_SIZE/len(dataloader.dataset), float(sum_correct/sum_total)))
    loss_value.append(sum_loss*BATCH_SIZE/len(dataloader.dataset))
    acc_value.append(float(sum_correct/sum_total))

    return loss_value, acc_value
'''


#グラフ描写の関数
def plot_loss_acc(train_loss_value, test_loss_value, train_acc_value, test_acc_value):
    plt.figure(figsize=(6,6))      #グラフ描画用

    save_dir = result_dir_path


    #以下グラフ描画
    plt.plot(range(EPOCH), train_loss_value)
    plt.plot(range(EPOCH), test_loss_value, c='#00ff00')
    plt.xlim(0, EPOCH)
    plt.ylim(0, 2.5)
    plt.xlabel('EPOCH')
    plt.ylabel('LOSS')
    plt.legend(['train loss', 'test loss'])
    plt.title('loss')
    plt.savefig(os.path.join(save_dir, EXPT_NUMBER + "_loss_image_.png"))
    plt.clf()

    plt.plot(range(EPOCH), train_acc_value)
    plt.plot(range(EPOCH), test_acc_value, c='#00ff00')
    plt.xlim(0, EPOCH)
    plt.ylim(0, 1)
    plt.xlabel('EPOCH')
    plt.ylabel('ACCURACY')
    plt.legend(['train acc', 'test acc'])
    plt.title('accuracy')
    plt.savefig(os.path.join(save_dir, EXPT_NUMBER + "_accuracy_image_.png"))

    plt.close()

    return

# csvに変数を記録

def writeTxt(file_name, myCheck):
    header = ['Date', 'Dataset', 'BATCH_SIZE', 'WEIGHT_DECAY', 'LEARNING_RATE', 'EPOCH', 'RESIZE', 'DEVICE', 'Accuracy_Image_Name', 'Loss_Image_Name', 'Log_File_Name']
    dt = datetime.datetime.now()
    Date_time = dt.strftime('%Y-%m-%d %H:%M:%S')
    date = dt.strftime('%Y%m%d')

    if not myCheck:   # notを付与することで、Falseの場合に実行（真(True)でない）
        with open(file_name, "w") as f:   # ファイルを作成
            writer = csv.writer(f)
            writer.writerow(header)

        with open(file_name, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([Date_time, DATASET_PATH, BATCH_SIZE, WEIGHT_DECAY, LEARNING_RATE, EPOCH, RESIZE, DEVICE, "accuracy_image_" + date + ".png","loss_image_" + date + ".png",'cnn_' + date + ".log"])

    else: # Trueの場合
        with open(file_name, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([Date_time, DATASET_PATH, BATCH_SIZE, WEIGHT_DECAY, LEARNING_RATE, EPOCH, RESIZE, DEVICE, "accuracy_image_" + date + ".png","loss_image_" + date + ".png",'cnn_' + date + ".log"])


def main():
    #start
    print("start")

    # make result dir
    os.makedirs(result_dir_path, exist_ok=True)

    #load Dataset
    train_dataset = MyDataset(DATASET_PATH + "/train", (RESIZE[0], RESIZE[1]))    #画像のリサイズはいくらにするか？　これは学習とテストに影響を与える
    test_dataset = MyDataset(DATASET_PATH + "/test", (RESIZE[0], RESIZE[1]))  #元画像に近い形にリサイズする　小さくする必要ない

    #load Dataloader
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, drop_last=True
    )

    test_dataloader = data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, drop_last=True
    )

    #loop of epoch

    device = torch.device(DEVICE)
    net = CNNs()
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)


    train_loss_value=[]      #trainingのlossを保持するlist
    train_acc_value=[]       #trainingのaccuracyを保持するlist
    test_loss_value=[]       #testのlossを保持するlist
    test_acc_value=[]        #testのaccuracyを保持するlist


    for epoch in range(1, EPOCH+1):
        dt_now = datetime.datetime.now()
        epoch_time = dt_now.strftime('%Y-%m-%d %H:%M:%S')
        path = result_dir_path + "/" + EXPT_NUMBER + '.log'
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

        # validation(パラメータ更新なし)
        # train_loss_value, train_acc_value = log_observe(train_dataloader, "train")

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
                sum_loss*BATCH_SIZE/len(train_dataloader.dataset), float(sum_correct/sum_total)), file=f)  #lossとaccuracy出力
            train_loss_value.append(sum_loss*BATCH_SIZE/len(train_dataloader.dataset))  #traindataのlossをグラフ描画のためにlistに保持
            train_acc_value.append(float(sum_correct/sum_total))   #traindataのaccuracyをグラフ描画のためにlistに保持


        #test dataを使ってテストをする
        # test_loss_value, test_acc_value = log_observe(test_dataloader, "test")

        sum_loss = 0.0
        sum_correct = 0
        sum_total = 0


        for (inputs, labels) in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            #labels = labels.view(-1, 1)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            sum_loss += loss.item()
            _, predicted = outputs.max(1)
            sum_total += labels.size(0)
            sum_correct += (predicted == labels).sum().item()


        with open(path, 'a') as f:

            print("test  mean loss={}, accuracy={}".format(
                sum_loss*BATCH_SIZE/len(test_dataloader.dataset), float(sum_correct/sum_total)), file=f)
            test_loss_value.append(sum_loss*BATCH_SIZE/len(test_dataloader.dataset))
            test_acc_value.append(float(sum_correct/sum_total))


    if EPOCH >= 10:
        # plot
        plot_loss_acc(train_loss_value, test_loss_value, train_acc_value, test_acc_value)

        # parameter
        file_name = EXPT_NUMBER + '_variable.csv'                             # 検索ファイル名を設定
        file_path = os.path.join(result_dir_path, file_name)
        myCheck = os.path.isfile(file_path)
        writeTxt(file_path, myCheck)


    print("finish")


    return

if __name__ == "__main__":
    main()