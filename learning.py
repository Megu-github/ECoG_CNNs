import datetime
import os

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
from sklearn.model_selection import KFold
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from statistics import mean
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


import model
import dataset
import graph
from parameters import Parameters1, Parameters2
import graph


def learning(parameter):

    # make result dir
    os.makedirs(parameter.RESULT_DIR_PATH, exist_ok=True)

    #load Dataset
    trainval_dataset = dataset.MyDataset(parameter.TRAIN_DATASET_PATH + "/train", (parameter.RESIZE[0], parameter.RESIZE[1]))    #画像のリサイズはいくらにするか？　これは学習とテストに影響を与える

    #trainval_dataset = dataset.pytorch_book(parameter.TRAIN_DATASET_PATH)
    '''
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    train_dir = os.path.join(parameter.TRAIN_DATASET_PATH, 'train')
    trainval_dataset = datasets.ImageFolder(train_dir,
                        transform=train_transform)
    '''

    device = torch.device(parameter.DEVICE)

    criterion = nn.CrossEntropyLoss()

    classes = ['EyesClosed', 'Anesthetized']


    ## cross val
    nets, accs, losses = [], [], []

    splits = KFold(n_splits=5, shuffle=True, random_state=26)   # random_stateの値は要検討
    for fold, (train_idx, val_idx) in enumerate(splits.split(trainval_dataset)):
        file_path = parameter.RESULT_DIR_PATH + "/" + parameter.EXPT_NUMBER + '.log'
        with open(file_path, 'a') as f:
            print("model name: model", fold + 1, file=f)

        net = model.CNNs(p_dropout1=0.25, p_dropout2=0.5, use_Barch_Norm=False).to(device)
        optimizer = optim.SGD(net.parameters(),lr=parameter.LEARNING_RATE, momentum=0.9, weight_decay=parameter.WEIGHT_DECAY)   #Adams検討

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_dataloader = torch.utils.data.DataLoader(
            trainval_dataset, batch_size=parameter.TRAIN_BATCH_SIZE,
            sampler=train_sampler, num_workers=2, drop_last=True
        )
        val_dataloader = torch.utils.data.DataLoader(
            trainval_dataset, batch_size=parameter.TRAIN_BATCH_SIZE,
            sampler=val_sampler, num_workers=2, drop_last=True
        )

        net, loss, acc, history = fit(net, optimizer, criterion, parameter.EPOCH, train_dataloader, val_dataloader, device, file_path)
        nets.append(net)
        losses.append(loss)
        accs.append(float(acc))

        #graph.plot_loss_acc(history[:,1], history[:,2], history[:,3], history[:,4], fold)
        graph.evaluate_history(history, fold, parameter=Parameters1)

        model_path = parameter.RESULT_DIR_PATH  + '/model' + str(fold+1) + '.pth'
        torch.save(net.state_dict(), model_path)


    with open(file_path, 'a') as f:
        print("oof loss: {:4f}".format(mean(losses)), file=f)
        print("oof acc: {:4f}".format(mean(accs)), file=f)



    #show_images_labels(test_dataloader, classes, net, device)

    return



def fit(net, optimizer, criterion, EPOCH, train_dataloader, val_dataloader, device, file_path):
    history = np.zeros((0,5))


    #loop of epoch
    for epoch in range(EPOCH):

        # training
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0


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


        # validation
        #予測フェーズ
        net.eval()
        used_datasize = 0

        for inputs, labels in val_dataloader:
            used_datasize += len(labels)
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            predicted = torch.max(outputs, 1)[1]

            val_acc += (predicted == labels).sum()

            avg_val_loss = val_loss / used_datasize
            avg_val_acc = val_acc / used_datasize


        dt_now = datetime.datetime.now()    #jstに設定しなおす。
        epoch_time = dt_now.strftime('%Y-%m-%d %H:%M:%S')
        #file_path = parameter.RESULT_DIR_PATH + "/" + parameter.EXPT_NUMBER + '.log'


        '''
        with open(file_path, 'a') as f:
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
           # print('Fold {}'.format(fold + 1), 'epoch', epoch, file=f)
            print(epoch_time, file=f)


            print (f'Epoch [{(epoch+1)}/{EPOCH}], loss: {avg_train_loss:.5f} acc: {avg_train_acc:.5f} val_loss: {avg_val_loss:.5f}, val_acc: {avg_val_acc:.5f}', file=f)
        item = np.array([epoch+1, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc])
        history = np.vstack((history, item))

    return net, history[-1,3], history[-1,4], history






if __name__ == "__main__":
    learning(parameter=Parameters1)
    # イメージとラベル表示