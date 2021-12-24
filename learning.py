import datetime
import os

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold
from statistics import mean
from tqdm.notebook import tqdm


import model
import dataset
import graph
import parameters
import test_smoothgrad


def learning(parameter):
    # make result dir
    os.makedirs(parameter.RESULT_DIR_PATH, exist_ok=True)

    trainval_dataset = dataset.get_dataset(
        dataset_class=parameter.DATASET_CLASS,
        dataset_dir=os.path.join(parameter.TRAIN_DATASET_PATH, 'train'),
        parameter=parameter,
        resize=parameter.RESIZE,
        # save_dir=parameter.RESULT_DIR_PATH,
        save_dir=None,
    )
    device = torch.device(parameter.DEVICE)
    criterion = nn.CrossEntropyLoss()

    ## cross validation
    nets, accs, losses = [], [], []
    splits = KFold(n_splits=parameter.N_SPLITS, shuffle=True,
                    random_state=496)

    for fold, (train_idx,
            val_idx) in enumerate(splits.split(trainval_dataset)):
        log_path = parameter.RESULT_DIR_PATH + "/" + \
            parameter.EXPT_NUMBER + '.log'
        with open(log_path, 'a') as f:
            print("model name: model", fold + 1, file=f)

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_dataloader = torch.utils.data.DataLoader(
            trainval_dataset,
            batch_size=parameter.TRAIN_BATCH_SIZE,
            sampler=train_sampler,
            num_workers=2,
            drop_last=True
        )
        val_dataloader = torch.utils.data.DataLoader(
            trainval_dataset,
            batch_size=parameter.TRAIN_BATCH_SIZE,
            sampler=val_sampler,
            num_workers=2,
            drop_last=True
        )
        net = model.CNNs(
            use_Barch_Norm=parameter.USE_BATCH_NORM,
            use_dropout=parameter.USE_DROPOUT,
            p_dropout1=parameter.P_DROPOUT1,
            p_dropout2=parameter.P_DROPOUT2,
        ).to(device)
        optimizer = get_optimizer(
            optimizer_class=parameter.OPTIMIZER_CLASS,
            net=net,
            parameter=parameter,
        )
        net, loss, acc, history = fit(
            net=net,
            optimizer=optimizer,
            criterion=criterion,
            epoch_size=parameter.EPOCH,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            decice=device,
            log_path=log_path,
            model_save_path=parameter.RESULT_DIR_PATH + '/model_fold' +
            str(fold + 1)
        )
        nets.append(net)
        losses.append(loss)
        accs.append(float(acc))

        graph.plot_loss_acc(
            train_loss_value=history[:, 1],
            train_acc_value=history[:, 2],
            val_loss_value=history[:, 3],
            val_acc_value=history[:, 4],
            fold=fold,
            parameter=parameter
        )
        graph.evaluate_history(history, fold, parameter=parameter)
        np.savetxt(parameter.RESULT_DIR_PATH + '/history_fold' +
                   str(fold + 1) + '.csv',
                   history,
                   delimiter=',')

        model_path = parameter.RESULT_DIR_PATH  + \
        '/model_fold' + str(fold+1) + '.pth'
        torch.save(net.state_dict(), model_path)

    with open(log_path, 'a') as f:
        print("all mean loss: {:4f}".format(mean(losses)), file=f)
        print("all mean acc: {:4f}".format(mean(accs)), file=f)

    #show_images_labels(test_dataloader, classes, net, device)

    return


def fit(net, optimizer, criterion, epoch_size, train_dataloader,
        val_dataloader, device, log_path, model_save_path):
    save_every = 1          # modelの保存を何EPOCHごとに行うか。 model.pthはテキストデータなので多めでおっけ。
    history = np.zeros((0,5))
    with open(log_path, 'a') as f:
        print(net, optimizer, file=f)

    for epoch in range(epoch_size):
        train_loss = 0
        train_acc = 0

        # training
        net.train()
        used_datasize = 0

        for inputs, labels in tqdm(train_dataloader):
            optimizer.zero_grad()

            used_datasize += len(labels)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = torch.max(outputs, 1)[1]
            train_acc += float((predicted == labels).sum())

        avg_train_loss = train_loss / used_datasize
        avg_train_acc = train_acc / used_datasize


        # validation
        avg_val_loss, avg_val_acc = test_smoothgrad.test(
            net=net,
            criterion=criterion,
            test_dataloader=val_dataloader,
            device=device,
        )

        # log
        epoch_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(log_path, 'a') as f:
            print(epoch_time, file=f)
            print(
                f'Epoch [{(epoch+1)}/{epoch_size}], loss: {avg_train_loss:.5f} acc: {avg_train_acc:.5f} val_loss: {avg_val_loss:.5f}, val_acc: {avg_val_acc:.5f}',
                file=f)

        # save
        item = np.array([
            epoch + 1,
            avg_train_loss,
            avg_train_acc,
            avg_val_loss,
            avg_val_acc,
        ])
        history = np.vstack((history, item))
        if (epoch + 1) % save_every == 0:
            model_path = model_save_path + '_epoch' + str(epoch + 1) + '.pth'
            torch.save(net.state_dict(), model_path)
    return net, history[-1, 3], history[-1, 4], history

def get_optimizer(optimizer_class, net, parameter):
    if optimizer_class == 'adam':
        optimizer = optim.Adam(
            net.parameters(),
            lr=parameter.LEARNING_RATE,
        )
    elif optimizer_class == 'sgd':
        optimizer = optim.SGD(
            net.parameters(),
            lr=parameter.LEARNING_RATE,
            momentum=0.9,
            weight_decay=parameter.WEIGHT_DECAY,
        )
    else:
        raise Exception('Wrong Optimizer Class', optimizer_class)

    return optimizer


if __name__ == "__main__":
    learning(parameter=parameters.Parameters1)
