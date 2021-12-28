import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import save_image
'''
# 結果を保存するpathを生成
dirname = os.path.dirname(os.path.abspath(__file__))
result_dir_path = dirname + '/Result/' + EXPT_NUMBER
'''



#グラフ描写の関数
def plot_loss_acc(train_loss_value, train_acc_value, val_loss_value,
                    val_acc_value, fold, parameter):
    # グラフ描写の関数
    epoch_plot = parameter.EPOCH


    # Loss
    plt.figure(figsize=(6,6))
    plt.plot(range(epoch_plot) , train_loss_value)
    plt.plot(range(epoch_plot), val_loss_value, c='#00ff00')
    plt.xlim(0, epoch_plot)
    plt.xlabel('EPOCH')
    plt.ylabel('LOSS')
    plt.legend(['train loss', 'val loss'])
    plt.title('loss')
    plt.savefig(
        os.path.join(
            parameter.RESULT_DIR_PATH, parameter.EXPT_NUMBER +
            "_own_loss_image" + str(fold + 1) + ".png"))
    plt.close()

    # Accuracy
    plt.figure(figsize=(6,6))
    plt.plot(range(epoch_plot) , train_acc_value)
    plt.plot(range(epoch_plot), val_acc_value, c='#00ff00')
    plt.xlim(0, epoch_plot)
    plt.ylim(0, 1)
    plt.xlabel('EPOCH')
    plt.ylabel('ACCURACY')
    plt.legend(['train acc', 'val acc'])
    plt.title('accuracy')
    plt.savefig(
        os.path.join(parameter.RESULT_DIR_PATH, parameter.EXPT_NUMBER +
        "_own_accuracy_image" + str(fold + 1) + ".png"))

    plt.close()

    return



# 学習ログ解析

def evaluate_history(history, fold, parameter):
    #損失と精度の確認
    file_path = parameter.RESULT_DIR_PATH + "/" + \
    parameter.EXPT_NUMBER + '.log'
    with open(file_path, 'a') as f:
        print(f'初期状態: 損失: {history[0,3]:.5f} 精度: {history[0,4]:.5f}', file=f)
        print(f'最終状態: 損失: {history[-1,3]:.5f} 精度: {history[-1,4]:.5f}', file=f)

    num_epochs = len(history)
    unit = num_epochs / 10

    # 学習曲線の表示 (損失)
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,1], 'b', label='train')
    plt.plot(history[:,0], history[:,3], 'k', label='val')
    plt.xticks(np.arange(0,num_epochs+1, unit))
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.title('learning graph(Loss)')
    plt.legend()
    plt.savefig(
        os.path.join(
            parameter.RESULT_DIR_PATH, parameter.EXPT_NUMBER +
            "_book_loss_image_fold" + str(fold + 1) + ".png"))
    plt.close()

    # 学習曲線の表示 (精度)
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,2], 'b', label='train')
    plt.plot(history[:,0], history[:,4], 'k', label='val')
    plt.xticks(np.arange(0,num_epochs+1,unit))
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('learning graph(Accuracy)')
    plt.legend()
    plt.savefig(
        os.path.join(
            parameter.RESULT_DIR_PATH, parameter.EXPT_NUMBER +
            "_book_accuracy_image_fold" + str(fold + 1) + ".png"))
    plt.close()

def plot_dataset(dataset, parameter):
    print("plot dataset.")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)
    save_dir = parameter.RESULT_DIR_PATH
    os.makedirs(save_dir + '/dataset/', exist_ok=True)

    for idx, (images, labels) in enumerate(dataloader):
        fname = save_dir + '/dataset/' \
            + 'plot_dataset_img_' + str(idx) + '.png'
        save_image(images, fname)

    return


def plot_dataset_with_label(dataset, parameter):
        import torch
        print("plot dataset with label.")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)

        save_dir = parameter.RESULT_DIR_PATH
        os.makedirs(save_dir + '/dataset/', exist_ok=True)
        classes = parameter.classes
        cnt = 0

        for idx_batch, (images, labels) in enumerate(dataloader):
            for image, label in zip(images, labels):
                cnt += 1
                fname = parameter.RESULT_DIR_PATH + '/dataset/' \
                    'plot_dataset_img_' + \
                    classes[label] + str(cnt) + '.png'
                img = image.to('cpu').numpy()
                img = np.transpose(img, (1, 2, 0))
                plt.imshow(img)
                plt.title(classes[label])
                plt.savefig(fname)

        return


def plot_dataset_using_old_source(dataset, parameter):
    """
    good plotting even if using `my_dataset`
    """
    print("plot dataset using old code.")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)

    save_dir = parameter.RESULT_DIR_PATH
    os.makedirs(save_dir + '/dataset/', exist_ok=True)
    classes = parameter.classes
    cnt = 0

    for idx_batch, (images, labels) in enumerate(dataloader):
        for image, label in zip(images, labels):
            cnt += 1

            # old version
            fname = save_dir + '/dataset/plot_dataset_img_old_' + \
                classes[label] + str(cnt) + '.png'
            plt.imsave(fname, image[0])  # `[0]` is very important.

            # correct version
            fname = save_dir + '/dataset/plot_dataset_img_correct_' + \
                classes[label] + str(cnt) + '.png'
            image = image.to('cpu').numpy()
            image = np.transpose(image, (1, 2, 0))
            plt.imsave(fname, image / 255)  # convertion of RGB to 0-1 range

    return
