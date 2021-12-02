import matplotlib.pyplot as plt
import numpy as np
from parameters import *
import os
import torch

def show_dataset(dataset, parameter):
    classes = ['EyesClosed', 'Anesthetized']

    plt.figure(figsize=(15, 4))
    for i in range(10):
        ax = plt.subplot(2, 10, i + 1)
        image, label = dataset[i]
        img = (np.transpose(image.numpy(), (1, 2, 0)) + 1)/2
        plt.imshow(img)
        ax.set_title(classes[label])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, 10, i + 11)
        image, label = dataset[-i-1]
        img = (np.transpose(image.numpy(), (1, 2, 0)) + 1)/2
        plt.imshow(img)
        ax.set_title(classes[label])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig(os.path.join(parameter.RESULT_DIR_PATH, parameter.EXPT_NUMBER + "dataset_image.png"))

    plt.close()

    return


def show_images_labels(loader, classes, net, device, parameter):

    # データローダーから最初の1セットを取得する
    for images, labels in loader:
        break
    # 表示数は50個とバッチサイズのうち小さい方
    n_size = min(len(images), 50)

    if net is not None:
    # デバイスの割り当て
        inputs = images.to(device)
        labels = labels.to(device)

        # 予測計算
        outputs = net(inputs)
        predicted = torch.max(outputs,1)[1]
        #images = images.to('cpu')

    # 最初のn_size個の表示
    plt.figure(figsize=(20, 15))
    for i in range(n_size):
        ax = plt.subplot(5, 10, i + 1)
        label_name = classes[labels[i]]
        # netがNoneでない場合は、予測結果もタイトルに表示する
        if net is not None:
            predicted_name = classes[predicted[i]]
            # 正解かどうかで色分けをする
            if label_name == predicted_name:
                c = 'k'
            else:
                c = 'b'
            ax.set_title(label_name + ':' + predicted_name, c=c, fontsize=20)
        # netがNoneの場合は、正解ラベルのみ表示
        else:
            ax.set_title(label_name, fontsize=20)
        # TensorをNumPyに変換
        image_np = images[i].numpy().copy()
        # 軸の順番変更 (channel, row, column) -> (row, column, channel)
        img = np.transpose(image_np, (1, 2, 0))
        # 値の範囲を[-1, 1] -> [0, 1]に戻す
        img = (img + 1)/2
        # 結果表示
        plt.imshow(img)
        ax.set_axis_off()
    plt.savefig(os.path.join(parameter.RESULT_DIR_PATH, parameter.EXPT_NUMBER + "predict_image.png"))

    plt.close()