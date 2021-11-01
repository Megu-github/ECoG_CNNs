import os
import matplotlib.pyplot as plt

from parameters import *




'''
# 結果を保存するpathを生成
dirname = os.path.dirname(os.path.abspath(__file__))
result_dir_path = dirname + '/Result/' + EXPT_NUMBER
'''
#グラフ描写の関数
def plot_loss_acc(train_loss_value, train_acc_value, val_loss_value, val_acc_value):   # , train_acc_value, test_acc_value
    plt.figure(figsize=(6,6))      #グラフ描画用

    epoch_plot = EPOCH * 5


    #以下グラフ描画
    plt.plot(range(epoch_plot) , train_loss_value)
    plt.plot(range(epoch_plot), val_loss_value, c='#00ff00')
    plt.xlim(0, epoch_plot)
    plt.ylim(0, 1.5)
    plt.xlabel('EPOCH')
    plt.ylabel('LOSS')
    plt.legend(['train loss', 'val loss'])
    plt.title('loss')
    plt.savefig(os.path.join(RESULT_DIR_PATH, EXPT_NUMBER + "_loss_image.png"))
    plt.clf()

    plt.plot(range(epoch_plot) , train_acc_value)
    plt.plot(range(epoch_plot), val_acc_value, c='#00ff00')
    plt.xlim(0, epoch_plot)
    plt.ylim(0, 1)
    plt.xlabel('EPOCH')
    plt.ylabel('ACCURACY')
    plt.legend(['train acc', 'val acc'])
    plt.title('accuracy')
    plt.savefig(os.path.join(RESULT_DIR_PATH, EXPT_NUMBER + "_accuracy_image.png"))

    plt.close()

    return