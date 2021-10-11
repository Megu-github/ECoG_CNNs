import os
import matplotlib.pyplot as plt

from parameters import *




'''
# 結果を保存するpathを生成
dirname = os.path.dirname(os.path.abspath(__file__))
result_dir_path = dirname + '/Result/' + TRAIN_EXPT_NUMBER
'''

#グラフ描写の関数
def plot_loss_acc(train_loss_value, train_acc_value):   # , train_acc_value, test_acc_value
    plt.figure(figsize=(6,6))      #グラフ描画用

    last_plot = EPOCH * 5 + 1

    #以下グラフ描画
    plt.plot(range(1, last_plot), train_loss_value)
    #plt.plot(range(EPOCH), test_loss_value, c='#00ff00')
    plt.xlim(0, EPOCH * 5)
    plt.ylim(0, 1.5)
    plt.xlabel('EPOCH')
    plt.ylabel('LOSS')
    plt.legend(['train loss', 'test loss'])
    plt.title('loss')
    plt.savefig(os.path.join(result_dir_path, EXPT_NUMBER + "_loss_image.png"))



    plt.clf()

    plt.plot(range(1, last_plot), train_acc_value)
    #plt.plot(range(EPOCH), test_acc_value, c='#00ff00')
    plt.xlim(0, EPOCH * 5)
    plt.ylim(0, 1)
    plt.xlabel('EPOCH')
    plt.ylabel('ACCURACY')
    plt.legend(['train acc', 'test acc'])
    plt.title('accuracy')
    plt.savefig(os.path.join(result_dir_path, EXPT_NUMBER + "_accuracy_image.png"))



    plt.close()

    return