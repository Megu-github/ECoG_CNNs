import os
import matplotlib.pyplot as plt
import numpy as np

from parameters import *




'''
# 結果を保存するpathを生成
dirname = os.path.dirname(os.path.abspath(__file__))
result_dir_path = dirname + '/Result/' + EXPT_NUMBER
'''
#グラフ描写の関数
def plot_loss_acc(train_loss_value, train_acc_value, val_loss_value, val_acc_value, fold):   # , train_acc_value, test_acc_value
    plt.figure(figsize=(6,6))      #グラフ描画用

    epoch_plot = EPOCH * 5


    #以下グラフ描画
    plt.plot(range(epoch_plot) , train_loss_value)
    plt.plot(range(epoch_plot), val_loss_value, c='#00ff00')
    plt.xlim(0, epoch_plot)
    plt.ylim(0, 1.0)
    plt.xlabel('EPOCH')
    plt.ylabel('LOSS')
    plt.legend(['train loss', 'val loss'])
    plt.title('loss')
    plt.savefig(os.path.join(RESULT_DIR_PATH, EXPT_NUMBER + "_own_loss_image" + str(fold) + ".png"))
    plt.close()

    plt.plot(range(epoch_plot) , train_acc_value)
    plt.plot(range(epoch_plot), val_acc_value, c='#00ff00')
    plt.xlim(0, epoch_plot)
    plt.ylim(0, 1)
    plt.xlabel('EPOCH')
    plt.ylabel('ACCURACY')
    plt.legend(['train acc', 'val acc'])
    plt.title('accuracy')
    plt.savefig(os.path.join(RESULT_DIR_PATH, EXPT_NUMBER + "_own_accuracy_image" + str(fold) + ".png"))

    plt.close()

    return

# 学習ログ解析

def evaluate_history(history, fold):
    #損失と精度の確認
    print(f'初期状態: 損失: {history[0,3]:.5f} 精度: {history[0,4]:.5f}')
    print(f'最終状態: 損失: {history[-1,3]:.5f} 精度: {history[-1,4]:.5f}' )

    num_epochs = len(history)
    unit = num_epochs / 10

    # 学習曲線の表示 (損失)
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,1], 'b', label='訓練')
    plt.plot(history[:,0], history[:,3], 'k', label='検証')
    plt.xticks(np.arange(0,num_epochs+1, unit))
    plt.xlabel('繰り返し回数')
    plt.ylabel('損失')
    plt.title('学習曲線(損失)')
    plt.legend()
    plt.savefig(os.path.join(RESULT_DIR_PATH, EXPT_NUMBER + "_book_loss_image" + str(fold) + ".png"))
    plt.close()

    # 学習曲線の表示 (精度)
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,2], 'b', label='訓練')
    plt.plot(history[:,0], history[:,4], 'k', label='検証')
    plt.xticks(np.arange(0,num_epochs+1,unit))
    plt.xlabel('繰り返し回数')
    plt.ylabel('精度')
    plt.title('学習曲線(精度)')
    plt.legend()
    plt.savefig(os.path.join(RESULT_DIR_PATH, EXPT_NUMBER + "_book_accuracy_image" + str(fold) + ".png"))
    plt.close()