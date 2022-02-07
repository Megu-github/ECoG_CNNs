import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

header_name = ('epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc')
header_name_fold = ('fold1', 'fold2', 'fold3', 'fold4', 'fold5')

def road_csv(reference_dir_path):

    df_fold1 = pd.read_csv(reference_dir_path + '/history_fold1.csv', names=header_name)
    df_fold2 = pd.read_csv(reference_dir_path + '/history_fold2.csv', names=header_name)
    df_fold3 = pd.read_csv(reference_dir_path + '/history_fold3.csv', names=header_name)
    df_fold4 = pd.read_csv(reference_dir_path + '/history_fold4.csv', names=header_name)
    df_fold5 = pd.read_csv(reference_dir_path + '/history_fold5.csv', names=header_name)
    
    return df_fold1, df_fold2, df_fold3, df_fold4, df_fold5


def df_np_value(train_or_val, loss_or_acc, dir_path):
    df_fold1, df_fold2, df_fold3, df_fold4, df_fold5 = road_csv(dir_path) #  
    
    df = pd.concat([df_fold1[train_or_val + '_' + loss_or_acc], 
                    df_fold2[train_or_val + '_' + loss_or_acc],
                    df_fold3[train_or_val + '_' + loss_or_acc],
                    df_fold4[train_or_val + '_' + loss_or_acc],
                    df_fold5[train_or_val + '_' + loss_or_acc]
                    ], axis=1,)
    df.columns = header_name_fold

    df_mean = df.mean(axis=1)
    df_std = df.std(axis=1)

    np_mean = df_mean.values
    np_std = df_std.values

    center = np_mean
    high = np_mean + np_std
    low = np_mean - np_std

    return center, high, low




def make_graph_mean_std(dir_path):

    epoch = np.arange(1, 101)

    train_loss_center, train_loss_high, train_loss_low = df_np_value('train', 'loss', dir_path)
    val_loss_center, val_loss_high, val_loss_low = df_np_value('val', 'loss', dir_path)

    plt.plot(epoch, train_loss_center, color='blue', marker='o', markersize=2, label='train loss mean')
    plt.fill_between(epoch, train_loss_high, train_loss_low, alpha=0.15, color='blue')
    plt.plot(epoch, val_loss_center, color='green', linestyle='--', marker='o', markersize=2, label='val loss mean')
    plt.fill_between(epoch, val_loss_high, val_loss_low, alpha=0.15, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    plt.savefig(dir_path + '/mean_std_loss.png')

    plt.close()


    train_acc_center, train_acc_high, train_acc_low = df_np_value('train', 'acc', dir_path)
    val_acc_center, val_acc_high, val_acc_low = df_np_value('val', 'acc', dir_path)

    plt.plot(epoch, train_acc_center, color='blue', marker='o', markersize=2, label='train acc mean')
    plt.fill_between(epoch, train_acc_high, train_acc_low, alpha=0.15, color='blue')
    plt.plot(epoch, val_acc_center, color='green', linestyle='--', marker='o', markersize=2, label='val acc mean')
    plt.fill_between(epoch, val_acc_high, val_acc_low, alpha=0.15, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.savefig(dir_path + '/mean_std_acc.png')

    plt.close()

    return



if __name__ == "__main__":
    make_graph_mean_std('/home/megu/ECoG_CNNs/Result/2022-01-19_Anesthesia_PF_Dropout')
