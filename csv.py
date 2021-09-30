import datetime
import csv
import datetime


# グローバル変数
BATCH_SIZE = 20
WEIGHT_DECAY = 0.005
LEARNING_RATE = 0.0001
EPOCH = 50
RESIZE = [224, 224]
DEVICE = "cuda" # サーバー上なら"cuda"

DATASET_PATH = '/home/megu/CNN_Dataset/MK12_expt.4' # セーバーにDATASETをコピーして、そのpathを書く
EXPT_NUMBER = 'MK12_expt.4_2'

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
