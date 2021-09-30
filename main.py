import os
from PIL.Image import Image
import matplotlib.pyplot as plt
import torchvision

import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn

from visualize_pytorch.src.smoothGrad import *
import model
import Dataset

classes = ('Anesthetized', 'EyesClosed')


# グローバル変数
BATCH_SIZE = 20
WEIGHT_DECAY = 0.005
LEARNING_RATE = 0.0001
EPOCH = 1
RESIZE = [224, 224]
DEVICE = "cuda" # サーバー上なら"cuda"

DATASET_PATH = '/home/megu/CNN_Dataset/MK12_expt.4' # セーバーにDATASETをコピーして、そのpathを書く
EXPT_NUMBER = 'move_test'


# 結果を保存するpathを生成
dirname = os.path.dirname(os.path.abspath(__file__))
result_dir_path = dirname + '/Result/' + EXPT_NUMBER
path = result_dir_path + "/" + EXPT_NUMBER + '.log'



def main():




    test_dataset = Dataset.MyDataset(DATASET_PATH + "/test", (RESIZE[0], RESIZE[1]))  #元画像に近い形にリサイズする　小さくする必要ない


    test_dataloader = data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, drop_last=True
    )



    device = torch.device(DEVICE)
    net = model.CNNs()
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)

    model_path = 'model.pth'
    net.load_state_dict(torch.load(model_path))


    for param in net.parameters():
        param.requires_grad = False  # 勾配を計算しない


    test_loss_value=[]       #testのlossを保持するlist
    test_acc_value=[]        #testのaccuracyを保持するlist



    for epoch in range(1, EPOCH+1):
        with open(path, 'a') as f:
            print('epoch', epoch, file=f)


        sum_loss = 0.0          #lossの合計
        sum_correct = 0         #正解率の合計
        sum_total = 0           #dataの数の合計




        #test dataを使ってテストをする
        # test_loss_value, test_acc_value = log_observe(test_dataloader, "test")

        sum_loss = 0.0
        sum_correct = 0
        sum_total = 0

        '''
        dataiter = iter(test_dataloader)
        image, label = dataiter.next()

        plt.imshow(torchvision.utils.make_grid(image))
        print('Truth:', ' '.join('%5s' % classes[label][j] for j in range(4)))


        '''
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

        images, batch = next(iter(test_dataloader))
        img = images[0]
        img = img.unsqueeze(0)

        smooth_grad = SmoothGrad(net, use_cuda=True, stdev_spread=0.2, n_samples=20)
        smooth_cam, _ = smooth_grad(img)
        cv2.imwrite("/home/megu/ECoG_CNNs/Result/move_test/smoothGrad_testimage.png", show_as_gray_image(smooth_cam))







    return

if __name__ == "__main__":
    main()