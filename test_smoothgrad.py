import datetime
import matplotlib.pyplot as plt
import cv2


import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn

from visualize_pytorch.src.smoothGrad import *
import model
import dataset
from parameters import *




classes = ('Anesthetized', 'EyesClosed')




def test_smoothgrad():

    test_dataset = dataset.MyDataset(TEST_DATASET_PATH + "/test", (RESIZE[0], RESIZE[1]))  #元画像に近い形にリサイズする　小さくする必要ない

    test_dataloader = data.DataLoader(
        test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False,
        num_workers=2, drop_last=True
    )

    device = torch.device(DEVICE)
    net = model.CNNs()
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()

    model_path = RESULT_DIR_PATH + '/model.pth'
    net.load_state_dict(torch.load(model_path))


    for param in net.parameters():
        param.requires_grad = False  # 勾配を計算しない

    test(net, criterion, test_dataloader, device)

    make_smoothgrad(net, test_dataloader)


def test(net, criterion, test_dataloader, device):
    """
    testのみ
    """
    test_loss_value=[]       #testのlossを保持するlist
    test_acc_value=[]        #testのaccuracyを保持するlist

    #test dataを使ってテストをする
    # test_loss_value, test_acc_value = log_observe(test_dataloader, "test")

    sum_loss = 0.0
    sum_correct = 0
    sum_total = 0


    for (inputs, labels) in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        sum_loss += loss.item()
        _, predicted = outputs.max(1)
        sum_total += labels.size(0)
        sum_correct += (predicted == labels).sum().item()

    path = RESULT_DIR_PATH + "/" + EXPT_NUMBER + '.log'
    dt_now = datetime.datetime.now()
    epoch_time = dt_now.strftime('%Y-%m-%d %H:%M:%S')

    with open(path, 'a') as f:
        print('test', epoch_time, file=f)
        print("test  mean loss={}, accuracy={}".format(
            sum_loss*TEST_BATCH_SIZE/len(test_dataloader.dataset), float(sum_correct/sum_total)), file=f)
        test_loss_value.append(sum_loss*TEST_BATCH_SIZE/len(test_dataloader.dataset))
        test_acc_value.append(float(sum_correct/sum_total))

    return



def make_smoothgrad(net, test_dataloader):
    for idx_data, (images, labels) in enumerate(test_dataloader):
        # images, batches = next(iter(test_dataloader))
        img_raw = images[idx_data]
        label = labels[idx_data]
        img = img_raw.unsqueeze(0)
        fname_common = RESULT_DIR_PATH + "/" + EXPT_NUMBER + str(label)
        fname_original = fname_common + '_original_image.png'
        fname_smooth_grad = fname_common + "_smoothGrad.png"
        # print(img.size())
        # print(img[0].size())
        # print(f"Label: {batch}")
        smooth_grad = SmoothGrad(net,
                                 use_cuda=True,
                                 stdev_spread=0.2,
                                 n_samples=20)
        smooth_cam, _ = smooth_grad(img)
        # plot
        plt.imsave(fname_original, img_raw)
        cv2.imwrite(fname_smooth_grad, show_as_gray_image(smooth_cam))
        syn_smoothgrad(fname_common=fname_common)

    return



def avarage_smoothgrad():   # 保留
    return



def syn_smoothgrad(fname_common):

    fname_original = fname_common + '_original_image.png'
    fname_smooth_grad = fname_common + "_smoothGrad.png"
    src1 = cv2.imread(fname_original)
    src2 = cv2.imread(fname_smooth_grad)
    dst = cv2.addWeighted(src1, 0.5, src2, 0.5, 0)
    cv2.imwrite(fname_common + '_syn_image.png', dst)
    return


if __name__ == "__test__":
    test_smoothgrad()
