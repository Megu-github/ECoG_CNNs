import datetime
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import os


import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn

from visualize_pytorch.src.smoothGrad import *
import model
import dataset
from parameters import *





def test_smoothgrad(parameter):

    test_dataset = dataset.MyDataset(parameter.TEST_DATASET_PATH + "/test", (parameter.RESIZE[0], parameter.RESIZE[1]))  #元画像に近い形にリサイズする　小さくする必要ない

    test_dataloader = data.DataLoader(
        test_dataset, batch_size=parameter.TEST_BATCH_SIZE, shuffle=False,
        num_workers=2, drop_last=True
    )

    device = torch.device(parameter.DEVICE)
    net = model.CNNs()
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()

    model_path = parameter.RESULT_DIR_PATH + '/model.pth'
    net.load_state_dict(torch.load(model_path))     # 5回ループを回す


    for param in net.parameters():
        param.requires_grad = False  # 勾配を計算しない

    test(net, criterion, test_dataloader, device)

    make_smoothgrad(net, test_dataloader)


def test(net, criterion, test_dataloader, device, parameter):
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

    path = parameter.RESULT_DIR_PATH + "/" + parameter.EXPT_NUMBER + '.log'
    dt_now = datetime.datetime.now()
    epoch_time = dt_now.strftime('%Y-%m-%d %H:%M:%S')

    with open(path, 'a') as f:
        print('test', epoch_time, file=f)
        print("test  mean loss={}, accuracy={}".format(
            sum_loss*parameter.TEST_BATCH_SIZE/len(test_dataloader.dataset), float(sum_correct/sum_total)), file=f)
        test_loss_value.append(sum_loss*parameter.TEST_BATCH_SIZE/len(test_dataloader.dataset))
        test_acc_value.append(float(sum_correct/sum_total))

    return



def make_smoothgrad(net, test_dataloader, parameter):
    cnt = 0
    for (images, labels) in test_dataloader:
        for img_raw, label in zip(images, labels):
            cnt += 1
            # images, batches = next(iter(test_dataloader))
            # img_raw = images[idx_data]
            # label = labels[idx_data]
            img = img_raw.unsqueeze(0)
            fname_common = parameter.RESULT_DIR_PATH + "/" + parameter.EXPT_NUMBER + "/" + parameter.classes[label]
            fname_original = fname_common + '/original/raw_image' +str(cnt) + ".png"
            fname_smooth_grad = fname_common + '/smooth_grad/smoothgrad' +str(cnt) + ".png"
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
            syn_smoothgrad(fname_common, cnt)

    for label in range(2):
        avarage_smoothgrad(parameter.RESULT_DIR_PATH + "/" + parameter.EXPT_NUMBER + "/" + parameter.classes[label] + '/smooth_grad')

    return



def avarage_smoothgrad(img_dir):
    input_dir = Path(img_dir)  # 画像があるディレクトリ

    img_dir = "save_dir"
    os.makedirs(img_dir, exist_ok=True)


    imgs = []
    for path in get_img_paths(input_dir):
        # 画像を読み込む。
        img = cv2.imread(str(path))
        imgs.append(img)


    imgs = np.array(imgs)
    assert imgs.ndim == 4, "すべての画像の大きさは同じでないといけない"

    mean_img = imgs.mean(axis=0)
    cv2.imwrite(img_dir + "/mean.png", mean_img)
    return


def get_img_paths(img_dir):
    """
    画像のパスを取得する。
    """
    IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
    img_paths = [p for p in img_dir.iterdir() if p.suffix in IMG_EXTENSIONS]

    return img_paths



def syn_smoothgrad(fname_common, cnt):
    fname_original = fname_common + '/original/raw_image' +str(cnt) + ".png"
    fname_smooth_grad = fname_common + '/smooth_grad/smoothgrad' +str(cnt) + ".png"
    fname_syn = fname_common + "/synth/syn_image" +str(cnt) + ".png"
    src1 = cv2.imread(fname_original)
    src2 = cv2.imread(fname_smooth_grad)
    dst = cv2.addWeighted(src1, 0.5, src2, 0.5, 0)
    cv2.imwrite(fname_syn, dst)
    return


if __name__ == "__test__":
    test_smoothgrad()
