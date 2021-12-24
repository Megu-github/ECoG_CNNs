import datetime
import os
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
from torchvision.utils import save_image

import visualize_pytorch.src.smoothGrad as smoothGrad
import model
import dataset
import parameters


def test_smoothgrad(parameter):
    file_path = parameter.RESULT_DIR_PATH + "/" + \
        parameter.EXPT_NUMBER + '.log'
    with open(file_path, 'a') as f:
        print("### TEST RESULT ###", file=f)

    target_epoch = parameter.TEST_EPOCH

    test_dataset = dataset.get_dataset(
        dataset_class=parameter.DATASET_CLASS,
        dataset_dir=os.path.join(parameter.TRAIN_DATASET_PATH, 'test'),
        parameter=parameter,
        resize=parameter.RESIZE,
        # save_dir=parameter.RESULT_DIR_PATH,
        save_dir=None,
    )
    test_dataloader = data.DataLoader(
        test_dataset,
        batch_size=parameter.TEST_BATCH_SIZE,
        shuffle=True,  # False,
        num_workers=2,
        drop_last=True,
    )
    device = torch.device(parameter.DEVICE)
    criterion = nn.CrossEntropyLoss()

    accs, losses = [], []

    for fold in range(parameter.N_SPLITS):
        # network setting
        net = model.CNNs(
            use_Barch_Norm=parameter.USE_BATCH_NORM,
            use_dropout=parameter.USE_DROPOUT,
            p_dropout1=parameter.P_DROPOUT1,
            p_dropout2=parameter.P_DROPOUT2,
        ).to(device)
        model_path = parameter.RESULT_DIR_PATH + \
            '/model_fold' + str(fold + 1) + \
            '_epoch' + str(target_epoch) + '.pth'
        net.load_state_dict(torch.load(model_path))
        for param in net.parameters():
            param.requires_grad = False  # 勾配を計算しない

        # test
        loss, acc = test(
            net=net,
            criterion=criterion,
            test_dataloader=test_dataloader,
            device=device
        )
        losses.append(loss)
        accs.append(acc)

        # log
        epoch_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(file_path, 'a') as f:
            print("fold: ", fold + 1, file=f)
            print(model_path, 'was loaded', file=f)
            print('test', epoch_time, file=f)
            print("test  mean loss={}, accuracy={}".format(loss, acc), file=f)
        print("test finish")

        # smooth grad
        make_smoothgrad(net, test_dataloader, parameter)

    with open(file_path, 'a') as f:
        print("losses: ", losses, "\naccs: ", accs, file=f)
        print("all mean loss: {:4f}".format(mean(losses)), file=f)
        print("all mean acc: {:4f}".format(mean(accs)), file=f)


def test(net, criterion, test_dataloader, device):
    """
    testのみ
    """
    #test dataを使ってテストをする
    # test_loss_value, test_acc_value = log_observe(test_dataloader, "test")

    net.eval()
    test_loss = 0.0
    test_acc = 0
    used_datasize = 0

    for inputs, labels in test_dataloader:
        used_datasize += len(labels)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        predicted = torch.max(outputs, 1)[1]
        test_acc += float((predicted == labels).sum())

    avg_test_loss = test_loss / used_datasize
    avg_test_acc = test_acc / used_datasize

    return avg_test_loss, avg_test_acc


def make_smoothgrad(net, test_dataloader, parameter):
    cnt = 0
    for (images, labels) in test_dataloader:
        # for idx in range(len(images)):
        for img_raw, label in zip(images, labels):
            cnt += 1

            if cnt < 300:
                # img_raw, label = images[idx], labels[idx]
                img = img_raw.unsqueeze(0)
                fname_common = parameter.RESULT_DIR_PATH + "/" + \
                    parameter.classes[label]
                fname_original = fname_common + \
                    '/original/raw_image' + str(cnt) + ".png"
                fname_smooth_grad = fname_common + \
                    '/smooth_grad/smoothgrad' + str(cnt) + ".png"

                os.makedirs(fname_common + '/original', exist_ok=True)
                os.makedirs(fname_common + '/smooth_grad', exist_ok=True)

                # smooth grad
                smooth_grad = smoothGrad.SmoothGrad(
                    net,
                    use_cuda=True,
                    stdev_spread=0.2,
                    n_samples=20,
                )
                smooth_cam, _ = smooth_grad(img)

                # plot
                save_image(img_raw, fname_original)
                cv2.imwrite(fname_smooth_grad,
                            smoothGrad.show_as_gray_image(smooth_cam))
                syn_smoothgrad(fname_common, cnt)

    for idx, label in enumerate(parameter.classes):
        fname_common = parameter.RESULT_DIR_PATH + "/" + \
            parameter.classes[idx]
        avarage_smoothgrad(fname_common + '/smooth_grad')

    subtract_average_image(fname_common=parameter.RESULT_DIR_PATH)

    return


def avarage_smoothgrad(img_dir):
    input_dir = Path(img_dir)  # 画像があるディレクトリ
    print("averaged directory: ", input_dir)
    os.makedirs(img_dir, exist_ok=True)

    imgs = []
    img_paths = get_img_paths(input_dir)
    if len(img_paths) > 0:
        for path in img_paths:
            img = cv2.imread(str(path))
            imgs.append(img)

        imgs = np.array(imgs)
        print(imgs.shape)
        assert imgs.ndim == 4, "error happen"

        mean_img = imgs.mean(axis=0)
        cv2.imwrite(img_dir + "/mean.png", mean_img)
    else:
        print("no image files.")
    return


def get_img_paths(img_dir):
    """
    画像のパスを取得する。
    """
    IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
    img_paths = [p for p in img_dir.iterdir() if p.suffix in IMG_EXTENSIONS]

    return img_paths


def syn_smoothgrad(fname_common, cnt):
    os.makedirs(fname_common + '/synth/', exist_ok=True)

    fname_original = fname_common + '/original/raw_image' +str(cnt) + ".png"
    fname_smooth_grad = fname_common + '/smooth_grad/smoothgrad' +str(
        cnt) + ".png"
    fname_syn = fname_common + "/synth/syn_image" +str(cnt) + ".png"

    src1 = cv2.imread(fname_original)
    src2 = cv2.imread(fname_smooth_grad)
    dst = cv2.addWeighted(src1, 0.5, src2, 0.5, 0)
    cv2.imwrite(fname_syn, dst)
    return


def subtract_average_image(fname_common):
    """
    Reference:
    https://note.nkmk.me/python-opencv-numpy-image-difference/
    """
    fname_original = fname_common + '/Anesthetized/smooth_grad/mean.png'
    fname_smooth_grad = fname_common + '/EyesClosed/smooth_grad/mean.png'
    img_original = cv2.imread(fname_original)
    img_smooth_grad = cv2.imread(fname_smooth_grad)

    assert img_original.shape == img_smooth_grad.shape, \
        "different shape images"

    img_diff = img_original.astype(int) - img_smooth_grad.astype(int)
    img_diff_abs = np.abs(img_diff)
    img_diff_norm = img_diff_abs / img_diff_abs.max() * 255
    img_diff_center = np.floor_divide(img_diff, 2) + 128
    img_diff_center_norm = img_diff / np.abs(img_diff).max() * 127.5 + 127.5

    cv2.imwrite(fname_common + "/subtracted_mean_normal.png", img_diff)
    cv2.imwrite(fname_common + "/subtracted_mean_abs.png", img_diff_abs)
    cv2.imwrite(fname_common + "/subtracted_mean_norm.png", img_diff_norm)
    cv2.imwrite(fname_common + "/subtracted_mean_center.png", img_diff_center)
    cv2.imwrite(fname_common + "/subtracted_mean_center_norm.png",
                img_diff_center_norm)

    return


if __name__ == "__main__":
    test_smoothgrad(parameter=parameters.Parameters1)
