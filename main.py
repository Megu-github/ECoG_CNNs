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




def main():

    test_dataset = dataset.MyDataset(TEST_DATASET_PATH + "/test", (RESIZE[0], RESIZE[1]))  #元画像に近い形にリサイズする　小さくする必要ない


    test_dataloader = data.DataLoader(
        test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False,
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
        optimizer.zero_grad()
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




    images, batches = next(iter(test_dataloader))


    img = images[0]
    plt.imsave(RESULT_DIR_PATH + '/original_image.png', img[0])

    img = img.unsqueeze(0)
    batch = batches[0]

    #print(img.size())
    #print(img[0].size())
    #print(f"Label: {batch}")
    '''
    plt.imshow(img[0].view(224,224,3))
    plt.show()
    plt.imsave('moto画像.png', img)
    '''
    smooth_grad = SmoothGrad(net, use_cuda=True, stdev_spread=0.2, n_samples=20)
    smooth_cam, _ = smooth_grad(img)
    cv2.imwrite(RESULT_DIR_PATH + "/smoothGrad.png", show_as_gray_image(smooth_cam))

    '''
    # 可視化して確認する
    fig, ax = plt.subplots()
    ax.imshow(images[0][0])
    ax.axis('off')
    ax.set_title(f'images, label={label[0]}', fontsize=20)
    plt.show()
    '''

def syn_image():


    src1 = cv2.imread(RESULT_DIR_PATH + '/original_image.png')
    src2 = cv2.imread(RESULT_DIR_PATH + "/smoothGrad.png")

    dst = cv2.addWeighted(src1, 0.5, src2, 0.5, 0)

    cv2.imwrite(RESULT_DIR_PATH + '/opencv_add_weighted.png', dst)

    dst = cv2.addWeighted(src1, 0.5, src2, 0.5, 0)

    cv2.imwrite(RESULT_DIR_PATH + '/opencv_add_weighted.png', dst)

    return

if __name__ == "__main__":
    main()
    syn_image()
