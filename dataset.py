from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import graph


class MyDataset(data.Dataset):
    def __init__(self, dir_path, input_size):
        super().__init__()

        self.dir_path = dir_path
        self.input_size = input_size

        self.image_paths = [
            str(p) for p in Path(self.dir_path).glob("**/*.png")
        ]
        self.len = len(self.image_paths)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        p = self.image_paths[index]

        # 入力
        image = Image.open(p)
        image = image.resize(self.input_size)
        image = np.array(image)
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).float()

        # ラベル (0: EyesClosed, 1: Anesthetized)
        label = p.split("/")[6]     #ここはpath名が変わると変更することになるので、いつかうまい具合に書き換える
        label = 1 if label == "Anesthetized" else 0

        return image, label


def get_dataset(dataset_class,
                dataset_dir,
                parameter,
                save_dir=None,
                resize=None):

    if dataset_class == 'my_dataset':
        load_dataset = MyDataset(dataset_dir, (resize, resize))

    elif dataset_class == 'image_folder':
        trainval_transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
        ])
        load_dataset = datasets.ImageFolder(
            root=dataset_dir,
            transform=trainval_transform,
        )

    elif dataset_class == 'pytorch_book':
        # load_dataset = pytorch_book(dataset_dir)
        pass
    else:
        load_dataset = None
        raise Exception('Wrong dataset_class: ', dataset_class)

    if save_dir is not None:
        graph.plot_dataset(dataset=load_dataset, parameter=parameter)
        graph.plot_dataset_with_label(dataset=load_dataset,
                                      parameter=parameter)

    return load_dataset


def pytorch_book(data_dir):
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    # 訓練データ用: 正規化に追加で反転とRandomErasingを実施
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    # 訓練データディレクトリと検証データディレクトリの指定
    import os
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    # 分類先クラスのリスト作成
    classes = ['EyesClosed', 'Anesthetized']

    trainval_dataset = datasets.ImageFolder(train_dir,
                        transform=train_transform)

    return trainval_dataset