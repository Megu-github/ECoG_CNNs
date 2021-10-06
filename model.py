import torch.nn as nn

RESIZE = [224, 224]


# PytorchでのCNNのモデル作り
# モデルは次のサイトを参考にした　https://qiita.com/mathlive/items/8e1f）9a8467fff8dfd03c
class CNNs(nn.Module):
    def __init__(self):
        super(CNNs, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = nn.Conv2d(3,16,3)
        self.conv2 = nn.Conv2d(16,32,3)

        self.dropout1 = nn.Dropout2d(p=0.2)
        self.dropout2 = nn.Dropout2d(p=0.5)

        self.fc1 = nn.Linear(32 * int(RESIZE[0]/4 - 1.5) * int(RESIZE[1]/4 - 1.5), 32*min(int(RESIZE[0]/4 - 1.5), int(RESIZE[1]/4 - 1.5)))
        self.fc2 = nn.Linear(32*min(int(RESIZE[0]/4 - 1.5), int(RESIZE[1]/4 - 1.5)), 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        #x = self.dropout2(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x