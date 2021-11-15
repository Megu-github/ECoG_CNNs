import torch.nn as nn
from parameters import RESIZE


# PytorchでのCNNのモデル作り
# モデルは次のサイトを参考にした　https://qiita.com/mathlive/items/8e1f）9a8467fff8dfd03c
class CNNs(nn.Module):
    def __init__(self):
        super(CNNs, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d(2, stride=2)



        self.conv1 = nn.Conv2d(3,16,3, padding=1)
        self.conv2 = nn.Conv2d(16,32,3, padding=1)
        self.conv3 = nn.Conv2d(32,64,3, padding=1)
        self.conv4 = nn.Conv2d(64,64,3, padding=1)
        self.conv5 = nn.Conv2d(64,32,3)


        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)


        self.dropout1 = nn.Dropout2d(p=0.25)
        self.dropout2 = nn.Dropout2d(p=0.5)

        '''
        self.fc1 = nn.Linear(32 * int(RESIZE[0]/4 - 1.5) * int(RESIZE[1]/4 - 1.5), 32*min(int(RESIZE[0]/4 - 1.5), int(RESIZE[1]/4 - 1.5)))
        self.fc2 = nn.Linear(32*min(int(RESIZE[0]/4 - 1.5), int(RESIZE[1]/4 - 1.5)), 32)
        self.fc3 = nn.Linear(32, 2)
        '''


        self.fc1 = nn.Linear(32 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 2)
        self.fc3 = nn.Linear(32, 2)



    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        #x = self.dropout1(x)

        x = self.conv4(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size()[0], -1)

        x = self.fc1(x)
        x = self.relu(x)
        #x = self.dropout2(x)
        x = self.fc2(x)
        #x = self.relu(x)
        #x = self.fc3(x)

        return x