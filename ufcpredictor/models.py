import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.optim import Adam
from sklearn.metrics import f1_score

    
class FighterNet(nn.Module):
    def __init__(self, input_size, dropout_prob=0):
        super(FighterNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 127)

        # Use the global dropout probability
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.dropout3 = nn.Dropout(p=dropout_prob)
        self.dropout4 = nn.Dropout(p=dropout_prob)
        self.dropout5 = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Apply dropout after the first ReLU
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # Apply dropout after the second ReLU
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)  # Apply dropout after the third ReLU
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)  # Apply dropout after the fourth ReLU
        x = F.relu(self.fc5(x))
        x = self.dropout5(x)  # Apply dropout after the fifth ReLU

        return x


class SymmetricFightNet(nn.Module):
    
    def __init__(self, input_size, dropout_prob=0):
        super(SymmetricFightNet, self).__init__()
        self.fighter_net = FighterNet(input_size=input_size, dropout_prob=dropout_prob)

        self.fc1 = nn.Linear(256, 512)
        # self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)

        # Use the global dropout probability
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.dropout3 = nn.Dropout(p=dropout_prob)
        self.dropout4 = nn.Dropout(p=dropout_prob)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, X1, X2, odds1, odds2):
        out1 = self.fighter_net(X1)
        out2 = self.fighter_net(X2)

        # import pdb; pdb.set_trace()

        out1 = torch.cat((out1, odds1), dim=1)
        out2 = torch.cat((out2, odds2), dim=1)

        x = torch.cat((out1 - out2, out2 - out1), dim=1)

        x = self.relu(self.fc1(x))
        x = self.dropout1(x)  # Apply dropout after the first ReLU
        # x = self.relu(self.fc2(x))
        # x = self.dropout2(x)  # Apply dropout after the second ReLU
        x = self.relu(self.fc3(x))
        x = self.dropout3(x)  # Apply dropout after the third ReLU
        x = self.relu(self.fc4(x))
        x = self.dropout4(x)  # Apply dropout after the fourth ReLU
        x = self.sigmoid(self.fc5(x))
        return x