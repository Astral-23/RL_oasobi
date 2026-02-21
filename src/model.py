import torch
import torch.nn as nn
import torch.functional as F


class AirRaidModel(nn.Module):
    def __init__(self, window_size=4):
        super().__init__()
        self.window_size = window_size

        self.conv1 = nn.Conv2d(self.window_size, 32, kernel_size=8, stride=4)
        self.ln1 = nn.GroupNorm(1, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.ln2 = nn.GroupNorm(1, 64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1)
        self.ln3 = nn.GroupNorm(1, 64)
        
        # 追加する3つの層
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.ln4 = nn.GroupNorm(1, 128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.ln5 = nn.GroupNorm(1, 256)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=1)
        self.ln6 = nn.GroupNorm(1, 512)

        self.relu = nn.ReLU()
        self.fc1 = nn.LazyLinear(512)
        self.fc2 = nn.Linear(512, 6)

    def forward(self, x):
        x = self.relu(self.ln1(self.conv1(x)))
        x = self.relu(self.ln2(self.conv2(x)))
        x = self.relu(self.ln3(self.conv3(x)))
        
        # 追加層のフォワードパス
        x = self.relu(self.ln4(self.conv4(x)))
        x = self.relu(self.ln5(self.conv5(x)))
        x = self.relu(self.ln6(self.conv6(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
    
class AirRaidModel_papar(nn.Module):
    def __init__(self, window_size=4):
        super().__init__()
        self.conv1 = nn.Conv2d(window_size, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.LazyLinear(256) 
        self.fc2 = nn.LazyLinear(6)   
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x