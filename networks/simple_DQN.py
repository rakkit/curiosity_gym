import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, h, w, inputs, outputs, device='cpu'):
        super(DQN, self).__init__()
        scale = 1
        self.inputs = inputs
        self.outputs = outputs

        self.conv1 = nn.Conv2d(inputs, 16, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16*scale)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32*scale)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(32 * scale)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(32*scale, outputs)

        self.device = device

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = torch.as_tensor(x, device=self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        # x = self.head(x.view(x.size(0), -1))
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        # print(x.shape)
        return x


# class DQN(nn.Module):
#     def __init__(self, h, w, inputs, outputs, device='cpu'):
#         super(DQN, self).__init__()
#         scale = h*w // 2
#         self.inputs = inputs
#         self.outputs = outputs
#
#         self.conv1 = nn.Linear(inputs*h*w, 16*scale)
#         self.bn1 = nn.BatchNorm1d(16*scale)
#
#         self.conv2 = nn.Linear(16*scale, 32*scale)
#         self.bn2 = nn.BatchNorm1d(32*scale)
#
#         self.conv3 = nn.Linear(32*scale, 32*scale//2)
#         self.bn3 = nn.BatchNorm1d(32*scale//2)
#
#         self.head = nn.Linear(32*scale//2, outputs)
#         self.device = device
#
#     # Called with either one element to determine next action, or a batch
#     # during optimization. Returns tensor([[left0exp,right0exp]...]).
#     def forward(self, x):
#         x = torch.as_tensor(x, device=self.device)
#         x = x.reshape(x.shape[0], -1)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         # x = self.head(x.view(x.size(0), -1))
#
#         x = self.head(x)
#         return x


if __name__ == '__main__':
    net = DQN(128, 128, inputs=1,outputs=5)
    inputTensor = torch.randn(1, 1, 128, 128)
    output = net(inputTensor)
    print(output.shape)