import torch
import torch.nn as nn
import torch.nn.functional as F
from uw_gym.networks.resnet import resnet18


class DQN(nn.Module):
    def __init__(self, inputs, outputs, device='cpu'):
        super(DQN, self).__init__()
        resnet_embedded_dim = 512
        self.resnet_encoder = resnet18(pretrained=False, in_channels=inputs)

        self.head = nn.Linear(resnet_embedded_dim, outputs)
        self.device = device

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = torch.as_tensor(x, device=self.device)
        embedded_features = self.resnet_encoder(x)
        return self.head(embedded_features)


if __name__ == '__main__':
    net = DQN(inputs=1, outputs=5)
    inputTensor = torch.randn(1, 1, 128, 128)
    output = net(inputTensor)
    print(output.shape)

