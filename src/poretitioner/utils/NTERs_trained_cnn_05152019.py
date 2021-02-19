import math as m

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        self.O_1 = 17
        self.O_2 = 18
        self.O_3 = 32
        self.O_4 = 37

        self.K_1 = 3
        self.K_2 = 1
        self.K_3 = 4
        self.K_4 = 2

        self.KP_1 = 4
        self.KP_2 = 4
        self.KP_3 = 1
        self.KP_4 = 1

        reshape = 141

        self.conv_linear_out = int(
            m.floor(
                (
                    m.floor(
                        (
                            m.floor(
                                (
                                    m.floor(
                                        (
                                            m.floor((reshape - self.K_1 + 1) / self.KP_1)
                                            - self.K_2
                                            + 1
                                        )
                                        / self.KP_2
                                    )
                                    - self.K_3
                                    + 1
                                )
                                / self.KP_3
                            )
                            - self.K_4
                            + 1
                        )
                        / self.KP_4
                    )
                    ** 2
                )
                * self.O_4
            )
        )

        self.FN_1 = 148

        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.O_1, self.K_1), nn.ReLU(), nn.MaxPool2d(self.KP_1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.O_1, self.O_2, self.K_2), nn.ReLU(), nn.MaxPool2d(self.KP_2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.O_2, self.O_3, self.K_3), nn.ReLU(), nn.MaxPool2d(self.KP_3)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.O_3, self.O_4, self.K_4), nn.ReLU(), nn.MaxPool2d(self.KP_4)
        )
        self.fc1 = nn.Linear(self.conv_linear_out, self.FN_1, nn.Dropout(0.2))
        self.fc2 = nn.Linear(self.FN_1, 10)

    def forward(self, x):
        x = x.float()
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.view(len(x), -1)
        x = F.logsigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


def load_cnn(state_dict_path, device="cpu"):
    cnn = CNN()
    state_dict = torch.load(state_dict_path, map_location=torch.device(device))
    cnn.load_state_dict(state_dict, strict=True)
    cnn.eval()
    return cnn
